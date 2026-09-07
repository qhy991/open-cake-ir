"""Emit Triton source from a Schedule.

The second backend. Triton owns placement -- registers, swizzles, the shape of the
tensor-core instruction it selects -- so a Schedule targeting it commits to the tiling
and the operation semantics and stops there. `verifier._verify_atom_placement` is what
keeps the two backends honest about that difference: a `tcgen05` contract must say where
its operands live, and a `triton.dot` contract must not, because the backend would not
honour the answer.

Everything emitted here comes from a declaration. Pointer arithmetic and masks come from
`access_maps`, the loop and its scheduling knobs from `tile_loops`, the reduction's tie
handling from the operation, and the host-side contract from the global buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .analysis import top_k_selection_structure
from .emit import BackendPrecondition, Emission, EmitError, require as _require
from .ir import (
    ElementwiseOp,
    LoadReuse,
    AccessIndexKind,
    AccessMap,
    Buffer,
    BufferMode,
    DType,
    IndexTieBreak,
    LoadMovement,
    MemorySpace,
    OperationKind,
    ProgramAxis,
    ReduceOp,
    ScanDirection,
    ScanOp,
    Schedule,
    TileLoop,
)
from .target import Target

_TL_DTYPE = {
    DType.BF16: "tl.bfloat16",
    DType.FP16: "tl.float16",
    DType.FP32: "tl.float32",
    DType.FP8_E4M3: "tl.float8e4nv",
    DType.INT32: "tl.int32",
}

_TORCH_DTYPE = {
    DType.BF16: "torch.bfloat16",
    DType.FP16: "torch.float16",
    DType.FP32: "torch.float32",
    DType.FP8_E4M3: "torch.float8_e4m3fn",
    DType.INT32: "torch.int32",
}

# What this backend can name, in both the places it has to name it.
SUPPORTED_DTYPES = frozenset(_TL_DTYPE) & frozenset(_TORCH_DTYPE)


@dataclass(frozen=True)
class _Reduction:
    """How one fold is written, in the two places a fold can appear.

    A loop that carries a reduction needs the identity before it starts, so a backend
    that assumed zero could only ever emit a sum -- max over an empty prefix is not zero.
    A reduction with no loop around it needs neither: it sees its whole input once, and
    writing it as an accumulation would read an accumulator nothing initialised.
    """

    identity: str
    accumulate: str
    once: str


REDUCTIONS: dict[ReduceOp, _Reduction] = {
    ReduceOp.SUM: _Reduction(
        identity="tl.zeros({shape}, tl.float32)",
        accumulate="{out} += tl.sum({src}.to(tl.float32), axis={axis})",
        once="{out} = tl.sum({src}.to(tl.float32), axis={axis})",
    ),
    ReduceOp.MAX: _Reduction(
        identity='tl.full({shape}, float("-inf"), tl.float32)',
        accumulate="{out} = tl.maximum({out}, tl.max({src}.to(tl.float32), axis={axis}))",
        once="{out} = tl.max({src}.to(tl.float32), axis={axis})",
    ),
}

SCANS: dict[ScanOp, str] = {
    ScanOp.SUM: "{out} = tl.cumsum({src}.to(tl.float32), axis={axis}, reverse={reverse})",
}


# Dispatch is scope-aware. Position restrictions share these tables with preflight;
# narrower in-loop effect contracts are checked below before any source is emitted.
OUTSIDE_LOOP_EMITTERS: dict[OperationKind, str] = {
    OperationKind.LOAD: "_emit_load",
    OperationKind.MMA: "_emit_mma",
    OperationKind.ELEMENTWISE: "_emit_elementwise",
    OperationKind.REDUCE: "_emit_reduce",
    OperationKind.SCAN: "_emit_scan",
    OperationKind.TOP_K: "_emit_top_k",
    OperationKind.INDEX_EXPAND: "_emit_index_expand",
    OperationKind.ATOMIC_RMW: "_emit_atomic_rmw",
    OperationKind.CAST: "_emit_cast",
    OperationKind.STORE: "_emit_store",
}

INSIDE_LOOP_EMITTERS: dict[OperationKind, str] = {
    OperationKind.STORE: "_emit_store",
    OperationKind.LOAD: "_emit_load",
    OperationKind.MMA: "_emit_mma",
    OperationKind.REDUCE_ARGMIN: "_emit_argmin",
    OperationKind.REDUCE: "_emit_reduce",
    OperationKind.TOP_K: "_emit_top_k",
    OperationKind.INDEX_EXPAND: "_emit_index_expand",
    OperationKind.ONLINE_SOFTMAX: "_emit_online_softmax",
    OperationKind.ELEMENTWISE: "_emit_elementwise",
    OperationKind.CAST: "_emit_cast",
}

# A kind outside this union can never be emitted, wherever it is placed, so the Compiler
# can say so before lowering. A kind inside it may still be refused for its position,
# which needs emission to discover and stays there.
SUPPORTED_OPERATION_KINDS = frozenset(OUTSIDE_LOOP_EMITTERS) | frozenset(
    INSIDE_LOOP_EMITTERS
)

_ATOMIC_RMW_CONTRACT = "triton.atomic_add.i32.relaxed.gpu"

_TRITON_MMA_CONTRACTS = frozenset(
    {
        "triton.dot.bf16_fp32",
        "triton.dot.fp32_ieee",
        "triton.dot.fp32_tf32",
        "triton.dot.fp8e4m3_block_scale_fp32",
    }
)

_TRITON_DOT_INPUT_PRECISION = {
    "triton.dot.fp32_ieee": "ieee",
    "triton.dot.fp32_tf32": "tf32",
}


def preflight(schedule: Schedule, target: Target) -> tuple[BackendPrecondition, ...]:
    """Return the constructor's backend-owned lowering requirements.

    The Compiler projects these into Findings and direct emitter users fail on the same
    list.  Keep semantic and dataflow legality in the common Verifier; these are only
    constraints imposed by this emitter's program shape and dispatch.
    """

    findings: list[BackendPrecondition] = []

    def add(condition: object, code: str, path: str, message: str) -> None:
        if not condition:
            findings.append(BackendPrecondition(code, path, message))

    add(
        target.compute_capability is not None,
        "BACKEND_TARGET_UNSUPPORTED", "target",
        "the current Triton backend emits CUDA kernels and cannot target Metal",
    )

    def arange(start: int, end: int, path: str) -> None:
        # Match the arange emitted below. Triton 3.7.1 checks end-start, so a
        # nonzero start is legal when the span is a power of two. block_type
        # limits this one-dimensional value to 2**20 elements.
        span = end - start
        add(
            0 <= start < end < (1 << 32)
            and span <= (1 << 20)
            and span & (span - 1) == 0,
            "TRITON_ARANGE_RANGE_UNSUPPORTED",
            path,
            f"Triton arange [{start}, {end}) requires 32-bit endpoints and a "
            "positive power-of-two span no larger than 1048576",
        )

    if schedule.program_map is not None:
        add(
            not schedule.program_map.persistent or target.occupancy is not None,
            "TRITON_PERSISTENT_TARGET_FACTS_MISSING", "program_map.persistent",
            "a persistent grid requires the Target's observed multiprocessor count",
        )
        for index, axis in enumerate(schedule.program_map.axes):
            if axis.is_tiled:
                arange(0, axis.tile, f"program_map.axes[{index}].tile")
    for index, loop in enumerate(schedule.tile_loops):
        arange(0, loop.tile, f"tile_loops[{index}].tile")
    for index, access in enumerate(schedule.access_maps):
        buffer = schedule.buffer(access.buffer)
        if buffer is None:
            continue  # The common Verifier owns unknown references.
        for position, component in enumerate(access.indices):
            if (
                component.source is AccessIndexKind.DIMENSION
                and component.dimension is not None
                and component.dimension < len(buffer.shape)
            ):
                start = component.offset if component.extent is None else 0
                end = buffer.shape[component.dimension] if component.extent is None else component.extent
                arange(start, end, f"access_maps[{index}].indices[{position}]")

    add(
        schedule.program_map is not None,
        "TRITON_PROGRAM_MAP_REQUIRED",
        "program_map",
        "the Triton backend requires a program map",
    )
    add(
        bool(schedule.access_maps),
        "TRITON_ACCESS_MAP_REQUIRED",
        "access_maps",
        "the Triton backend requires access maps",
    )
    add(
        len(schedule.tile_loops) <= 2,
        "TRITON_TILE_LOOP_COUNT",
        "tile_loops",
        "the Triton backend supports at most a two-deep tile-loop nest",
    )
    add(
        len(schedule.roles) == 1,
        "TRITON_ROLE_COUNT",
        "roles",
        "the Triton backend requires exactly one role",
    )

    counts = {
        kind: sum(operation.kind is kind for operation in schedule.operations)
        for kind in (OperationKind.STORE, OperationKind.REDUCE_ARGMIN)
    }
    add(
        counts[OperationKind.STORE] >= 1,
        "TRITON_STORE_COUNT",
        "operations",
        "the Triton backend requires at least one store operation",
    )
    add(
        counts[OperationKind.REDUCE_ARGMIN] <= 1,
        "TRITON_ARGMIN_COUNT",
        "operations",
        "the Triton backend supports at most one reduce_argmin operation",
    )
    for index, operation in enumerate(schedule.operations):
        if operation.kind is OperationKind.ELEMENTWISE:
            # Global arguments are pointers; _operand only names already-produced
            # values. Arithmetic does not implement access maps or memory effects.
            for edge in ("reads", "writes"):
                for position, name in enumerate(getattr(operation, edge)):
                    buffer = schedule.buffer(name)
                    if buffer is not None:
                        add(
                            buffer.space is MemorySpace.REGISTER,
                            "TRITON_ELEMENTWISE_STORAGE",
                            f"operations[{index}].{edge}[{position}]",
                            f"Triton elementwise arithmetic requires register values; "
                            f"{name!r} is {buffer.space.value}. Use explicit load/store operations.",
                        )
        if operation.kind is OperationKind.MMA:
            instruction = operation.parameters.instruction
            add(
                instruction is not None,
                "BACKEND_MMA_INSTRUCTION_REQUIRED",
                f"operations[{index}].parameters.instruction",
                "the Triton backend requires every mma to name an instruction contract",
            )
            add(
                instruction is None
                or instruction.contract not in target.instruction_contracts
                or instruction.contract in _TRITON_MMA_CONTRACTS,
                "TRITON_MMA_INSTRUCTION_UNSUPPORTED",
                f"operations[{index}].parameters.instruction.contract",
                "the Triton backend does not implement instruction contract "
                f"{None if instruction is None else instruction.contract!r}",
            )
        if operation.kind is OperationKind.LOAD:
            add(
                operation.parameters.movement is LoadMovement.GLOBAL,
                "TRITON_LOAD_MOVEMENT",
                f"operations[{index}].parameters.movement",
                "the Triton backend only emits direct global loads",
            )
        if operation.kind is OperationKind.ATOMIC_RMW:
            add(
                _ATOMIC_RMW_CONTRACT in target.instruction_contracts,
                "TRITON_ATOMIC_CONTRACT_UNSUPPORTED",
                f"operations[{index}].parameters",
                f"Target {target.target_id!r} does not admit "
                f"{_ATOMIC_RMW_CONTRACT!r}",
            )

    nested = len(schedule.tile_loops) > 1
    if nested:
        parent = schedule.loop_parent()
        add(
            len(schedule.tile_loops) == 2
            and len(parent) == 1
            and sorted(schedule.loop_depth(loop.name) for loop in schedule.tile_loops) == [0, 1],
            "TRITON_LOOP_NEST_UNSUPPORTED",
            "tile_loops",
            "the nested Triton slice requires one outer loop with one inner loop",
        )
        for index, loop in enumerate(schedule.tile_loops):
            add(
                loop.stop is None,
                "TRITON_NESTED_LOOP_STOP",
                f"tile_loops[{index}].stop",
                "nested Triton loops currently require static extents",
            )
            for option in ("flatten", "warp_specialize"):
                add(
                    not getattr(loop.range_options, option),
                    "TRITON_NESTED_LOOP_OPTION",
                    f"tile_loops[{index}].range_options.{option}",
                    f"nested Triton loops do not implement {option}=true",
                )

    for index, operation in enumerate(schedule.operations):
        chain = schedule.enclosing_loops(operation)
        admitted = (
            operation.kind in INSIDE_LOOP_EMITTERS
            if chain else operation.kind in OUTSIDE_LOOP_EMITTERS
        )
        add(
            admitted,
            "TRITON_OPERATION_POSITION",
            f"operations[{index}].kind",
            f"the Triton backend has no {operation.kind.value!r} body at this loop position",
        )
        if nested and chain:
            add(
                operation.kind not in {
                    OperationKind.REDUCE_ARGMIN,
                    OperationKind.TOP_K,
                    OperationKind.ONLINE_SOFTMAX,
                },
                "TRITON_NESTED_OPERATION_UNSUPPORTED",
                f"operations[{index}].kind",
                f"the two-deep Triton slice does not implement nested {operation.kind.value!r}",
            )
            if operation.kind is OperationKind.MMA:
                add(
                    len(operation.reads) == 2 and all(
                        any(producer.kind is OperationKind.LOAD
                            and operand in producer.writes for producer in schedule.operations)
                        for operand in operation.reads
                    ),
                    "TRITON_NESTED_MMA_OPERAND",
                    f"operations[{index}].reads",
                    "nested MMA currently requires two directly loaded operands so "
                    "the existing access-map query proves its accumulation axis",
                )
                add(
                    not any(schedule.mma_accumulates_over(operation, loop) for loop in chain[:-1]),
                    "TRITON_MMA_ANCESTOR_CARRY",
                    f"operations[{index}]",
                    "a nested MMA may accumulate only over its direct loop",
                )
        if chain and operation.kind is OperationKind.STORE and operation.writes:
            destination = schedule.buffer(operation.writes[0])
            access = schedule.access_map(operation.op_id, operation.writes[0])
            if destination is None or access is None:
                continue  # Common verification localizes missing declarations.
            coordinates = [
                component.name for component in access.indices
                if component.source in {
                    AccessIndexKind.PROGRAM, AccessIndexKind.PROGRAM_TILE,
                    AccessIndexKind.LOOP_TILE,
                }
            ]
            required = [loop.iterator for loop in chain]
            if schedule.program_map is not None:
                required.extend(axis.name for axis in schedule.program_map.axes)
            add(
                destination.mode is BufferMode.OUTPUT
                and all(component.source is not AccessIndexKind.BUFFER for component in access.indices)
                and all(coordinates.count(name) == 1 for name in required),
                "TRITON_LOOP_STORE_OWNERSHIP",
                f"operations[{index}]",
                "an in-loop Triton store requires an output buffer and affine "
                "coordinates covering every active loop and program axis exactly once",
            )

    return tuple(findings)


class _TritonEmitter:
    def __init__(
        self, schedule: Schedule, target: Target, entry_point: str | None = None
    ) -> None:
        self.schedule = schedule
        self.target = target
        self.lines: list[str] = []
        # The route owns the external symbol; the emitter derives its signature from
        # global Buffers rather than consulting an operator-named profile.
        self.entry_point = entry_point or schedule.lowering.entry_point

        failures = preflight(schedule, target)
        if failures:
            raise EmitError(failures[0].message)
        self.role = schedule.roles[0]

        # A kernel must write something, so a store is required of every Schedule -- but
        # how many is the host wrapper's business, not this constructor's. MMA operations
        # are independent DAG nodes; each validates its own Target/backend contract and
        # the declared order decides when its write becomes available. A reduction is not
        # required either: requiring one described the first operator this backend served.
        self.reduce = self._at_most_one(OperationKind.REDUCE_ARGMIN, "reduce_argmin")
        for operation in self.schedule.operations:
            if operation.kind is not OperationKind.MMA:
                continue
            instruction = operation.parameters.instruction
            assert instruction is not None
            _require(
                instruction.contract in target.instruction_contracts,
                f"instruction {instruction.contract!r} is not admitted by {target.target_id!r}",
            )

    # ---------------------------------------------------------------- derivation

    def _single(self, kind: OperationKind, label: str):
        matches = [op for op in self.schedule.operations if op.kind is kind]
        _require(len(matches) == 1, f"expected exactly one {label} operation")
        return matches[0]

    def _at_most_one(self, kind: OperationKind, label: str):
        matches = [op for op in self.schedule.operations if op.kind is kind]
        _require(len(matches) <= 1, f"expected at most one {label} operation")
        return matches[0] if matches else None

    def _extent(self, buffer_name: str, dimension: int) -> str:
        """Constexpr name for one global buffer dimension.

        Named after whichever axis or loop walks it, so a name in the emitted kernel
        points at the declaration that produced it.
        """

        assert self.schedule.program_map is not None
        for axis in self.schedule.program_map.axes:
            if axis.buffer == buffer_name and axis.dimension == dimension:
                return f"N_{axis.name.upper()}"
        for loop in self.schedule.tile_loops:
            if loop.buffer == buffer_name and loop.dimension == dimension:
                return f"N_{loop.name.upper()}"
        return f"D_{buffer_name.upper()}_{dimension}"

    def _tile(self, name: str) -> str:
        return f"BLOCK_{name.upper()}"

    def _axis(self, name: str) -> ProgramAxis:
        assert self.schedule.program_map is not None
        axis = self.schedule.program_map.axis(name)
        _require(axis is not None, f"unknown program axis {name!r}")
        return axis

    def constants(self) -> dict[str, int]:
        """Every constexpr the kernel takes, and where each one comes from."""

        values: dict[str, int] = {}
        for buffer in self.schedule.buffers:
            if buffer.space is not MemorySpace.GLOBAL:
                continue
            for dimension, extent in enumerate(buffer.shape):
                values.setdefault(self._extent(buffer.name, dimension), extent)
        assert self.schedule.program_map is not None
        for axis in self.schedule.program_map.axes:
            if axis.is_tiled:
                values[self._tile(axis.name)] = axis.tile
        for loop in self.schedule.tile_loops:
            values[self._tile(loop.name)] = loop.tile
        if len(self.schedule.tile_loops) == 1:
            values["NUM_STAGES"] = self.schedule.tile_loops[0].range_options.num_stages
        values["NUM_WARPS"] = len(self.role.warps)
        if self.schedule.program_map is not None and self.schedule.program_map.persistent:
            values["TOTAL_TILES"] = self.total_tiles()
            values["NUM_CTAS"] = self.grid()[0]
        return values

    def _axis_tiles(self, axis) -> int:
        buffer = self.schedule.buffer(axis.buffer)
        _require(buffer is not None, f"axis {axis.name!r} names an unknown buffer")
        return axis.tile_count(buffer.shape[axis.dimension])

    def grid(self) -> tuple[int, int, int]:
        assert self.schedule.program_map is not None
        program_map = self.schedule.program_map
        if program_map.persistent:
            # One CTA per resident slot, not one per tile. The count is the Target's
            # multiprocessor count times the residency the Schedule committed to, so the
            # launch and the commitment cannot disagree.
            residency = self.schedule.residency
            _require(
                residency is not None and residency.ctas_per_multiprocessor is not None,
                "a persistent grid is sized from residency.ctas_per_multiprocessor",
            )
            facts = self.target.occupancy
            _require(
                facts is not None,
                f"Target {self.target.target_id!r} declares no multiprocessor count",
            )
            launched = facts.multiprocessor_count * residency.ctas_per_multiprocessor
            return (min(launched, self.total_tiles()), 1, 1)
        extents = [1, 1, 1]
        for axis in program_map.axes:
            extents[axis.axis] = self._axis_tiles(axis)
        return tuple(extents)  # type: ignore[return-value]

    def total_tiles(self) -> int:
        assert self.schedule.program_map is not None
        total = 1
        for axis in self.schedule.program_map.axes:
            total *= self._axis_tiles(axis)
        return total

    # ---- addressing ---------------------------------------------------------

    def _components(self, access: AccessMap, buffer: Buffer) -> tuple[list[str], list[str]]:
        """Per-dimension index expressions, and which of them carry a tile axis."""

        expressions: list[str] = []
        vectors: list[str] = []
        for position, component in enumerate(access.indices):
            if component.source is AccessIndexKind.PROGRAM:
                expressions.append(str(component.name))
            elif component.source is AccessIndexKind.PROGRAM_TILE:
                name = f"{component.name}_offsets"
                expressions.append(name)
                vectors.append(name)
            elif component.source is AccessIndexKind.LOOP_TILE:
                name = f"{component.name}_offsets"
                expressions.append(name)
                vectors.append(name)
            else:
                name = self._dimension_vector(buffer, component)
                expressions.append(name)
                vectors.append(name)
        return expressions, vectors

    def _dimension_vector(self, buffer: Buffer, component) -> str:
        """The name of the offsets vector a `dimension` component walks.

        Two accesses may take different sub-ranges of the same axis, so the name carries
        the range -- otherwise the define-once dedup below would hand the second access
        the first one's offsets, silently. A whole-axis component keeps its original
        name, because every existing corpus case pins the emitted bytes.
        """

        base = f"{buffer.name}_d{component.dimension}"
        if component.offset == 0 and component.extent is None:
            return f"{base}_offsets"
        end = "end" if component.extent is None else component.extent
        return f"{base}_o{component.offset}_e{end}_offsets"

    def _bound(self, access: AccessMap, buffer: Buffer, vector: str) -> str | None:
        """The accessed Buffer extent that bounds one tiled coordinate.

        Program axes and loops own work decomposition, not the size of every Buffer
        indexed by their coordinate.  Using the axis owner's extent here made a shorter
        Buffer readable past its declaration.  Always derive the name from the accessed
        Buffer dimension; ``_extent`` reuses its canonical axis or loop name when one
        owns that same dimension.
        """

        for position, component in enumerate(access.indices):
            if component.source is AccessIndexKind.PROGRAM_TILE and f"{component.name}_offsets" == vector:
                self._axis(component.name)
                return self._extent(buffer.name, position)
            if (
                component.source is AccessIndexKind.LOOP_TILE
                and f"{component.name}_offsets" == vector
            ):
                # ACCESS_LOOP_UNKNOWN refuses this Schedule before it reaches emission,
                # so this cannot fire. It raises anyway because the alternative is a
                # silent fall-through to "needs no mask", which is a wrong kernel rather
                # than a refused one.
                _require(any(loop.iterator == component.name for loop in self.schedule.tile_loops),
                         f"{vector} indexes an unknown loop")
                return self._extent(buffer.name, position)
        return None  # a full-dimension index spans its axis and needs no mask

    def _address(self, access: AccessMap, pad: str) -> tuple[str, str]:
        """Pointer expression and mask for one access map."""

        if any(
            component.source is AccessIndexKind.BUFFER
            for component in access.indices
        ):
            return self._indexed_address(access, pad)

        buffer = self.schedule.buffer(access.buffer)
        _require(buffer is not None, f"access map names unknown buffer {access.buffer!r}")
        _require(
            len(access.indices) == len(buffer.shape),
            f"access map for {access.buffer!r} has the wrong rank",
        )
        expressions, vectors = self._components(access, buffer)
        strides: list[int] = []
        running = 1
        for extent in reversed(buffer.shape):
            strides.insert(0, running)
            running *= extent
        stride_names = [
            "1"
            if index == len(buffer.shape) - 1
            else " * ".join(
                self._extent(buffer.name, later)
                for later in range(index + 1, len(buffer.shape))
            )
            for index in range(len(buffer.shape))
        ]

        terms = []
        for expression, stride in zip(expressions, stride_names):
            broadcast = self._broadcast(expression, vectors)
            terms.append(expression + broadcast + ("" if stride == "1" else f" * {stride}"))
        pointer = f"{buffer.name} + " + " + ".join(terms)

        masks = []
        for vector in vectors:
            bound = self._bound(access, buffer, vector)
            if bound is not None:
                masks.append(f"{vector}{self._broadcast(vector, vectors)} < {bound}")
        relation = buffer.valid_extent
        if relation is not None:
            _require(
                len(relation.indexed_by) == 1,
                "the Triton valid-extent subset has one indexed axis",
            )
            extent_axis = relation.indexed_by[0]
            _require(
                extent_axis < len(expressions),
                "the valid-extent index axis is present in the access map",
            )
            extent_buffer = self.schedule.buffer(relation.buffer)
            _require(
                extent_buffer is not None and len(extent_buffer.shape) == 1,
                "the Triton valid-extent subset uses a rank-1 extent buffer",
            )
            extent_name = f"{access.operation}_{access.buffer}_valid_extent"
            self.line(
                f"{pad}{extent_name} = tl.load("
                f"{relation.buffer} + {expressions[extent_axis]})"
            )
            coordinate = expressions[relation.dimension]
            masks.append(
                f"{coordinate}{self._broadcast(coordinate, vectors)} < {extent_name}"
            )
        # `&` binds tighter than `<` in Python, so an unparenthesized conjunction of
        # comparisons silently becomes a chained comparison against a bitwise and. No
        # existing profile masked two axes at once, so the emitted text was correct
        # until an access map bounded more than one.
        if len(masks) > 1:
            return pointer, " & ".join(f"({mask})" for mask in masks)
        return pointer, " & ".join(masks)

    def _indexed_address(self, access: AccessMap, pad: str) -> tuple[str, str]:
        """Address one zipped runtime-index domain and ordinary independent axes.

        All buffer-valued components share the `_runtime_index` domain. Giving each
        component its own broadcast axis would produce a Cartesian product instead of
        the `[expert[k], row[k]]` tuples the Schedule declares.
        """

        buffer = self.schedule.buffer(access.buffer)
        _require(buffer is not None, f"access map names unknown buffer {access.buffer!r}")
        domains: list[str] = []
        expressions: list[str] = []
        component_domains: list[str | None] = []
        for component in access.indices:
            if component.source is AccessIndexKind.PROGRAM:
                expression = str(component.name)
                domain = None
            elif component.source is AccessIndexKind.PROGRAM_TILE:
                expression = f"{component.name}_offsets"
                domain = expression
            elif component.source is AccessIndexKind.LOOP_TILE:
                expression = f"{component.name}_offsets"
                domain = expression
            elif component.source is AccessIndexKind.DIMENSION:
                expression = self._dimension_vector(buffer, component)
                domain = expression
            else:
                expression = str(component.name)
                domain = "_runtime_index"
            expressions.append(expression)
            component_domains.append(domain)
            if domain is not None and domain not in domains:
                domains.append(domain)

        stride_names = [
            "1"
            if index == len(buffer.shape) - 1
            else " * ".join(
                self._extent(buffer.name, later)
                for later in range(index + 1, len(buffer.shape))
            )
            for index in range(len(buffer.shape))
        ]

        def shaped(expression: str, domain: str | None) -> str:
            if domain is None or len(domains) < 2:
                return expression
            position = domains.index(domain)
            suffix = "[" + ", ".join(
                ":" if index == position else "None"
                for index in range(len(domains))
            ) + "]"
            return expression + suffix

        terms = [
            shaped(expression, domain)
            + ("" if stride == "1" else f" * {stride}")
            for expression, domain, stride in zip(
                expressions, component_domains, stride_names
            )
        ]
        pointer = f"{buffer.name} + " + " + ".join(terms)

        masks: list[str] = []
        for position, (component, expression, domain) in enumerate(
            zip(access.indices, expressions, component_domains)
        ):
            coordinate = shaped(expression, domain)
            if component.source is AccessIndexKind.PROGRAM_TILE:
                self._axis(component.name)
                masks.append(f"{coordinate} < {self._extent(buffer.name, position)}")
            elif component.source is AccessIndexKind.LOOP_TILE:
                _require(any(loop.iterator == component.name for loop in self.schedule.tile_loops),
                         f"{expression} indexes an unknown loop")
                masks.append(f"{coordinate} < {self._extent(buffer.name, position)}")
            elif component.source is AccessIndexKind.BUFFER:
                bound = self._extent(buffer.name, position)
                masks.append(f"({coordinate} >= 0) & ({coordinate} < {bound})")

        relation = buffer.valid_extent
        if relation is not None:
            _require(
                len(relation.indexed_by) == 1,
                "the Triton valid-extent subset has one indexed axis",
            )
            extent_axis = relation.indexed_by[0]
            extent_name = f"{access.operation}_{access.buffer}_valid_extent"
            self.line(
                f"{pad}{extent_name} = tl.load("
                f"{relation.buffer} + {expressions[extent_axis]})"
            )
            coordinate = shaped(
                expressions[relation.dimension],
                component_domains[relation.dimension],
            )
            masks.append(f"{coordinate} < {extent_name}")

        return pointer, " & ".join(f"({mask})" for mask in masks)

    @staticmethod
    def _broadcast(expression: str, vectors: list[str]) -> str:
        """Give each tile axis its own dimension, in declaration order."""

        if expression not in vectors or len(vectors) < 2:
            return ""
        position = vectors.index(expression)
        return "[" + ", ".join("None" if i != position else ":" for i in range(len(vectors))) + "]"

    # ------------------------------------------------------------------- emission

    def line(self, text: str = "") -> None:
        self.lines.append(text)

    def emit(self) -> Emission:
        entry = self.entry_point
        kernel = f"_{entry}_kernel"
        self._emit_header()
        self._emit_kernel(kernel)
        self._emit_host(entry, kernel)
        return Emission(
            "\n".join(self.lines) + "\n",
            entry,
            dict(self.constants()),
            self._toolchain(kernel),
        )

    _POINTER = {
        DType.BF16: "*bf16",
        DType.FP16: "*fp16",
        DType.FP32: "*fp32",
        DType.FP8_E4M3: "*fp8e4nv",
        DType.INT32: "*i32",
    }

    def _toolchain(self, kernel: str) -> dict[str, object]:
        """The compile contract, derived rather than restated beside the source."""

        constants = self.constants()
        return {
            "kernel_entry_point": kernel,
            "signature": {
                buffer.name: self._POINTER[buffer.dtype] for buffer in self._globals()
            },
            "compile_constants": {
                name: value for name, value in constants.items() if name != "NUM_WARPS"
            },
            # A declared register budget is a cap the backend enforces, so it travels
            # with the other compile options rather than staying a claim the verifier
            # checked and then dropped.
            "compile_options": {
                "num_warps": constants["NUM_WARPS"],
                **(
                    {"num_stages": constants["NUM_STAGES"]}
                    if "NUM_STAGES" in constants
                    else {}
                ),
                **(
                    {"maxnreg": self.schedule.residency.registers_per_thread}
                    if self.schedule.residency is not None
                    and self.schedule.residency.registers_per_thread is not None
                    else {}
                ),
            },
            "grid": list(self.grid()),
        }

    def _emit_header(self) -> None:
        self.line(f"# Generated by open-cake-ir from {self.schedule.schedule_id}; DO NOT EDIT.")
        self.line("# schedule_sha256=__SCHEDULE_SHA256__")
        self.line("from __future__ import annotations")
        self.line()
        self.line("import torch")
        self.line("import triton")
        self.line("import triton.language as tl")
        if any(
            operation.kind is OperationKind.ELEMENTWISE
            and operation.parameters.op is ElementwiseOp.TANH
            and operation.parameters.instruction is not None
            and operation.parameters.instruction.contract == "libdevice.tanh.f32"
            for operation in self.schedule.operations
        ):
            self.line("from triton.language.extra import libdevice")
        self.line()
        self.line()

    def _globals(self) -> list[Buffer]:
        return [b for b in self.schedule.buffers if b.space is MemorySpace.GLOBAL]

    def _emit_kernel(self, kernel: str) -> None:
        constants = self.constants()
        self.line("@triton.jit")
        self.line(f"def {kernel}(")
        for buffer in self._globals():
            self.line(f"    {buffer.name},")
        for name in constants:
            if name != "NUM_WARPS":
                self.line(f"    {name}: tl.constexpr,")
        self.line("):")

        assert self.schedule.program_map is not None
        program_map = self.schedule.program_map
        if program_map.persistent:
            self._emit_persistent_header()
        else:
            for axis in program_map.axes:
                self.line(f"    {axis.name} = tl.program_id({axis.axis})")
        pad = self._body_pad()
        for axis in self.schedule.program_map.axes:
            if axis.is_tiled:
                tile = self._tile(axis.name)
                self.line(
                    f"{pad}{axis.name}_offsets = {axis.name} * {tile} + tl.arange(0, {tile})"
                )
        for access in self.schedule.access_maps:
            buffer = self.schedule.buffer(access.buffer)
            for component in access.indices:
                if component.source is AccessIndexKind.DIMENSION:
                    name = self._dimension_vector(buffer, component)
                    if f"{name} = " not in "\n".join(self.lines):
                        extent = self._extent(buffer.name, component.dimension)
                        if component.extent is None:
                            # Whole axis, or a tail of it. The offset==0 spelling is the
                            # one every existing case pins, so it stays literal.
                            walk = f"tl.arange({component.offset}, {extent})"
                        else:
                            walk = f"tl.arange(0, {component.extent})"
                            if component.offset:
                                walk = f"{walk} + {component.offset}"
                        self.line(f"{pad}{name} = {walk}")
        self.line()

        # A root loop replaces its contiguous expanded operation interval. Within
        # it, body order recursively owns child-loop and operation placement.
        emitted: set[str] = set()
        for operation in self.schedule.operations:
            chain = self.schedule.enclosing_loops(operation)
            if chain:
                root = chain[0]
                if root.name not in emitted:
                    self._emit_loop(root, pad)
                    emitted.add(root.name)
            else:
                self._emit_operation(operation, pad, inside=False)
        _require(
            all(loop.name in emitted for loop in self.schedule.tile_loops
                if loop.name not in self.schedule.loop_parent()),
            "a declared tile loop names no operation",
        )
        self.line("    # CAKE_KERNEL_END")
        self.line()
        self.line()

    def _body_pad(self) -> str:
        """A persistent walk puts the whole body one level deeper."""

        program_map = self.schedule.program_map
        return "        " if program_map is not None and program_map.persistent else "    "

    def _emit_persistent_header(self) -> None:
        """Decompose one linear work index into the declared axes.

        The traversal order is the order of division: the first axis named varies fastest,
        which is what makes adjacent CTAs share a tile of whichever operand the author
        wanted them to share.
        """

        program_map = self.schedule.program_map
        assert program_map is not None
        order = program_map.walk_order()
        self.line("    for _work in tl.range(tl.program_id(0), TOTAL_TILES, NUM_CTAS):")
        remainder = "_work"
        for position, axis in enumerate(order):
            extent = self._axis_tiles(axis)
            if position + 1 == len(order):
                self.line(f"        {axis.name} = {remainder}")
            else:
                self.line(f"        {axis.name} = {remainder} % {extent}")
                remainder = f"({remainder} // {extent})"
        self.line()

    def _emit_load(self, operation, pad: str) -> None:
        access = self.schedule.access_map(operation.op_id, operation.reads[0])
        _require(access is not None, f"load {operation.op_id!r} has no access map")
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        pointer, mask = self._address(access, pad)
        self.line(f"{pad}{operation.op_id}_ptrs = {pointer}")
        self.line(f"{pad}{operation.writes[0]} = tl.load(")
        self.line(f"{pad}    {operation.op_id}_ptrs,")
        if mask:
            self.line(f"{pad}    mask={mask},")
            self.line(f"{pad}    other=0.0,")
        # The declared intent, in the terms this backend uses to say it: an operand read
        # again is asked to stay resident, one read once is asked not to displace it.
        reuse = operation.parameters.reuse
        if reuse is LoadReuse.REUSED:
            self.line(f'{pad}    eviction_policy="evict_last",')
        elif reuse is LoadReuse.STREAMED:
            # `.cg` alone: it caches in L2 and bypasses L1, which is the no-allocate the
            # streamed operand wants. ptxas rejects `.cg` combined with `.evict_first`,
            # and an eviction hint would allocate the line this is trying not to take.
            self.line(f'{pad}    cache_modifier=".cg",')
        self.line(f"{pad})")

    def _state_shape(self, buffer: Buffer) -> str:
        # Preserve established single-loop source spelling where that axis owns the
        # result shape. Nested state is shaped by its result, never an unrelated axis.
        if len(self.schedule.tile_loops) == 1 and self.schedule.program_map is not None:
            for axis in self.schedule.program_map.axes:
                if axis.is_tiled:
                    if buffer.shape == (axis.tile,):
                        return f"({self._tile(axis.name)},)"
                    break
        return repr(tuple(buffer.shape))

    def _emit_reduction_state(self, loop: TileLoop, pad: str) -> None:
        """Initialize only state directly carried by this loop, at its entry."""

        if self.reduce is not None and self.reduce.op_id in loop.body:
            _require(self.reduce.parameters.across_loop,
                     "this backend carries the reduction across the loop")
            tile = self._tile(self._token_axis().name)
            self.line(f'{pad}best_distance = tl.full(({tile},), float("inf"), tl.float32)')
            self.line(f"{pad}{self.reduce.writes[0]} = tl.zeros(({tile},), tl.int32)")
            self.line()
        for operation in self.schedule.operations:
            if operation.op_id not in loop.body:
                continue
            if operation.kind is OperationKind.REDUCE:
                if not operation.parameters.across_loop:
                    continue
                result = self.schedule.buffer(operation.writes[0])
                _require(result is not None, "a carried reduction has no result buffer")
                shape = self._state_shape(result)
                identity = REDUCTIONS[operation.parameters.op].identity.format(shape=shape)
                self.line(f"{pad}{result.name} = {identity}")
                self.line()
            elif (
                operation.kind is OperationKind.TOP_K
                and operation.parameters.across_loop
            ):
                k = operation.parameters.k
                values, indices = operation.writes
                self.line(
                    f'{pad}{values} = tl.full(({k},), float("-inf"), tl.float32)'
                )
                self.line(
                    f"{pad}{indices} = tl.full(({k},), 2147483647, tl.int32)"
                )
                if operation.parameters.source_tiles_per_merge == 2:
                    source = self.schedule.buffer(operation.reads[0])
                    _require(
                        source is not None and len(source.shape) == 1,
                        "two-tile top_k has no resident rank-one source",
                    )
                    self.line(
                        f"{pad}{operation.op_id}_pending_keys = "
                        f"tl.zeros(({source.shape[0]},), tl.uint64)"
                    )
                self.line()
            elif operation.kind is OperationKind.ONLINE_SOFTMAX:
                maximum, normalizer, accumulator, _ = (
                    self.schedule.buffer(name) for name in operation.writes
                )
                _require(
                    maximum is not None
                    and normalizer is not None
                    and accumulator is not None,
                    "online_softmax state buffers are missing",
                )
                rows = maximum.shape[0]
                columns = accumulator.shape[1]
                self.line(
                    f'{pad}{maximum.name} = tl.full(({rows},), float("-inf"), tl.float32)'
                )
                self.line(
                    f"{pad}{normalizer.name} = tl.zeros(({rows},), tl.float32)"
                )
                self.line(
                    f"{pad}{accumulator.name} = tl.zeros(({rows}, {columns}), tl.float32)"
                )
                self.line()
            elif operation.kind is OperationKind.MMA and self.schedule.mma_accumulates_over(operation, loop):
                # A contraction summed across the loop needs its accumulator before the
                # loop, for the same reason a fold does: the first iteration adds to it.
                accumulator = self.schedule.buffer(operation.writes[0])
                _require(
                    accumulator is not None and len(accumulator.shape) == 2,
                    "an accumulated contraction writes a rank-2 buffer",
                )
                rows, columns = accumulator.shape
                self.line(
                    f"{pad}{accumulator.name} = tl.zeros(({rows}, {columns}), tl.float32)"
                )
                self.line()

    def _token_axis(self) -> ProgramAxis:
        assert self.schedule.program_map is not None
        for axis in self.schedule.program_map.axes:
            if axis.is_tiled:
                return axis
        raise EmitError("no tiled program axis")

    def _emit_operation(self, operation, pad: str, *, inside: bool) -> None:
        dispatch = INSIDE_LOOP_EMITTERS if inside else OUTSIDE_LOOP_EMITTERS
        method = dispatch.get(operation.kind)
        _require(method is not None,
                 f"operation {operation.op_id!r} has no Triton body at this scope")
        getattr(self, method)(operation, pad)
        if not inside and operation.kind is OperationKind.ELEMENTWISE:
            self.line()

    def _emit_loop(self, loop: TileLoop, pad: str) -> None:
        self._emit_reduction_state(loop, pad)
        options = loop.range_options
        extent = self._loop_stop(loop)
        tile = self._tile(loop.name)
        knobs = [f"num_stages={options.num_stages}"]
        if options.disallow_acc_multi_buffer:
            knobs.append("disallow_acc_multi_buffer=True")
        if options.flatten:
            knobs.append("flatten=True")
        if options.warp_specialize:
            knobs.append("warp_specialize=True")
        if options.disable_licm:
            knobs.append("disable_licm=True")
        if options.loop_unroll_factor != 1:
            knobs.append(f"loop_unroll_factor={options.loop_unroll_factor}")
        self.line(
            f"{pad}for {loop.iterator} in tl.range(0, {extent}, {tile}, "
            + ", ".join(knobs)
            + "):"
        )
        self.line(
            f"{pad}    {loop.iterator}_offsets = "
            f"{loop.iterator} + tl.arange(0, {tile})"
        )
        for op_id in loop.body:
            child = self.schedule.tile_loop(op_id)
            if child is not None:
                self._emit_loop(child, pad + "    ")
                continue
            operation = self.schedule.operation(op_id)
            _require(operation is not None, f"loop body names unknown operation {op_id!r}")
            self._emit_operation(operation, pad + "    ", inside=True)
        for op_id in loop.body:
            operation = self.schedule.operation(op_id)
            if operation is not None and operation.kind is OperationKind.ONLINE_SOFTMAX:
                self._emit_online_softmax_finalize(operation, pad)
            if (
                operation is not None
                and operation.kind is OperationKind.TOP_K
                and operation.parameters.across_loop
            ):
                if operation.parameters.source_tiles_per_merge == 2:
                    self._emit_two_tile_top_k_flush(operation, pad)
                self._emit_top_k_finalize(operation, pad)
        self.line()

    def _loop_stop(self, loop: TileLoop) -> str:
        """Return the loop's static or declared query-derived exclusive stop."""

        static = self._extent(loop.buffer, loop.dimension)
        relation = loop.stop
        if relation is None:
            return static
        self._axis(relation.program)
        expression = relation.program
        if relation.add > 0:
            expression = f"({expression} + {relation.add})"
        elif relation.add < 0:
            expression = f"({expression} - {-relation.add})"
        if relation.floor_div != 1:
            expression = f"({expression} // {relation.floor_div})"
        return f"tl.minimum(tl.maximum({expression}, 0), {static})"

    _ELEMENTWISE_TEXT = {
        ElementwiseOp.SQUARE: "{a} * {a}",
        ElementwiseOp.RSQRT: "tl.rsqrt({a})",
        ElementwiseOp.EXP: "tl.exp({a})",
        ElementwiseOp.RELU: "tl.maximum({a}, 0.0)",
        ElementwiseOp.ADD: "{a} + {b}",
        ElementwiseOp.SUB: "{a} - {b}",
        ElementwiseOp.MUL: "{a} * {b}",
        # Written as the division it is. A Schedule that wants the reciprocal-then-
        # multiply a fast kernel uses declares `rsqrt`-style reciprocal and `mul`, which
        # is a different Schedule with different numerics -- and says so.
        ElementwiseOp.DIV: "{a} / {b}",
        # Opaque inputs preserve producer rounding and nested FMA boundaries.
        # No .ftz or .sat modifier may alter the declared contract.
        ElementwiseOp.FMA: (
            'tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", '
            'constraints="=f,f,f,f", args=[{a}, {b}, {c}], dtype=tl.float32, '
            'is_pure=True, pack=1)'
        ),
    }

    def _emit_elementwise(self, operation, pad: str) -> None:
        """One arithmetic primitive, written once per backend rather than per operator.

        A narrower operand is broadcast against the wider one along its leading axes.
        The verifier has already held the pair to a legal broadcast, so the shapes here
        are known to line up and this only has to say how.
        """

        parameters = operation.parameters
        operands = [self._operand(name, operation) for name in operation.reads]
        if parameters.scalar is not None:
            operands.append(repr(parameters.scalar))
        if parameters.op is ElementwiseOp.FMA:
            instruction = parameters.instruction
            _require(
                instruction is not None
                and instruction.contract == "ptx.fma.rn.f32",
                "the Triton fma body requires ptx.fma.rn.f32",
            )
        if parameters.op is ElementwiseOp.TANH:
            instruction = parameters.instruction
            _require(
                instruction is not None
                and instruction.contract == "libdevice.tanh.f32",
                "the Triton tanh body requires the admitted libdevice.tanh.f32 contract",
            )
            expression = f"libdevice.tanh({operands[0]})"
        else:
            template = self._ELEMENTWISE_TEXT[parameters.op]
            expression = template.format(
                a=operands[0],
                b=operands[1] if len(operands) > 1 else "",
                c=operands[2] if len(operands) > 2 else "",
            )
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(f"{pad}{operation.writes[0]} = {expression}")

    def _operand(self, name: str, operation) -> str:
        """A read, indexed so it spans the declared axis of the wider operand."""

        buffer = self.schedule.buffer(name)
        _require(buffer is not None, f"elementwise reads unknown buffer {name!r}")
        result = self.schedule.buffer(operation.writes[0])
        _require(result is not None, "elementwise requires its verified result shape")
        widest = result.shape
        if buffer.is_scalar or len(buffer.shape) == len(widest):
            return name
        axis = operation.parameters.broadcast_axis
        _require(axis is not None, f"operand {name!r} needs a declared broadcast axis")
        index = ", ".join(":" if position == axis else "None" for position in range(len(widest)))
        return f"{name}[{index}]"

    def _emit_reduce(self, operation, pad: str) -> None:
        """Fold the declared axis of the input into the result.

        The axis is what the operation declares and the extent follows from the read
        buffer, so nothing here needs to know which operator asked for the reduction --
        and which fold it is comes from the operation too, rather than from this backend
        having been written when sum was the only one.
        """

        source = self.schedule.buffer(operation.reads[0])
        _require(source is not None, f"reduce reads unknown buffer {operation.reads[0]!r}")
        axis = operation.parameters.axis
        _require(
            axis < len(source.shape),
            f"reduce axis {axis} is outside {source.name!r}",
        )
        reduction = REDUCTIONS[operation.parameters.op]
        carried = (
            bool(self.schedule.enclosing_loops(operation))
            and operation.parameters.across_loop
        )
        template = reduction.accumulate if carried else reduction.once
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(
            pad
            + template.format(
                out=operation.writes[0], src=operation.reads[0], axis=axis
            )
        )

    def _emit_scan(self, operation, pad: str) -> None:
        """Accumulate one inclusive prefix along the declared resident axis."""

        source = self.schedule.buffer(operation.reads[0])
        _require(source is not None, f"scan reads unknown buffer {operation.reads[0]!r}")
        axis = operation.parameters.axis
        _require(axis < len(source.shape), f"scan axis {axis} is outside {source.name!r}")
        reverse = operation.parameters.direction is ScanDirection.REVERSE
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(
            pad
            + SCANS[operation.parameters.op].format(
                out=operation.writes[0],
                src=operation.reads[0],
                axis=axis,
                reverse=reverse,
            )
        )

    def _emit_online_softmax(self, operation, pad: str) -> None:
        """Update explicit FP32 online-softmax state from one logits/value tile."""

        logits, values = operation.reads[:2]
        validity = operation.reads[2] if len(operation.reads) == 3 else None
        maximum, normalizer, accumulator, _ = operation.writes
        prefix = operation.op_id
        tile_max = f"{prefix}_tile_max"
        new_max = f"{prefix}_new_max"
        empty = f"{prefix}_empty"
        old_scale = f"{prefix}_old_scale"
        weights = f"{prefix}_weights"
        new_sum = f"{prefix}_new_sum"
        new_accumulator = f"{prefix}_new_accumulator"

        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        effective_logits = logits
        if validity is not None:
            effective_logits = f"{prefix}_effective_logits"
            self.line(
                f'{pad}{effective_logits} = tl.where({validity}[None, :] != '
                f'{operation.parameters.sentinel}, {logits}, float("-inf"))'
            )
        self.line(f"{pad}{tile_max} = tl.max({effective_logits}, axis=1)")
        self.line(f"{pad}{new_max} = tl.maximum({maximum}, {tile_max})")
        self.line(f'{pad}{empty} = {new_max} == float("-inf")')
        self.line(
            f"{pad}{old_scale} = tl.where({empty}, 1.0, "
            f"tl.exp({maximum} - {new_max}))"
        )
        self.line(
            f'{pad}{weights} = tl.where({effective_logits} == float("-inf"), 0.0, '
            f"tl.exp({effective_logits} - {new_max}[:, None]))"
        )
        self.line(
            f"{pad}{new_sum} = {normalizer} * {old_scale} + "
            f"tl.sum({weights}, axis=1)"
        )
        self.line(
            f"{pad}{new_accumulator} = {accumulator} * {old_scale}[:, None] + "
            f"tl.sum({weights}[:, :, None] * {values}[None, :, :].to(tl.float32), axis=1)"
        )
        self.line(f"{pad}{maximum} = {new_max}")
        self.line(f"{pad}{normalizer} = {new_sum}")
        self.line(f"{pad}{accumulator} = {new_accumulator}")

    def _emit_online_softmax_finalize(self, operation, pad: str) -> None:
        """Materialize the declared zero-if-empty normalized accumulator."""

        _, normalizer, accumulator, normalized = operation.writes
        self.line(f"{pad}# CAKE_FINALIZE:{operation.op_id}")
        self.line(
            f"{pad}{normalized} = tl.where({normalizer}[:, None] > 0.0, "
            f"{accumulator} / {normalizer}[:, None], 0.0)"
        )

    def _emit_mma(self, operation, pad: str) -> None:
        """The contraction, and nothing else.

        This used to emit the dot and the distance formula together, because MmaFormula
        named the pair as one thing. Two of the three retained program slices declared
        a bare contraction with the arithmetic in a following operation; this is now the
        only shape, so the same kernel is written down one way.
        """

        tiles = [
            name
            for name in operation.reads
            if (buffer := self.schedule.buffer(name)) is not None
            and buffer.space is not MemorySpace.GLOBAL
        ]
        instruction = operation.parameters.instruction
        contract = instruction.contract if instruction is not None else None
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        if contract == "triton.dot.fp8e4m3_block_scale_fp32":
            _require(
                len(operation.reads) == 4 and len(tiles) == 4,
                "the block-scaled dot takes staged A, B, scale(A), scale(B)",
            )
            a = self.schedule.buffer(tiles[0])
            b = self.schedule.buffer(tiles[1])
            scale_a = self.schedule.buffer(tiles[2])
            _require(a is not None and b is not None and scale_a is not None, "missing tile")
            relation = scale_a.scale_of
            _require(relation is not None, "scale(A) has no relation")
            block_k = relation.granularity[1]
            groups = a.shape[1] // block_k
            _require(groups == 2, "the initial block-scale lowering takes two K blocks")
            output = operation.writes[0]
            self.line(
                f"{pad}{tiles[0]}_pairs = tl.permute(tl.reshape({tiles[0]}, "
                f"({a.shape[0]}, {groups}, {block_k})), (0, 2, 1))"
            )
            self.line(
                f"{pad}{tiles[1]}_pairs = tl.permute(tl.reshape({tiles[1]}, "
                f"({b.shape[0]}, {groups}, {block_k})), (0, 2, 1))"
            )
            self.line(
                f"{pad}{tiles[0]}_block_0, {tiles[0]}_block_1 = "
                f"tl.split({tiles[0]}_pairs)"
            )
            self.line(
                f"{pad}{tiles[1]}_block_0, {tiles[1]}_block_1 = "
                f"tl.split({tiles[1]}_pairs)"
            )
            self.line(
                f"{pad}{tiles[2]}_block_0, {tiles[2]}_block_1 = "
                f"tl.split(tl.trans({tiles[2]}))"
            )
            self.line(
                f"{pad}{tiles[3]}_block_0, {tiles[3]}_block_1 = tl.split({tiles[3]})"
            )
            if not self._accumulating(operation):
                self.line(
                    f"{pad}{output} = tl.zeros(({a.shape[0]}, {b.shape[0]}), tl.float32)"
                )
            for block in range(groups):
                self.line(
                    f"{pad}{output} += tl.dot({tiles[0]}_block_{block}, "
                    f"tl.trans({tiles[1]}_block_{block}), out_dtype=tl.float32) * "
                    f"{tiles[2]}_block_{block}[:, None] * {tiles[3]}_block_{block}"
                )
            return
        _require(
            len(operation.reads) == 2 and len(tiles) == 2,
            "the dot takes exactly two staged operands and reads nothing else",
        )
        assign = "+=" if self._accumulating(operation) else "="
        input_precision = _TRITON_DOT_INPUT_PRECISION.get(contract)
        precision = (
            f', input_precision="{input_precision}"'
            if input_precision is not None
            else ""
        )
        self.line(
            f"{pad}{operation.writes[0]} {assign} tl.dot("
            f"{tiles[0]}, tl.trans({tiles[1]}){precision})"
        )

    def _accumulating(self, operation) -> bool:
        """Whether this contraction sums across the loop it sits in."""

        chain = self.schedule.enclosing_loops(operation)
        return bool(chain) and self.schedule.mma_accumulates_over(operation, chain[-1])

    def _emit_argmin(self, operation, pad: str) -> None:
        chain = self.schedule.enclosing_loops(operation)
        _require(bool(chain), "loop-carried operation has no containing loop")
        loop = chain[-1]
        source = operation.reads[0]
        best = operation.writes[0]
        lowest = operation.parameters.tie_break is IndexTieBreak.LOWEST_INDEX
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(f"{pad}block_position = tl.argmin(")
        self.line(f"{pad}    {source}, axis=1, tie_break_left={lowest},")
        self.line(f"{pad})")
        self.line(f"{pad}block_distance = tl.min({source}, axis=1)")
        self.line(
            f"{pad}candidate_index = {loop.iterator} + block_position"
        )
        self.line(f"{pad}better = block_distance < best_distance")
        comparison = "<" if lowest else ">"
        self.line(f"{pad}tie = (block_distance == best_distance) & (")
        self.line(f"{pad}    candidate_index {comparison} {best}")
        self.line(f"{pad})")
        self.line(f"{pad}update = better | tie")
        self.line(f"{pad}best_distance = tl.where(update, block_distance, best_distance)")
        self.line(f"{pad}{best} = tl.where(update, candidate_index, {best})")

    def _emit_top_k(self, operation, pad: str) -> None:
        """Select a deterministic descending prefix from one resident score tile.

        One sortable uint64 carries both the IEEE-754 total order used here and the
        lowest-index tie break.  This lets Triton's bitonic top-k select values and
        indices together instead of emitting k serial max/min reduction chains.
        """

        source = self.schedule.buffer(operation.reads[0])
        _require(source is not None, f"top_k reads unknown buffer {operation.reads[0]!r}")
        if operation.parameters.across_loop:
            self._emit_loop_carried_top_k(operation, source, pad)
            return
        values, indices = operation.writes
        k = operation.parameters.k
        prefix = operation.op_id
        positions = f"{prefix}_source_positions"

        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(f"{pad}{positions} = tl.arange(0, {source.shape[0]})")
        keys = self._emit_top_k_keys(
            operation.reads[0], positions, "True", prefix, pad, source.dtype
        )
        ranked = f"{prefix}_ranked_keys"
        self.line(f"{pad}{ranked} = tl.topk({keys}, {k})")
        self._emit_top_k_decode(ranked, values, indices, prefix, pad, source.dtype)

    def _emit_loop_carried_top_k(self, operation, source, pad: str) -> None:
        """Merge one score tile into deterministic loop-carried top-k state."""

        chain = self.schedule.enclosing_loops(operation)
        _require(bool(chain), "loop-carried operation has no containing loop")
        loop = chain[-1]

        if operation.parameters.source_tiles_per_merge == 2:
            self._emit_two_tile_loop_carried_top_k(operation, source, pad)
            return
        values, indices = operation.writes
        k = operation.parameters.k
        prefix = operation.op_id
        source_positions = f"{prefix}_source_positions"
        source_valid = f"{prefix}_source_valid"
        state_valid = f"{prefix}_state_valid"
        previous_values = f"{prefix}_previous_values"
        previous_indices = f"{prefix}_previous_indices"

        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(f"{pad}{previous_values} = {values}")
        self.line(f"{pad}{previous_indices} = {indices}")
        self.line(
            f"{pad}{source_positions} = {loop.iterator} + "
            f"tl.arange(0, {source.shape[0]})"
        )
        if loop.stop is None:
            self.line(f"{pad}{source_valid} = tl.full(({source.shape[0]},), True, tl.int1)")
        else:
            self.line(
                f"{pad}{source_valid} = {source_positions} < {self._loop_stop(loop)}"
            )
        self.line(f"{pad}{state_valid} = {previous_indices} != 2147483647")
        source_keys = self._emit_top_k_keys(
            operation.reads[0], source_positions, source_valid, f"{prefix}_source", pad,
            source.dtype,
        )
        state_keys = self._emit_top_k_keys(
            previous_values, previous_indices, state_valid, f"{prefix}_state", pad,
            source.dtype,
        )
        width = max(source.shape[0], k)
        source_keys = self._emit_top_k_key_padding(
            source_keys, source.shape[0], width, f"{prefix}_source", pad
        )
        state_keys = self._emit_top_k_key_padding(
            state_keys, k, width, f"{prefix}_state", pad
        )
        ranked = self._emit_top_k_merge_selection(
            state_keys,
            source_keys,
            k,
            2 * width,
            operation.parameters.source_tiles_per_merge,
            prefix,
            pad,
        )
        self._emit_top_k_decode(ranked, values, indices, prefix, pad, source.dtype)

    def _emit_two_tile_loop_carried_top_k(self, operation, source, pad: str) -> None:
        """Retain one source-key tile and merge only on every second loop trip."""

        chain = self.schedule.enclosing_loops(operation)
        _require(bool(chain), "loop-carried operation has no containing loop")
        loop = chain[-1]

        prefix = operation.op_id
        source_positions = f"{prefix}_source_positions"
        source_valid = f"{prefix}_source_valid"
        pending = f"{prefix}_pending_keys"
        pair = f"{prefix}_pair_keys"

        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(
            f"{pad}{source_positions} = {loop.iterator} + "
            f"tl.arange(0, {source.shape[0]})"
        )
        # The access-map mask protects the load, while this mask protects selection.
        # They must use the same effective stop even for a static partial final tile;
        # otherwise the load's zero fill becomes a real score.
        self.line(
            f"{pad}{source_valid} = {source_positions} < {self._loop_stop(loop)}"
        )
        source_keys = self._emit_top_k_keys(
            operation.reads[0],
            source_positions,
            source_valid,
            f"{prefix}_source",
            pad,
            source.dtype,
        )
        tile = self._tile(loop.name)
        self.line(f"{pad}if (({loop.iterator} // {tile}) & 1) != 0:")
        self.line(f"{pad}    {pair} = tl.cat({pending}, {source_keys})")
        self._emit_two_tile_top_k_merge(
            operation,
            source,
            pair,
            f"{prefix}_pair",
            pad + "    ",
        )
        self.line(f"{pad}else:")
        self.line(f"{pad}    {pending} = {source_keys}")

    def _emit_two_tile_top_k_merge(
        self,
        operation,
        source,
        source_keys: str,
        prefix: str,
        pad: str,
    ) -> None:
        """Merge one pair of source-key tiles with the carried selected state."""

        values, indices = operation.writes
        k = operation.parameters.k
        state_valid = f"{prefix}_state_valid"
        self.line(f"{pad}{state_valid} = {indices} != 2147483647")
        state_keys = self._emit_top_k_keys(
            values,
            indices,
            state_valid,
            f"{prefix}_state",
            pad,
            source.dtype,
        )
        source_extent = 2 * source.shape[0]
        width = max(source_extent, k)
        source_keys = self._emit_top_k_key_padding(
            source_keys,
            source_extent,
            width,
            f"{prefix}_source",
            pad,
        )
        state_keys = self._emit_top_k_key_padding(
            state_keys,
            k,
            width,
            f"{prefix}_state",
            pad,
        )
        ranked = self._emit_top_k_merge_selection(
            state_keys,
            source_keys,
            k,
            2 * width,
            operation.parameters.source_tiles_per_merge,
            prefix,
            pad,
        )
        self._emit_top_k_decode(
            ranked,
            values,
            indices,
            prefix,
            pad,
            source.dtype,
        )

    def _emit_top_k_merge_selection(
        self,
        state_keys: str,
        source_keys: str,
        k: int,
        merge_width: int,
        source_tiles_per_merge: int,
        prefix: str,
        pad: str,
    ) -> str:
        """Select an exact descending top-k from carried state plus source keys.

        The carried half is descending by induction.  Sorting only the source half
        ascending makes their concatenation bitonic; one public Triton bitonic merge
        sorts the full vector, and splitting its first half keeps the exact top-k.
        """

        selection = top_k_selection_structure(
            k,
            merge_width,
            source_tiles_per_merge,
        )
        combined = f"{prefix}_combined_keys"
        ranked = f"{prefix}_ranked_keys"
        if selection.algorithm == "triton_topk":
            self.line(f"{pad}{combined} = tl.cat({state_keys}, {source_keys})")
            self.line(f"{pad}{ranked} = tl.topk({combined}, {k})")
            return ranked

        sorted_source = f"{prefix}_sorted_source_keys"
        merged = f"{prefix}_merged_keys"
        halves = f"{prefix}_merged_halves"
        pairs = f"{prefix}_merged_pairs"
        discarded = f"{prefix}_discarded_keys"
        self.line(f"{pad}{sorted_source} = tl.sort({source_keys}, descending=False)")
        self.line(f"{pad}{combined} = tl.cat({state_keys}, {sorted_source})")
        self.line(
            f"{pad}{merged} = tl.bitonic_merge({combined}, descending=True)"
        )
        self.line(f"{pad}{halves} = tl.reshape({merged}, (2, {k}))")
        self.line(f"{pad}{pairs} = tl.trans({halves})")
        self.line(f"{pad}{ranked}, {discarded} = tl.split({pairs})")
        return ranked

    def _emit_two_tile_top_k_flush(self, operation, pad: str) -> None:
        """Merge a pending odd final source tile after the declared tile loop."""

        chain = self.schedule.enclosing_loops(operation)
        _require(bool(chain), "loop-carried operation has no containing loop")
        loop = chain[-1]

        source = self.schedule.buffer(operation.reads[0])
        _require(
            source is not None and len(source.shape) == 1,
            "two-tile top_k flush has no resident rank-one source",
        )
        prefix = operation.op_id
        pending = f"{prefix}_pending_keys"
        count = f"{prefix}_source_tile_count"
        zeros = f"{prefix}_flush_zero_keys"
        pair = f"{prefix}_flush_pair_keys"
        tile = self._tile(loop.name)
        stop = self._loop_stop(loop)

        self.line(f"{pad}# CAKE_FLUSH:{operation.op_id}")
        self.line(f"{pad}{count} = ({stop} + {tile} - 1) // {tile}")
        self.line(f"{pad}if ({count} & 1) != 0:")
        self.line(
            f"{pad}    {zeros} = tl.zeros(({source.shape[0]},), tl.uint64)"
        )
        self.line(f"{pad}    {pair} = tl.cat({pending}, {zeros})")
        self._emit_two_tile_top_k_merge(
            operation,
            source,
            pair,
            f"{prefix}_flush",
            pad + "    ",
        )

    def _emit_top_k_keys(
        self,
        values: str,
        indices: str,
        valid: str,
        prefix: str,
        pad: str,
        dtype: DType,
    ) -> str:
        """Pack value order plus the lowest-index tie break into one uint64."""

        bits = f"{prefix}_score_bits"
        ordered = f"{prefix}_ordered_scores"
        ties = f"{prefix}_tie_keys"
        keys = f"{prefix}_keys"
        if dtype is DType.FP32:
            canonical = f"{prefix}_canonical_scores"
            self.line(f"{pad}{canonical} = tl.where({values} == 0.0, 0.0, {values})")
            self.line(f"{pad}{bits} = {canonical}.to(tl.uint32, bitcast=True)")
            self.line(
                f"{pad}{ordered} = tl.where(({bits} & 0x80000000) != 0, "
                f"{bits} ^ 0xffffffff, {bits} ^ 0x80000000)"
            )
        else:
            _require(dtype is DType.INT32, "top_k value dtype is unsupported")
            self.line(f"{pad}{bits} = {values}.to(tl.uint32, bitcast=True)")
            self.line(f"{pad}{ordered} = {bits} ^ 0x80000000")
        self.line(f"{pad}{ties} = 0xffffffff - {indices}.to(tl.uint32)")
        self.line(
            f"{pad}{keys} = tl.where({valid}, "
            f"({ordered}.to(tl.uint64) << 32) | {ties}.to(tl.uint64), 0)"
        )
        return keys

    def _emit_top_k_key_padding(
        self,
        keys: str,
        extent: int,
        target: int,
        prefix: str,
        pad: str,
    ) -> str:
        """Pad one power-of-two key vector to the common merge width."""

        current = keys
        while extent < target:
            zeros = f"{prefix}_padding_{extent}"
            padded = f"{prefix}_padded_{extent * 2}"
            self.line(f"{pad}{zeros} = tl.zeros(({extent},), tl.uint64)")
            self.line(f"{pad}{padded} = tl.cat({current}, {zeros})")
            current = padded
            extent *= 2
        return current

    def _emit_top_k_decode(
        self,
        keys: str,
        values: str,
        indices: str,
        prefix: str,
        pad: str,
        dtype: DType,
    ) -> None:
        """Recover source-typed values and int32 indices from ranked keys."""

        ordered = f"{prefix}_ranked_ordered_scores"
        bits = f"{prefix}_ranked_score_bits"
        decoded = f"{prefix}_ranked_scores"
        low = f"{prefix}_ranked_tie_keys"
        decoded_indices = f"{prefix}_ranked_indices"
        valid = f"{prefix}_ranked_valid"
        self.line(f"{pad}{valid} = {keys} != 0")
        self.line(f"{pad}{ordered} = ({keys} >> 32).to(tl.uint32)")
        if dtype is DType.FP32:
            self.line(
                f"{pad}{bits} = tl.where(({ordered} & 0x80000000) != 0, "
                f"{ordered} ^ 0x80000000, {ordered} ^ 0xffffffff)"
            )
            self.line(f"{pad}{decoded} = {bits}.to(tl.float32, bitcast=True)")
            invalid_value = 'float("-inf")'
        else:
            _require(dtype is DType.INT32, "top_k value dtype is unsupported")
            self.line(f"{pad}{bits} = {ordered} ^ 0x80000000")
            self.line(f"{pad}{decoded} = {bits}.to(tl.int32, bitcast=True)")
            invalid_value = "-2147483648"
        self.line(f"{pad}{low} = ({keys} & 0xffffffff).to(tl.uint32)")
        self.line(f"{pad}{decoded_indices} = (0xffffffff - {low}).to(tl.int32)")
        self.line(f"{pad}{values} = tl.where({valid}, {decoded}, {invalid_value})")
        self.line(
            f"{pad}{indices} = tl.where({valid}, {decoded_indices}, 2147483647)"
        )

    def _emit_top_k_finalize(self, operation, pad: str) -> None:
        """Normalize unfilled loop-carried positions to the canonical -1 sentinel."""

        indices = operation.writes[1]
        self.line(f"{pad}# CAKE_FINALIZE:{operation.op_id}")
        self.line(f"{pad}{indices} = tl.where({indices} == 2147483647, -1, {indices})")

    def _emit_index_expand(self, operation, pad: str) -> None:
        """Expand group indices into one flat affine run per selected group."""

        source = self.schedule.buffer(operation.reads[0])
        output = self.schedule.buffer(operation.writes[0])
        _require(source is not None and output is not None, "index_expand buffers are missing")
        extent = operation.parameters.extent
        offsets = f"{operation.op_id}_offsets"
        expanded = f"{operation.op_id}_matrix"
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(f"{pad}{offsets} = tl.arange(0, {extent})")
        self.line(
            f"{pad}{expanded} = tl.where({operation.reads[0]}[:, None] >= 0, "
            f"{operation.reads[0]}[:, None] * {operation.parameters.scale} + "
            f"{offsets}[None, :], {operation.parameters.sentinel})"
        )
        self.line(
            f"{pad}{operation.writes[0]} = tl.reshape({expanded}, ({output.shape[0]},))"
        )

    def _emit_cast(self, operation, pad: str) -> None:
        """Emit the conversion named by the written buffer's declared dtype."""

        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(
            f"{pad}{operation.writes[0]} = "
            f"{operation.reads[0]}.to({_TL_DTYPE[operation.parameters.to]})"
        )

    def _emit_atomic_rmw(self, operation, pad: str) -> None:
        """Emit the one admitted state transition and its returned old values."""

        target = operation.writes[0]
        result = operation.writes[1]
        access = self.schedule.access_map(operation.op_id, target)
        _require(access is not None, f"atomic_rmw {operation.op_id!r} has no access map")
        pointer, mask = self._address(access, pad)
        _require(bool(mask), "atomic_rmw requires a bounded runtime index")
        old = f"{operation.op_id}_old"
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(f"{pad}{old} = tl.atomic_add(")
        self.line(f"{pad}    {pointer},")
        self.line(f"{pad}    {operation.parameters.value},")
        self.line(f'{pad}    sem="relaxed",')
        self.line(f'{pad}    scope="gpu",')
        if mask:
            self.line(f"{pad}    mask={mask},")
        self.line(f"{pad})")
        self.line(f"{pad}{result} = tl.where({mask}, {old}, 0)")

    def _emit_store(self, operation, pad: str) -> None:
        access = self.schedule.access_map(operation.op_id, operation.writes[0])
        _require(access is not None, f"store {operation.op_id!r} has no access map")
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        pointer, mask = self._address(access, pad)
        value = self.schedule.buffer(operation.reads[0])
        _require(value is not None, f"store {operation.op_id!r} has no value buffer")
        if value.is_scalar and all(
            component.source is AccessIndexKind.PROGRAM for component in access.indices
        ):
            # A canonical [1] value can be a native scalar or a one-element block.
            # Give both the same one-element pointer domain; adding [0] preserves
            # the address/count and leaves reduction state and any mask unchanged.
            pointer = f"({pointer}) + tl.arange(0, 1)"
        self.line(f"{pad}tl.store(")
        self.line(f"{pad}    {pointer},")
        self.line(f"{pad}    {operation.reads[0]},")
        if mask:
            self.line(f"{pad}    mask={mask},")
        self.line(f"{pad})")

    def _emit_launch_options(self, constants: dict[str, int]) -> None:
        """Emit every Schedule-owned launch option for either host ABI."""

        for name, value in constants.items():
            if name != "NUM_WARPS":
                self.line(f"        {name}={value},")
        self.line(f"        num_warps={constants['NUM_WARPS']},")
        if "NUM_STAGES" in constants:
            # Pipelining depth belongs to the loop's range options. With no loop there is
            # nothing to pipeline and no declaration to carry, so the launch says nothing
            # rather than inventing a depth the Schedule never asked for.
            self.line(f"        num_stages={constants['NUM_STAGES']},")
        residency = self.schedule.residency
        if residency is not None and residency.registers_per_thread is not None:
            self.line(f"        maxnreg={residency.registers_per_thread},")

    def _emit_host(self, entry: str, kernel: str) -> None:
        globals_in_order = self._globals()
        inputs = [b for b in globals_in_order if b.mode.value == "input"]
        states = [b for b in globals_in_order if b.mode.value == "state"]
        if states:
            self._emit_host_with_state(entry, kernel, globals_in_order)
            return
        outputs = self._exported_outputs()
        output = outputs[0]
        names = ", ".join(b.name for b in inputs)
        constants = self.constants()

        self.line(f"def {entry}({names}, out=None):")
        self.line("    for tensor, shape, dtype in (")
        for buffer in inputs:
            self.line(
                f"        ({buffer.name}, {tuple(buffer.shape)}, {_TORCH_DTYPE[buffer.dtype]}),"
            )
        self.line("    ):")
        self.line("        if tuple(tensor.shape) != shape:")
        self.line("            raise ValueError(\"an input differs from the frozen shape\")")
        self.line("        if tensor.dtype != dtype:")
        self.line("            raise TypeError(\"an input differs from the frozen dtype\")")
        self.line("        if not tensor.is_cuda or not tensor.is_contiguous():")
        self.line("            raise ValueError(\"every input must be contiguous on CUDA\")")
        self.line(f"    if any(t.device != {inputs[0].name}.device for t in ({names},)):")
        self.line("        raise ValueError(\"every input must share one device\")")
        self._emit_output_binding(outputs, inputs[0].name)
        self.line(f"    {kernel}[{self.grid()}](")
        self._emit_launch_arguments(globals_in_order, outputs)
        self._emit_launch_options(constants)
        self.line("    )")
        self.line("    return out")

    def _exported_outputs(self) -> list[Buffer]:
        """The output Buffers in the order `Schedule.outputs` declares them.

        That field is what the caller reads its results back in, so it owns the order.
        Taking it from buffer declaration order instead gave the same Schedule two
        authorities for one fact, and they disagreed silently: a Schedule exporting
        `[y_hi, y_lo]` still handed back `y_lo` first.
        """

        return [
            buffer
            for buffer in (self.schedule.buffer(name) for name in self.schedule.outputs)
            if buffer is not None
        ]

    def _emit_output_binding(self, outputs: list[Buffer], anchor: str) -> None:
        """Allocate and check the caller's output tensors.

        One output binds the name `out` directly; several bind a sequence. The single
        case keeps its exact emitted bytes because the corpus pins the lowered source
        digest of every case, so drift there is a corpus-wide break for no gain.

        Both host wrappers route through here. They did not before, and the consequence
        was that the multi-output launch defect existed twice -- once per copy.
        """

        if len(outputs) == 1:
            output = outputs[0]
            self.line("    if out is None:")
            self.line(
                f"        out = torch.empty({tuple(output.shape)}, "
                f"dtype={_TORCH_DTYPE[output.dtype]}, device={anchor}.device)"
            )
            self.line(
                f"    if tuple(out.shape) != {tuple(output.shape)} "
                f"or out.dtype != {_TORCH_DTYPE[output.dtype]}:"
            )
            self.line("        raise ValueError(\"out differs from the frozen output contract\")")
            self.line(f"    if out.device != {anchor}.device or not out.is_contiguous():")
            self.line("        raise ValueError(\"out must be contiguous on the input device\")")
            return
        self.line("    if out is None:")
        self.line("        out = (")
        for buffer in outputs:
            self.line(
                f"            torch.empty({tuple(buffer.shape)}, "
                f"dtype={_TORCH_DTYPE[buffer.dtype]}, device={anchor}.device),"
            )
        self.line("        )")
        self.line("    out = tuple(out)")
        self.line(f"    if len(out) != {len(outputs)}:")
        self.line(
            f"        raise ValueError(\"out must provide {len(outputs)} output tensors\")"
        )
        self.line("    for tensor, shape, dtype in (")
        for index, buffer in enumerate(outputs):
            self.line(
                f"        (out[{index}], {tuple(buffer.shape)}, "
                f"{_TORCH_DTYPE[buffer.dtype]}),"
            )
        self.line("    ):")
        self.line("        if tuple(tensor.shape) != shape or tensor.dtype != dtype:")
        self.line(
            "            raise ValueError(\"out differs from the frozen output contract\")"
        )
        self.line(f"        if tensor.device != {anchor}.device or not tensor.is_contiguous():")
        self.line("            raise ValueError(\"out must be contiguous on the input device\")")

    def _emit_launch_arguments(
        self, globals_in_order: list[Buffer], outputs: list[Buffer]
    ) -> None:
        """Every global in declared order, with each output bound to its own slot."""

        positions = {buffer.name: index for index, buffer in enumerate(outputs)}
        for buffer in globals_in_order:
            if buffer.name in positions:
                slot = "out" if len(outputs) == 1 else f"out[{positions[buffer.name]}]"
            else:
                slot = buffer.name
            self.line(f"        {slot},")

    def _emit_host_with_state(
        self, entry: str, kernel: str, globals_in_order: list[Buffer]
    ) -> None:
        """Accept caller-owned mutable state without changing legacy wrappers."""

        caller_owned = [
            buffer
            for buffer in globals_in_order
            if buffer.mode in {BufferMode.INPUT, BufferMode.STATE}
        ]
        outputs = self._exported_outputs()
        names = ", ".join(buffer.name for buffer in caller_owned)
        constants = self.constants()
        anchor = caller_owned[0].name

        self.line(f"def {entry}({names}, out=None):")
        self.line("    for tensor, shape, dtype in (")
        for buffer in caller_owned:
            self.line(
                f"        ({buffer.name}, {tuple(buffer.shape)}, {_TORCH_DTYPE[buffer.dtype]}),"
            )
        self.line("    ):")
        self.line("        if tuple(tensor.shape) != shape:")
        self.line(
            '            raise ValueError("an input or state differs from the frozen shape")'
        )
        self.line("        if tensor.dtype != dtype:")
        self.line(
            '            raise TypeError("an input or state differs from the frozen dtype")'
        )
        self.line("        if not tensor.is_cuda or not tensor.is_contiguous():")
        self.line(
            '            raise ValueError("every input and state must be contiguous on CUDA")'
        )
        self.line(
            f"    if any(t.device != {anchor}.device for t in ({names},)):"
        )
        self.line('        raise ValueError("every input and state must share one device")')
        self._emit_output_binding(outputs, anchor)
        self.line(f"    {kernel}[{self.grid()}](")
        self._emit_launch_arguments(globals_in_order, outputs)
        self._emit_launch_options(constants)
        self.line("    )")
        self.line("    return out")


def emit(
    schedule: Schedule, target: Target, *, entry_point: str | None = None
) -> Emission:
    """Emit Triton source for one Schedule, or raise if it under-specifies."""

    return _TritonEmitter(schedule, target, entry_point).emit()
