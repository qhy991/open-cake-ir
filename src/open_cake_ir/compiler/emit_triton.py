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

from .emit import Emission, EmitError, require as _require
from .ir import (
    ElementwiseOp,
    LoadReuse,
    AccessIndexKind,
    AccessMap,
    Buffer,
    DType,
    IndexTieBreak,
    LoadMovement,
    MemorySpace,
    OperationKind,
    ProgramAxis,
    ReduceOp,
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
        identity="tl.zeros(({tile},), tl.float32)",
        accumulate="{out} += tl.sum({src}.to(tl.float32), axis={axis})",
        once="{out} = tl.sum({src}.to(tl.float32), axis={axis})",
    ),
    ReduceOp.MAX: _Reduction(
        identity='tl.full(({tile},), float("-inf"), tl.float32)',
        accumulate="{out} = tl.maximum({out}, tl.max({src}.to(tl.float32), axis={axis}))",
        once="{out} = tl.max({src}.to(tl.float32), axis={axis})",
    ),
}


# What this backend has a body for, and where. The two sets differ: an mma or a reduction
# is only emitted inside the tile loop, and a store only outside it. Keeping them as
# tables the dispatch reads means the coverage cannot drift from the code, and it makes
# the position-dependence a stated fact rather than the shape of two elif chains.
OUTSIDE_LOOP_EMITTERS: dict[OperationKind, str] = {
    OperationKind.LOAD: "_emit_load",
    OperationKind.MMA: "_emit_mma",
    OperationKind.ELEMENTWISE: "_emit_elementwise",
    OperationKind.REDUCE: "_emit_reduce",
    OperationKind.TOP_K: "_emit_top_k",
    OperationKind.STORE: "_emit_store",
}

INSIDE_LOOP_EMITTERS: dict[OperationKind, str] = {
    OperationKind.LOAD: "_emit_load",
    OperationKind.MMA: "_emit_mma",
    OperationKind.REDUCE_ARGMIN: "_emit_argmin",
    OperationKind.REDUCE: "_emit_reduce",
    OperationKind.TOP_K: "_emit_top_k",
    OperationKind.ELEMENTWISE: "_emit_elementwise",
}

# A kind outside this union can never be emitted, wherever it is placed, so the Compiler
# can say so before lowering. A kind inside it may still be refused for its position,
# which needs emission to discover and stays there.
SUPPORTED_OPERATION_KINDS = frozenset(OUTSIDE_LOOP_EMITTERS) | frozenset(
    INSIDE_LOOP_EMITTERS
)


class _TritonEmitter:
    def __init__(
        self, schedule: Schedule, target: Target, entry_point: str | None = None
    ) -> None:
        self.schedule = schedule
        self.target = target
        self.lines: list[str] = []
        # The entry point is the artifact's contract with whatever launches it, so the
        # Revision names it rather than the emitter inventing one from the profile.
        self.entry_point = entry_point or f"cake_{schedule.profile}"

        _require(schedule.program_map is not None, "a Triton Schedule maps its program")
        _require(schedule.access_maps, "a Triton Schedule declares its access maps")
        # At most one, not exactly one. A Schedule whose whole reduced axis is resident
        # has nothing to iterate, and requiring a loop there made the backend emit a trip
        # count of one with an unused iterator -- and forced a two-pass operator to carry
        # a value out of a loop, which is the one thing a loop must not do.
        _require(len(schedule.tile_loops) <= 1, "expected at most one tile loop")
        _require(len(schedule.roles) == 1, "expected exactly one role")
        self.loop = schedule.tile_loops[0] if schedule.tile_loops else None
        self.role = schedule.roles[0]

        # A kernel must write something, so a store is required of every Schedule. An
        # mma and a reduction are not: requiring them described the operator this backend
        # was written for rather than anything Triton needs, and a Schedule that reduces
        # without contracting was refused for missing a contraction it never claimed.
        self.store = self._single(OperationKind.STORE, "store")
        self.mma = self._at_most_one(OperationKind.MMA, "mma")
        self.reduce = self._at_most_one(OperationKind.REDUCE_ARGMIN, "reduce_argmin")
        if self.mma is not None:
            instruction = self.mma.parameters.instruction
            _require(instruction is not None, "the mma must name an instruction contract")
            _require(
                instruction.contract in target.instruction_contracts,
                f"instruction {instruction.contract!r} is not admitted by {target.target_id!r}",
            )
        for load in schedule.operations:
            if load.kind is OperationKind.LOAD:
                _require(
                    load.parameters.movement is LoadMovement.GLOBAL,
                    f"load {load.op_id!r} must move from global memory on this backend",
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
        if (
            self.loop is not None
            and self.loop.buffer == buffer_name
            and self.loop.dimension == dimension
        ):
            return f"N_{self.loop.name.upper()}"
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
        if self.loop is not None:
            values[self._tile(self.loop.name)] = self.loop.tile
            values["NUM_STAGES"] = self.loop.range_options.num_stages
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
                name = f"{buffer.name}_d{component.dimension}_offsets"
                expressions.append(name)
                vectors.append(name)
        return expressions, vectors

    def _bound(self, access: AccessMap, buffer: Buffer, vector: str) -> str | None:
        """The extent a masked tile axis is bounded by, if it is masked at all."""

        for position, component in enumerate(access.indices):
            if component.source is AccessIndexKind.PROGRAM_TILE and f"{component.name}_offsets" == vector:
                axis = self._axis(component.name)
                return self._extent(axis.buffer, axis.dimension)
            if (
                component.source is AccessIndexKind.LOOP_TILE
                and f"{component.name}_offsets" == vector
            ):
                # ACCESS_LOOP_UNKNOWN refuses this Schedule before it reaches emission,
                # so this cannot fire. It raises anyway because the alternative is a
                # silent fall-through to "needs no mask", which is a wrong kernel rather
                # than a refused one.
                _require(self.loop is not None, f"{vector} indexes a loop there is none of")
                return self._extent(self.loop.buffer, self.loop.dimension)
        return None  # a full-dimension index spans its axis and needs no mask

    def _address(self, access: AccessMap, pad: str) -> tuple[str, str]:
        """Pointer expression and mask for one access map."""

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
                    name = f"{buffer.name}_d{component.dimension}_offsets"
                    if f"{name} = " not in "\n".join(self.lines):
                        extent = self._extent(buffer.name, component.dimension)
                        self.line(f"{pad}{name} = tl.arange(0, {extent})")
        self.line()

        # Declared order is the authority. The loop is emitted where its body begins,
        # and everything else is dispatched by kind at that point in the sequence. The
        # previous shape emitted prologue loads, the loop, then the store, which silently
        # dropped any other operation declared outside the loop.
        emitted_loop = False
        for operation in self.schedule.operations:
            if self.loop is not None and operation.op_id in self.loop.body:
                if not emitted_loop:
                    self._emit_reduction_state(self._body_pad())
                    self._emit_loop()
                    emitted_loop = True
                continue
            method = OUTSIDE_LOOP_EMITTERS.get(operation.kind)
            if method is None:
                raise EmitError(
                    f"operation {operation.op_id!r} of kind "
                    f"{operation.kind.value!r} sits outside the loop and this backend "
                    "has no body for it there"
                )
            getattr(self, method)(operation, self._body_pad())
            if operation.kind is OperationKind.ELEMENTWISE:
                self.line()
        _require(
            emitted_loop or self.loop is None,
            "the declared tile loop names no operation",
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

    def _emit_reduction_state(self, pad: str) -> None:
        """A reduction carried across the loop needs its identity before the loop.

        Which identity depends on the reduction the Schedule declares, so this reads the
        operation rather than assuming the argmin the first profile happened to use.
        """

        tile = self._tile(self._token_axis().name)
        if self.reduce is not None:
            _require(
                self.reduce.parameters.across_loop,
                "this backend carries the reduction across the loop",
            )
            self.line(f"{pad}best_distance = tl.full(({tile},), float(\"inf\"), tl.float32)")
            self.line(f"{pad}{self.reduce.writes[0]} = tl.zeros(({tile},), tl.int32)")
            self.line()
        for operation in self.schedule.operations:
            if operation.op_id not in self.loop.body:
                continue
            if operation.kind is OperationKind.REDUCE:
                identity = REDUCTIONS[operation.parameters.op].identity
                self.line(f"{pad}{operation.writes[0]} = {identity.format(tile=tile)}")
                self.line()
            elif operation.kind is OperationKind.MMA and self._accumulating(operation):
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

    def _emit_loop(self) -> None:
        options = self.loop.range_options
        extent = self._extent(self.loop.buffer, self.loop.dimension)
        tile = self._tile(self.loop.name)
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
            f"{self._body_pad()}for {self.loop.iterator} in tl.range(0, {extent}, {tile}, "
            + ", ".join(knobs)
            + "):"
        )
        self.line(
            f"{self._body_pad()}    {self.loop.iterator}_offsets = "
            f"{self.loop.iterator} + tl.arange(0, {tile})"
        )
        for op_id in self.loop.body:
            operation = self.schedule.operation(op_id)
            _require(operation is not None, f"loop body names unknown operation {op_id!r}")
            method = INSIDE_LOOP_EMITTERS.get(operation.kind)
            if method is None:
                raise EmitError(
                    f"operation kind {operation.kind.value!r} has no Triton body emitter"
                )
            getattr(self, method)(operation, self._body_pad() + "    ")
        self.line()

    _ELEMENTWISE_TEXT = {
        ElementwiseOp.SQUARE: "{a} * {a}",
        ElementwiseOp.RSQRT: "tl.rsqrt({a})",
        ElementwiseOp.EXP: "tl.exp({a})",
        ElementwiseOp.ADD: "{a} + {b}",
        ElementwiseOp.SUB: "{a} - {b}",
        ElementwiseOp.MUL: "{a} * {b}",
        # Written as the division it is. A Schedule that wants the reciprocal-then-
        # multiply a fast kernel uses declares `rsqrt`-style reciprocal and `mul`, which
        # is a different Schedule with different numerics -- and says so.
        ElementwiseOp.DIV: "{a} / {b}",
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
                a=operands[0], b=operands[1] if len(operands) > 1 else ""
            )
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(f"{pad}{operation.writes[0]} = {expression}")

    def _operand(self, name: str, operation) -> str:
        """A read, indexed so it spans the declared axis of the wider operand."""

        buffer = self.schedule.buffer(name)
        _require(buffer is not None, f"elementwise reads unknown buffer {name!r}")
        widest = max(
            (
                other.shape
                for other in (self.schedule.buffer(read) for read in operation.reads)
                if other is not None
            ),
            key=len,
            default=(),
        )
        if len(buffer.shape) == len(widest):
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
        carried = self.loop is not None and operation.op_id in self.loop.body
        template = reduction.accumulate if carried else reduction.once
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(
            pad
            + template.format(
                out=operation.writes[0], src=operation.reads[0], axis=axis
            )
        )

    def _emit_mma(self, operation, pad: str) -> None:
        """The contraction, and nothing else.

        This used to emit the dot and the distance formula together, because MmaFormula
        named the pair as one thing. Two of the three admitted profiles already declared
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
        self.line(
            f"{pad}{operation.writes[0]} {assign} tl.dot({tiles[0]}, tl.trans({tiles[1]}))"
        )

    def _accumulating(self, operation) -> bool:
        """Whether this contraction sums across the loop it sits in."""

        return (
            self.loop is not None
            and operation.op_id in self.loop.body
            and self.schedule.mma_accumulates_over(operation, self.loop)
        )

    def _emit_argmin(self, operation, pad: str) -> None:
        source = operation.reads[0]
        best = operation.writes[0]
        lowest = operation.parameters.tie_break is IndexTieBreak.LOWEST_INDEX
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(f"{pad}block_position = tl.argmin(")
        self.line(f"{pad}    {source}, axis=1, tie_break_left={lowest},")
        self.line(f"{pad})")
        self.line(f"{pad}block_distance = tl.min({source}, axis=1)")
        self.line(
            f"{pad}candidate_index = {self.loop.iterator} + block_position"
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

        The current SM100 Triton path has one physical implementation: repeated maximum
        value and minimum matching-index reductions. An explicit selected-position mask
        keeps legal negative-infinity values distinct; replacing a winner with negative
        infinity alone would select it again when every remaining value is also -inf.
        """

        source = self.schedule.buffer(operation.reads[0])
        _require(source is not None, f"top_k reads unknown buffer {operation.reads[0]!r}")
        values, indices = operation.writes
        k = operation.parameters.k
        prefix = operation.op_id
        selected = f"{prefix}_selected"
        slots = f"{prefix}_slots"
        positions = f"{prefix}_source_positions"

        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        self.line(f'{pad}{values} = tl.full(({k},), float("-inf"), tl.float32)')
        self.line(f"{pad}{indices} = tl.zeros(({k},), tl.int32)")
        self.line(f"{pad}{slots} = tl.arange(0, {k})")
        self.line(f"{pad}{positions} = tl.arange(0, {source.shape[0]})")
        self.line(f"{pad}{selected} = tl.zeros(({source.shape[0]},), tl.int1)")
        for slot in range(k):
            value = f"{prefix}_value_{slot}"
            index = f"{prefix}_index_{slot}"
            candidates = f"{prefix}_candidates_{slot}"
            matching = f"{prefix}_matching_{slot}"
            self.line(
                f'{pad}{candidates} = tl.where(~{selected}, '
                f'{operation.reads[0]}, float("-inf"))'
            )
            self.line(f"{pad}{value} = tl.max({candidates}, axis=0)")
            self.line(
                f"{pad}{matching} = (~{selected}) & "
                f"({operation.reads[0]} == {value})"
            )
            self.line(
                f"{pad}{index} = tl.min(tl.where({matching}, {positions}, "
                f"{source.shape[0]}), axis=0)"
            )
            self.line(f"{pad}{values} = tl.where({slots} == {slot}, {value}, {values})")
            self.line(f"{pad}{indices} = tl.where({slots} == {slot}, {index}, {indices})")
            self.line(f"{pad}{selected} |= {positions} == {index}")

    def _emit_store(self, operation, pad: str) -> None:
        access = self.schedule.access_map(operation.op_id, operation.writes[0])
        _require(access is not None, f"store {operation.op_id!r} has no access map")
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        pointer, mask = self._address(access, pad)
        self.line(f"{pad}tl.store(")
        self.line(f"{pad}    {pointer},")
        self.line(f"{pad}    {operation.reads[0]},")
        if mask:
            self.line(f"{pad}    mask={mask},")
        self.line(f"{pad})")

    def _emit_host(self, entry: str, kernel: str) -> None:
        globals_in_order = self._globals()
        inputs = [b for b in globals_in_order if b.mode.value == "input"]
        output = next(b for b in globals_in_order if b.mode.value == "output")
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
        self.line("    if out is None:")
        self.line(
            f"        out = torch.empty({tuple(output.shape)}, "
            f"dtype={_TORCH_DTYPE[output.dtype]}, device={inputs[0].name}.device)"
        )
        self.line(
            f"    if tuple(out.shape) != {tuple(output.shape)} "
            f"or out.dtype != {_TORCH_DTYPE[output.dtype]}:"
        )
        self.line("        raise ValueError(\"out differs from the frozen output contract\")")
        self.line(f"    if out.device != {inputs[0].name}.device or not out.is_contiguous():")
        self.line("        raise ValueError(\"out must be contiguous on the input device\")")
        self.line(f"    {kernel}[{self.grid()}](")
        for buffer in globals_in_order:
            self.line(f"        {'out' if buffer is output else buffer.name},")
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
        self.line("    )")
        self.line("    return out")


def emit(
    schedule: Schedule, target: Target, *, entry_point: str | None = None
) -> Emission:
    """Emit Triton source for one Schedule, or raise if it under-specifies."""

    return _TritonEmitter(schedule, target, entry_point).emit()
