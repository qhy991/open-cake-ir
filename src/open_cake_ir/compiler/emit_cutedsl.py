"""Emit CuTe-DSL source from a Schedule.

This is lowering in the sense the paper means it: "a schedule states what is to happen
and lowering derives how -- barrier addresses, phase bits, TMEM offsets, descriptor
encodings, and warp identity are all computed from the declarations rather than written
out by the agent."

The backend is CuTe-DSL rather than raw PTX because CuTe-DSL sits at the abstraction
level the IR describes: it exposes SMEM view offsets, TMEM column ranges, swizzles, TMA
descriptors and warp roles, while owning register allocation and instruction selection.
Emitting Triton instead would hide four of the five commitments the IR exists to carry.

Identifiers are derived from Schedule names, so every name in the generated source
traces back to a declaration. The retained hand-written artifact used chosen names
(`operand_mbarriers` for the `tiles_ready` barrier); those do not derive, and
reproducing them would mean hardcoding the thing this module exists to compute.
"""

from __future__ import annotations

from .emit import Emission, EmitError, require as _require

from .ir import (
    AccessIndexKind,
    Barrier,
    BarrierMechanism,
    DType,
    LoadMovement,
    MemorySpace,
    OperationKind,
    PipelineKind,
    Role,
    Schedule,
    Swizzle,
    TileLoop,
)
from .target import Target


_CUTLASS_DTYPE = {
    DType.BF16: "cutlass.BFloat16",
    DType.FP16: "cutlass.Float16",
    DType.FP32: "cutlass.Float32",
    DType.FP8_E4M3: "cutlass.Float8E4M3FN",
    DType.INT32: "cutlass.Int32",
}

_TORCH_DTYPE = {
    DType.BF16: "torch.bfloat16",
    DType.FP16: "torch.float16",
    DType.FP32: "torch.float32",
    DType.INT32: "torch.int32",
}

_SWIZZLE_BYTES = {
    Swizzle.NONE: 0,
    Swizzle.B32: 32,
    Swizzle.B64: 64,
    Swizzle.B128: 128,
}


class _Emitter:
    def __init__(
        self, schedule: Schedule, target: Target, entry_point: str | None = None
    ) -> None:
        self.schedule = schedule
        self.target = target
        self.lines: list[str] = []
        # The entry point is the artifact's contract with whatever launches it, so the
        # Revision names it rather than the emitter inventing one from the profile.
        self.entry_point = entry_point or f"cake_{schedule.profile}"

        self.mma = self._single(OperationKind.MMA, "mma")
        self.epilogue = self._single(OperationKind.EPILOGUE, "epilogue")
        _require(
            self.mma.parameters.instruction is not None,
            "mma operation must commit to an instruction atom",
        )
        _require(
            self.mma.parameters.tile_shape is not None,
            "mma operation must commit to a tile shape",
        )
        self.atom = self.mma.parameters.instruction
        self.tile = self.mma.parameters.tile_shape

        _require(len(schedule.pipelines) == 1, "expected exactly one pipeline")
        self.pipeline = schedule.pipelines[0]

        self.loads = tuple(
            op for op in schedule.operations if op.kind is OperationKind.LOAD
        )
        _require(self.loads, "expected at least one load")
        for load in self.loads:
            _require(
                load.parameters.movement is LoadMovement.TMA,
                f"load {load.op_id!r} must move by TMA on this backend",
            )
            _require(
                load.parameters.descriptor_box is not None,
                f"load {load.op_id!r} must commit to a descriptor box",
            )

        self.roles = {role.name: role for role in schedule.roles}
        self.nest = self._loop_nest()

    # ---------------------------------------------------------------- derivation

    def _single(self, kind: OperationKind, label: str):
        matches = [op for op in self.schedule.operations if op.kind is kind]
        _require(len(matches) == 1, f"expected exactly one {label} operation")
        return matches[0]

    def _loop_nest(self) -> tuple[TileLoop, ...]:
        """Outermost-first chain of tile loops."""

        parent = self.schedule.loop_parent()
        ordered = sorted(
            self.schedule.tile_loops, key=lambda loop: self.schedule.loop_depth(loop.name)
        )
        for child, enclosing in parent.items():
            _require(
                self.schedule.loop_depth(child) == self.schedule.loop_depth(enclosing) + 1,
                f"loop {child!r} is not directly inside {enclosing!r}",
            )
        return tuple(ordered)

    def _trip_count(self, loop: TileLoop) -> int:
        buffer = self.schedule.buffer(loop.buffer)
        _require(buffer is not None, f"loop {loop.name!r} names an unknown buffer")
        extent = buffer.shape[loop.dimension]
        _require(
            extent % loop.tile == 0,
            f"loop {loop.name!r} tile {loop.tile} does not divide extent {extent}",
        )
        return extent // loop.tile

    @property
    def io_dtype(self) -> DType:
        operands = [self.schedule.buffer(name) for name in self.mma.reads]
        dtypes = {b.dtype for b in operands if b is not None and b.space is MemorySpace.SHARED}
        _require(len(dtypes) == 1, "MMA operands must share one element type")
        return next(iter(dtypes))

    @property
    def accumulator(self):
        writes = [self.schedule.buffer(name) for name in self.mma.writes]
        tensor = [b for b in writes if b is not None and b.space is MemorySpace.TENSOR]
        _require(len(tensor) == 1, "mma must write exactly one tensor-memory accumulator")
        return tensor[0]

    def role_constant(self, role: Role) -> str:
        upper = role.name.upper()
        return f"{upper}_WARP" if len(role.warps) == 1 else f"{upper}_WARPS"

    def constants(self) -> dict[str, object]:
        values: dict[str, object] = {
            "IO_DTYPE": _CUTLASS_DTYPE[self.io_dtype],
            "ACC_DTYPE": _CUTLASS_DTYPE[self.mma.parameters.accumulator],
            "MMA_INSTRUCTION_SHAPE": tuple(self.atom.shape),
            "MMA_TILE": tuple(self.tile),
            "PIPELINE_STAGES": self.pipeline.stages,
            "THREADS_PER_CTA": self.schedule.total_warp_extent * self.target.warp_size,
        }
        for loop in self.nest:
            values[f"NUM_{loop.name.upper()}_TRIPS"] = self._trip_count(loop)
        for role in self.schedule.roles:
            name = self.role_constant(role)
            values[name] = role.warps[0] if len(role.warps) == 1 else tuple(role.warps)
            if len(role.warps) > 1:
                values[f"{role.name.upper()}_THREADS"] = (
                    len(role.warps) * self.target.warp_size
                )
        allocation = next(
            (a for a in self.schedule.allocations if a.space is MemorySpace.TENSOR), None
        )
        if allocation is not None and allocation.tensor_columns is not None:
            values["TMEM_COLUMNS"] = allocation.tensor_columns
        return values

    # ------------------------------------------------------------------- emission

    def line(self, text: str = "") -> None:
        self.lines.append(text)

    def emit(self) -> Emission:
        self._emit_header()
        self._emit_constants()
        self._emit_shared_storage()
        self._emit_kernel()
        self._emit_host()
        entry = self.entry_point
        return Emission("\n".join(self.lines) + "\n", entry, self.constants())

    def _emit_header(self) -> None:
        self.line(f"# Generated by open-cake-ir from {self.schedule.schedule_id}; DO NOT EDIT.")
        self.line("# schedule_sha256=__SCHEDULE_SHA256__")
        self.line("import cutlass")
        self.line("import cutlass.cute as cute")
        self.line("import cutlass.pipeline as pipeline")
        self.line("import cutlass.utils as utils")
        self.line("import cutlass.utils.blackwell_helpers as sm100_utils")
        self.line("from cutlass.cute.nvgpu import cpasync, tcgen05")
        self.line("from cutlass.cute.runtime import from_dlpack")
        self.line()
        self.line()

    def _emit_constants(self) -> None:
        for name, value in self.constants().items():
            rendered = value if isinstance(value, str) else repr(value)
            self.line(f"{name} = {rendered}")
        self.line()
        self.line()

    def _barrier_field(self, barrier: Barrier) -> str:
        return f"{barrier.name}_mbarriers"

    def _barrier_slots(self, barrier: Barrier) -> str:
        """Two mbarriers per stage: one full, one empty."""

        if barrier.pipeline is not None:
            return "PIPELINE_STAGES * 2"
        return "2"

    def _emit_shared_storage(self) -> None:
        self.line("@cute.struct")
        self.line("class SharedStorage:")
        emitted = False
        for barrier in self.schedule.barriers:
            if not self._is_mbarrier(barrier):
                continue
            self.line(
                f"    {self._barrier_field(barrier)}: "
                f"cute.struct.MemRange[cutlass.Int64, {self._barrier_slots(barrier)}]"
            )
            emitted = True
        if any(a.space is MemorySpace.TENSOR for a in self.schedule.allocations):
            self.line("    tmem_holding_buffer: cutlass.Int32")
            emitted = True
        if not emitted:
            self.line("    pass")
        self.line()
        self.line()

    def _is_mbarrier(self, barrier: Barrier) -> bool:
        """Only an mbarrier occupies shared-memory state; a named CTA barrier does not."""

        _require(
            barrier.mechanism is not None,
            f"barrier {barrier.name!r} must declare how it is realized",
        )
        return barrier.mechanism is BarrierMechanism.MBARRIER

    def _kernel_name(self) -> str:
        return f"_cake_{self.schedule.profile}_kernel"

    def _emit_kernel(self) -> None:
        self.line("@cute.kernel")
        self.line(f"def {self._kernel_name()}(")
        self.line("    tiled_mma: cute.TiledMma,")
        for load in self.loads:
            self.line(f"    {load.op_id}_atom: cute.CopyAtom,")
            self.line(f"    {load.op_id}_source: cute.Tensor,")
        for buffer in self.schedule.buffers:
            if buffer.space is MemorySpace.GLOBAL and buffer.name not in self._tma_sources():
                self.line(f"    {buffer.name}: cute.Tensor,")
        for load in self.loads:
            self.line(f"    {load.op_id}_smem_layout: cute.ComposedLayout,")
        self.line("):")
        self.line("    thread_idx, _, _ = cute.arch.thread_idx()")
        self.line("    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())")
        self.line()
        self.line("    smem = utils.SmemAllocator()")
        self.line("    storage = smem.allocate(SharedStorage)")
        for load in self.loads:
            staged = load.writes[0]
            buffer = self.schedule.buffer(staged)
            swizzle = "" if buffer is None or buffer.swizzle is None else (
                f"\n        swizzle={load.op_id}_smem_layout.inner,"
            )
            self.line(f"    {staged} = smem.allocate_tensor(")
            self.line("        element_type=IO_DTYPE,")
            self.line(f"        layout={load.op_id}_smem_layout.outer,")
            self.line("        byte_alignment=128,")
            if swizzle:
                self.line(f"        swizzle={load.op_id}_smem_layout.inner,")
            self.line("    )")
        self.line()
        self._emit_prologue()
        self._emit_role_dispatch()

    def _tma_sources(self) -> set[str]:
        return {load.reads[0] for load in self.loads}

    # ---- derived pipeline and scope facts --------------------------------------

    def _pipeline_class(self, barrier: Barrier) -> str:
        """Which CuTe pipeline realizes a handshake, from typed producer semantics.

        A TMA producer feeding the MMA is a TmaUmma pipeline; the MMA producing an
        accumulator for threads is UmmaAsync. The verifier requires every signaller to
        resolve to the same kind; direct emitter calls fail closed on the same condition.
        """

        signallers = [
            op for op in self.schedule.operations if barrier.name in op.signals
        ]
        kinds = {operation.produced_pipeline_kind for operation in signallers}
        if None in kinds or len(kinds) != 1:
            rendered = sorted(
                "unsupported" if kind is None else kind.value for kind in kinds
            )
            raise EmitError(
                f"barrier {barrier.name!r} needs one lowerable pipeline kind; got "
                f"{rendered or ['none']}"
            )
        kind = kinds.pop()
        return {
            PipelineKind.TMA_TO_UMMA: "PipelineTmaUmma",
            PipelineKind.UMMA_TO_THREAD: "PipelineUmmaAsync",
        }[kind]

    def _participants(self, barrier: Barrier) -> tuple[str, str]:
        return f"{barrier.name}_producer", f"{barrier.name}_consumer"

    def _mbarriers(self) -> tuple[Barrier, ...]:
        return tuple(b for b in self.schedule.barriers if self._is_mbarrier(b))

    def _scope_of(self, op_id: str) -> str | None:
        """The innermost loop whose body names this operation, if any."""

        for loop in self.schedule.tile_loops:
            if op_id in loop.body:
                return loop.name
        return None

    def _ops_in_scope(self, role: Role, scope: str | None) -> list:
        return [
            op
            for op in self.schedule.operations
            if op.role == role.name and self._scope_of(op.op_id) == scope
        ]

    def _loops_for_role(self, role: Role) -> tuple[TileLoop, ...]:
        """Loops this role actually enters, outermost first."""

        names = {
            self._scope_of(op.op_id)
            for op in self.schedule.operations
            if op.role == role.name
        }
        wanted: set[str] = set()
        parent = self.schedule.loop_parent()
        for name in names:
            while name is not None:
                wanted.add(name)
                name = parent.get(name)
        return tuple(loop for loop in self.nest if loop.name in wanted)

    def _emit_prologue(self) -> None:
        """Partitioning, TMEM custody and pipeline participants, all derived.

        Every quantity here follows from a declaration: the tile from the MMA atom, the
        staging depth from the pipeline, the participant classes from who produces and
        consumes each barrier, the TMEM allocator's owning warp from the role that reads
        tensor memory.
        """

        a, b = self.loads[0], self.loads[1]
        self.line(f"    tiled_{a.op_id} = cute.local_tile(")
        self.line(f"        {a.op_id}_source, MMA_TILE, (0, None, None), proj=(1, None, 1)")
        self.line("    )")
        self.line(f"    tiled_{b.op_id} = cute.local_tile(")
        self.line(f"        {b.op_id}_source, MMA_TILE, (None, None, None), proj=(None, 1, 1)")
        self.line("    )")
        scratch = self._epilogue_global()
        self.line(f"    tiled_{scratch} = cute.local_tile(")
        self.line(f"        {scratch}, MMA_TILE, (0, None, None), proj=(1, 1, None)")
        self.line("    )")
        self.line("    mma_slice = tiled_mma.get_slice(0)")
        self.line(f"    mma_{a.op_id} = mma_slice.partition_A(tiled_{a.op_id})")
        self.line(f"    mma_{b.op_id} = mma_slice.partition_B(tiled_{b.op_id})")
        self.line(f"    mma_{scratch} = mma_slice.partition_C(tiled_{scratch})")
        self.line("    identity = cute.make_identity_tensor(MMA_TILE[:2])")
        self.line("    mma_identity = mma_slice.partition_C(identity)")
        self.line(f"    fragment_a = tiled_mma.make_fragment_A({a.writes[0]})")
        self.line(f"    fragment_b = tiled_mma.make_fragment_B({b.writes[0]})")
        self.line("    accumulator_template = tiled_mma.make_fragment_C(")
        self.line("        tiled_mma.partition_shape_C(MMA_TILE[:2])")
        self.line("    )")
        self.line()
        for load, partitioned in ((a, f"mma_{a.op_id}"), (b, f"mma_{b.op_id}")):
            self.line(
                f"    tma_shared_{load.op_id}, tma_global_{load.op_id} = "
                "cute.nvgpu.cpasync.tma_partition("
            )
            self.line(f"        {load.op_id}_atom,")
            self.line("        0,")
            self.line("        cute.make_layout(1),")
            self.line(f"        cute.group_modes({load.writes[0]}, 0, 3),")
            self.line(f"        cute.group_modes({partitioned}, 0, 3),")
            self.line("    )")
        self.line()

        owner = self._tmem_owner()
        self.line("    tmem_sync = pipeline.NamedBarrier(")
        self.line("        barrier_id=1,")
        self.line(
            f"        num_threads=(len({self.role_constant(owner)}) + 1) * 32,"
            if len(owner.warps) > 1
            else "        num_threads=2 * 32,"
        )
        self.line("    )")
        self.line("    tmem = utils.TmemAllocator(")
        self.line("        storage.tmem_holding_buffer,")
        self.line("        barrier_for_retrieve=tmem_sync,")
        self.line(
            f"        allocator_warp_id={self.role_constant(owner)}[0],"
            if len(owner.warps) > 1
            else f"        allocator_warp_id={self.role_constant(owner)},"
        )
        self.line("    )")
        self.line()
        self.line("    transaction_bytes = " + " + ".join(
            f"cute.size_in_bytes(IO_DTYPE, cute.select({load.op_id}_smem_layout, mode=[0, 1, 2]))"
            for load in self.loads
        ))
        for barrier in self._mbarriers():
            producer, consumer = self._participants(barrier)
            klass = self._pipeline_class(barrier)
            stages = "PIPELINE_STAGES" if barrier.pipeline is not None else "1"
            self.line(f"    {producer}, {consumer} = pipeline.{klass}.create(")
            self.line(f"        barrier_storage=storage.{barrier.name}_mbarriers.data_ptr(),")
            self.line(f"        num_stages={stages},")
            self.line("        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),")
            consumer_role = self.roles[barrier.consumers[0]]
            if klass == "PipelineUmmaAsync" and len(consumer_role.warps) > 1:
                self.line("        consumer_group=pipeline.CooperativeGroup(")
                self.line(
                    f"            pipeline.Agent.Thread, size={consumer_role.name.upper()}_THREADS"
                )
                self.line("        ),")
            else:
                self.line("        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),")
            if klass == "PipelineTmaUmma":
                self.line("        tx_count=transaction_bytes,")
            self.line("    ).make_participants()")
        self.line()

    def _epilogue_global(self) -> str:
        for name in self.epilogue.writes:
            buffer = self.schedule.buffer(name)
            if buffer is not None and buffer.space is MemorySpace.GLOBAL:
                return name
        raise EmitError("epilogue must write one global buffer")

    def _tmem_owner(self) -> Role:
        """The role that allocates tensor memory: the one that reads the accumulator."""

        for operation in self.schedule.operations:
            for name in operation.reads:
                buffer = self.schedule.buffer(name)
                if buffer is not None and buffer.space is MemorySpace.TENSOR:
                    return self.roles[operation.role]
        raise EmitError("no role reads tensor memory")

    def _named_barriers(self) -> tuple[Barrier, ...]:
        return tuple(
            b
            for b in self.schedule.barriers
            if b.mechanism is BarrierMechanism.NAMED
        )

    def _dispatch_phases(self) -> list[tuple[Barrier | None, list[Role]]]:
        """Split the roles around every CTA-wide barrier.

        A named barrier is a whole-CTA rendezvous, so it must sit outside the warp
        dispatch -- every warp has to arrive. Emitting it inside one role's branch
        deadlocks the other six. The consumers of a named barrier therefore run in a
        second dispatch block behind it.
        """

        active = [
            role
            for role in sorted(self.schedule.roles, key=lambda r: r.warps[0])
            if any(op.role == role.name for op in self.schedule.operations)
        ]
        deferred: dict[str, Barrier] = {}
        for barrier in self._named_barriers():
            for name in barrier.consumers:
                deferred[name] = barrier
        phases: list[tuple[Barrier | None, list[Role]]] = [
            (None, [r for r in active if r.name not in deferred])
        ]
        for barrier in self._named_barriers():
            waiting = [r for r in active if deferred.get(r.name) is barrier]
            if waiting:
                phases.append((barrier, waiting))
        return phases

    def _emit_role_dispatch(self) -> None:
        """Warp identity is computed from the declared roles, not written by the agent."""

        for barrier, roles in self._dispatch_phases():
            if barrier is not None:
                self.line(
                    f"    # {barrier.name}: a CTA-wide rendezvous, outside the dispatch"
                )
                self.line("    cute.arch.sync_threads()")
            first = True
            for role in roles:
                keyword = "if" if first else "elif"
                first = False
                if len(role.warps) == 1:
                    condition = f"warp_idx == {self.role_constant(role)}"
                else:
                    low, high = min(role.warps), max(role.warps)
                    condition = f"{low} <= warp_idx <= {high}"
                self.line(f"    {keyword} {condition}:")
                self._emit_role_body(role, indent=8)
        self.line("    # CAKE_KERNEL_END")
        self.line()
        self.line()

    def _emit_role_body(self, role: Role, indent: int) -> None:
        pad = " " * indent
        if role is self._tmem_owner():
            self.line(f"{pad}tmem.allocate(TMEM_COLUMNS)")
        if any(
            self.schedule.buffer(name) is not None
            and self.schedule.buffer(name).space is MemorySpace.TENSOR
            for op in self.schedule.operations
            if op.role == role.name
            for name in list(op.reads) + list(op.writes)
        ):
            self.line(f"{pad}tmem.wait_for_alloc()")
            self.line(f"{pad}accumulator = cute.make_tensor(")
            self.line(f"{pad}    tmem.retrieve_ptr(ACC_DTYPE), accumulator_template.layout")
            self.line(f"{pad})")

        loops = self._loops_for_role(role)
        for operation in self._ops_in_scope(role, None):
            if self._emitted_with_producer(operation):
                continue
            self._emit_operation(operation, indent)
        self._emit_loop_nest(role, loops, 0, indent)

        for barrier in self._mbarriers():
            if role.name in barrier.producers:
                self.line(f"{pad}{self._participants(barrier)[0]}.tail()")
        if role is self._tmem_owner():
            self.line(f"{pad}tmem.relinquish_alloc_permit()")
            self.line(f"{pad}tmem.free(tmem.retrieve_ptr(ACC_DTYPE))")

    def _emit_loop_nest(self, role: Role, loops, depth: int, indent: int) -> None:
        if depth >= len(loops):
            return
        loop = loops[depth]
        pad = " " * indent
        trips = f"NUM_{loop.name.upper()}_TRIPS"
        # Always bind the iterator. An operation in an inner scope may still address an
        # outer axis -- the B operand is tiled in N by the outer loop and read by a load
        # that lives in the inner one.
        self.line(f"{pad}for {loop.iterator} in cutlass.range({trips}):")

        acquired = [
            b
            for b in self._mbarriers()
            if role.name in b.producers and self._barrier_scope(b) == loop.name
        ]
        for barrier in acquired:
            self.line(
                f"{pad}    {barrier.name}_empty = "
                f"{self._participants(barrier)[0]}.acquire_and_advance()"
            )
        for operation in self._ops_in_scope(role, loop.name):
            if self._emitted_with_producer(operation):
                continue
            self._emit_operation(operation, indent + 4)
        self._emit_loop_nest(role, loops, depth + 1, indent + 4)
        for barrier in acquired:
            if self._pipeline_class(barrier) == "PipelineTmaUmma":
                continue  # the TMA transaction completes this barrier itself
            self.line(f"{pad}    {barrier.name}_empty.commit()")

    def _barrier_scope(self, barrier: Barrier) -> str | None:
        """The loop a barrier is acquired in: the scope of whatever signals it."""

        signallers = [op for op in self.schedule.operations if barrier.name in op.signals]
        scopes = {self._scope_of(op.op_id) for op in signallers}
        if len(scopes) != 1:
            raise EmitError(
                f"barrier {barrier.name!r} is signalled from more than one loop scope"
            )
        scope = scopes.pop()
        if barrier.pipeline is not None:
            return scope
        parent = self.schedule.loop_parent()
        return parent.get(scope, scope) if scope else scope

    def _loop_axis(self, loop: TileLoop) -> str:
        """Whether a loop walks the MMA tile's N or K axis, by matching its tile."""

        if loop.tile == self.tile[1]:
            return "N"
        if loop.tile == self.tile[2]:
            return "K"
        raise EmitError(
            f"loop {loop.name!r} tile {loop.tile} matches neither the MMA tile N "
            f"({self.tile[1]}) nor K ({self.tile[2]})"
        )

    def _axis_iterators(self) -> dict[str, str]:
        return {self._loop_axis(loop): loop.iterator for loop in self.nest}

    def _emit_operation(self, operation, indent: int) -> None:
        pad = " " * indent
        self.line(f"{pad}# CAKE_OP:{operation.op_id}")
        kind = operation.kind
        if kind is OperationKind.LOAD:
            self._emit_load(operation, pad)
        elif kind is OperationKind.MMA:
            self._emit_mma(operation, pad)
        elif kind is OperationKind.EPILOGUE:
            self._emit_epilogue(operation, pad)
        elif kind is OperationKind.REDUCE_ARGMIN:
            self._emit_argmin(operation, pad)
        elif kind is OperationKind.STORE:
            self._emit_store(operation, pad)
        else:
            raise EmitError(
                f"operation kind {kind.value!r} has no CuTe-DSL body emitter"
            )

    def _emit_load(self, operation, pad: str) -> None:
        """A TMA copy into the stage the barrier this load signals is gating."""

        barrier = next(
            (b for b in self._mbarriers() if b.name in operation.signals), None
        )
        if barrier is None:
            raise EmitError(f"load {operation.op_id!r} signals no mbarrier")
        axes = self._axis_iterators()
        # Operand A is tiled in K alone; operand B is tiled in N and K. Which one this
        # load feeds follows from its position in the MMA's reads.
        index = list(self.mma.reads).index(operation.writes[0])
        coordinate = (
            f"(None, {axes['K']})"
            if index == 0
            else f"(None, {axes['N']}, {axes['K']})"
        )
        self.line(f"{pad}cute.copy(")
        self.line(f"{pad}    {operation.op_id}_atom,")
        self.line(f"{pad}    tma_global_{operation.op_id}[{coordinate}],")
        self.line(
            f"{pad}    tma_shared_{operation.op_id}"
            f"[(None, {barrier.name}_empty.index)],"
        )
        self.line(f"{pad}    tma_bar_ptr={barrier.name}_empty.barrier,")
        self.line(f"{pad})")

    def _emit_mma(self, operation, pad: str) -> None:
        waited = next(
            (b for b in self._mbarriers() if b.name in operation.waits), None
        )
        if waited is None:
            raise EmitError(f"mma {operation.op_id!r} waits on no mbarrier")
        axes = self._axis_iterators()
        consumer = self._participants(waited)[1]
        self.line(f"{pad}{waited.name}_full = {consumer}.wait_and_advance()")
        self.line(
            f"{pad}tiled_mma.set(tcgen05.Field.ACCUMULATE, {axes['K']} != 0)"
        )
        self.line(f"{pad}stage = (None, None, None, {waited.name}_full.index)")
        self.line(f"{pad}cute.gemm(")
        self.line(f"{pad}    tiled_mma, accumulator, fragment_a[stage], fragment_b[stage], accumulator")
        self.line(f"{pad})")
        self.line(f"{pad}{waited.name}_full.release()")

    def _emit_epilogue(self, operation, pad: str) -> None:
        """Read the accumulator through the declared copy atom, apply the formula."""

        parameters = operation.parameters
        if parameters.subtile is None or parameters.source_atom is None:
            raise EmitError(
                f"epilogue {operation.op_id!r} must commit to a subtile and a source atom"
            )
        waited = next((b for b in self._mbarriers() if b.name in operation.waits), None)
        if waited is None:
            raise EmitError(f"epilogue {operation.op_id!r} waits on no mbarrier")
        axes = self._axis_iterators()
        scratch = self._epilogue_global()
        norm = next(
            name
            for name in operation.reads
            if (buffer := self.schedule.buffer(name)) is not None
            and buffer.space is MemorySpace.GLOBAL
        )
        atom = parameters.source_atom
        self.line(f"{pad}{self._participants(waited)[1]}.wait_and_advance()")
        self.line(f"{pad}epilogue_tiler = ({parameters.subtile[0]}, {parameters.subtile[1]})")
        self.line(f"{pad}accumulator_tiles = cute.zipped_divide(accumulator, (epilogue_tiler,))")
        self.line(f"{pad}coordinate_tiles = cute.zipped_divide(mma_identity, (epilogue_tiler,))")
        self.line(f"{pad}source_atom = cute.make_copy_atom(")
        self.line(
            f"{pad}    tcgen05.Ld32x32bOp(tcgen05.Repetition.x{atom.repetition}), ACC_DTYPE"
        )
        self.line(f"{pad})")
        self.line(
            f"{pad}source_copy = tcgen05.make_tmem_copy(source_atom, accumulator_tiles[None, 0])"
        )
        self.line(f"{pad}thread_copy = source_copy.get_slice(thread_idx)")
        self.line(f"{pad}tmem_source = thread_copy.partition_S(accumulator_tiles)")
        self.line(f"{pad}coordinate_source = thread_copy.partition_D(coordinate_tiles)")
        self.line(f"{pad}output_tile = mma_{scratch}[(None, None, None, {axes['N']})]")
        self.line(f"{pad}output_tiles = cute.zipped_divide(output_tile, (epilogue_tiler,))")
        self.line(f"{pad}destination = thread_copy.partition_D(output_tiles)")
        self.line(f"{pad}registers = cute.make_rmem_tensor(")
        self.line(f"{pad}    destination[None, None, 0].shape, ACC_DTYPE")
        self.line(f"{pad})")
        self.line(f"{pad}base = {axes['N']} * MMA_TILE[1]")
        self.line(f"{pad}for subtile in cutlass.range(cute.size(tmem_source, mode=[2])):")
        self.line(f"{pad}    cute.copy(source_copy, tmem_source[None, None, subtile], registers)")
        self.line(f"{pad}    coordinates = coordinate_source[None, None, subtile]")
        self.line(f"{pad}    for value in range(cute.size(registers)):")
        self.line(f"{pad}        column = base + coordinates[value][1]")
        self.line(
            f"{pad}        registers[value] = {norm}[column] - 2.0 * registers[value]"
        )
        self.line(
            f"{pad}    cute.autovec_copy(registers, destination[None, None, subtile])"
        )
        self.line(f"{pad}{self._participants(waited)[1]}.release()")

    def _emit_argmin(self, operation, pad: str) -> None:
        source = self.schedule.buffer(operation.reads[0])
        if source is None or len(source.shape) != 2:
            raise EmitError(f"argmin {operation.op_id!r} needs a rank-2 source")
        rows, columns = source.shape
        groups = rows // 32
        self.line(f"{pad}lane = cute.arch.lane_idx()")
        self.line(f"{pad}for group in cutlass.range({groups}, unroll_full=True):")
        self.line(f"{pad}    row = lane + group * 32")
        self.line(f"{pad}    best_value = {source.name}[row, 0]")
        self.line(f"{pad}    best_index = cutlass.Int32(0)")
        self.line(
            f"{pad}    for column in cutlass.range(1, {columns}, 1, unroll=1):"
        )
        self.line(f"{pad}        candidate = {source.name}[row, column]")
        self.line(f"{pad}        if candidate < best_value:")
        self.line(f"{pad}            best_value = candidate")
        self.line(f"{pad}            best_index = column")

        # The reduction produces one index per row inside a loop this emitter opened, so
        # a store that consumes it belongs in that loop rather than beside it.
        for consumer in self._consumers_of(operation):
            self.line(f"{pad}    # CAKE_OP:{consumer.op_id}")
            self.line(f"{pad}    {consumer.writes[0]}[row] = best_index")

    def _consumers_of(self, operation):
        produced = set(operation.writes)
        return [
            item
            for item in self.schedule.operations
            if item.op_id != operation.op_id and produced & set(item.reads)
        ]

    def _emitted_with_producer(self, operation) -> bool:
        """A store fed by a reduction is emitted inside that reduction's row loop."""

        if operation.kind is not OperationKind.STORE:
            return False
        return any(
            producer.kind is OperationKind.REDUCE_ARGMIN
            and set(producer.writes) & set(operation.reads)
            for producer in self.schedule.operations
        )

    def _emit_store(self, operation, pad: str) -> None:
        raise EmitError(
            f"store {operation.op_id!r} is emitted with the reduction that feeds it"
        )

    def _iterator_uses(self, operation) -> set[str]:
        return {
            index.name
            for access in self.schedule.access_maps
            if access.operation == operation.op_id
            for index in access.indices
            if index.source is AccessIndexKind.LOOP_TILE and index.name
        } | {
            loop.iterator
            for loop in self.schedule.tile_loops
            if operation.op_id in loop.body
        }

    def _emit_host(self) -> None:
        globals_in_order = [
            b for b in self.schedule.buffers if b.space is MemorySpace.GLOBAL
        ]
        entry = self.entry_point
        signature = ", ".join(f"{b.name}: cute.Tensor" for b in globals_in_order)
        self.line("@cute.jit")
        self.line(f"def {entry}({signature}):")
        self.line("    operation = tcgen05.MmaF16BF16Op(")
        self.line("        IO_DTYPE,")
        self.line("        ACC_DTYPE,")
        self.line("        MMA_INSTRUCTION_SHAPE,")
        self.line(f"        tcgen05.CtaGroup.{'ONE' if self.atom.cta_group == 1 else 'TWO'},")
        self.line(
            f"        tcgen05.OperandSource."
            f"{'SMEM' if self.atom.operand_source.value == 'shared' else 'TMEM'},"
        )
        for mode in self.atom.operand_major:
            self.line(f"        tcgen05.OperandMajorMode.{mode.value.upper()},")
        self.line("    )")
        self.line("    tiled_mma = cute.make_tiled_mma(operation)")
        for index, load in enumerate(self.loads):
            maker = "make_smem_layout_a" if index == 0 else "make_smem_layout_b"
            self.line(f"    {load.op_id}_smem_layout = sm100_utils.{maker}(")
            self.line(f"        tiled_mma, MMA_TILE, {load.reads[0]}.element_type, PIPELINE_STAGES")
            self.line("    )")
        self.line("    tma_operation = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(")
        self.line(f"        tcgen05.CtaGroup.{'ONE' if self.atom.cta_group == 1 else 'TWO'}")
        self.line("    )")
        for index, load in enumerate(self.loads):
            maker = "make_tiled_tma_atom_A" if index == 0 else "make_tiled_tma_atom_B"
            self.line(f"    {load.op_id}_atom, {load.op_id}_source = cute.nvgpu.{maker}(")
            self.line("        tma_operation,")
            self.line(f"        {load.reads[0]},")
            self.line(f"        cute.select({load.op_id}_smem_layout, mode=[0, 1, 2]),")
            self.line("        MMA_TILE,")
            self.line("        tiled_mma,")
            self.line("    )")
        self.line(f"    {self._kernel_name()}(")
        self.line("        tiled_mma,")
        for load in self.loads:
            self.line(f"        {load.op_id}_atom,")
            self.line(f"        {load.op_id}_source,")
        for buffer in globals_in_order:
            if buffer.name not in self._tma_sources():
                self.line(f"        {buffer.name},")
        for load in self.loads:
            self.line(f"        {load.op_id}_smem_layout,")
        grid = self.schedule.grid or (1, 1, 1)
        self.line(
            f"    ).launch(grid={tuple(grid)}, block=(THREADS_PER_CTA, 1, 1))"
        )
        self.line()
        self.line()
        self._emit_binding(globals_in_order, entry)

    def _emit_binding(self, globals_in_order, entry: str) -> None:
        names = ", ".join(b.name for b in globals_in_order)
        self.line(f"def compile_and_bind({names}):")
        self.line("    import torch")
        self.line()
        for buffer in globals_in_order:
            self.line(
                f"    if tuple({buffer.name}.shape) != {tuple(buffer.shape)}:"
            )
            self.line(
                f"        raise ValueError("
                f"\"{buffer.name} must have shape {list(buffer.shape)}\")"
            )
            self.line(
                f"    if {buffer.name}.dtype != {_TORCH_DTYPE[buffer.dtype]}:"
            )
            self.line(
                f"        raise TypeError("
                f"\"{buffer.name} must be {buffer.dtype.value}\")"
            )
        self.line(f"    tensors = ({names},)")
        self.line("    if not all(tensor.is_cuda for tensor in tensors):")
        self.line("        raise ValueError(\"every tensor must be on CUDA\")")
        self.line("    if not all(tensor.is_contiguous() for tensor in tensors):")
        self.line("        raise ValueError(\"every tensor must be contiguous\")")
        self.line("    if any(tensor.device != tensors[0].device for tensor in tensors):")
        self.line("        raise ValueError(\"every tensor must share one device\")")
        self.line()
        for buffer in globals_in_order:
            align = 32 if len(buffer.shape) > 1 else 16
            self.line(f"    {buffer.name}_cute = from_dlpack(")
            self.line(f"        {buffer.name}, assumed_align={align}")
            if len(buffer.shape) > 1:
                self.line("    ).mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(")
                self.line(f"        mode=1, divisibility={buffer.shape[1]}")
                self.line("    )")
            else:
                self.line("    ).mark_layout_dynamic()")
        self.line("    compiled = cute.compile(")
        self.line(f"        {entry},")
        for buffer in globals_in_order:
            self.line(f"        {buffer.name}_cute,")
        self.line("        options=\"--generate-line-info\",")
        self.line("    )")
        arguments = ", ".join(f"{b.name}_cute" for b in globals_in_order)
        self.line(f"    return compiled, ({arguments})")
        self.line()
        self.line()
        self.line(f"def launch_once({names}):")
        self.line(f"    compiled, arguments = compile_and_bind({names})")
        self.line("    compiled(*arguments)")
        self.line(f"    return {globals_in_order[-1].name}")


def emit(
    schedule: Schedule, target: Target, *, entry_point: str | None = None
) -> Emission:
    """Emit CuTe-DSL source for one Schedule, or raise if it under-specifies."""

    return _Emitter(schedule, target, entry_point).emit()
