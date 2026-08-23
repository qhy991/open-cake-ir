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

from dataclasses import dataclass

from .ir import (
    Barrier,
    BarrierMechanism,
    DType,
    LoadMovement,
    MemorySpace,
    OperationKind,
    Role,
    Schedule,
    Swizzle,
    TileLoop,
)
from .target import Target


class EmitError(ValueError):
    """The Schedule does not determine the source to emit."""


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


@dataclass(frozen=True)
class Emission:
    source: str
    entry_point: str
    constants: dict[str, object]
    """Every value derived from the Schedule, so a test can compare them against the
    module constants the hand-written artifact carries."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise EmitError(message)


class _Emitter:
    def __init__(self, schedule: Schedule, target: Target) -> None:
        self.schedule = schedule
        self.target = target
        self.lines: list[str] = []

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
        entry = f"cake_{self.schedule.profile}"
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
        self._emit_role_dispatch()

    def _tma_sources(self) -> set[str]:
        return {load.reads[0] for load in self.loads}

    def _emit_role_dispatch(self) -> None:
        """Warp identity is computed from the declared roles, not written by the agent."""

        self.line("    # warp identity derived from the declared roles")
        first = True
        for role in sorted(self.schedule.roles, key=lambda r: r.warps[0]):
            operations = [
                op for op in self.schedule.operations if op.role == role.name
            ]
            if not operations:
                continue
            keyword = "if" if first else "elif"
            first = False
            if len(role.warps) == 1:
                condition = f"warp_idx == {self.role_constant(role)}"
            else:
                low, high = min(role.warps), max(role.warps)
                condition = f"{low} <= warp_idx <= {high}"
            self.line(f"    {keyword} {condition}:")
            for operation in operations:
                self.line(f"        # CAKE_OP:{operation.op_id}")
                self.line(
                    f"        pass  # body for {operation.kind.value} "
                    f"{operation.op_id!r} in role {role.name!r}"
                )
        self.line("    # CAKE_KERNEL_END")
        self.line()
        self.line()

    def _emit_host(self) -> None:
        globals_in_order = [
            b for b in self.schedule.buffers if b.space is MemorySpace.GLOBAL
        ]
        entry = f"cake_{self.schedule.profile}"
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


def emit(schedule: Schedule, target: Target) -> Emission:
    """Emit CuTe-DSL source for one Schedule, or raise if it under-specifies."""

    return _Emitter(schedule, target).emit()
