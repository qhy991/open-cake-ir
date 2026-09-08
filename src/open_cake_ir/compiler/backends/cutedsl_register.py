"""Single-warp register MMA lowering for explicitly mapped B300 schedules.

The instruction fixes the lane/value partition. No warp layout is inferred: a
single declared warp traverses the declared tile using the atom's own partition.
Global copies are scalar and masked; neither shared staging nor coalesced stores
are promised. This domain is selected by register MMA, never by an operator name.
"""

from __future__ import annotations

from dataclasses import dataclass

from .common import python_name_findings, safe_python_identifier, Emission, EmitError, refusal, vocabulary_findings
from ..diagnostics import Finding
from ..ir import (
    AccessIndexKind, BoundaryPolicy, Buffer, BufferMode, DType, ElementwiseOp,
    LoadMovement, LoadReuse, MemorySpace, OperandMajorMode, OperandSource,
    Operation, OperationKind, ProgramAxis, Schedule, TileLoop,
)
from ..target import Target


MMA_CONTRACT = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
SUPPORTED_DTYPES = frozenset({DType.BF16, DType.FP32})
SUPPORTED_OPERATION_KINDS = frozenset({
    OperationKind.LOAD, OperationKind.MMA, OperationKind.ELEMENTWISE, OperationKind.STORE,
})


def applies(schedule: Schedule) -> bool:
    """Register MMA is a distinct hardware domain from the legacy TMA/UMMA path."""
    return any(
        operation.kind is OperationKind.MMA and (
            (operation.parameters.instruction is not None
             and operation.parameters.instruction.operand_source is OperandSource.REGISTER)
            or any(schedule.buffer(name) is not None
                   and schedule.buffer(name).space is MemorySpace.REGISTER
                   for name in operation.writes)
        )
        for operation in schedule.operations
    )


def requirements(schedule: Schedule) -> tuple[Finding, ...]:
    return vocabulary_findings(schedule, SUPPORTED_DTYPES, SUPPORTED_OPERATION_KINDS) + python_name_findings(schedule, register_route=True)


@dataclass(frozen=True)
class _Plan:
    a: Buffer
    b: Buffer
    bias: Buffer
    output: Buffer
    load_a: Operation
    load_b: Operation
    load_bias: Operation
    mma: Operation
    add: Operation
    store: Operation
    m_axis: ProgramAxis
    n_axis: ProgramAxis
    loop: TileLoop


def _plan(schedule: Schedule, target: Target) -> tuple[_Plan | None, tuple[Finding, ...]]:
    findings = list(requirements(schedule))

    def check(condition: object, code: str, path: str, message: str) -> None:
        if not condition:
            findings.append(refusal(code, path, message))

    check(schedule.target == target.target_id == "sm_103a", "CUTE_REGISTER_TARGET",
          "target", "register CuTe lowering requires the exact sm_103a target")
    check(len(schedule.roles) == 1 and schedule.roles[0].warps == (0,),
          "CUTE_REGISTER_ROLE", "roles", "register CuTe lowering requires one role with warps=[0]")
    check(schedule.residency is None and all(r.registers_per_thread is None for r in schedule.roles),
          "CUTE_REGISTER_RESIDENCY", "residency", "register CuTe lowering does not implement register or residency caps")
    for name in ("allocations", "pipelines", "barriers"):
        check(not getattr(schedule, name), "CUTE_REGISTER_STORAGE", name,
              "register CuTe lowering has no shared/tensor allocation or asynchronous pipeline")
    check(len(schedule.tile_loops) == 1, "CUTE_REGISTER_LOOP", "tile_loops",
          "register CuTe lowering requires one complete K loop")
    pm = schedule.program_map
    check(pm is not None and not pm.persistent and len(pm.axes) == 2 and schedule.grid is None,
          "CUTE_REGISTER_PROGRAM_MAP", "program_map",
          "register CuTe lowering requires two nonpersistent program axes and a derived grid")
    groups = {kind: tuple(op for op in schedule.operations if op.kind is kind)
              for kind in SUPPORTED_OPERATION_KINDS}
    check({kind: len(ops) for kind, ops in groups.items()} == {
        OperationKind.LOAD: 3, OperationKind.MMA: 1,
        OperationKind.ELEMENTWISE: 1, OperationKind.STORE: 1,
    }, "CUTE_REGISTER_OPERATIONS", "operations",
        "register CuTe lowering requires three global loads, one register MMA, one broadcast ADD and one store")
    for index, op in enumerate(schedule.operations):
        check(not op.waits and not op.signals and op.pipeline is None,
              "CUTE_REGISTER_SYNC", f"operations[{index}]",
              "register CuTe operations execute in program order without asynchronous synchronization")
        check(len(op.writes) == 1 and len(op.reads) == (2 if op.kind in (
            OperationKind.MMA, OperationKind.ELEMENTWISE) else 1),
            "CUTE_REGISTER_ARITY", f"operations[{index}]", "unsupported register operation arity")
        check(len(schedule.roles) == 1 and op.role == schedule.roles[0].name,
              "CUTE_REGISTER_ROLE", f"operations[{index}].role", "every operation must use the declared single warp")
    if findings:
        return None, tuple(findings)

    mma, add, store = (groups[k][0] for k in (
        OperationKind.MMA, OperationKind.ELEMENTWISE, OperationKind.STORE))
    loads = {op.writes[0]: op for op in groups[OperationKind.LOAD]}
    instruction, tile = mma.parameters.instruction, mma.parameters.tile_shape
    check(instruction is not None and instruction.contract == MMA_CONTRACT
          and instruction.shape == (16, 8, 16) and instruction.cta_group == 1
          and instruction.operand_source is OperandSource.REGISTER
          and instruction.operand_major == (OperandMajorMode.K, OperandMajorMode.K),
          "CUTE_REGISTER_MMA", f"operations[{schedule.operations.index(mma)}].parameters.instruction",
          "register CuTe lowering requires the declared BF16 m16n8k16 row.col register atom")
    check(mma.parameters.accumulator is DType.FP32
          and tile is not None and tile[0] == 16 and tile[1] % 8 == 0 and tile[2] % 16 == 0,
          "CUTE_REGISTER_TILE", f"operations[{schedule.operations.index(mma)}].parameters.tile_shape",
          "register CuTe tiles require M=16, N a multiple of 8 and K a multiple of 16")
    check(add.parameters.op is ElementwiseOp.ADD and add.parameters.broadcast_axis == 1
          and add.parameters.scalar is None and add.parameters.instruction is None,
          "CUTE_REGISTER_ELEMENTWISE", f"operations[{schedule.operations.index(add)}].parameters",
          "register CuTe lowering implements FP32 ADD with an explicit axis-1 vector broadcast")
    check(not store.parameters.coalesced, "CUTE_REGISTER_STORE_COALESCING",
          f"operations[{schedule.operations.index(store)}].parameters.coalesced",
          "the register CuTe scalar store does not promise coalescing; declare coalesced=false")
    check(all(name in loads for name in mma.reads) and mma.writes[0] == add.reads[0]
          and add.reads[1] in loads and add.writes == store.reads,
          "CUTE_REGISTER_DATAFLOW", "operations",
          "loads must feed MMA operands and broadcast bias, then ADD must feed the sole store")
    names = [name for op in schedule.operations for name in (*op.reads, *op.writes)]
    check(len({b.name for b in schedule.buffers}) == 9 and len(schedule.buffers) == 9
          and set(names) == {b.name for b in schedule.buffers},
          "CUTE_REGISTER_BUFFERS", "buffers", "every declared buffer must belong to the complete register dataflow")
    if findings:
        return None, tuple(findings)

    la, lb, l_bias = (loads[name] for name in (*mma.reads, add.reads[1]))
    a, b, bias = (schedule.buffer(op.reads[0]) for op in (la, lb, l_bias))
    output = schedule.buffer(store.writes[0])
    globals_ = (a, b, bias, output)
    check(all(buf is not None for buf in globals_), "CUTE_REGISTER_BUFFERS", "buffers",
          "register CuTe global buffers must be declared")
    if findings:
        return None, tuple(findings)
    assert a is not None and b is not None and bias is not None and output is not None
    assert tile is not None and pm is not None
    expected = {
        a.name: (MemorySpace.GLOBAL, DType.BF16, BufferMode.INPUT, None),
        b.name: (MemorySpace.GLOBAL, DType.BF16, BufferMode.INPUT, None),
        bias.name: (MemorySpace.GLOBAL, DType.FP32, BufferMode.INPUT, None),
        output.name: (MemorySpace.GLOBAL, DType.FP32, BufferMode.OUTPUT, None),
        la.writes[0]: (MemorySpace.REGISTER, DType.BF16, BufferMode.SCRATCH, (tile[0], tile[2])),
        lb.writes[0]: (MemorySpace.REGISTER, DType.BF16, BufferMode.SCRATCH, (tile[1], tile[2])),
        mma.writes[0]: (MemorySpace.REGISTER, DType.FP32, BufferMode.SCRATCH, tile[:2]),
        l_bias.writes[0]: (MemorySpace.REGISTER, DType.FP32, BufferMode.SCRATCH, (tile[1],)),
        add.writes[0]: (MemorySpace.REGISTER, DType.FP32, BufferMode.SCRATCH, tile[:2]),
    }
    for index, buf in enumerate(schedule.buffers):
        wanted = expected.get(buf.name)
        check(wanted is not None and (buf.space, buf.dtype, buf.mode) == wanted[:3]
              and (wanted[3] is None or buf.shape == wanted[3]),
              "CUTE_REGISTER_BUFFER", f"buffers[{index}]", "buffer type, storage or tile shape does not match its register dataflow role")
        check(buf.allocation is None and buf.byte_offset == 0 and buf.stages == 1
              and buf.swizzle is None and buf.scale_of is None and buf.valid_extent is None,
              "CUTE_REGISTER_BUFFER_OPTIONS", f"buffers[{index}]",
              "register CuTe lowering supports unstaged dense buffers without storage aliases or runtime extents")
    check(len(a.shape) == len(b.shape) == len(output.shape) == 2 and len(bias.shape) == 1
          and a.shape[1] == b.shape[1] and bias.shape == (b.shape[0],)
          and output.shape == (a.shape[0], b.shape[0]),
          "CUTE_REGISTER_SHAPE", "buffers", "global shapes must be A(M,K), B(N,K), bias(N), C(M,N)")
    check(all(buf.elements <= 2**31 - 1 and all(extent <= 2**31 - 1 - max(tile) for extent in buf.shape)
              for buf in globals_), "CUTE_REGISTER_INDEX_RANGE", "buffers",
          "register CuTe global addressing and padded tile coordinates must fit signed 32-bit indices")
    check(schedule.outputs == (output.name,), "CUTE_REGISTER_OUTPUT", "outputs",
          "register CuTe lowering requires the sole stored output")
    loop = schedule.tile_loops[0]
    check(loop.buffer == a.name and loop.dimension == 1 and loop.tile == tile[2]
          and len(a.shape) == 2 and a.shape[1] > loop.tile
          and loop.body == (la.op_id, lb.op_id, mma.op_id) and loop.stop is None,
          "CUTE_REGISTER_LOOP", "tile_loops[0]",
          "the K loop must load A and B then accumulate MMA over A's whole K dimension in at least two trips")
    for field, value in (("num_stages", 1), ("loop_unroll_factor", 1), ("flatten", False),
                         ("warp_specialize", False), ("disable_licm", False)):
        check(getattr(loop.range_options, field) == value, "CUTE_RANGE_OPTION_UNSUPPORTED",
              f"tile_loops[0].range_options.{field}", f"register CuTe lowering requires {field}={value!r}")
    # Both values of disallow_acc_multi_buffer are realized by the one accumulator.
    canonical = (la, lb, mma, l_bias, add, store)
    check(schedule.operations == canonical, "CUTE_REGISTER_ORDER", "operations",
          "register CuTe operations must follow load A, load B, MMA, bias load, ADD, store")
    dependencies = ((), (), (la.op_id, lb.op_id), (), (mma.op_id, l_bias.op_id), (add.op_id,))
    for op, required in zip(canonical, dependencies):
        index = schedule.operations.index(op)
        check(set(op.depends_on) == set(required), "CUTE_REGISTER_DEPENDENCIES",
              f"operations[{index}].depends_on", "dependencies must match the complete register producer-consumer graph")
    for op in (la, lb, l_bias):
        index = schedule.operations.index(op)
        check(op.parameters.movement is LoadMovement.GLOBAL and op.parameters.descriptor_box is None,
              "CUTE_REGISTER_LOAD", f"operations[{index}].parameters", "register operands are loaded directly from global memory")
    m_axes = [axis for axis in pm.axes if axis.buffer == a.name and axis.dimension == 0 and axis.tile == tile[0]]
    n_axes = [axis for axis in pm.axes if axis.buffer == b.name and axis.dimension == 0 and axis.tile == tile[1]]
    check(len(m_axes) == len(n_axes) == 1 and {axis.axis for axis in pm.axes} == {0, 1},
          "CUTE_REGISTER_PROGRAM_MAP", "program_map", "program axes 0 and 1 must tile A's M and B's N dimensions")
    if findings:
        return None, tuple(findings)
    ma, na = m_axes[0], n_axes[0]
    expected_accesses = {
        (la.op_id, a.name): ((AccessIndexKind.PROGRAM_TILE, ma.name), (AccessIndexKind.LOOP_TILE, loop.iterator)),
        (lb.op_id, b.name): ((AccessIndexKind.PROGRAM_TILE, na.name), (AccessIndexKind.LOOP_TILE, loop.iterator)),
        (l_bias.op_id, bias.name): ((AccessIndexKind.PROGRAM_TILE, na.name),),
        (store.op_id, output.name): ((AccessIndexKind.PROGRAM_TILE, ma.name), (AccessIndexKind.PROGRAM_TILE, na.name)),
    }
    check(len(schedule.access_maps) == 4 and
          {(access.operation, access.buffer) for access in schedule.access_maps} == set(expected_accesses),
          "CUTE_REGISTER_ACCESS", "access_maps", "each global transfer requires exactly its declared access map")
    for index, access in enumerate(schedule.access_maps):
        check(tuple((part.source, part.name) for part in access.indices)
              == expected_accesses.get((access.operation, access.buffer))
              and access.boundary is BoundaryPolicy.MASK_TILED_AXES
              and all(part.offset == 0 and part.extent is None for part in access.indices),
              "CUTE_REGISTER_ACCESS", f"access_maps[{index}]",
              "register CuTe access must follow the mapped M/N/K tiles with masked boundaries")
    if findings:
        return None, tuple(findings)
    return _Plan(a, b, bias, output, la, lb, l_bias, mma, add, store, ma, na, loop), ()


def preflight(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    return _plan(schedule, target)[1]


def emit(schedule: Schedule, target: Target, *, entry_point: str | None = None) -> Emission:
    plan, findings = _plan(schedule, target)
    if findings:
        raise EmitError(findings[0].message)
    assert plan is not None
    entry = entry_point or schedule.lowering.entry_point
    globals_ = tuple(buf for mode in (BufferMode.INPUT, BufferMode.OUTPUT)
                     for buf in schedule.buffers if buf.space is MemorySpace.GLOBAL and buf.mode is mode)
    for name in (entry, *(buf.name for buf in globals_)):
        if not safe_python_identifier(name, register_route=True):
            raise EmitError(f"CuTe kernel symbol {name!r} conflicts with Python syntax or the fixed DSL imports")
    if any(buf.name == entry for buf in globals_):
        raise EmitError("CuTe pointer arguments cannot shadow the kernel entry point")
    prefix = "_cake_"
    suffix = 0
    while any(name.startswith(prefix) for name in (entry, *(buf.name for buf in globals_))):
        suffix += 1
        prefix = f"_cake{suffix}_"
    def v(name: str) -> str:
        return prefix + name
    tile = plan.mma.parameters.tile_shape
    assert tile is not None
    bm, bn, bk = tile
    m, k = plan.a.shape
    n = plan.b.shape[0]
    lines = [f"# Generated by open-cake-ir from {schedule.schedule_id!r}; DO NOT EDIT.",
             "# schedule_sha256=__SCHEDULE_SHA256__", "import cutlass", "import cutlass.cute as cute",
             "from cutlass.cute.nvgpu import warp", "", "", "@cute.kernel",
             f"def {entry}({', '.join(buf.name + ': cute.Pointer' for buf in globals_)}):"]
    def line(value: str, indent: int = 1) -> None:
        lines.append("    " * indent + value)
    line(f"{v('tid')}, {v('ty')}, {v('tz')} = cute.arch.thread_idx()")
    line(f"{v('bx')}, {v('by')}, {v('bz')} = cute.arch.block_idx()")
    line(f"{v('m0')} = {v('bx' if plan.m_axis.axis == 0 else 'by')} * {bm}")
    line(f"{v('n0')} = {v('bx' if plan.n_axis.axis == 0 else 'by')} * {bn}")
    line(f"{v('mma')} = cute.make_tiled_mma(warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)))")
    line(f"{v('thread_mma')} = {v('mma')}.get_slice({v('tid')})")
    for operand, shape, dtype in (("a", (bm, bk), "BFloat16"), ("b", (bn, bk), "BFloat16"), ("c", (bm, bn), "Float32")):
        line(f"{v('coord_' + operand)} = {v('thread_mma')}.partition_{operand.upper()}(cute.make_identity_tensor({shape!r}))")
        line(f"{v('r_' + operand)} = cute.make_rmem_tensor({v('coord_' + operand)}.shape, cutlass.{dtype})")
    line(f"{v('r_c')}.fill(0.0)")
    line(f"{v('tmp_bf16')} = cute.make_rmem_tensor((1,), cutlass.BFloat16)")
    line(f"{v('tmp_fp32')} = cute.make_rmem_tensor((1,), cutlass.Float32)")
    for label, op, dtype, bits in (("a", plan.load_a, "BFloat16", 16), ("b", plan.load_b, "BFloat16", 16), ("bias", plan.load_bias, "Float32", 32)):
        hint = ""
        if op.parameters.reuse is LoadReuse.STREAMED:
            hint = ", load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL"
        elif op.parameters.reuse is LoadReuse.REUSED:
            hint = ", l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_LAST"
        line(f"{v('copy_' + label)} = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), cutlass.{dtype}, num_bits_per_copy={bits}{hint})")
    line(f"{v('copy_out')} = cute.make_copy_atom(cute.nvgpu.CopyR2GOp(), cutlass.Float32, num_bits_per_copy=32)")
    line(f"for {v('kt')} in cutlass.range({(k + bk - 1) // bk}):")
    line(f"{v('k0')} = {v('kt')} * {bk}", 2)
    for label, op, buf, first, extent in (("a", plan.load_a, plan.a, "m0", m), ("b", plan.load_b, plan.b, "n0", n)):
        line(f"# CAKE_OP:{op.op_id}", 2)
        line(f"{v('r_' + label)}.fill(0.0)", 2)
        line(f"for {v('i')} in cutlass.range_constexpr(cute.size({v('r_' + label)})):", 2)
        line(f"{v('coord')} = {v('coord_' + label)}[{v('i')}]", 3)
        line(f"{v('row')} = {v(first)} + {v('coord')}[0]", 3)
        line(f"{v('col')} = {v('k0')} + {v('coord')}[1]", 3)
        line(f"if ({v('row')} < {extent}) and ({v('col')} < {k}):", 3)
        line(f"{v('src')} = cute.make_tensor({buf.name} + {v('row')} * {k} + {v('col')}, cute.make_layout((1,)))", 4)
        line(f"cute.copy({v('copy_' + label)}, {v('src')}, {v('tmp_bf16')})", 4)
        line(f"{v('r_' + label)}[{v('i')}] = {v('tmp_bf16')}[0]", 4)
    line(f"# CAKE_OP:{plan.mma.op_id}", 2)
    line(f"cute.gemm({v('mma')}, {v('r_c')}, {v('r_a')}, {v('r_b')}, {v('r_c')})", 2)
    line(f"# CAKE_OP:{plan.load_bias.op_id}")
    line(f"{v('r_bias')} = cute.make_rmem_tensor({v('coord_c')}.shape, cutlass.Float32)")
    line(f"{v('r_bias')}.fill(0.0)")
    line(f"for {v('i')} in cutlass.range_constexpr(cute.size({v('r_bias')})):")
    line(f"{v('coord')} = {v('coord_c')}[{v('i')}]", 2)
    line(f"{v('col')} = {v('n0')} + {v('coord')}[1]", 2)
    line(f"if {v('col')} < {n}:", 2)
    line(f"{v('src')} = cute.make_tensor({plan.bias.name} + {v('col')}, cute.make_layout((1,)))", 3)
    line(f"cute.copy({v('copy_bias')}, {v('src')}, {v('tmp_fp32')})", 3)
    line(f"{v('r_bias')}[{v('i')}] = {v('tmp_fp32')}[0]", 3)
    line(f"# CAKE_OP:{plan.add.op_id}")
    line(f"{v('r_out')} = cute.make_rmem_tensor({v('coord_c')}.shape, cutlass.Float32)")
    line(f"for {v('i')} in cutlass.range_constexpr(cute.size({v('r_out')})):")
    line(f"{v('r_out')}[{v('i')}] = {v('r_c')}[{v('i')}] + {v('r_bias')}[{v('i')}]", 2)
    line(f"# CAKE_OP:{plan.store.op_id}")
    line(f"for {v('i')} in cutlass.range_constexpr(cute.size({v('r_out')})):")
    line(f"{v('coord')} = {v('coord_c')}[{v('i')}]", 2)
    line(f"{v('row')} = {v('m0')} + {v('coord')}[0]", 2)
    line(f"{v('col')} = {v('n0')} + {v('coord')}[1]", 2)
    line(f"if ({v('row')} < {m}) and ({v('col')} < {n}):", 2)
    line(f"{v('dst')} = cute.make_tensor({plan.output.name} + {v('row')} * {n} + {v('col')}, cute.make_layout((1,)))", 3)
    line(f"{v('tmp_fp32')}[0] = {v('r_out')}[{v('i')}]", 3)
    line(f"cute.copy({v('copy_out')}, {v('tmp_fp32')}, {v('dst')})", 3)
    lines.append("# CAKE_KERNEL_END")
    grid = [1, 1, 1]
    grid[plan.m_axis.axis] = (m + bm - 1) // bm
    grid[plan.n_axis.axis] = (n + bn - 1) // bn
    toolchain = {"source_language": "python", "compiler": "cutlass_cute_dsl", "target": target.target_id,
                 "kernel_entry_point": entry,
                 "signature": [{"name": buf.name, "dtype": buf.dtype.value} for buf in globals_],
                 "grid": grid, "block": [32, 1, 1], "dynamic_shared_memory_bytes": 0}
    return Emission("\n".join(lines) + "\n", entry,
                    {"MMA_TILE": tile, "MMA_INSTRUCTION_SHAPE": (16, 8, 16)}, toolchain)
