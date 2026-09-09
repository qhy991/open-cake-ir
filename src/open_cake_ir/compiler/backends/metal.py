"""Compositional SIMD Metal lowering over canonical Schedule coordinates.

Flattened private value i belongs to lane i % 32, slot i // 32. All 32 lanes
execute every collective round; padded lanes hold initialized values and contribute
reduction identities. Reduction order is ascending local slots followed by the Metal
SIMD collective, whose cross-lane addition order is not a serial or PTX RN contract.
Logical lane storage is derived from these slots and IR lifetimes, not Apple occupancy.
"""

from __future__ import annotations

import math
import re

from .common import refusal, vocabulary_findings, Emission, EmitError
from ..ir import (
    AccessIndexKind, BufferMode, DType, ElementwiseOp, LoadMovement,
    LoweringBackend, MemorySpace, OperationKind, ReduceOp, ReductionScope, Schedule,
)
from ..target import Target
from ..diagnostics import Finding, FindingCategory, FindingSeverity
from ..verifier import verify

SUPPORTED_DTYPES = frozenset({DType.FP32})
SUPPORTED_OPERATION_KINDS = frozenset({
    OperationKind.LOAD, OperationKind.ELEMENTWISE, OperationKind.REDUCE, OperationKind.STORE,
})
_BINARY = {ElementwiseOp.ADD: "+", ElementwiseOp.SUB: "-", ElementwiseOp.MUL: "*", ElementwiseOp.DIV: "/"}
# C++ language keywords and Metal address-space / scalar type spellings cannot be
# used as the externally visible kernel symbol. Buffer ids are separately mangled.
_RESERVED = frozenset("""
alignas alignof and and_eq asm auto bitand bitor bool break case catch char char16_t
char32_t class compl const constexpr const_cast continue decltype default delete do
double dynamic_cast else enum explicit export extern false float for friend goto if
inline int kernel long metal mutable namespace new noexcept not not_eq nullptr
operator or or_eq private protected public register reinterpret_cast return short
signed sizeof static static_assert static_cast struct switch template this thread
threadgroup constant device throw true try typedef typeid typename uchar uint ulong
union unsigned ushort using virtual void volatile wchar_t while xor xor_eq half
sampler texture2d array vector matrix INFINITY NAN
""".split())
_UNARY = {
    ElementwiseOp.SQUARE: "({x} * {x})",
    ElementwiseOp.RELU: "max({x}, 0.0f)",
    ElementwiseOp.RSQRT: "precise::rsqrt({x})",
}


SIMD_WIDTH = 32
MAXIMUM_SIMD_GROUPS = 32


def lane_width(schedule: Schedule) -> int:
    """Threads per threadgroup this schedule's single role occupies.

    One SIMD group keeps the original route exactly. More groups widen the stripe,
    which is the only way a fixed reduction width can own fewer values per lane.
    """
    warps = schedule.roles[0].warps if schedule.roles else (0,)
    return SIMD_WIDTH * len(warps)


def _slots(buffer, lanes: int = SIMD_WIDTH):
    return (buffer.elements + lanes - 1) // lanes


def _share_slots(schedule: Schedule, lanes: int) -> int:
    """Threadgroup floats needed to combine SIMD groups and publish scalars."""
    if lanes == SIMD_WIDTH:
        return 0
    scalars = max((sum(1 for read in operation.reads
                       if operation.kind is OperationKind.ELEMENTWISE
                       and next(b for b in schedule.buffers if b.name == read).is_scalar
                       and not next(b for b in schedule.buffers if b.name == operation.writes[0]).is_scalar)
                   for operation in schedule.operations), default=0)
    return max(lanes // SIMD_WIDTH, scalars, 1)


def private_values_per_thread(schedule: Schedule, lanes: int = SIMD_WIDTH) -> int:
    """Peak simultaneously live lane-owned FP32 values, without physical allocation claims.

    Count source and destination at their common operation boundary. No speculative
    in-place aliasing or compiler register reuse is assumed. Shuffle/reduction scalar
    temporaries and backend spills are outside this declared-Buffer domain.
    """
    intervals = []
    for buffer in schedule.buffers:
        if buffer.space is not MemorySpace.REGISTER:
            continue
        uses = [index for index, operation in enumerate(schedule.operations)
                if buffer.name in (*operation.reads, *operation.writes)]
        intervals.append((uses[0], uses[-1], _slots(buffer, lanes)) if uses
                         else (0, len(schedule.operations) - 1, _slots(buffer, lanes)))
    return max((sum(slots for first, last, slots in intervals if first <= index <= last)
                for index in range(len(schedule.operations))), default=0)


def requirements(schedule: Schedule) -> tuple[Finding, ...]:
    """Target-independent backend requirements, including unsupported vocabulary."""
    return vocabulary_findings(schedule, SUPPORTED_DTYPES, SUPPORTED_OPERATION_KINDS)


def preflight(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    """Refuse every declaration this emitter cannot faithfully realize."""
    findings = list(requirements(schedule))
    if findings:
        return tuple(findings)

    def check(condition, code, path, message):
        if not condition:
            findings.append(refusal(code, path, message))

    check(schedule.lowering.backend is LoweringBackend.METAL
          and schedule.target == target.target_id
          and (target.target_id, target.architecture, target.device_names) in {
              ("apple_gpu_family7", "apple7", ("Apple M1 Pro",)),
              ("apple_gpu_family8", "apple8", ("Apple M2",)),
          },
          "METAL_TARGET_UNSUPPORTED", "target",
          "Metal requires the exact Apple M1 Pro / apple_gpu_family7 or Apple M2 / apple_gpu_family8 target")
    check(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", schedule.lowering.entry_point)
          and "CAKE_KERNEL_END" not in schedule.lowering.entry_point
          and "__SCHEDULE_SHA256__" not in schedule.lowering.entry_point
          and schedule.lowering.entry_point not in _RESERVED
          and not re.fullmatch(r"(?:bool|char|uchar|short|ushort|int|uint|long|ulong|half|float)(?:[234](?:x[234])?)", schedule.lowering.entry_point),
          "METAL_ENTRY_POINT_UNSUPPORTED", "lowering.entry_point", "Metal requires a non-reserved function identifier")
    check(len(schedule.roles) == 1
          and schedule.roles[0].warps == tuple(range(len(schedule.roles[0].warps)))
          and 1 <= len(schedule.roles[0].warps) <= MAXIMUM_SIMD_GROUPS,
          "METAL_ROLE_UNSUPPORTED", "roles",
          "SIMD program tiles require one role occupying consecutive SIMD groups from [0], "
          f"at most {MAXIMUM_SIMD_GROUPS}")
    for index, role in enumerate(schedule.roles):
        check(role.registers_per_thread is None, "METAL_REGISTER_CAP_UNSUPPORTED",
              f"roles[{index}].registers_per_thread", "Metal has no CUDA warpgroup register redistribution")
    for field in ("allocations", "pipelines", "barriers", "tile_loops"):
        check(not getattr(schedule, field), "METAL_DECLARATION_UNSUPPORTED", field,
              f"SIMD Metal lowering does not implement {field}")
    lanes = lane_width(schedule)
    if lanes > SIMD_WIDTH:
        # Widening the stripe replaces every SIMD collective with a threadgroup one.
        # Reduce and scalar broadcast are implemented; the cross-lane exchange that
        # serves a narrower non-scalar operand is not, and is refused rather than
        # emitted through a shuffle that only reaches one group.
        for index, operation in enumerate(schedule.operations):
            if operation.kind is not OperationKind.ELEMENTWISE:
                continue
            destination = next(b for b in schedule.buffers if b.name == operation.writes[0])
            for read in operation.reads:
                source = next(b for b in schedule.buffers if b.name == read)
                check(source.shape == destination.shape or source.is_scalar,
                      "METAL_BROADCAST_WIDTH_UNSUPPORTED", f"operations[{index}]",
                      "multi-group Metal lowering implements matching-shape and scalar "
                      "operands only; a narrower non-scalar operand needs a cross-group "
                      "exchange this route does not emit")
    check(schedule.residency is None, "METAL_RESIDENCY_UNSUPPORTED", "residency",
          "Apple occupancy and register caps are not modeled or enforced")
    check(schedule.program_map is not None, "METAL_PROGRAM_MAP_REQUIRED", "program_map",
          "Metal addresses require a declared scalar ProgramMap")
    if schedule.program_map is not None:
        check(not schedule.program_map.persistent, "METAL_PERSISTENCE_UNSUPPORTED",
              "program_map.persistent", "Metal does not implement persistent scheduling")
        for index, axis in enumerate(schedule.program_map.axes):
            check(axis.tile == 1, "METAL_PROGRAM_TILE_UNSUPPORTED",
                  f"program_map.axes[{index}].tile", "this Metal route requires scalar program indices (tile=1); dimension extents may be odd")
    globals_ = [buffer for buffer in schedule.buffers if buffer.space is MemorySpace.GLOBAL]
    check(len(globals_) <= 31, "METAL_BUFFER_ARGUMENT_LIMIT", "buffers", "Metal admits at most 31 global buffer arguments")
    private_values = private_values_per_thread(schedule, lane_width(schedule))
    check(private_values <= 1024, "METAL_PRIVATE_STORAGE_LIMIT", "buffers",
          "Metal supports at most 1024 simultaneously live lane-owned FP32 values; "
          "this backend limit is not an Apple register capacity or occupancy estimate")
    for index, buffer in enumerate(schedule.buffers):
        path = f"buffers[{index}]"
        check(buffer.space in {MemorySpace.GLOBAL, MemorySpace.REGISTER},
              "METAL_STORAGE_UNSUPPORTED", path + ".space", "Metal first slice uses global buffers and private register values only")
        check(buffer.allocation is None and buffer.byte_offset == 0 and buffer.stages == 1
              and buffer.swizzle is None and buffer.scale_of is None and buffer.valid_extent is None,
              "METAL_STORAGE_DECLARATION_UNSUPPORTED", path,
              "Metal does not implement buffer allocations, offsets, staging, swizzles, scales or runtime valid extents")
        check(buffer.mode in ({BufferMode.INPUT, BufferMode.OUTPUT} if buffer.space is MemorySpace.GLOBAL else {BufferMode.SCRATCH}),
              "METAL_BUFFER_MODE_UNSUPPORTED", path + ".mode", "Metal requires global input/output and private scratch values; state/aliasing is unsupported")
        check(buffer.size_bytes < 2**63, "METAL_ADDRESS_RANGE_UNSUPPORTED", path + ".shape", "Metal buffer addressing requires a signed 64-bit byte extent")
    for index, access in enumerate(schedule.access_maps):
        for component_index, component in enumerate(access.indices):
            check(component.source in {AccessIndexKind.PROGRAM, AccessIndexKind.DIMENSION},
                  "METAL_ACCESS_UNSUPPORTED", f"access_maps[{index}].indices[{component_index}]",
                  "Metal supports scalar program indices and finite contiguous dimension slices only")
    for index, operation in enumerate(schedule.operations):
        path = f"operations[{index}]"
        check(not any(ord(c) < 32 or ord(c) == 127 for c in operation.op_id)
              and "CAKE_OP:" not in operation.op_id and "CAKE_KERNEL_END" not in operation.op_id
              and operation.op_id.strip() == operation.op_id
              and operation.op_id.splitlines() == [operation.op_id]
              and "__SCHEDULE_SHA256__" not in operation.op_id
              and not operation.op_id.endswith(("\\", "??/")),
              "METAL_OPERATION_ID_UNSUPPORTED", path + ".id", "operation id contains a source-map delimiter, reserved substitution token, "
              "or lexical line continuation")
        check(not operation.waits and not operation.signals and operation.pipeline is None,
              "METAL_SYNCHRONIZATION_UNSUPPORTED", path, "Metal does not implement declared barriers or pipelines")
        if operation.kind in {OperationKind.ELEMENTWISE, OperationKind.REDUCE}:
            check(len(operation.writes) == 1 and all(
                schedule.buffer(name) is not None and schedule.buffer(name).space is MemorySpace.REGISTER
                for name in (*operation.reads, *operation.writes)
            ), "METAL_ARITHMETIC_STORAGE_UNSUPPORTED", path,
                "Metal arithmetic writes one private result and reads private values only")
        if operation.kind is OperationKind.REDUCE:
            check(len(operation.reads) == 1, "METAL_REDUCTION_ARITY_UNSUPPORTED", path + ".reads",
                  "Metal reductions read exactly one private value")
        parameters = operation.parameters
        if operation.kind is OperationKind.LOAD:
            check(parameters.movement is LoadMovement.GLOBAL and parameters.reuse is None,
                  "METAL_LOAD_UNSUPPORTED", path + ".parameters", "Metal supports direct loads without CUDA cache or TMA declarations")
            check(len(operation.reads) == len(operation.writes) == 1
                  and schedule.buffer(operation.reads[0]).space is MemorySpace.GLOBAL
                  and schedule.buffer(operation.reads[0]).mode is BufferMode.INPUT
                  and schedule.buffer(operation.writes[0]).space is MemorySpace.REGISTER,
                  "METAL_LOAD_STORAGE_UNSUPPORTED", path, "Metal loads one global input into one private value")
        elif operation.kind is OperationKind.STORE:
            check(not parameters.coalesced, "METAL_COALESCING_UNSUPPORTED", path + ".parameters.coalesced",
                  "this generic SIMD address route does not promise coalescing; declare coalesced=false")
            check(len(operation.reads) == len(operation.writes) == 1
                  and schedule.buffer(operation.reads[0]).space is MemorySpace.REGISTER
                  and schedule.buffer(operation.writes[0]).space is MemorySpace.GLOBAL,
                  "METAL_STORE_STORAGE_UNSUPPORTED", path, "Metal stores one private value into one global output")
        elif operation.kind is OperationKind.ELEMENTWISE:
            check(parameters.op in _BINARY or parameters.op in _UNARY,
                  "METAL_ELEMENTWISE_UNSUPPORTED", path + ".parameters.op",
                  "Metal supports add/sub/mul/div/square/relu/rsqrt; PTX FMA and other transcendental contracts are not implemented")
            check(parameters.instruction is None, "METAL_INSTRUCTION_UNSUPPORTED", path + ".parameters.instruction", "Metal does not implement a CUDA/PTX instruction contract")
            check(parameters.scalar is None or abs(parameters.scalar) <= 3.4028234663852886e38,
                  "METAL_SCALAR_RANGE_UNSUPPORTED", path + ".parameters.scalar", "literal must be representable as finite FP32")
        elif operation.kind is OperationKind.REDUCE:
            check(parameters.scope is ReductionScope.CTA and not parameters.across_loop,
                  "METAL_REDUCTION_UNSUPPORTED", path + ".parameters", "Metal folds one private tile in CTA scope without loop-carried state")
        if operation.kind in {OperationKind.LOAD, OperationKind.STORE} and len(operation.reads) == len(operation.writes) == 1:
            global_name, private_name = (
                (operation.reads[0], operation.writes[0])
                if operation.kind is OperationKind.LOAD
                else (operation.writes[0], operation.reads[0])
            )
            matches = [(i, access) for i, access in enumerate(schedule.access_maps)
                       if (access.operation, access.buffer) == (operation.op_id, global_name)]
            edge = "reads" if operation.kind is OperationKind.LOAD else "writes"
            check(len(matches) == 1, "METAL_ACCESS_MAP_REQUIRED", f"{path}.{edge}[0]",
                  f"Metal requires exactly one access map for global buffer {global_name!r}")
            if len(matches) != 1:
                continue
            access_index, access = matches[0]
            global_buffer, private_buffer = schedule.buffer(global_name), schedule.buffer(private_name)
            # Common verification owns coordinate validity and duplicate maps. The Metal
            # body additionally needs the complete value domain: it cannot reshape,
            # broadcast, or discard a private axis when visiting global addresses.
            if global_buffer is not None and private_buffer is not None and all(
                component.source is AccessIndexKind.PROGRAM or (
                    component.source is AccessIndexKind.DIMENSION
                    and component.dimension is not None
                    and component.dimension < len(global_buffer.shape)
                ) for component in access.indices
            ):
                shape = tuple(component.span(global_buffer.shape[component.dimension])
                              for component in access.indices
                              if component.source is AccessIndexKind.DIMENSION) or (1,)
                check(private_buffer.shape == shape, "METAL_ACCESS_VALUE_SHAPE",
                      f"access_maps[{access_index}].indices",
                      f"access has value shape {list(shape)}, but private buffer "
                      f"{private_name!r} has shape {list(private_buffer.shape)}; Metal "
                      "loads/stores do not reshape, broadcast or truncate values")
            if operation.kind is OperationKind.STORE and schedule.program_map is not None:
                owned = {component.name for component in access.indices
                         if component.source is AccessIndexKind.PROGRAM}
                unowned = [axis.name for axis in schedule.program_map.axes
                           if (owner := schedule.buffer(axis.buffer)) is not None
                           and axis.dimension < len(owner.shape)
                           and axis.tile_count(owner.shape[axis.dimension]) > 1
                           and axis.name not in owned]
                # With direct scalar program coordinates, differing programs must
                # differ in a destination coordinate. Common verification proves the
                # coordinates are in bounds and the output has only one writer.
                check(not unowned, "METAL_STORE_OWNERSHIP", f"access_maps[{access_index}].indices",
                      f"store does not own varying program axes {unowned}; different "
                      "threadgroups could write the same non-atomic output addresses")
    if not findings:
        findings.append(Finding(
            "METAL_SIMD_EXECUTION", "lowering",
            f"Metal stripes flattened values over {lane_width(schedule)} lanes with uniform SIMD "
            "collectives and uniquely owned stores. Peak live lane-owned Buffer "
            f"storage: {private_values} FP32 values; "
            "temporary registers and spills are unmodeled. No occupancy, cost or "
            "GPU correctness is inferred. Local-slot then SIMD reduction order and "
            "precise rsqrt use Metal rounding/denormal behavior, without PTX RN equivalence.",
            FindingCategory.HARDWARE_CONFORMANCE, FindingSeverity.REPORT,
        ))
    return tuple(findings)


def _coordinates(flat, shape):
    """Row-major private-array coordinates, including extent-one dimensions."""
    return [f"(({flat} / {math.prod(shape[index + 1:])}u) % {extent}u)"
            for index, extent in enumerate(shape)]


def _flat(coordinates, shape):
    return " + ".join(f"({coordinate}) * {math.prod(shape[index + 1:])}ul"
                      for index, coordinate in enumerate(coordinates)) or "0ul"


def emit(schedule: Schedule, target: Target, *, entry_point: str | None = None) -> Emission:
    """Emit a verified composition; direct callers receive the same refusals."""
    invalid = [finding for finding in verify(schedule, target) if finding.severity is FindingSeverity.BLOCKING]
    if invalid:
        raise EmitError(f"{invalid[0].path}: {invalid[0].message}")
    failures = [finding for finding in preflight(schedule, target) if finding.blocks_lowering]
    if failures:
        raise EmitError(f"{failures[0].path}: {failures[0].message}")
    if entry_point is not None and entry_point != schedule.lowering.entry_point:
        raise EmitError("entry point differs from the canonical Lowering route")
    buffers = {buffer.name: buffer for buffer in schedule.buffers}
    # Do not interpolate user buffer/role names as source identifiers.
    names = {buffer.name: f"v{index}" for index, buffer in enumerate(schedule.buffers)}
    globals_ = [buffer for buffer in schedule.buffers if buffer.space is MemorySpace.GLOBAL]
    axes = {axis.name: axis for axis in schedule.program_map.axes}
    accesses = {(access.operation, access.buffer): access for access in schedule.access_maps}
    grid = [1, 1, 1]
    for axis in axes.values():
        grid[axis.axis] = buffers[axis.buffer].shape[axis.dimension]
    lanes = lane_width(schedule)
    groups = lanes // SIMD_WIDTH
    share_slots = _share_slots(schedule, lanes)
    lines = ["#include <metal_stdlib>", "using namespace metal;", "#pragma METAL fp contract(off)",
             f"// SIMD program tile: i belongs to lane i % {lanes}, private slot i / {lanes}.",
             "// Every collective has uniform participation, including padded tail lanes.",
             "// FP32 local-slot accumulation then SIMD sum/max; precise rsqrt; fast math disabled."]
    if groups > 1:
        lines.append(f"// {groups} SIMD groups: each collective completes in threadgroup memory "
                     "under uniform barriers.")
    lines += [f"// Peak live lane-owned Buffer values: {private_values_per_thread(schedule, lanes)} FP32; temporaries/spills unmodeled.",
              f"kernel void {schedule.lowering.entry_point}("]
    for index, buffer in enumerate(globals_):
        const = "const " if buffer.mode is BufferMode.INPUT else ""
        lines.append(f"    device {const}float* {names[buffer.name]} [[buffer({index})]],")
    lines.append("    uint3 program [[threadgroup_position_in_grid]],")
    if groups > 1:
        # Metal requires every position attribute in one kernel to be the same
        # scalar or vector form, and the program position is already uint3.
        lines += ["    uint3 thread_position [[thread_position_in_threadgroup]],",
                  "    uint simd_lane [[thread_index_in_simdgroup]],",
                  "    uint simd_group [[simdgroup_index_in_threadgroup]]) {",
                  "    uint lane = thread_position.x;",
                  f"    threadgroup float share[{share_slots}];"]
    else:
        lines.append("    uint lane [[thread_index_in_simdgroup]]) {")
    for buffer in schedule.buffers:
        if buffer.space is MemorySpace.REGISTER:
            lines.append(f"    float {names[buffer.name]}[{_slots(buffer, lanes)}] = {{}};")

    def address(operation, global_buffer, local_buffer):
        access = accesses[(operation.op_id, global_buffer.name)]
        local_coordinates = iter(_coordinates("i", local_buffer.shape))
        coordinates = []
        for component in access.indices:
            if component.source is AccessIndexKind.PROGRAM:
                coordinates.append(f"ulong(program.{'xyz'[axes[component.name].axis]})")
            else:
                coordinates.append(f"(ulong({next(local_coordinates)}) + {component.offset}ul)")
        return _flat(coordinates, global_buffer.shape)

    for operation in schedule.operations:
        lines += [f"    // CAKE_OP: {operation.op_id}", "    {"]
        src = buffers[operation.reads[0]]
        dst = buffers[operation.writes[0]]
        if operation.kind is OperationKind.REDUCE:
            parameters = operation.parameters
            extent = src.shape[parameters.axis]
            inner = math.prod(src.shape[parameters.axis + 1:])
            identity = "0.0f" if parameters.op is ReduceOp.SUM else "-INFINITY"
            intrinsic = "simd_sum" if parameters.op is ReduceOp.SUM else "simd_max"
            # Every output coordinate is uniform across the SIMD group. Each lane
            # contributes only its own source elements; no cross-lane array indexing.
            # Restrict visits to the containing contiguous source slab. A final-axis
            # reduction has inner=1 and visits exactly its row, including misaligned tails.
            lines += [f"        for (uint output = 0u; output < {dst.elements}u; ++output) {{",
                      f"            float partial = {identity};",
                      f"            uint begin = (output / {inner}u) * {extent * inner}u;",
                      f"            uint end = begin + {extent * inner}u;",
                      f"            for (uint s = begin / {lanes}u; s < (end + {lanes - 1}u) / {lanes}u; ++s) {{",
                      f"                uint i = s * {lanes}u + lane;",
                      f"                if (i >= begin && i < end && i % {inner}u == output % {inner}u) {{"]
            value = f"{names[src.name]}[s]"
            expression = f"partial + {value}" if parameters.op is ReduceOp.SUM else f"max(partial, {value})"
            lines += [f"                    partial = {expression};", "                }", "            }"]
            if groups > 1:
                combine = "reduced + share[g]" if parameters.op is ReduceOp.SUM else "max(reduced, share[g])"
                lines += [f"            float grouped = {intrinsic}(partial);",
                          # The first barrier retires the previous output's reads before
                          # this one overwrites the same threadgroup slots.
                          "            threadgroup_barrier(mem_flags::mem_threadgroup);",
                          "            if (simd_lane == 0u) share[simd_group] = grouped;",
                          "            threadgroup_barrier(mem_flags::mem_threadgroup);",
                          f"            float reduced = {identity};",
                          f"            for (uint g = 0u; g < {groups}u; ++g) {{ reduced = {combine}; }}"]
            else:
                lines.append(f"            float reduced = {intrinsic}(partial);")
            lines += [f"            if (lane == output % {lanes}u) {names[dst.name]}[output / {lanes}u] = reduced;",
                      "        }", "    }"]
            continue

        count = src.elements if operation.kind is OperationKind.STORE else dst.elements
        parameters = operation.parameters
        scalar_reads = {}
        if operation.kind is OperationKind.ELEMENTWISE:
            for position, read in enumerate(operation.reads):
                if buffers[read].is_scalar and not dst.is_scalar:
                    scalar_reads[read] = f"scalar{position}"
                    if groups > 1:
                        # Element zero is owned by thread zero, whose SIMD group cannot
                        # broadcast to the others; publish it to the threadgroup instead.
                        slot = len(scalar_reads) - 1
                        lines += ["        threadgroup_barrier(mem_flags::mem_threadgroup);",
                                  f"        if (lane == 0u) share[{slot}] = {names[read]}[0];",
                                  "        threadgroup_barrier(mem_flags::mem_threadgroup);",
                                  f"        float scalar{position} = share[{slot}];"]
                    else:
                        lines.append(f"        float scalar{position} = simd_broadcast({names[read]}[0], 0u);")
        lines += [f"        for (uint s = 0u; s < {(count + lanes - 1) // lanes}u; ++s) {{",
                  f"            uint i = s * {lanes}u + lane;"]
        if operation.kind is OperationKind.LOAD:
            lines.append(f"            {names[dst.name]}[s] = i < {count}u ? {names[src.name]}[{address(operation, src, dst)}] : 0.0f;")
        elif operation.kind is OperationKind.STORE:
            lines.append(f"            if (i < {count}u) {names[dst.name]}[{address(operation, dst, src)}] = {names[src.name]}[s];")
        else:
            operands = []
            for position, read in enumerate(operation.reads):
                buffer = buffers[read]
                if buffer.shape == dst.shape:
                    operands.append(f"{names[read]}[s]")
                elif read in scalar_reads:
                    operands.append(scalar_reads[read])
                else:
                    coordinates = _coordinates("i", dst.shape)
                    start = parameters.broadcast_axis
                    subscript = _flat(coordinates[start:start + len(buffer.shape)], buffer.shape)
                    # A lane may request a value from a different private slot than
                    # its source lane. Visit slots uniformly; shuffling src[index/32]
                    # directly would select the requesting lane's slot at the source.
                    lines += [f"            uint index{position} = uint({subscript});",
                              f"            float operand{position} = 0.0f;",
                              f"            for (uint k = 0u; k < {_slots(buffer, lanes)}u; ++k) {{",
                              f"                float exchanged = simd_shuffle({names[read]}[k], ushort(index{position} % 32u));",
                              f"                if (k == index{position} / 32u) operand{position} = exchanged;",
                              "            }"]
                    operands.append(f"operand{position}")
            if parameters.scalar is not None:
                operands.append(f"{float(parameters.scalar)!r}f")
            expression = (f"({operands[0]} {_BINARY[parameters.op]} {operands[1]})"
                          if parameters.op in _BINARY else _UNARY[parameters.op].format(x=operands[0]))
            lines.append(f"            {names[dst.name]}[s] = i < {count}u ? {expression} : 0.0f;")
        lines += ["        }", "    }"]
    lines += ["    // CAKE_KERNEL_END", "}", ""]
    return Emission("\n".join(lines), schedule.lowering.entry_point, {}, {
        "buffer_order": [buffer.name for buffer in globals_],
        "threadgroups_per_grid": grid, "threads_per_threadgroup": [lanes, 1, 1],
        "threadgroup_memory_bytes": share_slots * 4, "language_standard": "metal2.3",
        "fast_math_enabled": False, "execution_model": "simd_program_tile",
        "active_threads_per_threadgroup": lanes,
    })
