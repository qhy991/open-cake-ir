"""Compositional, serial-per-program Metal lowering over the canonical Schedule.

One SIMD group is launched per program coordinate, and lane zero evaluates its complete
FP32 tile using private arrays. This deliberately small correctness route has no hidden
threadgroup storage, collective reduction, operator registry or performance estimate.
Reductions fold in increasing index order; contraction and fast math must stay disabled.
The 4096-value private-storage limit is this backend's bounded first slice, not an Apple
register-file capacity claim. Physical allocation and spills remain toolchain evidence.
"""

from __future__ import annotations

import math
import re

from .emit import BackendPrecondition, Emission, EmitError
from .ir import (
    AccessIndexKind, BufferMode, DType, ElementwiseOp, LoadMovement,
    LoweringBackend, MemorySpace, OperationKind, ReduceOp, ReductionScope, Schedule,
)
from .target import Target
from .verifier import FindingSeverity, verify

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
_UNARY = {ElementwiseOp.SQUARE: "({x} * {x})", ElementwiseOp.RELU: "max({x}, 0.0f)"}


def preflight(schedule: Schedule, target: Target) -> tuple[BackendPrecondition, ...]:
    """Refuse every declaration this emitter cannot faithfully realize."""
    findings = []

    def check(condition, code, path, message):
        if not condition:
            findings.append(BackendPrecondition(code, path, message))

    check(schedule.lowering.backend is LoweringBackend.METAL
          and schedule.target == target.target_id == "apple_gpu_family8"
          and target.architecture == "apple8" and target.device_names == ("Apple M2",),
          "METAL_TARGET_UNSUPPORTED", "target", "Metal requires the exact Apple M2 / apple_gpu_family8 target")
    check(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", schedule.lowering.entry_point)
          and "CAKE_KERNEL_END" not in schedule.lowering.entry_point
          and "__SCHEDULE_SHA256__" not in schedule.lowering.entry_point
          and schedule.lowering.entry_point not in _RESERVED
          and not re.fullmatch(r"(?:bool|char|uchar|short|ushort|int|uint|long|ulong|half|float)(?:[234](?:x[234])?)", schedule.lowering.entry_point),
          "METAL_ENTRY_POINT_UNSUPPORTED", "lowering.entry_point", "Metal requires a non-reserved function identifier")
    check(len(schedule.roles) == 1 and schedule.roles[0].warps == (0,),
          "METAL_ROLE_UNSUPPORTED", "roles", "serial program tiles require one role occupying SIMD group [0]")
    for index, role in enumerate(schedule.roles):
        check(role.registers_per_thread is None, "METAL_REGISTER_CAP_UNSUPPORTED",
              f"roles[{index}].registers_per_thread", "Metal has no CUDA warpgroup register redistribution")
    for field in ("allocations", "pipelines", "barriers", "tile_loops"):
        check(not getattr(schedule, field), "METAL_DECLARATION_UNSUPPORTED", field,
              f"serial Metal lowering does not implement {field}")
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
    private_values = sum(buffer.elements for buffer in schedule.buffers if buffer.space is MemorySpace.REGISTER)
    check(private_values <= 4096, "METAL_PRIVATE_STORAGE_LIMIT", "buffers",
          "serial Metal lowering supports at most 4096 FP32 private values per program tile")
    for index, buffer in enumerate(schedule.buffers):
        path = f"buffers[{index}]"
        check(buffer.dtype in SUPPORTED_DTYPES, "BACKEND_DTYPE_UNEMITTABLE", path + ".dtype", "Metal first slice supports FP32 only")
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
        check(operation.kind in SUPPORTED_OPERATION_KINDS, "BACKEND_OPERATION_UNEMITTABLE", path + ".kind", "operation has no Metal body")
        check(not operation.waits and not operation.signals and operation.pipeline is None,
              "METAL_SYNCHRONIZATION_UNSUPPORTED", path, "Metal serial operations do not implement synchronization or pipelines")
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
                  "lane-zero serial stores require coalesced=false")
            check(len(operation.reads) == len(operation.writes) == 1
                  and schedule.buffer(operation.reads[0]).space is MemorySpace.REGISTER
                  and schedule.buffer(operation.writes[0]).space is MemorySpace.GLOBAL,
                  "METAL_STORE_STORAGE_UNSUPPORTED", path, "Metal stores one private value into one global output")
        elif operation.kind is OperationKind.ELEMENTWISE:
            check(parameters.op in _BINARY or parameters.op in _UNARY,
                  "METAL_ELEMENTWISE_UNSUPPORTED", path + ".parameters.op",
                  "Metal supports add/sub/mul/div/square/relu; PTX FMA and transcendental contracts are not implemented")
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
    failures = preflight(schedule, target)
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
    lines = ["#include <metal_stdlib>", "using namespace metal;", "#pragma METAL fp contract(off)",
             "// Serial program tile: lane 0 owns all private values; other lanes do no work.",
             "// Finite FP32 inputs; increasing-index reductions; fast math must be disabled.",
             f"kernel void {schedule.lowering.entry_point}("]
    for index, buffer in enumerate(globals_):
        const = "const " if buffer.mode is BufferMode.INPUT else ""
        lines.append(f"    device {const}float* {names[buffer.name]} [[buffer({index})]],")
    lines += ["    uint3 program [[threadgroup_position_in_grid]],",
              "    uint lane [[thread_index_in_threadgroup]]) {", "    if (lane != 0u) return;"]
    for buffer in schedule.buffers:
        if buffer.space is MemorySpace.REGISTER:
            lines.append(f"    float {names[buffer.name]}[{buffer.elements}];")

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
        lines.append(f"    // CAKE_OP: {operation.op_id}")
        src = buffers[operation.reads[0]]
        dst = buffers[operation.writes[0]]
        count = src.elements if operation.kind is OperationKind.STORE else dst.elements
        lines.append(f"    for (uint i = 0u; i < {count}u; ++i) {{")
        if operation.kind is OperationKind.LOAD:
            lines.append(f"        {names[dst.name]}[i] = {names[src.name]}[{address(operation, src, dst)}];")
        elif operation.kind is OperationKind.STORE:
            lines.append(f"        {names[dst.name]}[{address(operation, dst, src)}] = {names[src.name]}[i];")
        elif operation.kind is OperationKind.ELEMENTWISE:
            parameters = operation.parameters
            operands = []
            for read in operation.reads:
                buffer = buffers[read]
                if buffer.shape == dst.shape:
                    subscript = "i"
                elif buffer.elements == 1:
                    subscript = "0"
                else:
                    coordinates = _coordinates("i", dst.shape)
                    start = parameters.broadcast_axis if parameters.broadcast_axis is not None else len(dst.shape) - len(buffer.shape)
                    subscript = _flat(coordinates[start:start + len(buffer.shape)], buffer.shape)
                operands.append(f"{names[read]}[{subscript}]")
            if parameters.scalar is not None:
                operands.append(f"{float(parameters.scalar)!r}f")
            if parameters.op in _BINARY:
                expression = f"({operands[0]} {_BINARY[parameters.op]} {operands[1]})"
            else:
                expression = _UNARY[parameters.op].format(x=operands[0])
            lines.append(f"        {names[dst.name]}[i] = {expression};")
        else:
            parameters = operation.parameters
            extent = src.shape[parameters.axis]
            coordinates = [] if len(src.shape) == 1 else _coordinates("i", dst.shape)
            coordinates.insert(parameters.axis, "j")
            initial = "0.0f" if parameters.op is ReduceOp.SUM else "-INFINITY"
            lines.append(f"        float accumulator = {initial};")
            lines.append(f"        for (uint j = 0u; j < {extent}u; ++j) {{")
            value = f"{names[src.name]}[{_flat(coordinates, src.shape)}]"
            expression = f"accumulator + {value}" if parameters.op is ReduceOp.SUM else f"max(accumulator, {value})"
            lines += [f"            accumulator = {expression};", "        }", f"        {names[dst.name]}[i] = accumulator;"]
        lines.append("    }")
    lines += ["    // CAKE_KERNEL_END", "}", ""]
    return Emission("\n".join(lines), schedule.lowering.entry_point, {}, {
        "buffer_order": [buffer.name for buffer in globals_],
        "threadgroups_per_grid": grid, "threads_per_threadgroup": [32, 1, 1],
        "threadgroup_memory_bytes": 0, "language_standard": "metal2.3",
        "fast_math_enabled": False, "execution_model": "serial_program_tile",
        "active_threads_per_threadgroup": 1,
    })
