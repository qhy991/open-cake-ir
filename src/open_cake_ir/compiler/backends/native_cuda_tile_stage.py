"""Schedule-driven tensor-core stage body for a persistent CUDA worker.

The worker owns task claims and the lifetime of TMEM. This emitter owns the
TMA/MMA/readout/store body and the shared/barrier offsets, using the same
lowering routines as a standalone native CUDA Schedule. Its exact admission
scope is the evidenced B300 model-width FFN tile.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from . import native_cuda
from .common import EmitError
from ..ir import DType, LoadMovement, LoweringBackend, MemorySpace, OperationKind


_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
_ROUTE_EVIDENCE = frozenset({'sm_103a'})


@dataclass(frozen=True)
class TileStageEmission:
    source: str
    instruction_helpers: str
    function_name: str
    dynamic_shared_bytes: int
    tmem_columns: int
    mapped_operations: tuple[str, ...]


def emit_tensor_tile_stage(schedule, target, *, function_name: str) -> TileStageEmission:
    """Emit one CTA work unit from a complete Cake TMA/MMA Schedule.

    ``maps_a`` addresses the current logical tile's descriptor and ``map_b``
    the selected expert weight descriptor. The caller owns both bindings,
    task publication, and one TMEM allocation for the persistent CTA.
    """
    if (_IDENTIFIER.fullmatch(function_name) is None):
        raise EmitError('worker stage function needs an ASCII C identifier')
    if (schedule.target != target.target_id or target.target_id not in _ROUTE_EVIDENCE
            or schedule.lowering.backend is not LoweringBackend.NATIVE_CUDA):
        raise EmitError('worker tensor tile requires the evidenced exact B300 native CUDA route')
    for check in (native_cuda.verify, native_cuda.preflight):
        failures = [finding for finding in check(schedule, target)
                    if finding.blocks_lowering or finding.blocks_acceptance]
        if failures:
            raise EmitError('; '.join(f'{f.code} at {f.path}: {f.message}'
                                      for f in failures))
    mapping = schedule.program_map
    if (mapping is None or mapping.persistent or len(mapping.axes) != 2
            or tuple(axis.tile for axis in mapping.axes) != (128, 64)
            or tuple(axis.axis for axis in mapping.axes) != (0, 1)):
        raise EmitError('worker tensor tile needs M128/N64 finite program axes')
    globals_ = [b for b in schedule.buffers if b.space is MemorySpace.GLOBAL]
    loads = [op for op in schedule.operations if op.kind is OperationKind.LOAD
             and op.parameters.movement is LoadMovement.TMA]
    other = [op for op in schedule.operations if op not in loads]
    if (len(globals_) != 3 or len(loads) != 2
            or tuple(op.kind for op in other)
            != (OperationKind.MMA, OperationKind.LOAD, OperationKind.STORE)
            or len(schedule.outputs) != 1):
        raise EmitError('worker tensor tile needs exactly two TMA loads, MMA, TMEM read and store')
    a = schedule.buffer(loads[0].reads[0])
    b = schedule.buffer(loads[1].reads[0])
    output = schedule.buffer(schedule.outputs[0])
    if (a not in globals_ or b not in globals_ or output not in globals_
            or a.dtype is not DType.BF16 or b.dtype is not DType.BF16
            or output.dtype is not DType.FP32
            or len(a.shape) != 2 or len(b.shape) != 2
            or a.shape[0] != 128 or b.shape[1] != a.shape[1]
            or output.shape != (128, b.shape[0])
            or b.shape[0] % 64 or a.shape[1] % 64):
        raise EmitError('worker tensor tile BF16 M128/K/N global shapes differ')
    tensor = [alloc for alloc in schedule.allocations
              if alloc.space is MemorySpace.TENSOR]
    if len(tensor) != 1 or tensor[0].tensor_columns != 64:
        raise EmitError('worker tensor tile needs one externally owned 64-column TMEM allocation')
    emitter = native_cuda._Emitter(schedule, target, function_name)
    if emitter.shared_bytes > 49200:
        raise EmitError('worker tensor tile exceeds the evidenced CTA shared-memory footprint')
    emitter.axisvars = {mapping.axes[0].name: '0', mapping.axes[1].name: 'n_tile'}
    emitter.mapnames = {loads[0].op_id: 'maps_a[0]',
                        loads[1].op_id: 'map_b[0]'}
    emitter.tmemvars = {tensor[0].name: 'tm1'}
    emitter.names[output.name] = 'output'
    emitter.begin(f'__device__ __forceinline__ void {function_name}('
                  'unsigned char* smem, uint32_t* tm1, int warp, float* output, '
                  'const CUtensorMap* maps_a, const CUtensorMap* map_b, int n_tile)')
    for barrier in schedule.barriers:
        key = 'barrier:' + barrier.name
        emitter.line(f'uint64_t* {emitter.barvars[barrier.name]} = '
                     f'reinterpret_cast<uint64_t*>(smem + {emitter.offsets[key]});')
    for index, pipeline in enumerate(schedule.pipelines):
        key = 'free:' + pipeline.name
        emitter.line(f'uint64_t* free{index} = '
                     f'reinterpret_cast<uint64_t*>(smem + {emitter.offsets[key]});')
    for register in (b for b in schedule.buffers if b.space is MemorySpace.REGISTER):
        slots = native_cuda._slots(schedule, register)
        emitter.line(f'{native_cuda._TYPES[register.dtype]} '
                     f'{emitter.names[register.name]}[{slots}];')
    emitter.sequence(None)
    emitter.line('__syncthreads();')
    emitter.invalidate_completions(None)
    emitter.end()
    source = '\n'.join(emitter.lines) + '\n'
    mapped = tuple(line.split('// CAKE_OP: ', 1)[1].strip()
                   for line in emitter.lines if '// CAKE_OP: ' in line)
    required = tuple(op.op_id for op in schedule.operations)
    if mapped != required:
        raise EmitError('worker stage source map differs from its Schedule operations')
    return TileStageEmission(source, native_cuda._INSTRUCTIONS,
                             function_name, emitter.shared_bytes,
                             tensor[0].tensor_columns, mapped)
