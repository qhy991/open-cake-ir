"""Operation-driven CUDA C++/PTX for explicit single-CTA Blackwell schedules.

No framework compiler or kernel template participates. A bounded K loop has independent
TMA and MMA warp roles connected by the declared circular pipeline. Other operations
compose over a row-owned register tile. See docs/NATIVE_CUDA_DESIGN.md for the admitted
hardware domain; preflight is also consumed by direct emission.
"""
from __future__ import annotations

import math

from .common import Emission, EmitError, refusal, vocabulary_findings
from ..diagnostics import Finding
from ..ir import (
    AccessIndexKind, BarrierMechanism, BufferMode, DType, ElementwiseOp,
    LoadMovement, LoweringBackend, MemorySpace, OperandMajorMode, OperandSource,
    OperationKind, Schedule, Swizzle,
)
from ..target import Target
from ..verifier import verify

SUPPORTED_DTYPES = frozenset({DType.BF16, DType.FP16, DType.FP32, DType.INT32})
SUPPORTED_OPERATION_KINDS = frozenset({OperationKind.LOAD, OperationKind.MMA,
    OperationKind.ELEMENTWISE, OperationKind.CAST, OperationKind.REDUCE_ARGMIN,
    OperationKind.STORE})
_TYPES = {DType.BF16: '__nv_bfloat16', DType.FP16: '__half',
          DType.FP32: 'float', DType.INT32: 'int32_t'}
_SWIZZLE = {Swizzle.B32: (32, 6), Swizzle.B64: (64, 4), Swizzle.B128: (128, 2)}
_TARGETS = {'sm_100a': (10, 0), 'sm_103a': (10, 3)}
_CONTRACT = 'tcgen05.mma.cta_group::1.kind::f16'


def _scope(s, op):
    return next((loop for loop in s.tile_loops if op.op_id in loop.body), None)


def _pipeline_loop(s, pipeline):
    scopes = {_scope(s, op).name if _scope(s, op) else None
              for op in s.operations if op.pipeline == pipeline.name}
    return s.tile_loop(next(iter(scopes))) if len(scopes) == 1 and None not in scopes else None


def _publication_scope(s, operation):
    """The final accumulator is consumed in its contraction's parent scope."""
    loop = _scope(s, operation)
    return s.loop_parent().get(loop.name) if loop is not None else None


def _scalar_row(s, buffer):
    return any(op.kind is OperationKind.REDUCE_ARGMIN and buffer.name in op.writes
               for op in s.operations)


def _slots(s, buffer):
    return 1 if _scalar_row(s, buffer) else buffer.shape[-1]


def _storage(s):
    """Shared allocations own operand offsets. Barrier/control bytes are mechanical."""
    offsets, cursor = {}, 0
    for allocation in s.allocations:
        if allocation.space is MemorySpace.SHARED:
            cursor = (cursor + 1023) // 1024 * 1024
            offsets[allocation.name] = cursor
            cursor += allocation.size_bytes
    cursor = (cursor + 7) // 8 * 8
    for barrier in s.barriers:
        slots = next((p.stages for p in s.pipelines if p.name == barrier.pipeline), 1)
        offsets['barrier:' + barrier.name] = cursor
        cursor += slots * 8
    for pipeline in s.pipelines:
        offsets['free:' + pipeline.name] = cursor
        cursor += pipeline.stages * 8
    for allocation in s.allocations:
        if allocation.space is MemorySpace.TENSOR:
            offsets['tmem:' + allocation.name] = cursor
            cursor += 4
    return offsets, (cursor + 15) // 16 * 16


def requirements(s: Schedule) -> tuple[Finding, ...]:
    """Backend-owned vocabulary admission, shared by public and direct emission."""
    return vocabulary_findings(s, SUPPORTED_DTYPES, SUPPORTED_OPERATION_KINDS)


def preflight(s: Schedule, target: Target) -> tuple[Finding, ...]:
    failures = list(requirements(s))
    if failures:
        return tuple(failures)
    def check(ok, code, path, message):
        if not ok:
            failures.append(refusal(code, path, message))
    check(s.lowering.backend is LoweringBackend.NATIVE_CUDA,
          'NATIVE_ROUTE_UNSUPPORTED', 'lowering.backend',
          'native CUDA emission requires the canonical native_cuda route')
    check(s.target in _TARGETS and target.target_id == s.target
          and target.compute_capability == _TARGETS.get(s.target),
          'NATIVE_TARGET_UNSUPPORTED', 'target',
          'native CUDA requires an exact sm_100a or sm_103a Target and matching compute capability')
    check(s.program_map is None or not s.program_map.persistent,
          'NATIVE_PERSISTENCE_UNSUPPORTED', 'program_map', 'native CUDA does not implement persistent traversal')
    check(s.residency is None, 'NATIVE_RESIDENCY_UNSUPPORTED', 'residency',
          'native CUDA does not yet enforce register caps or requested CTA residency')
    check(s.grid in (None, (1, 1, 1)), 'NATIVE_GRID_UNSUPPORTED', 'grid',
          'multi-CTA ownership must be expressed by ProgramMap')
    check(bool(s.pipelines), 'NATIVE_PIPELINE_REQUIRED', 'pipelines',
          'explicit TMA/MMA lowering requires at least one pipeline')
    for field in ('allocations','barriers','pipelines'):
        for i,item in enumerate(getattr(s,field)):
            check(not any(c in item.name for c in '\\\r\n'), 'NATIVE_NAME_UNSUPPORTED',
                  f'{field}[{i}].name', 'source-map resource names cannot contain line breaks or backslashes')
    roles = {r.name: r for r in s.roles}
    buffers = {b.name: b for b in s.buffers}
    writers = {name: op for op in s.operations for name in op.writes}
    # Public Compiler verification normally diagnoses these first; the backend's
    # direct preflight must remain total over structurally parsed Schedules as well.
    unknown = []
    for i,op in enumerate(s.operations):
        if op.role not in roles:
            unknown.append(refusal('NATIVE_REFERENCE_UNKNOWN', f'operations[{i}].role', f'unknown role {op.role!r}'))
        for field in ('reads','writes'):
            for j,name in enumerate(getattr(op,field)):
                if name not in buffers:
                    unknown.append(refusal('NATIVE_REFERENCE_UNKNOWN', f'operations[{i}].{field}[{j}]', f'unknown buffer {name!r}'))
    for i,access in enumerate(s.access_maps):
        if access.buffer not in buffers:
            unknown.append(refusal('NATIVE_REFERENCE_UNKNOWN', f'access_maps[{i}].buffer', f'unknown buffer {access.buffer!r}'))
    if unknown:
        return tuple(failures+unknown)
    for i, role in enumerate(s.roles):
        check(all(c not in role.name for c in '\\\r\n'), 'NATIVE_NAME_UNSUPPORTED',
              f'roles[{i}].name', 'source-map names must occupy one line')
        check(role.registers_per_thread is None, 'NATIVE_ROLE_REGISTERS_UNSUPPORTED',
              f'roles[{i}].registers_per_thread', 'setmaxnreg redistribution is not implemented')
    for i, b in enumerate(s.buffers):
        path = f'buffers[{i}]'
        check(all(c not in b.name for c in '\\\r\n'), 'NATIVE_NAME_UNSUPPORTED', path+'.name', 'source-map names must occupy one line')
        check(b.dtype in SUPPORTED_DTYPES, 'NATIVE_DTYPE_UNSUPPORTED', path+'.dtype', 'unsupported native storage type')
        check(b.mode is not BufferMode.STATE and b.valid_extent is None and b.scale_of is None,
              'NATIVE_BUFFER_REFINEMENT_UNSUPPORTED', path,
              'native CUDA does not implement state, valid-extent or block-scale relations')
        check(b.space in (MemorySpace.SHARED, MemorySpace.TENSOR) or
              b.stages == 1 and b.swizzle is None and b.byte_offset == 0,
              'NATIVE_BUFFER_REFINEMENT_UNSUPPORTED', path,
              'register/global buffers cannot carry unused staging or placement refinements')
        if b.space is MemorySpace.GLOBAL:
            check(b.elements <= 2147483647, 'NATIVE_INDEX_RANGE', path,
                  'native contiguous addressing currently requires a signed-32-bit element domain')
        if b.space is MemorySpace.SHARED:
            sw = _SWIZZLE.get(b.swizzle)
            check(sw is not None and len(b.shape) == 2 and b.dtype in (DType.BF16, DType.FP16)
                  and b.shape[1]*b.dtype.itemsize == (sw[0] if sw else 0),
                  'NATIVE_SMEM_LAYOUT', path,
                  'K-major native TMA tiles require a complete 32/64/128-byte swizzle row')
            check(b.byte_offset % 1024 == 0, 'NATIVE_SMEM_ALIGNMENT', path+'.byte_offset',
                  'native stage bases require 1024-byte alignment for zero swizzle base offset')
        if b.space is MemorySpace.TENSOR:
            check(b.swizzle is None, 'NATIVE_TMEM_SWIZZLE_UNSUPPORTED', path+'.swizzle',
                  'native TMEM addressing uses columns and physical lanes, not an SMEM swizzle')
            check(len(b.shape) == 2 and b.shape[0] == 128 and b.dtype is DType.FP32
                  and b.byte_offset % 512 == 0 and b.stages == 1,
                  'NATIVE_TMEM_LAYOUT', path, 'native accumulators use 128 FP32 rows and whole TMEM columns')
        if b.space is MemorySpace.REGISTER:
            check(len(b.shape) in (1,2) and (len(b.shape) == 1 or b.shape[0] == 128),
                  'NATIVE_REGISTER_LAYOUT', path, 'register tiles use 128 row-owning threads or replicated vectors')
    for i, allocation in enumerate(s.allocations):
        check(allocation.space in (MemorySpace.SHARED, MemorySpace.TENSOR),
              'NATIVE_ALLOCATION_SPACE', f'allocations[{i}]', 'only explicit shared and tensor allocations are emitted')
        if allocation.space is MemorySpace.TENSOR:
            check(allocation.tensor_columns in (32,64,128,256,512), 'NATIVE_TMEM_ALLOCATION',
                  f'allocations[{i}]', 'tcgen05.alloc requires a power-of-two column count from 32 through 512')
    check(_storage(s)[1] <= target.resource_limits.maximum_shared_memory_bytes,
          'NATIVE_SHARED_MEMORY_LIMIT', 'allocations',
          'operand allocations plus derived barriers/control exceed target shared-memory capacity')
    for i, access in enumerate(s.access_maps):
        b = buffers[access.buffer]
        check(b.space is MemorySpace.GLOBAL, 'NATIVE_ACCESS_SPACE', f'access_maps[{i}]',
              'local storage is addressed by its declared view, not a global AccessMap')
        for j, component in enumerate(access.indices):
            check(component.source in (AccessIndexKind.DIMENSION, AccessIndexKind.PROGRAM, AccessIndexKind.PROGRAM_TILE,
                                        AccessIndexKind.LOOP_TILE),
                  'NATIVE_ACCESS_UNSUPPORTED', f'access_maps[{i}].indices[{j}]',
                  'native accesses admit contiguous dimension, program_tile and loop_tile coordinates')
    if s.program_map:
        for i, axis in enumerate(s.program_map.axes):
            check(axis.tile >= 1, 'NATIVE_PROGRAM_AXIS', f'program_map.axes[{i}]',
                  'native program coordinates use positive tiles')
    pipe_loops = {}
    owned_barriers = set()
    for i, pipeline in enumerate(s.pipelines):
        tagged = [op for op in s.operations if op.pipeline == pipeline.name]
        scopes = {_scope(s, op).name if _scope(s, op) else None for op in tagged}
        check(len(scopes) == 1, 'NATIVE_PIPELINE_SCOPE', f'pipelines[{i}]',
              'each pipeline belongs to one contraction loop or one contiguous root group')
        if len(scopes) != 1:
            continue
        loop = _pipeline_loop(s, pipeline)
        if loop is not None:
            pipe_loops[loop.name] = pipeline
        body = s.loop_operations(loop) if loop is not None else tagged
        body_path = f'tile_loops[{s.tile_loops.index(loop)}].body' if loop is not None else f'pipelines[{i}]'
        if loop is None:
            positions = [s.operations.index(op) for op in tagged]
            check(positions == list(range(positions[0],positions[-1]+1)),
                  'NATIVE_PIPELINE_SCOPE', body_path, 'a root pipeline group must be contiguous in the operation DAG')
            check(pipeline.stages == 1, 'NATIVE_PIPELINE_STAGES', body_path,
                  'one contraction tile without a K loop uses one explicitly declared stage')
        loads = [op for op in body if op.kind is OperationKind.LOAD and op.parameters.movement in (LoadMovement.TMA, LoadMovement.GLOBAL)]
        mmas = [op for op in body if op.kind is OperationKind.MMA]
        check(bool(loads) and bool(mmas) and len(loads)+len(mmas) == len(body)
              and all(op.pipeline == pipeline.name for op in body)
              and (loop is None or list(loop.body) == [op.op_id for op in body]),
              'NATIVE_PIPELINE_BODY', body_path,
              'pipeline loops contain direct TMA loads followed by one or more MMA operations')
        check([op.kind for op in body] == [OperationKind.LOAD]*len(loads)+[OperationKind.MMA]*len(mmas),
              'NATIVE_PIPELINE_ORDER', body_path,
              'all stage producers must precede its MMA consumers')
        check(len({op.role for op in loads}) == 1 and len({op.role for op in mmas}) == 1
              and {op.role for op in loads}.isdisjoint({op.role for op in mmas}),
              'NATIVE_PIPELINE_ROLES', f'pipelines[{i}]',
              'a pipeline requires separate single-warp TMA and MMA roles')
        ready = [b for b in s.barriers if b.pipeline == pipeline.name]
        check(len(ready) == 1, 'NATIVE_PIPELINE_BARRIER', f'pipelines[{i}]',
              'one declared ready barrier owns the stage transaction arrivals')
        if len(ready) == 1:
            b = ready[0]; owned_barriers.add(b.name)
            check(b.mechanism is BarrierMechanism.MBARRIER and b.count == len(loads)
                  and all(op.signals == (b.name,) and not op.waits for op in loads)
                  and all(op.waits == (b.name,) for op in mmas),
                  'NATIVE_PIPELINE_BARRIER', f'barriers[{s.barriers.index(b)}]',
                  'ready arrival count and exact TMA/MMA edges must match every stage producer/consumer')
        staged = {name for op in loads for name in op.writes}
        consumed = {name for op in mmas for name in op.reads}
        check(staged == consumed, 'NATIVE_PIPELINE_STAGE_OWNERSHIP', f'pipelines[{i}]',
              'every staged operand is produced and consumed by this pipeline')
        for name in staged:
            check(buffers[name].stages == pipeline.stages,
                  'NATIVE_PIPELINE_STAGES', f'buffers[{s.buffers.index(buffers[name])}].stages',
                  'every operand has exactly the pipeline stage count')
            check(all(op in mmas for op in s.operations if name in op.reads),
                  'NATIVE_PIPELINE_STAGE_ESCAPE', f'buffers[{s.buffers.index(buffers[name])}]',
                  'stage readers cannot escape the completion-protected pipeline')
        for mma in mmas:
            check(loop is None or (s.mma_accumulates_over(mma, loop)
                  and all(s._staged_axis_filled_by(name, loop) == 1 for name in mma.reads)),
                  'NATIVE_MMA_CONTRACTION_SCOPE', f'operations[{s.operations.index(mma)}]',
                  'both operand AccessMaps must use this loop for the K dimension')
    for i, loop in enumerate(s.tile_loops):
        options = loop.range_options
        required_stages = pipe_loops[loop.name].stages if loop.name in pipe_loops else 1
        for field, expected in (('num_stages', required_stages), ('loop_unroll_factor',1),
                                ('flatten',False), ('warp_specialize',True),
                                ('disallow_acc_multi_buffer',True), ('disable_licm',False)):
            check(getattr(options, field) == expected, 'NATIVE_RANGE_OPTION_UNSUPPORTED',
                  f'tile_loops[{i}].range_options.{field}',
                  f'native lowering requires {field}={expected!r} for this loop')
        check(loop.stop is None, 'NATIVE_LOOP_STOP_UNSUPPORTED', f'tile_loops[{i}].stop',
              'dynamic loop stops are not implemented')
    for i, op in enumerate(s.operations):
        path = f'operations[{i}]'
        check(all(c not in op.op_id for c in '\\\r\n'), 'NATIVE_NAME_UNSUPPORTED', path+'.id', 'source-map names must occupy one line')
        check(op.kind in SUPPORTED_OPERATION_KINDS, 'NATIVE_OPERATION_UNSUPPORTED', path+'.kind', 'unsupported native operation')
        if op.kind not in SUPPORTED_OPERATION_KINDS:
            continue
        check(len(op.writes) == 1, 'NATIVE_OPERATION_ARITY', path+'.writes', 'native operations have one explicit result')
        if not op.writes or any(name not in buffers for name in op.reads+op.writes):
            continue
        dst = buffers[op.writes[0]]
        role = roles[op.role]
        asynchronous = op.kind is OperationKind.MMA or (op.kind is OperationKind.LOAD and dst.space is MemorySpace.SHARED)
        check(len(role.warps) == (1 if asynchronous else 4), 'NATIVE_ROLE_WIDTH', path+'.role',
              'TMA/MMA roles use one warp; row-owned arithmetic/transfer/store roles use four')
        if not asynchronous:
            check(role.warps[0] % 4 == 0, 'NATIVE_ROLE_ALIGNMENT', path+'.role',
                  'TMEM row ownership requires an aligned group of four physical warps')
            check(op.pipeline is None and not op.signals, 'NATIVE_OPERATION_SYNC', path,
                  'register operations do not produce asynchronous stage signals')
            if not (op.kind is OperationKind.LOAD and op.parameters.movement is LoadMovement.TMEM):
                check(not op.waits, 'NATIVE_OPERATION_SYNC', path+'.waits', 'this operation has no asynchronous wait protocol')
        for name in op.reads:
            b = buffers[name]
            if b.space is MemorySpace.REGISTER and name in writers:
                check(writers[name].role == op.role, 'NATIVE_REGISTER_ROLE_OWNERSHIP', path+'.reads',
                      'a register value must stay within its producer role')
        if op.kind is OperationKind.MMA:
            p = op.parameters; instruction = p.instruction
            good = (instruction is not None and instruction.contract == _CONTRACT
                    and instruction.cta_group == 1 and instruction.operand_source is OperandSource.SHARED
                    and instruction.operand_major == (OperandMajorMode.K, OperandMajorMode.K)
                    and p.tile_shape is not None and instruction.shape == (128,p.tile_shape[1],16)
                    and p.tile_shape[0] == 128 and 8 <= p.tile_shape[1] <= 256 and p.tile_shape[1] % 8 == 0
                    and len(op.reads) == 2 and all(buffers[n].space is MemorySpace.SHARED for n in op.reads)
                    and all(buffers[n].dtype in (DType.BF16,DType.FP16) for n in op.reads)
                    and buffers[op.reads[0]].dtype == buffers[op.reads[1]].dtype
                    and dst.space is MemorySpace.TENSOR)
            check(good, 'NATIVE_MMA_CONTRACT', path+'.parameters.instruction',
                  'native MMA requires explicit M128/N8..256/K16 f16-family, matching half/BF16 shared K-major operands and FP32 TMEM')
            if p.k_ranges is not None:
                check(good and all(endpoint % instruction.shape[2] == 0
                                   for interval in p.k_ranges for endpoint in interval),
                      'NATIVE_MMA_K_RANGES', path+'.parameters.k_ranges',
                      'native K contributions require the f16-family contract and endpoints aligned to its declared instruction K')
            if p.tile_shape is not None:
                check(len(op.reads) == 2
                      and buffers[op.reads[0]].shape == (p.tile_shape[0], p.tile_shape[2])
                      and buffers[op.reads[1]].shape == (p.tile_shape[1], p.tile_shape[2])
                      and dst.shape == p.tile_shape[:2],
                      'NATIVE_MMA_TILE_DOMAIN', path+'.parameters.tile_shape',
                      'the full input and result tile domains must match A[M,K], B[N,K] and result[M,N]')
            check(len(op.reads) == 2 and all(name in writers and writers[name].kind is OperationKind.LOAD for name in op.reads),
                  'NATIVE_MMA_OPERAND_WRITER', path+'.reads', 'MMA operands must be explicitly staged by loads')
            check(op.pipeline is not None, 'NATIVE_MMA_PIPELINE', path+'.pipeline', 'MMA must belong to an explicit contraction pipeline')
            check(len(op.signals) == 1, 'NATIVE_MMA_COMPLETION', path+'.signals', 'each MMA needs one distinct completion mbarrier')
            for name in op.signals:
                b = next((b for b in s.barriers if b.name == name), None)
                if b:
                    owned_barriers.add(name)
                    check(b.pipeline is None and b.count == 1 and b.mechanism is BarrierMechanism.MBARRIER
                          and sum(name in other.signals for other in s.operations) == 1,
                          'NATIVE_MMA_COMPLETION', path+'.signals', 'MMA completion must have one owning issuer and one arrival')
                    for consumer in (other for other in s.operations if name in other.waits):
                        check(consumer.kind is OperationKind.LOAD
                              and consumer.parameters.movement is LoadMovement.TMEM
                              and consumer.reads == op.writes and consumer.waits == (name,)
                              and (_scope(s,consumer).name if _scope(s,consumer) else None) == _publication_scope(s,op),
                              'NATIVE_MMA_COMPLETION_CONSUMER',
                              f'operations[{s.operations.index(consumer)}].waits',
                              'completion consumers must read this final TMEM accumulator in the contraction parent scope; per-K notifications are not admitted')

        elif op.kind is OperationKind.LOAD:
            p = op.parameters; src = buffers[op.reads[0]]
            check(len(op.reads) == 1 and p.reuse is None,
                  'NATIVE_LOAD_REFINEMENT', path,
                  'native loads use one source without cache/reuse refinements; no cache policy is emitted')
            if dst.space is MemorySpace.SHARED:
                scope = _scope(s, op)
                check(op.pipeline is not None and (scope is None or scope.name in pipe_loops),
                      'NATIVE_SHARED_LOAD_SCOPE', path+'.pipeline',
                      'native global-to-shared staging must belong to an explicit contraction pipeline')
            if p.movement is LoadMovement.TMA:
                check(src.space is MemorySpace.GLOBAL and src.mode is BufferMode.INPUT
                      and len(src.shape) in (2,3) and dst.space is MemorySpace.SHARED
                      and p.descriptor_box == dst.shape and src.dtype == dst.dtype
                      and src.shape[-1]*src.dtype.itemsize % 16 == 0 and op.pipeline is not None,
                      'NATIVE_TMA_DESCRIPTOR', path+'.parameters',
                      'TMA requires a contiguous rank-2 input with 16-byte row stride and a declared shared box')
            elif p.movement is LoadMovement.TMEM:
                producer = writers.get(src.name)
                check(producer is not None and producer.kind is OperationKind.MMA and op.waits == producer.signals
                      and len(src.shape) == 2 and src.shape[0] == 128
                      and p.source_atom is not None and src.shape[1] % p.source_atom.repetition == 0,
                      'NATIVE_TMEM_COMPLETION', path,
                      'TMEM load must wait its MMA completion and read full atom repetitions')
                if producer:
                    check((_scope(s, op).name if _scope(s, op) else None) == _publication_scope(s, producer),
                          'NATIVE_TMEM_LIFETIME', path, 'TMEM readout must share the contraction loop parent scope, before the next output tile overwrites it')
            else:
                check(src.space is MemorySpace.GLOBAL and dst.space in (MemorySpace.REGISTER, MemorySpace.SHARED)
                      and src.dtype == dst.dtype and len(dst.shape) in (1,2),
                      'NATIVE_GLOBAL_LOAD', path, 'global loads preserve dtype into row-owned registers')
            if src.space is MemorySpace.GLOBAL:
                check(s.access_map(op.op_id, src.name) is not None, 'NATIVE_ACCESS_REQUIRED', path,
                      'every global transfer requires an explicit AccessMap')
                access=s.access_map(op.op_id,src.name)
                if access is not None:
                    check(sum(c.source is not AccessIndexKind.PROGRAM for c in access.indices) == len(dst.shape),
                          'NATIVE_ACCESS_RANK', path, 'global access vector axes must match the local tile rank')
                    if p.movement is LoadMovement.TMA:
                        check(all(c.source is AccessIndexKind.PROGRAM for c in access.indices[:-2])
                              and all(c.source is not AccessIndexKind.PROGRAM for c in access.indices[-2:]),
                              'NATIVE_TMA_COORDINATES', path, 'TMA scalar program axes precede the two tiled matrix axes')

        elif op.kind is OperationKind.ELEMENTWISE:
            check(op.parameters.op in (ElementwiseOp.ADD, ElementwiseOp.SUB, ElementwiseOp.MUL,
                                        ElementwiseOp.DIV, ElementwiseOp.RELU, ElementwiseOp.SQUARE),
                  'NATIVE_ARITHMETIC_UNSUPPORTED', path+'.parameters.op', 'native arithmetic currently admits add/sub/mul/div/relu/square')
            check(dst.dtype is DType.FP32 and all(buffers[n].dtype is DType.FP32 for n in op.reads),
                  'NATIVE_ARITHMETIC_DTYPE', path, 'native elementwise operations use FP32 values')
            check(dst.space is MemorySpace.REGISTER
                  and all(buffers[n].space is MemorySpace.REGISTER for n in op.reads),
                  'NATIVE_ARITHMETIC_STORAGE', path,
                  'native elementwise operations read and write row-owned registers; load global inputs explicitly')
            check(op.parameters.broadcast_axis in (None,1), 'NATIVE_BROADCAST_UNSUPPORTED', path,
                  'native row ownership currently admits trailing-column broadcasts only')
            check(op.parameters.scalar is None or math.isfinite(op.parameters.scalar)
                  and abs(op.parameters.scalar) <= 3.4028234663852886e38,
                  'NATIVE_SCALAR_FINITE', path+'.parameters.scalar', 'native scalar literals must be finite FP32 values')
        elif op.kind is OperationKind.REDUCE_ARGMIN:
            src = buffers[op.reads[0]]
            check(len(src.shape) == 2 and src.shape[0] == 128 and src.space is MemorySpace.REGISTER
                  and dst.shape == (128,) and dst.space is MemorySpace.REGISTER,
                  'NATIVE_ARGMIN_LAYOUT', path, 'argmin maps each FP32 register row to one INT32 index')
            scope = _scope(s, op)
            check(not op.parameters.across_loop or scope is not None and scope.tile == src.shape[1]
                  and scope.name not in pipe_loops,
                  'NATIVE_ARGMIN_SCOPE', path, 'streaming argmin must reduce columns in its enclosing output-tile loop')
            check(s.argmin_domain(op) is not None, 'NATIVE_ARGMIN_DOMAIN', path+'.reads',
                  'argmin requires one proven static zero-origin candidate domain: a complete resident tile or its enclosing candidate loop')
        elif op.kind is OperationKind.CAST:
            check(dst.space is MemorySpace.REGISTER and buffers[op.reads[0]].space is MemorySpace.REGISTER,
                  'NATIVE_CAST_SPACE', path, 'cast preserves row-owned register storage')
        elif op.kind is OperationKind.STORE:
            src = buffers[op.reads[0]]
            check(dst.space is MemorySpace.GLOBAL and src.space is MemorySpace.REGISTER
                  and dst.dtype == src.dtype,
                  'NATIVE_STORE_CONTRACT', path, 'native stores preserve register dtype to global output')
            check(len(src.shape) == 2 and src.shape[0] == 128 or _scalar_row(s,src),
                  'NATIVE_STORE_ROLE_OWNERSHIP', path+'.reads',
                  'native stores require a row-owned matrix or one argmin result per row; replicated vectors have no unique writing thread')
            check(not op.parameters.coalesced, 'NATIVE_STORE_COALESCING', path+'.parameters.coalesced',
                  'row-owned native stores require an explicit coalesced=false commitment')
            access = s.access_map(op.op_id, dst.name)
            check(access is not None, 'NATIVE_ACCESS_REQUIRED', path,
                  'every global store requires an explicit AccessMap')
            if access is not None and s.program_map is not None:
                owned = {component.name for component in access.indices
                         if component.source in (AccessIndexKind.PROGRAM,AccessIndexKind.PROGRAM_TILE)}
                missing = [axis.name for axis in s.program_map.axes
                           if (owner := s.buffer(axis.buffer)) is not None
                           and axis.dimension < len(owner.shape)
                           and axis.tile_count(owner.shape[axis.dimension]) > 1
                           and axis.name not in owned]
                check(not missing, 'NATIVE_STORE_PROGRAM_OWNERSHIP', path,
                      f'native store omits varying program axes {missing}, allowing different CTAs to write the same output')
    for i, b in enumerate(s.barriers):
        check(b.name in owned_barriers, 'NATIVE_BARRIER_UNSUPPORTED', f'barriers[{i}]',
              'native barriers belong to a TMA stage or an MMA completion')
    return tuple(failures)


class _Emitter:
    def __init__(self, s, target, entry):
        self.s, self.target, self.entry = s, target, entry
        self.lines, self.indent = [], 0
        # Ordinal identifiers avoid both C++ keywords and sanitized-name collisions.
        self.names = {b.name: f'b{i}' for i,b in enumerate(s.buffers)}
        self.roles = {r.name: r for r in s.roles}
        self.offsets, self.shared_bytes = _storage(s)
        self.globals = [b for b in s.buffers if b.space is MemorySpace.GLOBAL]
        self.loads = [op for op in s.operations if op.kind is OperationKind.LOAD and op.parameters.movement is LoadMovement.TMA]
        self.mapnames = {op.op_id: f'map{i}' for i,op in enumerate(self.loads)}
        self.pipeloops = {loop.name:p for p in s.pipelines if (loop := _pipeline_loop(s,p)) is not None}
        self.rootpipes = [p for p in s.pipelines if _pipeline_loop(s,p) is None]
        self.loopvars = {loop.iterator:f'it{i}' for i,loop in enumerate(s.tile_loops)}
        self.axisvars = {axis.name:f'blockIdx.{"xyz"[axis.axis]}' for axis in s.program_map.axes} if s.program_map else {}
        self.barvars = {b.name:f'bar{i}' for i,b in enumerate(s.barriers)}
        self.tmemvars = {a.name:f'tm{i}' for i,a in enumerate(s.allocations) if a.space is MemorySpace.TENSOR}

    def line(self, text=''):
        self.lines.append('  '*self.indent + text)
    def begin(self, text):
        self.line(text+' {'); self.indent += 1
    def end(self):
        self.indent -= 1; self.line('}')
    def role_condition(self, name):
        r = self.roles[name]
        return f'warp >= {r.warps[0]} && warp <= {r.warps[-1]}'
    def row(self, op):
        return f'(int(threadIdx.x) - {self.roles[op.role].warps[0]*32})'
    def b(self, name):
        return self.s.buffer(name)
    def trip(self, loop):
        extent = self.b(loop.buffer).shape[loop.dimension]
        return (extent+loop.tile-1)//loop.tile
    def access(self, op, buffer, local):
        access = self.s.access_map(op.op_id, buffer.name)
        coords = []
        local_axis = 0
        for axis, component in enumerate(access.indices):
            if component.source is AccessIndexKind.PROGRAM:
                coords.append(f'int({self.axisvars[component.name]})')
                continue
            value = local[local_axis]
            local_axis += 1
            if component.source is AccessIndexKind.DIMENSION:
                coords.append(f'({component.offset} + {value})')
            elif component.source is AccessIndexKind.LOOP_TILE:
                loop = next(l for l in self.s.tile_loops if l.iterator == component.name)
                coords.append(f'({self.loopvars[component.name]} * {loop.tile} + {value})')
            else:
                axis_decl = self.s.program_map.axis(component.name)
                coords.append(f'({self.axisvars[component.name]} * {axis_decl.tile} + {value})')
        return coords
    def address(self, op, buffer, local):
        coords = self.access(op,buffer,local)
        index = ' + '.join(f'({c}) * {math.prod(buffer.shape[i+1:])}' for i,c in enumerate(coords))
        mask = ' && '.join(f'({c}) < {extent}' for c,extent in zip(coords, buffer.shape))
        return f'{self.names[buffer.name]}[{index}]', mask
    def pointer(self, buffer, stage):
        size = buffer.elements*buffer.dtype.itemsize
        return f'(smem + {self.offsets[buffer.allocation]+buffer.byte_offset} + ({stage})*{size})'
    def taddr(self, buffer):
        return f'(*{self.tmemvars[buffer.allocation]} + {buffer.byte_offset//512})'

    def emit(self):
        self.line('// Generated by Open-Cake native CUDA; schedule_sha256=__SCHEDULE_SHA256__')
        self.line('#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n#include <cstdint>\n#include <new>\n#include <cstring>\n#include <cmath>')
        arch = _TARGETS[self.s.target][0]*100 + _TARGETS[self.s.target][1]*10
        self.line(f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}\n#error "Schedule requires exact {self.s.target}"\n#endif')
        self.line(_INSTRUCTIONS)
        params = [f'{_TYPES[b.dtype]}* {self.names[b.name]}' for b in self.globals]
        params += [f'const __grid_constant__ CUtensorMap {self.mapnames[op.op_id]}' for op in self.loads]
        self.begin(f'extern "C" __global__ void {self.entry}_kernel('+', '.join(params)+')')
        self.line('extern __shared__ __align__(1024) unsigned char smem[];')
        self.line('const int warp = int(threadIdx.x) / 32;')
        for b in self.s.barriers:
            self.line(f'// CAKE_OP: resource.barrier.{b.name}')
            self.line(f'uint64_t* {self.barvars[b.name]} = reinterpret_cast<uint64_t*>(smem + {self.offsets["barrier:"+b.name]});')
        for i,p in enumerate(self.s.pipelines):
            self.line(f'// CAKE_OP: resource.pipeline.{p.name}')
            self.line(f'uint64_t* free{i} = reinterpret_cast<uint64_t*>(smem + {self.offsets["free:"+p.name]});')
        for a in self.s.allocations:
            self.line(f'// CAKE_OP: resource.allocation.{a.name}')
            if a.space is MemorySpace.SHARED:
                self.line(f'// shared allocation: offset={self.offsets[a.name]}, bytes={a.size_bytes}')
            else:
                name = self.tmemvars[a.name]
                self.line(f'uint32_t* {name} = reinterpret_cast<uint32_t*>(smem + {self.offsets["tmem:"+a.name]});')
                self.begin(f'if (warp == {self.roles[a.allocating_role].warps[0]})')
                self.line(f'asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(cake_smem({name})), "n"({a.tensor_columns}) : "memory");')
                self.end()
        self.line('__syncthreads();')
        for b in self.s.buffers:
            if b.space is MemorySpace.REGISTER:
                self.line(f'{_TYPES[b.dtype]} {self.names[b.name]}[{_slots(self.s,b)}];')
        for i, op in enumerate(self.s.operations):
            if op.kind is OperationKind.REDUCE_ARGMIN:
                self.line(f'float best{i};')
        self.sequence(None)
        self.line('__syncthreads();')
        self.invalidate_completions(None)
        for a in self.s.allocations:
            if a.space is MemorySpace.TENSOR:
                self.begin(f'if (warp == {self.roles[a.allocating_role].warps[0]})')
                self.line(f'asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(*{self.tmemvars[a.name]}), "n"({a.tensor_columns}) : "memory");')
                self.end()
        for owner in sorted({a.allocating_role for a in self.s.allocations if a.space is MemorySpace.TENSOR}):
            self.begin(f'if (warp == {self.roles[owner].warps[0]})')
            self.line('asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");')
            self.end()
        self.end(); self.line('// CAKE_KERNEL_END')
        self.host()
        dims = [1,1,1]
        if self.s.program_map:
            for axis in self.s.program_map.axes:
                dims[axis.axis] = axis.tile_count(self.b(axis.buffer).shape[axis.dimension])
        flags = ['-std=c++17',f'--gpu-architecture={self.s.target.replace("sm_","compute_")}',f'--gpu-code={self.s.target}','-O3','--fmad=false','-lineinfo','-Xptxas=-v']
        metadata = {'kernel_entry_point':self.entry+'_kernel', 'threads_per_cta':self.s.total_warp_extent*32,
                    'dynamic_shared_bytes':self.shared_bytes, 'grid':dims, 'nvcc_flags':flags,
                    'argument_order':[b.name for b in self.globals],
                    'arguments':[{'name':b.name,'dtype':b.dtype.value,'shape':list(b.shape),'mode':b.mode.value} for b in self.globals],
                    'host_abi':{'create':self.entry+'_create','launch':self.entry+'_launch','destroy':self.entry+'_destroy'},
                    'link_libraries':['cuda','cudart'], 'target_device_names':list(self.target.device_names),
                    'register_mapping':'one thread per row; replicated vector operands',
                    'source_language':'cuda_cpp', 'signature':{b.name:'*'+b.dtype.value for b in self.globals},
                    'block':[self.s.total_warp_extent*32,1,1], 'minimum_cuda_version': '12.9' if self.s.target == 'sm_103a' else '12.8'}
        return Emission('\n'.join(self.lines)+'\n',self.entry,{'shared_bytes':self.shared_bytes},metadata)

    def sequence(self, scope):
        ops = list(self.s.operations) if scope is None else list(self.s.loop_operations(scope))
        children = [loop for loop in self.s.tile_loops if self.s.loop_parent().get(loop.name) == (scope.name if scope else None)]
        first = {self.s.loop_operations(loop)[0].op_id:loop for loop in children}
        covered = {op.op_id for loop in children for op in self.s.loop_operations(loop)}
        rootgroups = {next(op.op_id for op in ops if op.pipeline == p.name):p for p in self.rootpipes} if scope is None else {}
        covered.update(op.op_id for op in ops if any(op.pipeline == p.name for p in rootgroups.values()))
        for op in ops:
            if op.op_id in first:
                self.loop(first[op.op_id])
            elif op.op_id in rootgroups:
                self.pipeline(None,rootgroups[op.op_id])
            elif op.op_id not in covered:
                self.operation(op)

    def loop(self, loop):
        if loop.name in self.pipeloops:
            self.pipeline(loop,self.pipeloops[loop.name]); return
        # A carried argmin is initialized once per dynamic entry to its reduction
        # loop, including when another output-coordinate loop invokes it repeatedly.
        for op in self.s.operations:
            if op.kind is OperationKind.REDUCE_ARGMIN and op.parameters.across_loop and _scope(self.s, op) == loop:
                self.begin(f'if ({self.role_condition(op.role)})')
                self.line(f'best{self.s.operations.index(op)} = INFINITY; {self.names[op.writes[0]]}[0] = -1;')
                self.end()
        self.line('#pragma unroll 1')
        self.begin(f'for (int {self.loopvars[loop.iterator]}=0; {self.loopvars[loop.iterator]}<{self.trip(loop)}; ++{self.loopvars[loop.iterator]})')
        self.sequence(loop)
        # Ensure all TMEM readers have completed before the next output tile overwrites it.
        self.line('__syncthreads();')
        self.invalidate_completions(loop)
        self.end()

    def invalidate_completions(self, scope):
        # Only the immediate parent of a contraction owns its final-publication
        # barrier. Ancestors must not invalidate the same object a second time.
        parent = scope.name if scope is not None else None
        barriers = [name for op in self.s.operations if op.kind is OperationKind.MMA
                    and _publication_scope(self.s, op) == parent
                    for name in op.signals]
        if barriers:
            self.begin('if (threadIdx.x == 0)')
            for name in barriers:
                self.line(f'cake_inval({self.barvars[name]});')
            self.end()
            self.line('__syncthreads();')

    def pipeline(self, loop, p):
        ops = self.s.loop_operations(loop) if loop is not None else [op for op in self.s.operations if op.pipeline == p.name]
        trips = self.trip(loop) if loop is not None else 1
        loads = [op for op in ops if op.kind is OperationKind.LOAD]
        mmas = [op for op in ops if op.kind is OperationKind.MMA]
        ready = next(b for b in self.s.barriers if b.pipeline == p.name)
        readyvar = self.barvars[ready.name]
        freevar = f'free{self.s.pipelines.index(p)}'
        self.begin('if (threadIdx.x == 0)')
        self.begin(f'for (int stage=0; stage<{p.stages}; ++stage)')
        self.line(f'cake_init({readyvar}+stage, {ready.count}); cake_init({freevar}+stage, 1);')
        self.end()
        for mma in mmas:
            self.line(f'cake_init({self.barvars[mma.signals[0]]}, 1);')
        self.line('asm volatile("fence.proxy.async.shared::cta;" ::: "memory");')
        self.end(); self.line('__syncthreads();')
        var = self.loopvars[loop.iterator] if loop is not None else f'once{self.s.pipelines.index(p)}'
        for producer in (True,False):
            role = loads[0].role if producer else mmas[0].role
            self.begin(f'if ({self.role_condition(role)}'+(')' if producer else ' && (threadIdx.x & 31) == 0)'))
            self.line('#pragma unroll 1')
            self.begin(f'for (int {var}=0; {var}<{trips}; ++{var})')
            self.line(f'const int stage = {var} % {p.stages};')
            if producer:
                self.line(f'if ((threadIdx.x & 31) == 0 && {var} >= {p.stages}) cake_wait({freevar}+stage, ({var}/{p.stages}-1)&1);')
                self.line('__syncwarp();')
                for op in loads:
                    dst = self.b(op.writes[0]); src = self.b(op.reads[0])
                    coords = self.access(op,src,['0','0'])
                    self.line(f'// CAKE_OP: {op.op_id}')
                    if op.parameters.movement is LoadMovement.TMA:
                        self.begin('if ((threadIdx.x & 31) == 0)')
                        self.line(f'cake_expect({readyvar}+stage, {dst.elements*dst.dtype.itemsize});')
                        self.line(f'cake_tma{len(src.shape)}({self.pointer(dst,"stage")}, &{self.mapnames[op.op_id]}, '+', '.join(reversed(coords))+f', {readyvar}+stage);')
                        self.end()
                    else:
                        width,_ = _SWIZZLE[dst.swizzle]
                        self.begin(f'for (int e=int(threadIdx.x & 31); e<{dst.elements}; e+=32)')
                        address,mask=self.address(op,src,[f'e/{dst.shape[1]}',f'e%{dst.shape[1]}'])
                        self.line(f'const int byte = e * {dst.dtype.itemsize};')
                        self.line(f'const int swizzled = byte ^ (((byte >> 7) & {(width//16)-1}) << 4);')
                        self.line(f'*reinterpret_cast<{_TYPES[dst.dtype]}*>({self.pointer(dst,"stage")} + swizzled) = ({mask}) ? {address} : {_TYPES[dst.dtype]}(0);')
                        self.end()
                        self.line('asm volatile("fence.proxy.async.shared::cta;" ::: "memory");')
                        self.line('__syncwarp();')
                        self.line(f'if ((threadIdx.x & 31) == 0) cake_arrive({readyvar}+stage);')
            else:
                self.line(f'cake_wait({readyvar}+stage, ({var}/{p.stages})&1);')
                self.line('asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");')
                for op in mmas:
                    self.line(f'// CAKE_OP: {op.op_id}')
                    a,b = [self.b(n) for n in op.reads]; dst=self.b(op.writes[0])
                    m,n,k = op.parameters.tile_shape
                    dtype_bit = 1 if a.dtype is DType.BF16 else 0
                    desc = (1<<4) | (dtype_bit<<7) | (dtype_bit<<10) | ((n>>3)<<17) | ((m>>4)<<24)
                    atom_k = op.parameters.instruction.shape[2]
                    ranges = op.parameters.contribution_ranges
                    first_atom = ranges[0][0] // atom_k
                    for start,end in ranges:
                        self.line('#pragma unroll')
                        self.begin(f'for (int atom={start//atom_k}; atom<{end//atom_k}; ++atom)')
                        expr=[]
                        for buf in (a,b):
                            width,mode = _SWIZZLE[buf.swizzle]
                            expr.append(f'cake_desc(cake_smem({self.pointer(buf,"stage")}) + atom*{atom_k*buf.dtype.itemsize}, {width*8}, {mode})')
                        # A nonzero physical atom can be this result's first contribution.
                        self.line(f'cake_mma({self.taddr(dst)}, {expr[0]}, {expr[1]}, {desc}u, {var} != 0 || atom != {first_atom});')
                        self.end()
                self.line(f'cake_commit({freevar}+stage);')
            self.end()
            if not producer:
                # Readout is admitted only in the contraction parent scope. Publish
                # each final accumulator once, after all K iterations are issued;
                # there is no unobserved intermediate completion-barrier phase.
                for op in mmas:
                    self.line(f'cake_commit({self.barvars[op.signals[0]]});')
                # Drain each used ring slot before invalidating its barrier. A wait
                # on another slot does not prove this object's arrival has completed.
                for stage in range(min(p.stages, trips)):
                    final_iteration = stage + ((trips-1-stage)//p.stages)*p.stages
                    self.line(f'cake_wait({freevar}+{stage}, {(final_iteration//p.stages)&1});')
            self.end()
        self.line('__syncthreads();')
        # Ready/free slots are drained before reinitialization in a parent iteration.
        self.begin('if (threadIdx.x == 0)')
        self.begin(f'for (int stage=0; stage<{p.stages}; ++stage)')
        self.line(f'cake_inval({readyvar}+stage); cake_inval({freevar}+stage);')
        self.end(); self.end()

    def operation(self, op):
        self.line(f'// CAKE_OP: {op.op_id}')
        self.begin(f'if ({self.role_condition(op.role)})')
        dst=self.b(op.writes[0]); d=self.names[dst.name]
        src=self.b(op.reads[0]); a=self.names[src.name]
        row=self.row(op)
        if op.kind is OperationKind.LOAD and op.parameters.movement is LoadMovement.TMEM:
            self.line(f'cake_wait({self.barvars[op.waits[0]]}, 0);')
            self.line('asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");')
            rep=op.parameters.source_atom.repetition
            self.line('#pragma unroll')
            self.begin(f'for (int col=0; col<{src.shape[1]}; col+={rep})')
            operands=', '.join(f'%{i}' for i in range(rep))
            outputs=', '.join(f'"=f"({d}[col+{i}])' for i in range(rep))
            self.line(f'asm volatile("tcgen05.ld.sync.aligned.32x32b.x{rep}.b32 {{{operands}}}, [%{rep}];" : {outputs} : "r"({self.taddr(src)} + (({row}/32)*32 << 16) + col) : "memory");')
            self.line('asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");')
            self.end()
        elif op.kind is OperationKind.LOAD:
            self.line('#pragma unroll')
            self.begin(f'for (int col=0; col<{_slots(self.s,dst)}; ++col)')
            local=[row,'col'] if len(dst.shape)==2 else ['col']
            address,mask=self.address(op,src,local)
            self.line(f'{d}[col] = ({mask}) ? {address} : {_TYPES[dst.dtype]}(0);')
            self.end()
        elif op.kind is OperationKind.ELEMENTWISE:
            p=op.parameters
            self.line('#pragma unroll')
            self.begin(f'for (int col=0; col<{_slots(self.s,dst)}; ++col)')
            x=f'{a}[{"0" if src.is_scalar else "col"}]'
            y=f'{p.scalar!r}f' if p.scalar is not None else (f'{self.names[op.reads[1]]}[{"0" if self.b(op.reads[1]).is_scalar else "col"}]' if len(op.reads)>1 else x)
            expr={ElementwiseOp.ADD:f'__fadd_rn({x},{y})',ElementwiseOp.SUB:f'__fsub_rn({x},{y})',
                  ElementwiseOp.MUL:f'__fmul_rn({x},{y})',ElementwiseOp.DIV:f'__fdiv_rn({x},{y})',
                  ElementwiseOp.RELU:f'fmaxf({x},0.0f)',ElementwiseOp.SQUARE:f'__fmul_rn({x},{x})'}[p.op]
            self.line(f'{d}[col] = {expr};'); self.end()
        elif op.kind is OperationKind.CAST:
            convert = {DType.BF16:'__float2bfloat16_rn',DType.FP16:'__float2half_rn',DType.FP32:'float',DType.INT32:'int32_t'}[dst.dtype]
            self.line('#pragma unroll')
            self.begin(f'for (int col=0; col<{_slots(self.s,dst)}; ++col)')
            self.line(f'{d}[col] = {convert}({a}[col]);'); self.end()
        elif op.kind is OperationKind.REDUCE_ARGMIN:
            idx=self.s.operations.index(op); loop=_scope(self.s,op)
            offset=f'{self.loopvars[loop.iterator]}*{loop.tile}' if op.parameters.across_loop else '0'
            domain=self.s.argmin_domain(op)
            assert domain is not None  # Shared preflight proved the index owner and bound.
            extent=domain[1]
            if not op.parameters.across_loop:
                self.line(f'best{idx} = INFINITY; {d}[0] = -1;')
            self.line('#pragma unroll 1')
            self.begin(f'for (int col=0; col<{src.shape[1]}; ++col)')
            self.line(f'const int index = {offset}+col;')
            self.begin(f'if (index < {extent} && ({d}[0] < 0 || {a}[col] < best{idx} || ({a}[col] == best{idx} && index < {d}[0])))')
            self.line(f'best{idx} = {a}[col]; {d}[0] = index;')
            self.end();self.end()
        elif op.kind is OperationKind.STORE:
            self.line('#pragma unroll')
            self.begin(f'for (int col=0; col<{_slots(self.s,src)}; ++col)')
            local=[row,'col'] if len(src.shape)==2 else [row if _scalar_row(self.s,src) else 'col']
            address,mask=self.address(op,dst,local)
            self.line(f'if ({mask}) {address} = {a}[col];'); self.end()
        self.end()

    def host(self):
        handle=self.entry+'_handle'
        self.begin(f'struct {handle}')
        self.line('int device;')
        for b in self.globals:
            self.line(f'{_TYPES[b.dtype]}* {self.names[b.name]};')
        for op in self.loads:
            self.line(f'CUtensorMap {self.mapnames[op.op_id]};')
        self.end(); self.lines[-1]+=';'
        self.begin(f'extern "C" int {self.entry}_create(void** buffers, void** result)')
        self.line('if (!buffers || !result) return int(cudaErrorInvalidValue);')
        self.line('*result = nullptr;')
        self.line('int device; cudaDeviceProp prop; cudaError_t error = cudaGetDevice(&device);')
        self.line('if (error != cudaSuccess) return int(error);')
        self.line('error = cudaGetDeviceProperties(&prop, device); if (error != cudaSuccess) return int(error);')
        major,minor=_TARGETS[self.s.target]
        names=' && '.join(f'std::strcmp(prop.name, "{name}") != 0' for name in self.target.device_names)
        self.line(f'if (prop.major != {major} || prop.minor != {minor} || ({names})) return int(cudaErrorInvalidDevice);')
        self.line(f'auto* h = new(std::nothrow) {handle}; if (!h) return int(cudaErrorMemoryAllocation);')
        self.line('h->device = device;')
        for i,b in enumerate(self.globals):
            self.line(f'if (!buffers[{i}]) {{ delete h; return int(cudaErrorInvalidValue); }}')
            self.line(f'h->{self.names[b.name]} = static_cast<{_TYPES[b.dtype]}*>(buffers[{i}]);')
        for i,op in enumerate(self.loads):
            src=self.b(op.reads[0]);dst=self.b(op.writes[0]);w,_=_SWIZZLE[dst.swizzle]
            dt='CU_TENSOR_MAP_DATA_TYPE_BFLOAT16' if src.dtype is DType.BF16 else 'CU_TENSOR_MAP_DATA_TYPE_FLOAT16'
            self.begin('')
            rank=len(src.shape)
            self.line(f'cuuint64_t dims[{rank}] = {{'+', '.join(map(str,reversed(src.shape)))+'};')
            self.line(f'cuuint64_t strides[{rank-1}] = {{'+', '.join(str(math.prod(src.shape[-i:])*src.dtype.itemsize) for i in range(1,rank))+'};')
            self.line(f'cuuint32_t box[{rank}] = {{'+', '.join(map(str,list(reversed(dst.shape))+[1]*(rank-2)))+f'}}, steps[{rank}] = {{'+','.join(['1']*rank)+'};')
            self.line(f'CUresult status = cuTensorMapEncodeTiled(&h->{self.mapnames[op.op_id]}, {dt}, {rank}, h->{self.names[src.name]}, dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_{w}B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);')
            self.line('if (status != CUDA_SUCCESS) { delete h; return 10000 + int(status); }')
            self.end()
        self.line(f'error = cudaFuncSetAttribute({self.entry}_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, {self.shared_bytes});')
        self.line('if (error != cudaSuccess) { delete h; return int(error); }')
        self.line('*result = h; return 0;'); self.end()
        self.begin(f'extern "C" int {self.entry}_launch(void* handle, void* stream)')
        self.line('if (!handle) return int(cudaErrorInvalidValue);')
        self.line(f'auto* h = static_cast<{handle}*>(handle);')
        self.line('int device; cudaError_t error = cudaGetDevice(&device); if (error != cudaSuccess) return int(error);')
        self.line('if (device != h->device) return int(cudaErrorInvalidDevice);')
        dims=[1,1,1]
        if self.s.program_map:
            for axis in self.s.program_map.axes:
                dims[axis.axis]=axis.tile_count(self.b(axis.buffer).shape[axis.dimension])
        args=[f'h->{self.names[b.name]}' for b in self.globals]+[f'h->{self.mapnames[op.op_id]}' for op in self.loads]
        self.line(f'{self.entry}_kernel<<<dim3({dims[0]},{dims[1]},{dims[2]}), {self.s.total_warp_extent*32}, {self.shared_bytes}, reinterpret_cast<cudaStream_t>(stream)>>>('+', '.join(args)+');')
        self.line('return int(cudaGetLastError());');self.end()
        self.begin(f'extern "C" int {self.entry}_destroy(void* handle)')
        self.line(f'delete static_cast<{handle}*>(handle); return 0;');self.end()


# Low-level instruction wrappers only: no loop, resource choice, role or workload math.
_INSTRUCTIONS = r'''
__device__ __forceinline__ uint32_t cake_smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cake_init(uint64_t* p, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(cake_smem(p)), "r"(count) : "memory");
}
__device__ __forceinline__ void cake_inval(uint64_t* p) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(cake_smem(p)) : "memory");
}
__device__ __forceinline__ void cake_wait(uint64_t* p, uint32_t phase) {
  asm volatile("{ .reg .pred done; WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [%0], %1; @!done bra WAIT; }" :: "r"(cake_smem(p)), "r"(phase) : "memory");
}
__device__ __forceinline__ void cake_expect(uint64_t* p, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(cake_smem(p)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cake_tma2(void* dst, const CUtensorMap* map, int x, int y, uint64_t* barrier) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" :: "r"(cake_smem(dst)), "l"(map), "r"(x), "r"(y), "r"(cake_smem(barrier)) : "memory");
}
__device__ __forceinline__ void cake_arrive(uint64_t* p) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(cake_smem(p)) : "memory");
}
__device__ __forceinline__ void cake_tma3(void* dst, const CUtensorMap* map, int x, int y, int z, uint64_t* barrier) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" :: "r"(cake_smem(dst)), "l"(map), "r"(x), "r"(y), "r"(z), "r"(cake_smem(barrier)) : "memory");
}
__device__ __forceinline__ uint64_t cake_desc(uint32_t address, uint32_t stride, uint32_t swizzle) {
  return uint64_t(address >> 4) | (1ull << 16) | (uint64_t(stride >> 4) << 32) | (1ull << 46) | (uint64_t(swizzle) << 61);
}
__device__ __forceinline__ void cake_mma(uint32_t dst, uint64_t a, uint64_t b, uint32_t desc, bool accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }" :: "r"(dst), "l"(a), "l"(b), "r"(desc), "r"(int(accumulate)) : "memory");
}
__device__ __forceinline__ void cake_commit(uint64_t* p) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(cake_smem(p)) : "memory");
}
'''


def emit(schedule: Schedule, target: Target, *, entry_point: str | None = None) -> Emission:
    for check in (verify,preflight):
        failures = [finding for finding in check(schedule,target)
                    if finding.blocks_lowering or finding.blocks_acceptance]
        if failures:
            raise EmitError('; '.join(f'{f.code} at {f.path}: {f.message}' for f in failures))
    if entry_point is not None and entry_point != schedule.lowering.entry_point:
        raise EmitError('Entry point differs from the canonical route.')
    return _Emitter(schedule,target,schedule.lowering.entry_point).emit()
