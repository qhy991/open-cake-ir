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
    OperationKind, ReduceOp, ReductionScope, Schedule, Swizzle,
)
from ..target import CodeObject, Target
from ..verifier import verify
from ..verifier.carried_tmem import analyze_carried_tmem

SUPPORTED_DTYPES = frozenset({DType.BF16, DType.FP16, DType.FP32, DType.INT32})
CODE_OBJECTS = frozenset({CodeObject.CUBIN})
SUPPORTED_OPERATION_KINDS = frozenset({OperationKind.LOAD, OperationKind.MMA,
    OperationKind.ELEMENTWISE, OperationKind.CAST, OperationKind.REDUCE,
    OperationKind.REDUCE_ARGMIN,
    OperationKind.STORE, OperationKind.TMEM_STORE,
    OperationKind.FORWARD_SUBSTITUTE})
_TYPES = {DType.BF16: '__nv_bfloat16', DType.FP16: '__half',
          DType.FP32: 'float', DType.INT32: 'int32_t'}
_SWIZZLE = {Swizzle.B32: (32, 6), Swizzle.B64: (64, 4), Swizzle.B128: (128, 2)}
_CONTRACT = 'tcgen05.mma.cta_group::1.kind::f16'
# Exact B300 device proof: BF16 TMEM-A M128 x MN-major B N32 x K128,
# swizzle-64B shared B, in F-2026-09-24-003. Other geometries remain refused.
_TMEM_A_MN_B_EVIDENCE = frozenset({'sm_103a'})


def _scope(s, op):
    return next((loop for loop in s.tile_loops if op.op_id in loop.body), None)


def _pipeline_loop(s, pipeline):
    scopes = {_scope(s, op).name if _scope(s, op) else None
              for op in s.operations if op.pipeline == pipeline.name}
    return s.tile_loop(next(iter(scopes))) if len(scopes) == 1 and None not in scopes else None


def _publication_scope(s, operation):
    """The final accumulator is consumed in its contraction's parent scope."""
    loop = _scope(s, operation)
    if loop is not None and loop.carried_buffers:
        # Each carried-state chunk consumes its result before overwriting the
        # accumulator in the next trip, so completion belongs to that trip.
        return loop.name
    return s.loop_parent().get(loop.name) if loop is not None else None


def _scalar_row(s, buffer):
    """Prove a rank-one tile has one scalar per physical row, not 128 replicas."""
    def walk(name, seen):
        value = s.buffer(name)
        if value is None or value.shape != (128,) or name in seen:
            return False
        writers = [op for op in s.operations if name in op.writes]
        if len(writers) != 1:
            return False
        writer = writers[0]
        if writer.kind is OperationKind.REDUCE_ARGMIN:
            return True
        if writer.kind is OperationKind.REDUCE:
            return writer.parameters.axis == 1 and not writer.parameters.across_loop
        if writer.kind not in (OperationKind.CAST, OperationKind.ELEMENTWISE):
            return False
        return bool(writer.reads) and all(
            (source := s.buffer(source_name)) is not None
            and (source.shape == (1,) or walk(source_name, seen | {name}))
            for source_name in writer.reads
        )
    return walk(buffer.name, set())


def _fused_decayed_state(s):
    """Map an exclusive typed BF16-state epilogue to one register pass.

    The Schedule keeps all four operations and their FP32 rounding boundaries.
    This is only a native row mapping; a shared IR rewrite is not inferred from
    one KDA witness.
    """
    writer = {name: op for op in s.operations for name in op.writes}
    readers = {b.name: [op for op in s.operations if b.name in op.reads]
               for b in s.buffers}
    dependents = {op.op_id: [other for other in s.operations
                             if op.op_id in other.depends_on]
                  for op in s.operations}
    result = {}
    for rounded in s.operations:
        if (rounded.kind is not OperationKind.CAST
                or rounded.parameters.to is not DType.BF16
                or len(rounded.reads) != 1 or len(rounded.writes) != 1):
            continue
        combined = writer.get(rounded.reads[0])
        if (combined is None or combined.kind is not OperationKind.ELEMENTWISE
                or combined.parameters.op is not ElementwiseOp.ADD
                or combined.parameters.scalar is not None
                or combined.parameters.broadcast_axis is not None
                or len(combined.reads) != 2):
            continue
        scaled = writer.get(combined.reads[0])
        if (scaled is None or scaled.kind is not OperationKind.ELEMENTWISE
                or scaled.parameters.op is not ElementwiseOp.MUL
                or scaled.parameters.scalar is not None
                or scaled.parameters.broadcast_axis != 1
                or len(scaled.reads) != 2):
            continue
        converted = writer.get(scaled.reads[0])
        if (converted is None or converted.kind is not OperationKind.CAST
                or converted.parameters.to is not DType.FP32
                or len(converted.reads) != 1):
            continue
        chain = (converted, scaled, combined, rounded)
        scope = _scope(s, rounded)
        if (scope is None or not scope.carried_buffers
                or any(_scope(s, op) != scope or op.role != rounded.role
                       or op.waits or op.signals or op.pipeline for op in chain)):
            continue
        positions = [scope.body.index(op.op_id) for op in chain]
        if positions != list(range(positions[0], positions[0] + 4)):
            continue
        if any(readers[op.writes[0]] != [next_op]
               or dependents[op.op_id] != [next_op]
               for op, next_op in zip(chain, chain[1:])):
            continue
        state = s.buffer(converted.reads[0])
        state_fp = s.buffer(converted.writes[0])
        prefix = s.buffer(scaled.reads[1])
        scaled_value = s.buffer(scaled.writes[0])
        correction = s.buffer(combined.reads[1])
        sum_value = s.buffer(combined.writes[0])
        destination = s.buffer(rounded.writes[0])
        if (any(b is None or b.space is not MemorySpace.REGISTER for b in
                (state, state_fp, prefix, scaled_value, correction, sum_value, destination))
                or state.dtype is not DType.BF16 or destination.dtype is not DType.BF16
                or prefix.dtype is not DType.FP32
                or any(b.dtype is not DType.FP32 for b in
                       (state_fp, scaled_value, correction, sum_value))
                or any(b.shape != (128, 128) for b in
                       (state, state_fp, scaled_value, correction, sum_value, destination))
                or prefix.shape != (128,)):
            continue
        for op in chain:
            result[op.op_id] = (chain, state.name, prefix.name,
                                correction.name, destination.name)
    return result


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
    # The code object decides which targets this backend serves, held against
    # CODE_OBJECTS by the Compiler before preflight; the emitted `__CUDA_ARCH__` guard
    # and device check read the capability the Target declares.
    check(target.target_id == s.target and target.compute_capability is not None,
          'NATIVE_TARGET_UNSUPPORTED', 'target',
          'native CUDA requires the Schedule\'s own Target and its declared compute capability')
    check(s.program_map is None or not s.program_map.persistent,
          'NATIVE_PERSISTENCE_UNSUPPORTED', 'program_map', 'native CUDA does not implement persistent traversal')
    check(s.residency is None, 'NATIVE_RESIDENCY_UNSUPPORTED', 'residency',
          'native CUDA does not yet enforce register caps or requested CTA residency')
    check(s.grid in (None, (1, 1, 1)), 'NATIVE_GRID_UNSUPPORTED', 'grid',
          'multi-CTA ownership must be expressed by ProgramMap')
    check(bool(s.pipelines) or any(op.kind is OperationKind.FORWARD_SUBSTITUTE
                                   for op in s.operations),
          'NATIVE_PIPELINE_REQUIRED', 'pipelines',
          'TMA/MMA lowering requires a pipeline; the root row solve has its own P stage')
    for field in ('allocations','barriers','pipelines'):
        for i,item in enumerate(getattr(s,field)):
            check(not any(c in item.name for c in '\\\r\n'), 'NATIVE_NAME_UNSUPPORTED',
                  f'{field}[{i}].name', 'source-map resource names cannot contain line breaks or backslashes')
    roles = {r.name: r for r in s.roles}
    buffers = {b.name: b for b in s.buffers}
    writers = {name: op for op in s.operations for name in op.writes}
    carried_pairs, _ = analyze_carried_tmem(s)
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
        check(b.valid_extent is None and b.scale_of is None,
              'NATIVE_BUFFER_REFINEMENT_UNSUPPORTED', path,
              'native CUDA does not implement valid-extent or block-scale relations')
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
            check(len(b.shape) == 2 and b.shape[0] == 128
                  and b.dtype in (DType.BF16, DType.FP32)
                  and b.byte_offset % 512 == 0 and b.stages == 1,
                  'NATIVE_TMEM_LAYOUT', path,
                  'native FP32 accumulators or BF16 state use 128 rows and whole TMEM columns')
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
                                        AccessIndexKind.LOOP, AccessIndexKind.LOOP_TILE),
                  'NATIVE_ACCESS_UNSUPPORTED', f'access_maps[{i}].indices[{j}]',
                  'native accesses admit contiguous dimension, program and loop coordinates')
    if s.program_map:
        for i, axis in enumerate(s.program_map.axes):
            check(axis.tile >= 1, 'NATIVE_PROGRAM_AXIS', f'program_map.axes[{i}]',
                  'native program coordinates use positive tiles')
    pipe_loops = {}
    owned_barriers = set()
    for loop_index, loop in enumerate(s.tile_loops):
        if not loop.carried_buffers:
            continue
        owners = [p for p in s.pipelines if _pipeline_loop(s, p) == loop]
        check(len(owners) in (1, 2), 'NATIVE_CARRIED_PIPELINE_COUNT',
              f'tile_loops[{loop_index}].body',
              'a carried-state loop has one contraction or two ordered contractions')
        for owner in owners:
            check(owner.stages == 1, 'NATIVE_CARRIED_PIPELINE_STAGES',
                  f'pipelines[{s.pipelines.index(owner)}].stages',
                  'each carried contraction completes one stage per chunk')
        if len(owners) == 2:
            owners.sort(key=lambda p: next(
                (loop.body.index(op.op_id) for op in s.operations
                 if op.pipeline == p.name and op.op_id in loop.body),
                len(loop.body),
            ))
            first, second = owners
            first_ops = [op for op in s.operations if op.pipeline == first.name]
            second_ops = [op for op in s.operations if op.pipeline == second.name]
            first_mma = [op for op in first_ops if op.kind is OperationKind.MMA]
            second_mma = [op for op in second_ops if op.kind is OperationKind.MMA]
            solves = [op for op in s.loop_operations(loop)
                      if op.kind is OperationKind.FORWARD_SUBSTITUTE]
            second_state = (second_mma[0].reads[0]
                            if len(second_mma) == 1 and second_mma[0].reads else None)
            update = writers.get(second_state) if second_state else None
            first_end = max((loop.body.index(op.op_id) for op in first_ops
                             if op.op_id in loop.body), default=-1)
            second_start = min((loop.body.index(op.op_id) for op in second_ops
                                if op.op_id in loop.body), default=-1)
            solve_at = loop.body.index(solves[0].op_id) if len(solves) == 1 else -1
            update_at = (loop.body.index(update.op_id) if update is not None
                         and update.op_id in loop.body else -1)
            check(len(first_mma) == len(second_mma) == len(solves) == 1
                  and bool(first_mma[0].reads)
                  and first_mma[0].reads[0] in loop.carried_buffers
                  and second_state not in loop.carried_buffers
                  and update is not None and update.kind is OperationKind.TMEM_STORE
                  and first_end < solve_at < update_at < second_start,
                  'NATIVE_TWO_PHASE_ORDER', f'tile_loops[{loop_index}].body',
                  'the carried-state MMA precedes the row solve and U publication, '
                  'which precede the correction MMA')
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
        carried = loop is not None and bool(loop.carried_buffers)
        body = tagged if carried else (s.loop_operations(loop) if loop is not None else tagged)
        body_path = f'tile_loops[{s.tile_loops.index(loop)}].body' if loop is not None else f'pipelines[{i}]'
        if loop is None:
            positions = [s.operations.index(op) for op in tagged]
            check(positions == list(range(positions[0],positions[-1]+1)),
                  'NATIVE_PIPELINE_SCOPE', body_path, 'a root pipeline group must be contiguous in the operation DAG')
            check(pipeline.stages == 1, 'NATIVE_PIPELINE_STAGES', body_path,
                  'one contraction tile without a K loop uses one explicitly declared stage')
        loads = [op for op in body if op.kind is OperationKind.LOAD and op.parameters.movement in (LoadMovement.TMA, LoadMovement.GLOBAL)]
        mmas = [op for op in body if op.kind is OperationKind.MMA]
        body_ids = [op.op_id for op in body]
        body_matches = loop is None or (
            bool(body_ids) and body_ids[0] in loop.body
            and list(loop.body[loop.body.index(body_ids[0]):
                               loop.body.index(body_ids[0]) + len(body)]) == body_ids
            if carried else list(loop.body) == body_ids
        )
        check(bool(loads) and bool(mmas) and len(loads)+len(mmas) == len(body)
              and all(op.pipeline == pipeline.name for op in body)
              and body_matches,
              'NATIVE_PIPELINE_BODY', body_path,
              'contraction stages contain TMA loads followed by MMA; a carried loop may follow with state update operations')
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
            def expected_waits(mma):
                tensor_stores = [writers.get(name) for name in mma.reads
                                 if buffers[name].space is MemorySpace.TENSOR]
                return {b.name} | {name for store in tensor_stores if store is not None
                                   for name in store.signals}
            check(b.mechanism is BarrierMechanism.MBARRIER and b.count == len(loads)
                  and all(op.signals == (b.name,) and not op.waits for op in loads)
                  and all(set(op.waits) == expected_waits(op) and
                          len(op.waits) == len(expected_waits(op)) for op in mmas),
                  'NATIVE_PIPELINE_BARRIER', f'barriers[{s.barriers.index(b)}]',
                  'ready arrival count and waits for staged and TMEM operands must match')
        staged = {name for op in loads for name in op.writes}
        consumed = {name for op in mmas for name in op.reads
                    if buffers[name].space is MemorySpace.SHARED}
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
            if any(buffers[name].space is MemorySpace.TENSOR for name in mma.reads):
                check(loop is None or carried, 'NATIVE_TMEM_MMA_SCOPE',
                      f'operations[{s.operations.index(mma)}].pipeline',
                      'TMEM-A MMA admits a root contraction or one declared carried chunk loop')
            if carried:
                tensor_a = (len(mma.reads) == 2
                            and buffers[mma.reads[0]].space is MemorySpace.TENSOR)
                staged_b = buffers[mma.reads[1]] if len(mma.reads) == 2 else None
                producers = [load for load in loads
                             if staged_b is not None and load.writes == (staged_b.name,)]
                producer = producers[0] if len(producers) == 1 else None
                source_b = (buffers[producer.reads[0]]
                            if producer is not None and len(producer.reads) == 1 else None)
                access_b = s.access_map(producer.op_id, source_b.name) if source_b else None
                loop_source = buffers.get(loop.buffer)
                rank2_domain = (source_b is not None and loop_source is not None
                                and staged_b is not None and access_b is not None
                                and len(source_b.shape) == len(loop_source.shape) == 2
                                and source_b.shape[0] == loop_source.shape[0]
                                and loop.dimension == 0 and staged_b.shape[0] == loop.tile
                                and len(access_b.indices) == 2
                                and access_b.indices[0].source is AccessIndexKind.LOOP_TILE
                                and access_b.indices[0].name == loop.iterator
                                and access_b.indices[1].source is AccessIndexKind.DIMENSION)
                rank3_domain = (source_b is not None and loop_source is not None
                                and staged_b is not None and access_b is not None
                                and len(source_b.shape) == len(loop_source.shape) == 3
                                and source_b.shape[0] == loop_source.shape[0]
                                and loop.dimension == 0 and loop.tile == 1
                                and staged_b.shape == source_b.shape[1:]
                                and len(access_b.indices) == 3
                                and access_b.indices[0].source is AccessIndexKind.LOOP
                                and access_b.indices[0].name == loop.iterator
                                and all(component.source is AccessIndexKind.DIMENSION
                                        and component.dimension == axis
                                        for axis, component in enumerate(access_b.indices[1:], 1)))
                check(tensor_a and producer is not None and (rank2_domain or rank3_domain),
                      'NATIVE_CARRIED_MMA_DOMAIN', f'operations[{s.operations.index(mma)}]',
                      'each carried MMA consumes one staged B tile selected by the outer chunk loop')
            check(loop is None or carried or (s.mma_accumulates_over(mma, loop)
                  and all(s._staged_axis_filled_by(name, loop) == 1 for name in mma.reads
                          if buffers[name].space is MemorySpace.SHARED)),
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
        check(len(role.execution_groups) == (1 if asynchronous else 4), 'NATIVE_ROLE_WIDTH', path+'.role',
              'TMA/MMA roles use one warp; row-owned arithmetic/transfer/store roles use four')
        if not asynchronous:
            check(role.execution_groups[0] % 4 == 0, 'NATIVE_ROLE_ALIGNMENT', path+'.role',
                  'TMEM row ownership requires an aligned group of four physical warps')
            check(op.pipeline is None and
                  (not op.signals or op.kind is OperationKind.TMEM_STORE),
                  'NATIVE_OPERATION_SYNC', path,
                  'register operations do not produce asynchronous stage signals')
            if not ((op.kind is OperationKind.LOAD and op.parameters.movement is LoadMovement.TMEM)
                    or op.kind is OperationKind.FORWARD_SUBSTITUTE):
                check(not op.waits, 'NATIVE_OPERATION_SYNC', path+'.waits', 'this operation has no asynchronous wait protocol')
        for name in op.reads:
            b = buffers[name]
            if b.space is MemorySpace.REGISTER and name in writers:
                check(writers[name].role == op.role, 'NATIVE_REGISTER_ROLE_OWNERSHIP', path+'.reads',
                      'a register value must stay within its producer role')
        if op.kind is OperationKind.MMA:
            p = op.parameters; instruction = p.instruction
            tensor_a = (len(op.reads) == 2
                        and buffers[op.reads[0]].space is MemorySpace.TENSOR)
            major = instruction.operand_major if instruction is not None else None
            mn_b = major == (OperandMajorMode.K, OperandMajorMode.MN)
            placement = (
                (tensor_a and instruction is not None
                 and instruction.operand_source is OperandSource.TENSOR
                 and buffers[op.reads[1]].space is MemorySpace.SHARED)
                or (not tensor_a and instruction is not None
                    and instruction.operand_source is OperandSource.SHARED
                    and all(buffers[n].space is MemorySpace.SHARED for n in op.reads))
            )
            good = (instruction is not None and instruction.contract == _CONTRACT
                    and instruction.cta_group == 1 and placement
                    and (major == (OperandMajorMode.K, OperandMajorMode.K)
                         or tensor_a and mn_b)
                    and p.tile_shape is not None and instruction.shape == (128,p.tile_shape[1],16)
                    and p.tile_shape[0] == 128 and 8 <= p.tile_shape[1] <= 256 and p.tile_shape[1] % 8 == 0
                    and len(op.reads) == 2
                    and all(buffers[n].dtype in (DType.BF16,DType.FP16) for n in op.reads)
                    and buffers[op.reads[0]].dtype == buffers[op.reads[1]].dtype
                    and dst.space is MemorySpace.TENSOR)
            check(good, 'NATIVE_MMA_CONTRACT', path+'.parameters.instruction',
                  'native MMA requires explicit M128/N8..256/K16 f16-family and either shared/shared or TMEM-A/shared-B operands')
            if mn_b:
                b_operand = buffers[op.reads[1]] if len(op.reads) == 2 else None
                check(tensor_a and target.target_id in _TMEM_A_MN_B_EVIDENCE
                      and p.tile_shape == (128, 32, 128)
                      and b_operand is not None and b_operand.dtype is DType.BF16
                      and b_operand.swizzle is Swizzle.B64,
                      'NATIVE_MN_MAJOR_B_UNQUALIFIED', path+'.parameters.instruction.operand_major',
                      'MN-major shared B is qualified only for BF16 TMEM-A M128/N32/K128 on sm_103a with 64-byte swizzle')
            if p.k_ranges is not None:
                check(good and all(endpoint % instruction.shape[2] == 0
                                   for interval in p.k_ranges for endpoint in interval),
                      'NATIVE_MMA_K_RANGES', path+'.parameters.k_ranges',
                      'native K contributions require the f16-family contract and endpoints aligned to its declared instruction K')
            if p.tile_shape is not None:
                b_shape = ((p.tile_shape[2], p.tile_shape[1]) if mn_b
                           else (p.tile_shape[1], p.tile_shape[2]))
                check(len(op.reads) == 2
                      and buffers[op.reads[0]].shape == (p.tile_shape[0], p.tile_shape[2])
                      and buffers[op.reads[1]].shape == b_shape
                      and dst.shape == p.tile_shape[:2],
                      'NATIVE_MMA_TILE_DOMAIN', path+'.parameters.tile_shape',
                      'the full tile domains must match A[M,K], B[N,K] for K-major or B[K,N] for MN-major, and result[M,N]')
            check(len(op.reads) == 2 and all(
                name in writers and writers[name].kind is (
                    OperationKind.TMEM_STORE if buffers[name].space is MemorySpace.TENSOR
                    else OperationKind.LOAD) for name in op.reads),
                  'NATIVE_MMA_OPERAND_WRITER', path+'.reads',
                  'MMA operands must be explicitly staged by loads or a TMEM store')
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
                solve_readers = [reader for reader in s.operations
                                 if reader.kind is OperationKind.FORWARD_SUBSTITUTE
                                 and dst.name in reader.reads]
                if op.pipeline is None and len(solve_readers) == 1:
                    barrier = next((b for b in s.barriers if b.name in op.signals), None)
                    solve_scope = _scope(s, solve_readers[0])
                    check((scope is None or scope.carried_buffers)
                          and solve_scope == scope
                          and p.movement is LoadMovement.GLOBAL
                          and len(op.signals) == 1 and barrier is not None
                          and barrier.count == 1 and barrier.pipeline is None
                          and barrier.mechanism is BarrierMechanism.MBARRIER
                          and solve_readers[0].waits == op.signals,
                          'NATIVE_SOLVE_P_STAGE', path,
                          'one root global-to-shared P stage publishes one mbarrier to its row solve')
                    if barrier is not None:
                        owned_barriers.add(barrier.name)
                else:
                    check(op.pipeline is not None and (scope is None or scope.name in pipe_loops),
                          'NATIVE_SHARED_LOAD_SCOPE', path+'.pipeline',
                          'other native global-to-shared staging belongs to a contraction pipeline')
            if p.movement is LoadMovement.TMA:
                check(src.space is MemorySpace.GLOBAL and src.mode is BufferMode.INPUT
                      and len(src.shape) in (2,3) and dst.space is MemorySpace.SHARED
                      and p.descriptor_box == dst.shape and src.dtype == dst.dtype
                      and src.shape[-1]*src.dtype.itemsize % 16 == 0 and op.pipeline is not None,
                      'NATIVE_TMA_DESCRIPTOR', path+'.parameters',
                      'TMA requires a contiguous rank-2 input with 16-byte row stride and a declared shared box')
            elif p.movement is LoadMovement.TMEM:
                producer = writers.get(src.name)
                if src.dtype is DType.BF16:
                    pair = carried_pairs.get(src.name)
                    scope = _scope(s, op)
                    check(pair is not None and scope is not None
                          and scope.name == pair.loop
                          and producer is not None and producer.op_id == pair.updater
                          and op.waits == (pair.barrier,)
                          and p.source_atom is not None
                          and len(src.shape) == 2 and src.shape[0] == 128
                          and src.shape[1] % (2 * p.source_atom.repetition) == 0,
                          'NATIVE_TMEM_BF16_STATE', path,
                          'BF16 TMEM reads require one proven carried state phase and whole packed 32-bit copy atoms')
                else:
                    check(producer is not None and producer.kind is OperationKind.MMA and op.waits == producer.signals
                          and len(src.shape) == 2 and src.shape[0] == 128
                          and p.source_atom is not None and src.shape[1] % p.source_atom.repetition == 0,
                          'NATIVE_TMEM_COMPLETION', path,
                          'FP32 TMEM load must wait its MMA completion and read full atom repetitions')
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
                    check(sum(c.is_vector for c in access.indices) == len(dst.shape),
                          'NATIVE_ACCESS_RANK', path, 'global access vector axes must match the local tile rank')
                    if p.movement is LoadMovement.TMA:
                        check(all(c.source in {AccessIndexKind.PROGRAM, AccessIndexKind.LOOP}
                                  for c in access.indices[:-2])
                              and all(c.is_vector for c in access.indices[-2:]),
                              'NATIVE_TMA_COORDINATES', path, 'TMA scalar program axes precede the two tiled matrix axes')

        elif op.kind is OperationKind.FORWARD_SUBSTITUTE:
            coefficient = buffers[op.reads[0]] if len(op.reads) == 2 else None
            rhs = buffers[op.reads[1]] if len(op.reads) == 2 else None
            producer = writers.get(coefficient.name) if coefficient is not None else None
            check(coefficient is not None and rhs is not None
                  and coefficient.space is MemorySpace.SHARED
                  and coefficient.swizzle is Swizzle.B64
                  and coefficient.dtype is DType.BF16
                  and coefficient.shape == (32, 32)
                  and rhs.space is MemorySpace.REGISTER
                  and rhs.dtype is DType.FP32 and rhs.shape == (128, 32)
                  and dst.space is MemorySpace.REGISTER
                  and dst.dtype is DType.FP32 and dst.shape == (128, 32),
                  'NATIVE_FORWARD_SOLVE_DOMAIN', path,
                  'native row solve requires shared BF16 P[32,32] and register FP32 RHS/U[128,32]')
            check(producer is not None and producer.kind is OperationKind.LOAD
                  and producer.parameters.movement is LoadMovement.GLOBAL
                  and producer.writes == (coefficient.name,)
                  and len(op.waits) == 1 and op.waits == producer.signals
                  and op.pipeline is None
                  and _scope(s, op) == _scope(s, producer),
                  'NATIVE_FORWARD_SOLVE_OWNER', path,
                  'one root P stage owns the solve input and its completion barrier')

        elif op.kind is OperationKind.TMEM_STORE:
            src = buffers[op.reads[0]]
            atom = op.parameters.destination_atom
            scope = _scope(s, op)
            transient = (scope is not None and dst.name not in scope.carried_buffers
                         and all(_scope(s, reader) == scope
                                 and reader.kind is OperationKind.MMA
                                 for reader in s.operations if dst.name in reader.reads)
                         and any(dst.name in reader.reads for reader in s.operations))
            check(scope is None or dst.name in scope.carried_buffers or transient,
                  'NATIVE_TMEM_STORE_SCOPE', path,
                  'an in-loop TMEM store updates carried state or feeds a same-loop MMA')
            check(src.space is MemorySpace.REGISTER and dst.space is MemorySpace.TENSOR
                  and src.dtype is dst.dtype is DType.BF16
                  and src.shape == dst.shape and len(src.shape) == 2
                  and src.shape[0] == 128 and src.shape[1] % 16 == 0
                  and atom.op == 'tcgen05.St32x32b' and atom.repetition == 8,
                  'NATIVE_TMEM_STORE_CONTRACT', path,
                  'native TMEM store packs pairs of BF16 values in 128 rows using x8 atoms')
            barrier = next((b for b in s.barriers if b.name in op.signals), None)
            check(len(op.signals) == 1 and barrier is not None
                  and barrier.mechanism is BarrierMechanism.MBARRIER
                  and barrier.count == 4 and barrier.pipeline is None,
                  'NATIVE_TMEM_STORE_COMPLETION', path+'.signals',
                  'four writing warps must publish through one count-four mbarrier')
            if barrier is not None:
                owned_barriers.add(barrier.name)

        elif op.kind is OperationKind.ELEMENTWISE:
            check(op.parameters.op in (ElementwiseOp.ADD, ElementwiseOp.SUB, ElementwiseOp.MUL,
                                        ElementwiseOp.DIV, ElementwiseOp.RELU, ElementwiseOp.SQUARE,
                                        ElementwiseOp.EXP, ElementwiseOp.RSQRT,
                                        ElementwiseOp.RECIPROCAL),
                  'NATIVE_ARITHMETIC_UNSUPPORTED', path+'.parameters.op',
                  'native arithmetic admits FP32 add/sub/mul/div/relu/square/exp/rsqrt/reciprocal')
            check(dst.dtype is DType.FP32 and all(buffers[n].dtype is DType.FP32 for n in op.reads),
                  'NATIVE_ARITHMETIC_DTYPE', path, 'native elementwise operations use FP32 values')
            check(dst.space is MemorySpace.REGISTER
                  and all(buffers[n].space is MemorySpace.REGISTER for n in op.reads),
                  'NATIVE_ARITHMETIC_STORAGE', path,
                  'native elementwise operations read and write row-owned registers; load global inputs explicitly')
            check(op.parameters.broadcast_axis in (None,0,1), 'NATIVE_BROADCAST_UNSUPPORTED', path,
                  'native row ownership admits explicit row or column broadcasts')
            if op.parameters.broadcast_axis == 0:
                narrow = [buffers[name] for name in op.reads
                          if buffers[name].shape == (dst.shape[0],)]
                check(len(dst.shape) == 2 and dst.shape[0] == 128
                      and bool(narrow) and all(_scalar_row(s,value) for value in narrow),
                      'NATIVE_ROW_BROADCAST', path+'.parameters.broadcast_axis',
                      'row broadcast requires a proven one-scalar-per-row producer')
            check(op.parameters.scalar is None or math.isfinite(op.parameters.scalar)
                  and abs(op.parameters.scalar) <= 3.4028234663852886e38,
                  'NATIVE_SCALAR_FINITE', path+'.parameters.scalar', 'native scalar literals must be finite FP32 values')
        elif op.kind is OperationKind.REDUCE:
            src = buffers[op.reads[0]]
            p = op.parameters
            check(p.op is ReduceOp.SUM and p.axis == 1
                  and p.scope is ReductionScope.CTA and not p.across_loop
                  and src.space is MemorySpace.REGISTER and src.dtype is DType.FP32
                  and len(src.shape) == 2 and src.shape[0] == 128
                  and dst.space is MemorySpace.REGISTER and dst.dtype is DType.FP32
                  and dst.shape == (128,),
                  'NATIVE_ROW_REDUCE', path,
                  'native sum folds each FP32 register row to one FP32 row-owned scalar')
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
        self.state_fusions = _fused_decayed_state(s)

    def line(self, text=''):
        self.lines.append('  '*self.indent + text)
    def begin(self, text):
        self.line(text+' {'); self.indent += 1
    def end(self):
        self.indent -= 1; self.line('}')
    def role_condition(self, name):
        r = self.roles[name]
        return f'warp >= {r.execution_groups[0]} && warp <= {r.execution_groups[-1]}'
    def row(self, op):
        return f'(int(threadIdx.x) - {self.roles[op.role].execution_groups[0]*self.target.warp_size})'
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
            if component.source is AccessIndexKind.LOOP:
                coords.append(self.loopvars[component.name])
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
        major, minor = self.target.compute_capability
        arch = major*100 + minor*10
        self.line(f'#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != {arch}\n#error "Schedule requires exact {self.s.target}"\n#endif')
        self.line(_INSTRUCTIONS)
        if any(op.kind is OperationKind.MMA and
               any(self.b(name).space is MemorySpace.TENSOR for name in op.reads)
               for op in self.s.operations):
            self.line(_TMEM_A_INSTRUCTIONS)
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
                self.begin(f'if (warp == {self.roles[a.allocating_role].execution_groups[0]})')
                self.line(f'asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(cake_smem({name})), "n"({a.tensor_columns}) : "memory");')
                self.end()
        self.line('__syncthreads();')
        root_barriers = {
            name for op in self.s.operations if op.kind is OperationKind.TMEM_STORE
            for name in op.signals
        } | {
            name for op in self.s.operations
            if op.kind is OperationKind.LOAD and op.pipeline is None
            and self.b(op.writes[0]).space is MemorySpace.SHARED
            for name in op.signals
        }
        if root_barriers:
            self.begin('if (threadIdx.x == 0)')
            for name in sorted(root_barriers):
                barrier = next(b for b in self.s.barriers if b.name == name)
                self.line(f'cake_init({self.barvars[name]}, {barrier.count});')
            self.line('asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");')
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
        if root_barriers:
            self.begin('if (threadIdx.x == 0)')
            for name in sorted(root_barriers):
                self.line(f'cake_inval({self.barvars[name]});')
            self.end()
        for a in self.s.allocations:
            if a.space is MemorySpace.TENSOR:
                self.begin(f'if (warp == {self.roles[a.allocating_role].execution_groups[0]})')
                self.line(f'asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(*{self.tmemvars[a.name]}), "n"({a.tensor_columns}) : "memory");')
                self.end()
        for owner in sorted({a.allocating_role for a in self.s.allocations if a.space is MemorySpace.TENSOR}):
            self.begin(f'if (warp == {self.roles[owner].execution_groups[0]})')
            self.line('asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");')
            self.end()
        self.end(); self.line('// CAKE_KERNEL_END')
        self.host()
        dims = [1,1,1]
        if self.s.program_map:
            for axis in self.s.program_map.axes:
                dims[axis.axis] = axis.tile_count(self.b(axis.buffer).shape[axis.dimension])
        flags = ['-std=c++17',f'--gpu-architecture={self.s.target.replace("sm_","compute_")}',f'--gpu-code={self.s.target}','-O3','--fmad=false','-lineinfo','-Xptxas=-v']
        metadata = {'kernel_entry_point':self.entry+'_kernel', 'threads_per_cta':self.s.total_execution_group_extent*self.target.warp_size,
                    'dynamic_shared_bytes':self.shared_bytes, 'grid':dims, 'nvcc_flags':flags,
                    'argument_order':[b.name for b in self.globals],
                    'arguments':[{'name':b.name,'dtype':b.dtype.value,'shape':list(b.shape),'mode':b.mode.value} for b in self.globals],
                    'host_abi':{'create':self.entry+'_create','launch':self.entry+'_launch','destroy':self.entry+'_destroy'},
                    'link_libraries':['cuda','cudart'], 'target_device_names':list(self.target.device_names),
                    'register_mapping':'one thread per row; replicated vector operands',
                    'source_language':'cuda_cpp', 'signature':{b.name:'*'+b.dtype.value for b in self.globals},
                    'block':[self.s.total_execution_group_extent*self.target.warp_size,1,1]}
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
        if loop.carried_buffers:
            self.carried_loop(loop)
            return
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

    def carried_loop(self, loop):
        """Run one complete chunk per trip, publishing the next TMEM phase last."""
        pipelines = [p for p in self.s.pipelines if _pipeline_loop(self.s, p) == loop]
        groups = {
            next(op.op_id for op in self.s.loop_operations(loop) if op.pipeline == p.name): p
            for p in pipelines
        }
        covered = {op.op_id for op in self.s.loop_operations(loop)
                   if op.pipeline in {p.name for p in pipelines}}
        variable = self.loopvars[loop.iterator]
        self.line('#pragma unroll 1')
        self.begin(f'for (int {variable}=0; {variable}<{self.trip(loop)}; ++{variable})')
        for operation in self.s.loop_operations(loop):
            if operation.op_id in groups:
                self.pipeline(None, groups[operation.op_id], carried_phase=f'({variable}&1)')
            elif operation.op_id not in covered:
                self.operation(operation)
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

    def pipeline(self, loop, p, *, carried_phase=None):
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
            if not producer:
                for mma in mmas:
                    for name in mma.reads:
                        if self.b(name).space is MemorySpace.TENSOR:
                            writer = next(op for op in self.s.operations if name in op.writes)
                            phase = carried_phase if carried_phase is not None else '0'
                            self.line(f'cake_wait({self.barvars[writer.signals[0]]}, {phase});')
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
                    mn_b = op.parameters.instruction.operand_major[1] is OperandMajorMode.MN
                    desc = ((1<<4) | (dtype_bit<<7) | (dtype_bit<<10)
                            | (int(mn_b)<<16) | ((n>>3)<<17) | ((m>>4)<<24))
                    atom_k = op.parameters.instruction.shape[2]
                    ranges = op.parameters.contribution_ranges
                    first_atom = ranges[0][0] // atom_k
                    for start,end in ranges:
                        self.line('#pragma unroll')
                        self.begin(f'for (int atom={start//atom_k}; atom<{end//atom_k}; ++atom)')
                        if a.space is MemorySpace.TENSOR:
                            width,mode = _SWIZZLE[b.swizzle]
                            b_step = atom_k*b.dtype.itemsize*(n if mn_b else 1)
                            b_desc = f'cake_desc(cake_smem({self.pointer(b,"stage")}) + atom*{b_step}, {width*8}, {mode})'
                            self.line(f'cake_mma_tmem_a({self.taddr(dst)}, {self.taddr(a)} + atom*{atom_k//2}, {b_desc}, {desc}u, {var} != 0 || atom != {first_atom});')
                        else:
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
        fusion = self.state_fusions.get(op.op_id)
        if fusion is not None:
            chain, state, prefix, correction, output = fusion
            if op is not chain[-1]:
                self.line(f'// Lowered together at {chain[-1].op_id}')
                return
            self.begin(f'if ({self.role_condition(op.role)})')
            self.line('#pragma unroll')
            self.begin('for (int col=0; col<128; ++col)')
            self.line(f'float state_value = __bfloat162float({self.names[state]}[col]);')
            self.line(f'float decayed = __fmul_rn(state_value, {self.names[prefix]}[col]);')
            self.line(f'float corrected = __fadd_rn(decayed, {self.names[correction]}[col]);')
            self.line(f'{self.names[output]}[col] = __float2bfloat16_rn(corrected);')
            self.end(); self.end()
            return
        self.begin(f'if ({self.role_condition(op.role)})')
        dst=self.b(op.writes[0]); d=self.names[dst.name]
        src=self.b(op.reads[0]); a=self.names[src.name]
        row=self.row(op)
        if op.kind is OperationKind.LOAD and op.parameters.movement is LoadMovement.TMEM:
            scope = _scope(self.s, op)
            phase = (f'({self.loopvars[scope.iterator]}&1)'
                     if src.dtype is DType.BF16 and scope is not None else '0')
            self.line(f'cake_wait({self.barvars[op.waits[0]]}, {phase});')
            self.line('asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");')
            rep=op.parameters.source_atom.repetition
            self.line('#pragma unroll')
            self.begin(f'for (int col=0; col<{src.shape[1]}; col+={rep * (2 if src.dtype is DType.BF16 else 1)})')
            operands=', '.join(f'%{i}' for i in range(rep))
            if src.dtype is DType.BF16:
                for i in range(rep):
                    self.line(f'uint32_t word{i};')
                outputs=', '.join(f'"=r"(word{i})' for i in range(rep))
                address=f'{self.taddr(src)} + (({row}/32)*32 << 16) + col/2'
            else:
                outputs=', '.join(f'"=f"({d}[col+{i}])' for i in range(rep))
                address=f'{self.taddr(src)} + (({row}/32)*32 << 16) + col'
            self.line(f'asm volatile("tcgen05.ld.sync.aligned.32x32b.x{rep}.b32 {{{operands}}}, [%{rep}];" : {outputs} : "r"({address}) : "memory");')
            self.line('asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");')
            if src.dtype is DType.BF16:
                for i in range(rep):
                    self.line(f'{d}[col+{2*i}] = __ushort_as_bfloat16(uint16_t(word{i}));')
                    self.line(f'{d}[col+{2*i+1}] = __ushort_as_bfloat16(uint16_t(word{i} >> 16));')
            self.end()
        elif op.kind is OperationKind.TMEM_STORE:
            self.line('#pragma unroll')
            self.begin(f'for (int group=0; group<{src.shape[1]//16}; ++group)')
            for i in range(8):
                col = f'(group*16+{i*2})'
                self.line(f'uint32_t word{i} = uint32_t(__bfloat16_as_ushort({a}[{col}])) | (uint32_t(__bfloat16_as_ushort({a}[{col}+1])) << 16);')
            inputs = ', '.join(f'"r"(word{i})' for i in range(8))
            self.line('asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 '
                      '[%0], {%1,%2,%3,%4,%5,%6,%7,%8};" :: '
                      f'"r"({self.taddr(dst)} + (({row}/32)*32 << 16) + group*8), '
                      f'{inputs} : "memory");')
            self.end()
            self.line('asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");')
            self.begin('if ((threadIdx.x & 31) == 0)')
            self.line(f'cake_arrive({self.barvars[op.signals[0]]});')
            self.end()
        elif op.kind is OperationKind.LOAD and dst.space is MemorySpace.SHARED:
            width, _ = _SWIZZLE[dst.swizzle]
            self.begin(f'for (int e=int(threadIdx.x & 31); e<{dst.elements}; e+=32)')
            address, mask = self.address(op, src, [f'e/{dst.shape[1]}', f'e%{dst.shape[1]}'])
            self.line(f'const int byte = e * {dst.dtype.itemsize};')
            self.line(f'const int swizzled = byte ^ (((byte >> 7) & {(width//16)-1}) << 4);')
            self.line(f'*reinterpret_cast<{_TYPES[dst.dtype]}*>({self.pointer(dst,"0")} + swizzled) = ({mask}) ? {address} : {_TYPES[dst.dtype]}(0);')
            self.end()
            self.line('asm volatile("fence.proxy.async.shared::cta;" ::: "memory");')
            self.line('__syncwarp();')
            self.begin('if ((threadIdx.x & 31) == 0)')
            self.line(f'cake_arrive({self.barvars[op.signals[0]]});')
            self.end()
        elif op.kind is OperationKind.LOAD:
            self.line('#pragma unroll')
            self.begin(f'for (int col=0; col<{_slots(self.s,dst)}; ++col)')
            local=[row,'col'] if len(dst.shape)==2 else ['col']
            address,mask=self.address(op,src,local)
            self.line(f'{d}[col] = ({mask}) ? {address} : {_TYPES[dst.dtype]}(0);')
            self.end()
        elif op.kind is OperationKind.FORWARD_SUBSTITUTE:
            coefficient = src
            rhs = self.names[op.reads[1]]
            solve_scope = _scope(self.s, op)
            phase = (f'({self.loopvars[solve_scope.iterator]}&1)'
                     if solve_scope is not None and solve_scope.carried_buffers else '0')
            self.line(f'cake_wait({self.barvars[op.waits[0]]}, {phase});')
            for token in range(32):
                self.line(f'float u{token} = {rhs}[{token}];')
                for prior in range(token):
                    byte = (token * 32 + prior) * 2
                    swizzled = byte ^ (((byte >> 7) & 3) << 4)
                    self.line(
                        f'u{token} = __fmaf_rn(__bfloat162float('
                        f'*reinterpret_cast<const __nv_bfloat16*>('
                        f'{self.pointer(coefficient,"0")} + {swizzled})), '
                        f'u{prior}, u{token});'
                    )
                self.line(f'{d}[{token}] = u{token};')
        elif op.kind is OperationKind.ELEMENTWISE:
            p=op.parameters
            self.line('#pragma unroll')
            self.begin(f'for (int col=0; col<{_slots(self.s,dst)}; ++col)')
            x=f'{a}[{"0" if src.is_scalar or _scalar_row(self.s,src) else "col"}]'
            y=(f'{p.scalar!r}f' if p.scalar is not None else
               (f'{self.names[op.reads[1]]}[{"0" if self.b(op.reads[1]).is_scalar or _scalar_row(self.s,self.b(op.reads[1])) else "col"}]'
                if len(op.reads)>1 else x))
            expr={ElementwiseOp.ADD:f'__fadd_rn({x},{y})',ElementwiseOp.SUB:f'__fsub_rn({x},{y})',
                  ElementwiseOp.MUL:f'__fmul_rn({x},{y})',ElementwiseOp.DIV:f'__fdiv_rn({x},{y})',
                  ElementwiseOp.RELU:f'fmaxf({x},0.0f)',ElementwiseOp.SQUARE:f'__fmul_rn({x},{x})',
                  ElementwiseOp.EXP:f'expf({x})',ElementwiseOp.RSQRT:f'rsqrtf({x})',
                  ElementwiseOp.RECIPROCAL:f'__fdiv_rn(1.0f,{x})'}[p.op]
            self.line(f'{d}[col] = {expr};'); self.end()
        elif op.kind is OperationKind.REDUCE:
            self.line(f'{d}[0] = 0.0f;')
            self.line('#pragma unroll')
            self.begin(f'for (int col=0; col<{src.shape[1]}; ++col)')
            self.line(f'{d}[0] = __fadd_rn({d}[0], {a}[col]);')
            self.end()
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
        major,minor=self.target.compute_capability
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
        self.line(f'{self.entry}_kernel<<<dim3({dims[0]},{dims[1]},{dims[2]}), {self.s.total_execution_group_extent*self.target.warp_size}, {self.shared_bytes}, reinterpret_cast<cudaStream_t>(stream)>>>('+', '.join(args)+');')
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

_TMEM_A_INSTRUCTIONS = r'''
__device__ __forceinline__ void cake_mma_tmem_a(uint32_t dst, uint32_t a, uint64_t b, uint32_t desc, bool accumulate) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p; }" :: "r"(dst), "r"(a), "l"(b), "r"(desc), "r"(int(accumulate)) : "memory");
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
