"""Nonblocking dependency facts about reductions repeated across program axes.

This counts logical executions of an invariant read-only subgraph. It is not
physical traffic, measured redundancy after native optimization, or a cost model.
Unknown effects, loops, aliasing storage and indirect access abstain.
"""
from collections import Counter
from math import prod

from ..diagnostics import Finding, FindingCategory, FindingSeverity
from ..ir import AccessIndexKind, BufferMode, MemorySpace, OperationKind


def program_repetition_guidance(schedule):
    mapping = schedule.program_map
    pure = {OperationKind.ELEMENTWISE, OperationKind.CAST, OperationKind.REDUCE}
    if (mapping is None or mapping.persistent or schedule.tile_loops
            or len(schedule.roles) != 1
            or any(op.kind not in pure | {OperationKind.LOAD,OperationKind.STORE}
                   for op in schedule.operations)
            or any(b.allocation is not None or b.valid_extent is not None
                   or b.space not in {MemorySpace.GLOBAL,MemorySpace.REGISTER}
                   or (b.space is MemorySpace.GLOBAL and b.mode not in {BufferMode.INPUT,BufferMode.OUTPUT})
                   for b in schedule.buffers)):
        return ()
    counts = {}
    for axis in mapping.axes:
        buffer = schedule.buffer(axis.buffer)
        if buffer is None or axis.dimension >= len(buffer.shape):
            return ()
        extent = buffer.shape[axis.dimension]
        if extent % axis.tile:
            return ()  # The first domain excludes partial program tiles.
        counts[axis.name] = axis.tile_count(extent)
    writers = Counter(name for op in schedule.operations for name in op.writes)
    dependencies = {}
    output = []
    for index, op in enumerate(schedule.operations):
        if op.kind is OperationKind.STORE:
            continue
        if len(op.writes) != 1 or writers[op.writes[0]] != 1:
            return ()
        destination = schedule.buffer(op.writes[0])
        if destination is None or destination.space is not MemorySpace.REGISTER:
            return ()
        if op.kind is OperationKind.LOAD:
            if len(op.reads) != 1:
                return ()
            source = schedule.buffer(op.reads[0])
            access = schedule.access_map(op.op_id,op.reads[0])
            if (source is None or source.space is not MemorySpace.GLOBAL
                    or source.mode is not BufferMode.INPUT or writers[source.name]
                    or access is None
                    or any(i.source not in {AccessIndexKind.PROGRAM,AccessIndexKind.PROGRAM_TILE,
                                            AccessIndexKind.DIMENSION} for i in access.indices)):
                return ()
            used = set(access.program_axes)
        else:
            if not op.reads or any(name not in dependencies for name in op.reads):
                return ()
            used = set().union(*(dependencies[name] for name in op.reads))
        dependencies[op.writes[0]] = used
        invariant = {name:count for name,count in counts.items() if count>1 and name not in used}
        if op.kind is OperationKind.REDUCE and invariant:
            copies = prod(invariant.values())
            axes = ', '.join(f'{name}={count}' for name,count in invariant.items())
            output.append(Finding(
                'PROGRAM_INVARIANT_REDUCTION',f'operations[{index}]',
                f"reduction {op.op_id!r} reads the same declared input subgraph across program axes "
                f"{axes}; each fixed coordinate of the other axes repeats it {copies} times. "
                "This is logical work before native optimization, not measured memory traffic "
                "or a performance prediction. Duplication may trade resources for parallelism.",
                FindingCategory.DATA_CONSISTENCY,FindingSeverity.HINT,
            ))
    return tuple(output)
