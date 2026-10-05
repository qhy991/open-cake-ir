"""Prove when an entire Triton program has no observable output writes.

The existing valid-prefix relation owns the domain. This does not infer a domain
from masked inputs, change launch geometry, or admit a new Schedule spelling.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ir import (AccessIndexKind, BufferMode, LoweringBackend, MemorySpace,
                 DType, OperationKind, Schedule)


@dataclass(frozen=True)
class OutputTileDomain:
    lengths: str
    group: str
    tile_axis: str


def output_tile_domain(schedule: Schedule) -> OutputTileDomain | None:
    """All global effects must be stores sharing one scalar-indexed prefix.

    A dense output or state transition keeps the original execution. The tile
    coordinate is program-owned and nonnegative; a loop/runtime index cannot
    authorize skipping the enclosing program. Persistent walks stay unchanged.
    """
    if (schedule.lowering.backend is not LoweringBackend.TRITON
            or schedule.program_map is None or schedule.program_map.persistent
            or schedule.barriers or schedule.pipelines):
        return None
    domain = None
    for operation in schedule.operations:
        if operation.kind is OperationKind.ATOMIC_RMW:
            return None
        for name in operation.writes:
            buffer = schedule.buffer(name)
            if buffer is None:
                return None
            if buffer.space is not MemorySpace.GLOBAL:
                continue
            if operation.kind is not OperationKind.STORE or buffer.mode is not BufferMode.OUTPUT:
                return None
            relation = buffer.valid_extent
            access = schedule.access_map(operation.op_id, name)
            if relation is None or access is None or len(relation.indexed_by) != 1:
                return None
            if max(relation.dimension, relation.indexed_by[0]) >= len(access.indices):
                return None
            lengths = schedule.buffer(relation.buffer)
            if (lengths is None or lengths.mode is not BufferMode.INPUT
                    or lengths.space is not MemorySpace.GLOBAL or lengths.dtype is not DType.INT32
                    or len(lengths.shape) != 1):
                return None
            row = access.indices[relation.dimension]
            group = access.indices[relation.indexed_by[0]]
            if row.source is not AccessIndexKind.PROGRAM_TILE or group.source is not AccessIndexKind.PROGRAM:
                return None
            axis = schedule.program_map.axis(row.name)
            group_axis = schedule.program_map.axis(group.name)
            if axis is None or not axis.is_tiled or group_axis is None or group_axis.is_tiled:
                return None
            current = OutputTileDomain(relation.buffer, group.name, row.name)
            if domain is not None and current != domain:
                return None
            domain = current
    return domain
