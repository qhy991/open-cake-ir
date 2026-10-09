"""Prove bounded, ordinary output stores partition their complete output.

This is a proof over concrete AccessMaps, not a new layout representation. Each
program owns one rectangle; its stores must partition that rectangle exactly.
The first subset has no loops, mutable state, atomics, output readback or aliasing.
"""
from __future__ import annotations

from math import prod

from ..ir import AccessIndexKind, BufferMode, MemorySpace, OperationKind, Schedule


def partition_refusal(schedule: Schedule, name: str) -> tuple[str, str] | None:
    """Return the missing proof, or None when all writes form an exact partition.

    The caller uses this only for multiple writers. Other verifier rules still
    own typing, value/address shape agreement, dependencies and backend support.
    """
    buffer = schedule.buffer(name)
    writers = tuple(op for op in schedule.operations if name in op.writes)
    unsupported = "BUFFER_MULTIPLE_WRITERS"
    if (buffer is None or buffer.mode is not BufferMode.OUTPUT
            or buffer.space is not MemorySpace.GLOBAL or buffer.allocation is not None
            or buffer.byte_offset != 0 or buffer.stages != 1
            or buffer.swizzle is not None or buffer.valid_extent is not None
            or buffer.scale_of is not None):
        return unsupported, "static partitions require an ordinary nonaliasing global output"
    if (schedule.tile_loops or any(b.mode is BufferMode.STATE for b in schedule.buffers)
            or any(op.kind is OperationKind.ATOMIC_RMW for op in schedule.operations)):
        return unsupported, "static output partitions do not admit loops, state or atomic operations"
    if any(name in op.reads for op in schedule.operations):
        return unsupported, "static output partitions do not admit output readback"
    if (any(op.kind is not OperationKind.STORE or op.writes != (name,) for op in writers)
            or len({op.role for op in writers}) != 1):
        return unsupported, "static output partitions require ordinary stores from one role"
    mapping = schedule.program_map
    if mapping is None or mapping.persistent:
        return unsupported, "static output partitions require a nonpersistent ProgramMap"

    ownership = None
    rectangles: list[tuple[tuple[int, int], ...]] = []
    for operation in writers:
        matches = tuple(a for a in schedule.access_maps
                        if a.operation == operation.op_id and a.buffer == name)
        if len(matches) != 1 or len(matches[0].indices) != len(buffer.shape):
            return unsupported, f"store {operation.op_id!r} needs one complete output AccessMap"
        access = matches[0]
        rectangle = []
        owner = []
        for dimension, (component, size) in enumerate(zip(access.indices, buffer.shape)):
            if component.source is AccessIndexKind.DIMENSION:
                if component.dimension != dimension:
                    return unsupported, "static partitions require dimension components in their addressed dimensions"
                start, end = component.offset, component.offset + component.span(size)
                if not 0 <= start < end <= size:
                    return "OUTPUT_PARTITION_BOUNDS", f"store {operation.op_id!r} leaves output dimension {dimension}: [{start}, {end}) of {size}"
                rectangle.append((start, end))
                owner.append(None)
                continue
            if component.source not in {AccessIndexKind.PROGRAM, AccessIndexKind.PROGRAM_TILE}:
                return unsupported, "static partitions require direct program coordinates and static dimension intervals"
            axis = mapping.axis(component.name or "")
            anchor = schedule.buffer(axis.buffer) if axis is not None else None
            if axis is None or anchor is None or axis.dimension >= len(anchor.shape):
                return unsupported, "static partition program ownership is undeclared"
            expected = AccessIndexKind.PROGRAM_TILE if axis.is_tiled else AccessIndexKind.PROGRAM
            if (component.source is not expected or anchor.shape[axis.dimension] != size
                    or size % axis.tile != 0):
                return "OUTPUT_PARTITION_OWNERSHIP", "each program coordinate must tile the complete output dimension exactly, without a masked tail"
            owner.append((axis.name, component.source))
            rectangle.append((0, axis.tile))
        # Every axis is consumed exactly once, even singleton axes. This makes
        # ownership injective and excludes diagonal, broadcast and omitted maps.
        if (len(access.program_axes) != len(mapping.axes)
                or set(access.program_axes) != {axis.name for axis in mapping.axes}):
            return "OUTPUT_PARTITION_OWNERSHIP", "each output store must consume every program axis exactly once"
        if ownership is not None and tuple(owner) != ownership:
            return "OUTPUT_PARTITION_OWNERSHIP", "all partition stores must have identical program ownership in each output dimension"
        ownership = tuple(owner)
        rectangle = tuple(rectangle)
        for previous in rectangles:
            if all(max(a, c) < min(b, d) for (a, b), (c, d) in zip(previous, rectangle)):
                return "OUTPUT_PARTITION_OVERLAP", f"store {operation.op_id!r} overlaps a preceding store to {name!r}"
        rectangles.append(rectangle)

    # Pairwise disjoint in-bounds rectangles cover the finite integer domain iff
    # their total volume equals its volume. No enumeration of output elements.
    assert ownership is not None
    domain = prod(size if owner is None else mapping.axis(owner[0]).tile
                  for size, owner in zip(buffer.shape, ownership))
    written = sum(prod(end - start for start, end in rectangle) for rectangle in rectangles)
    if written != domain:
        return "OUTPUT_PARTITION_COVERAGE", f"partition stores cover {written} of {domain} elements per program in output {name!r}"
    return None
