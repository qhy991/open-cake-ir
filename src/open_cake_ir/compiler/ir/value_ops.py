"""Typing of register coordinates, comparisons and selection, without hardware lookup."""
from __future__ import annotations

import math
from .vocabulary import DType, MemorySpace, OperationKind

VALUE_KINDS = frozenset({OperationKind.COORDINATE, OperationKind.COMPARE, OperationKind.SELECT})
NUMERIC = frozenset({DType.INT32, DType.FP32, DType.FP16, DType.BF16})


def _broadcast_shape(buffers):
    shapes = {buffer.shape for buffer in buffers if not buffer.is_scalar}
    if len(shapes) > 1:
        raise ValueError('value operands require equal shapes or canonical [1] scalars')
    return next(iter(shapes), (1,))


def _scalar(value, dtype):
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError('numeric scalar must be finite')
    if dtype is DType.INT32 and (value != int(value) or not -(2**31) <= value < 2**31):
        raise ValueError('INT32 scalar must be an exactly representable signed integer')


def result_type(schedule, operation):
    """Return one result dtype/shape or refuse the value operation by its actual rule."""
    p = operation.parameters
    if len(operation.writes) != 1:
        raise ValueError('value operation writes exactly one register result')
    if operation.kind is OperationKind.COORDINATE:
        if operation.reads:
            raise ValueError('coordinate reads declared program/loop axes, not buffers')
        if p.source == 'range':
            return DType.INT32, (p.extent,)
        if p.source.startswith('program'):
            axis = schedule.program_map.axis(p.name) if schedule.program_map else None
            if axis is None:
                raise ValueError(f'coordinate names undeclared program axis {p.name!r}')
            return DType.INT32, (axis.tile if p.source == 'program_tile' else 1,)
        loop = next((loop for loop in schedule.tile_loops if loop.iterator == p.name), None)
        if loop is None or operation.op_id not in loop.body:
            raise ValueError(f'coordinate loop {p.name!r} does not enclose this operation')
        return DType.INT32, (loop.tile if p.source == 'loop_tile' else 1,)
    reads = [schedule.buffer(name) for name in operation.reads]
    if any(buffer is None or buffer.space is not MemorySpace.REGISTER for buffer in reads):
        raise ValueError('value inputs must name declared register buffers')
    if operation.kind is OperationKind.COMPARE:
        if len(reads) != (1 if p.scalar is not None else 2):
            raise ValueError('compare needs two operands, optionally a scalar second operand')
        if len({buffer.dtype for buffer in reads}) != 1 or reads[0].dtype not in NUMERIC:
            raise ValueError('compare requires matching numeric dtypes')
        if p.scalar is not None:
            _scalar(p.scalar, reads[0].dtype)
        return DType.INT32, _broadcast_shape(reads)
    if operation.kind is OperationKind.SELECT:
        if len(reads) != (2 if p.false_value is not None else 3):
            raise ValueError('select reads predicate, true value and false value or literal')
        if reads[0].dtype is not DType.INT32:
            raise ValueError('select predicate must be INT32; zero is false')
        dtype = reads[1].dtype
        if dtype not in NUMERIC or any(buffer.dtype is not dtype for buffer in reads[1:]):
            raise ValueError('select values require matching numeric dtypes')
        if p.false_value == 'negative_infinity':
            if dtype is DType.INT32:
                raise ValueError('negative infinity is a floating selection value')
        elif p.false_value is not None:
            _scalar(p.false_value, dtype)
        return dtype, _broadcast_shape(reads)
    raise ValueError('not a typed register value operation')
