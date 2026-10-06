"""Explicit MACA engineering starters with the original task Workload contract."""
from __future__ import annotations

from open_cake_ir.compiler.target import CodeObject, declared_target


def native_event_starter(workload, source: str) -> str:
    """Bound GEMM+Bias's resident tile without changing its full mathematical task.

    The original row-wide 256x1024 product tile exceeds the observed C550 launch
    memory capacity. One output column per program uses the same FP32 product and
    reduction, full K extent, public buffers and independent oracle. This is a new
    engineering baseline, not an optimization gain against the old starter.
    """
    if declared_target(workload.target).code_object is not CodeObject.MCFATBIN:
        raise ValueError('MACA task starters require a declared mcfatbin target')
    if workload.document['operator'] != 'gemm_bias_fp32':
        return source
    from .gemm.authoring import starter_source
    ordinary = starter_source(workload)
    if source != ordinary:
        raise ValueError('GEMM+Bias starter differs from its task-owned source')
    return (ordinary.replace('    with compute:',
                '    column = lm.program(b, axis=1, dimension=1, tile=1)\n    with compute:')
            .replace('b[:, :]', 'b[:, column]')
            .replace('b_tile * lm.broadcast(a_row, axis=0)', 'b_tile * a_row')
            .replace('bias[:]', 'bias[column]')
            .replace('out[row, :]', 'out[row, column]'))
