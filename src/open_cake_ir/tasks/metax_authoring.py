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
    bounded = (ordinary.replace('    with compute:',
                '    column = lm.program(b, axis=1, dimension=1, tile=1)\n    with compute:')
            .replace('b[:, :]', 'b[:, column]')
            .replace('b_tile * lm.broadcast(a_row, axis=0)', 'b_tile * a_row')
            .replace('bias[:]', 'bias[column]')
            .replace('out[row, :]', 'out[row, column]'))
    # Keep rounded FP32 products, but avoid losing small terms during cancellation.
    # Eight-exponent bands make each power-of-two mixed-magnitude subtotal exact
    # at K=256. Neumaier compensation also handles arbitrary finite products.
    bands = [2.0**-16, 2.0**-8, 1.0, 256.0]
    body = ['        product_positive = lm.relu(products)',
            '        product_negative = lm.relu(products * -1.0)',
            '        product_abs = product_positive + product_negative']
    for i in range(5):
        if i == 0:
            body.append(f'        mask_{i} = lm.compare(product_abs, {bands[0]!r}, op="lt")')
        elif i == 4:
            body.append(f'        mask_{i} = lm.compare(product_abs, {bands[-1]!r}, op="ge")')
        else:
            body.extend([f'        low_{i} = lm.compare(product_abs, {bands[i-1]!r}, op="ge")',
                         f'        high_{i} = lm.compare(product_abs, {bands[i]!r}, op="lt")',
                         f'        mask_{i} = low_{i} & high_{i}'])
        body.extend([f'        band_{i} = lm.select(mask_{i}, products, 0.0)',
            f'        sum_{i} = lm.reduce(band_{i}, op="sum", axis=0, scope="cta", across_loop=False, id="sum_band_{i}")'])
    for i in range(1, 5):
        before = 'sum_0' if i == 1 else f'total_{i-1}'
        body.extend([f'        total_{i} = {before} + sum_{i}',
            f'        before_abs_{i} = lm.relu({before}) + lm.relu({before} * -1.0)',
            f'        next_abs_{i} = lm.relu(sum_{i}) + lm.relu(sum_{i} * -1.0)',
            f'        larger_{i} = lm.compare(before_abs_{i}, next_abs_{i}, op="ge")',
            f'        left_error_{i} = ({before} - total_{i}) + sum_{i}',
            f'        right_error_{i} = (sum_{i} - total_{i}) + {before}',
            f'        error_{i} = lm.select(larger_{i}, left_error_{i}, right_error_{i})',
            f'        correction_{i} = '+(f'error_{i}' if i == 1 else f'correction_{i-1} + error_{i}')])
    body.append('        totals = total_4 + correction_4')
    reduction = '        totals = lm.reduce(products, op="sum", axis=0, scope="cta", across_loop=False, id="sum_k")'
    if reduction not in bounded:
        raise ValueError('GEMM+Bias reduction site differs from its task-owned starter')
    return bounded.replace(reduction, '\n'.join(body))
