"""Readable baseline Schedule generation, bound to the task-owned Workload ABI."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, backend_for_target
from .workload import TASKS, row_spans, validate_solx_fib_contract


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    """Emit high-level Python for the selected shape, with frozen Workload metadata.

    The BF16 ABI is explicit on both sides: every operand widens on load, the reduction
    and the affine scale happen in FP32, and one narrowing cast produces the result. That
    is what the upstream definition's reference does and what the tolerance assumes.
    """
    return _rowwise_source(workload, case_id, execution_groups=1)


def rmsnorm025_cta_source(workload: WorkloadContract, case_id: str = "primary", *,
                          execution_groups: int = 8) -> str:
    """Keep FIB 025's row algorithm while assigning more lanes to its 4096 columns."""
    validate_solx_fib_contract(workload.document)
    if (workload.target != "sm_103a"
            or workload.document["operator"] != TASKS["fib_rmsnorm_h4096"][0]):
        raise ValueError("RMSNorm 025 CTA-width mapping requires its exact B300 Workload")
    if type(execution_groups) is not int or execution_groups not in (4, 8, 16):
        raise ValueError("RMSNorm 025 CTA-width mapping admits 4, 8 or 16 groups")
    return _rowwise_source(workload, case_id, execution_groups=execution_groups,
                           schedule_suffix=f"-cta-w{execution_groups}",
                           entry_point=f"cake_fib025_cta_w{execution_groups}")


def _rowwise_source(workload: WorkloadContract, case_id: str, *,
                    execution_groups: int, schedule_suffix: str = "",
                    entry_point: str | None = None) -> str:
    validate_solx_fib_contract(workload.document)
    operator = workload.document["operator"]
    args = workload.tensor_abi(case_id)
    names = {arg.name for arg in args}
    width = args[0].shape[-1]
    epsilon = workload.document["semantics"]["epsilon"]
    body = ['stored = lm.load(x[row, :], id="load_x")',
            'values = lm.cast(stored, to="fp32", id="widen_x")']
    operand = "values"
    if "residual" in names:
        body += ['stored_residual = lm.load(residual[row, :], id="load_residual")',
                 'residual_values = lm.cast(stored_residual, to="fp32", id="widen_residual")',
                 'combined = values + residual_values']
        operand = "combined"
    body += [
        'stored_weight = lm.load(weight[:], id="load_weight")',
        'weights = lm.cast(stored_weight, to="fp32", id="widen_weight")',
        f'squares = lm.square({operand}, id="square")',
        'square_sum = lm.reduce(squares, op="sum", axis=0, scope="cta", across_loop=False, id="sum_square")',
        f'mean_square = square_sum / {float(width)!r}',
        f'inverse = lm.rsqrt(mean_square + {epsilon!r}, id="inverse")',
        f'normalized = {operand} * inverse',
        'weighted = normalized * weights',
        'narrowed = lm.cast(weighted, to="bf16", id="narrow_out")',
        'lm.store(out[row, :], narrowed, coalesced=False, id="store_out")',
    ]
    if len(row_spans(width)) > 1:
        body = _partitioned_body(width, epsilon, residual="residual" in names)
    device = BACKENDS[backend_for_target(workload.target)]
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
                    + (', mode="output")' if arg.mode == "output" else ')') for arg in args]
    schedule_name = workload.workload_id + schedule_suffix
    entry = entry_point or f"cake_{operator}"
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{schedule_name}", target="{workload.target}",\n'
            f'               backend="{device["route"]}", entry_point="{entry}")\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            f'    compute = lm.role(execution_groups={list(range(execution_groups))!r})\n'
            '    row = lm.program(x, axis=0, dimension=0, tile=1)\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')


def _partitioned_body(width: int, epsilon: float, *, residual: bool) -> list[str]:
    """Sum all disjoint slices before normalizing any slice with the whole-row mean."""
    spans = row_spans(width)
    body = []
    for i, (start, stop) in enumerate(spans):
        body.extend([
            f'stored_{i} = lm.load(x[row, {start}:{stop}], id="load_x_{i}")',
            f'values_{i} = lm.cast(stored_{i}, to="fp32", id="widen_x_{i}")',
        ])
        if residual:
            body.extend([
                f'residual_{i} = lm.load(residual[row, {start}:{stop}], id="load_residual_{i}")',
                f'residual_fp32_{i} = lm.cast(residual_{i}, to="fp32", id="widen_residual_{i}")',
                f'combined_{i} = values_{i} + residual_fp32_{i}',
            ])
        operand = f'combined_{i}' if residual else f'values_{i}'
        body.extend([
            f'squares_{i} = lm.square({operand}, id="square_{i}")',
            f'sum_{i} = lm.reduce(squares_{i}, op="sum", axis=0, scope="cta", across_loop=False, id="sum_square_{i}")',
        ])
    body.extend([
        'square_sum = ' + ' + '.join(f'sum_{i}' for i in range(len(spans))),
        f'mean_square = square_sum / {float(width)!r}',
        f'inverse = lm.rsqrt(mean_square + {epsilon!r}, id="inverse")',
    ])
    tile = width & -width
    body.append(f'for column in lm.range(x, name="columns", dimension=1, tile={tile}, num_stages=1):')
    write = [
        'stored_output = lm.load(x[row, column], id="reload_x")',
        'output_values = lm.cast(stored_output, to="fp32", id="widen_output_x")',
    ]
    if residual:
        write.extend([
            'residual_output = lm.load(residual[row, column], id="reload_residual")',
            'residual_values = lm.cast(residual_output, to="fp32", id="widen_output_residual")',
            'output_combined = output_values + residual_values',
        ])
    operand = 'output_combined' if residual else 'output_values'
    write.extend([
        'output_weight = lm.load(weight[column], id="load_weight")',
        'output_weight_fp32 = lm.cast(output_weight, to="fp32", id="widen_weight")',
        f'normalized = {operand} * inverse',
        'weighted = normalized * output_weight_fp32',
        'narrowed = lm.cast(weighted, to="bf16", id="narrow_out")',
        'lm.store(out[row, column], narrowed, coalesced=False, id="store_out")',
    ])
    body.extend('    ' + statement for statement in write)
    return body
