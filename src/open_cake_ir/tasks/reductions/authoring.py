"""Readable baseline Schedule generation, bound to the task-owned Workload ABI."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, backend_for_target
from .workload import EPSILON, FP8_E4M3_MAX, validate_reductions_contract


def _body(operator: str, rows: int) -> list[str]:
    """Return one task's statements, including its own stores.

    Both tasks write more than the row-shaped output the other families do, so the store
    is part of the body rather than one trailing expression.
    """
    if operator == "per_channel_moments_fp32":
        return ['values = lm.load(x[:, column], id="load_x")',
                'squares = lm.square(values, id="square")',
                'total = lm.reduce(values, op="sum", axis=0, scope="cta",'
                ' across_loop=False, id="sum_x")',
                'square_total = lm.reduce(squares, op="sum", axis=0, scope="cta",'
                ' across_loop=False, id="sum_square")',
                f'lm.store(mean[column], total / {float(rows)!r}, coalesced=False, id="store_mean")',
                f'lm.store(mean_square[column], square_total / {float(rows)!r},'
                ' coalesced=False, id="store_mean_square")']
    if operator == "channel_absmax_scale_fp32":
        return ['values = lm.load(x[:, column], id="load_x")',
                'positive = lm.relu(values, id="relu_positive")',
                'negative = lm.relu(values * -1.0, id="relu_negative")',
                'magnitude = positive + negative',
                'peak = lm.reduce(magnitude, op="max", axis=0, scope="cta",'
                ' across_loop=False, id="max_abs")',
                f'regularized = peak + {EPSILON!r}',
                'lm.store(amax[column], regularized, coalesced=False, id="store_amax")',
                f'lm.store(scale[column], regularized / {FP8_E4M3_MAX!r},'
                ' coalesced=False, id="store_scale")']
    if operator == "bias_gradient_reduction_fp32":
        return ['upstream = lm.load(dout[:, column], id="load_dout")',
                'prior = lm.load(bias[column], id="load_bias")',
                'total = lm.reduce(upstream, op="sum", axis=0, scope="cta",'
                ' across_loop=False, id="sum_dout")',
                'lm.store(dbias[column], total + prior, coalesced=False, id="store_dbias")']
    return ['upstream = lm.load(dy[:, column], id="load_dy")',
            'values = lm.load(x[:, column], id="load_x")',
            'averages = lm.load(mean[:], id="load_mean")',
            'inverses = lm.load(rstd[:], id="load_rstd")',
            'centered = (values - averages) * inverses',
            'products = upstream * centered',
            'scale_sum = lm.reduce(products, op="sum", axis=0, scope="cta",'
            ' across_loop=False, id="sum_products")',
            'shift_sum = lm.reduce(upstream, op="sum", axis=0, scope="cta",'
            ' across_loop=False, id="sum_dy")',
            'lm.store(dgamma[column], scale_sum, coalesced=False, id="store_dgamma")',
            'lm.store(dbeta[column], shift_sum, coalesced=False, id="store_dbeta")']


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    """Emit high-level Python for the selected shape, with frozen Workload metadata.

    The program axis is the feature extent, so the tile each program loads is a column and
    the reduction runs down it. Neither task names an instruction contract, so one body
    serves every route.
    """
    validate_reductions_contract(workload.document)
    operator = workload.document["operator"]
    device = BACKENDS[backend_for_target(workload.target)]
    args = workload.tensor_abi(case_id)
    primary = args[0].name
    rows = next(arg.shape[0] for arg in args if len(arg.shape) == 2)
    body = _body(operator, rows)
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
                    + (', mode="output")' if arg.mode == "output" else ')') for arg in args]
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}", target="{workload.target}",\n'
            f'               backend="{device["route"]}", entry_point="cake_{operator}",\n'
            f'               metadata={{"workload_contract_sha256": "{workload.canonical_sha256}"}})\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            '    compute = lm.role(warps=[0])\n'
            f'    column = lm.program({primary}, axis=0, dimension=1, tile=1)\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')
