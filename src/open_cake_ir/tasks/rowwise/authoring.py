"""Readable baseline Schedule generation, bound to the task-owned Workload ABI."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, backend_for_target
from .workload import EPSILON, OUTPUTS, validate_rowwise_contract


def _body(operator: str, width: int) -> tuple[list[str], str]:
    """Return one task's operation statements and the expression its store writes.

    The per-row statistics arrive as rank-1 buffers indexed by the program axis, so the
    starter loads a scalar from each rather than reducing to recover it. Both reductions
    fold a product of two loaded tiles, which is what this family is for.
    """
    if operator == "softmax_backward_fp32":
        return (['probabilities = lm.load(p[row, :], id="load_p")',
                 'gradients = lm.load(dp[row, :], id="load_dp")',
                 'total = lm.reduce(gradients, op="sum", axis=0, scope="cta",'
                 ' across_loop=False, id="sum_dp")'],
                'gradients - probabilities * total')
    if operator == "absmax_rescale_fp32":
        return (['values = lm.load(x[row, :], id="load_x")',
                 'positive = lm.relu(values, id="relu_positive")',
                 'negative = lm.relu(values * -1.0, id="relu_negative")',
                 'magnitude = positive + negative',
                 'peak = lm.reduce(magnitude, op="max", axis=0, scope="cta",'
                 ' across_loop=False, id="max_abs")',
                 f'inverse = lm.reciprocal(peak + {EPSILON!r}, id="reciprocal")'],
                'values * inverse')
    if operator == "cosine_similarity_fp32":
        return (['left = lm.load(a[row, :], id="load_a")',
                 'right = lm.load(b[row, :], id="load_b")',
                 'left_squares = lm.square(left, id="square_a")',
                 'right_squares = lm.square(right, id="square_b")',
                 'products = left * right',
                 'aa = lm.reduce(left_squares, op="sum", axis=0, scope="cta",'
                 ' across_loop=False, id="sum_aa")',
                 'bb = lm.reduce(right_squares, op="sum", axis=0, scope="cta",'
                 ' across_loop=False, id="sum_bb")',
                 'ab = lm.reduce(products, op="sum", axis=0, scope="cta",'
                 ' across_loop=False, id="sum_ab")',
                 f'denominator = (aa + {EPSILON!r}) * (bb + {EPSILON!r})',
                 'inverse = lm.rsqrt(denominator, id="rsqrt")'],
                'ab * inverse')
    if operator == "rmsnorm_input_gradient_fp32":
        return (['upstream = lm.load(dy[row, :], id="load_dy")',
                 'values = lm.load(x[row, :], id="load_x")',
                 'scales = lm.load(gamma[:], id="load_gamma")',
                 'root = lm.load(rrms[row], id="load_rrms")',
                 'weighted = upstream * scales',
                 'products = weighted * values',
                 'total = lm.reduce(products, op="sum", axis=0, scope="cta",'
                 ' across_loop=False, id="sum_products")',
                 'cubed = lm.square(root, id="square_root") * root',
                 f'second = cubed * total / {float(-width)!r}'],
                'weighted * root + values * second')
    return (['upstream = lm.load(dy[row, :], id="load_dy")',
             'values = lm.load(x[row, :], id="load_x")',
             'scales = lm.load(gamma[:], id="load_gamma")',
             'average = lm.load(mean[row], id="load_mean")',
             'inverse = lm.load(rstd[row], id="load_rstd")',
             'centered = (values - average) * inverse',
             'weighted = upstream * scales',
             'products = weighted * centered',
             'first_sum = lm.reduce(products, op="sum", axis=0, scope="cta",'
             ' across_loop=False, id="sum_products")',
             f'first = first_sum / {float(width)!r}',
             'second_sum = lm.reduce(weighted, op="sum", axis=0, scope="cta",'
             ' across_loop=False, id="sum_weighted")',
             f'second = second_sum / {float(width)!r}',
             'residual = weighted - second',
             'corrected = residual - centered * first'],
            'corrected * inverse')


def operator_key(operator: str) -> str:
    """Map the registered operator id back to its task name."""
    from .workload import TASKS
    return next(name for name, (registered, _) in TASKS.items() if registered == operator)


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    """Emit high-level Python for the selected shape, with frozen Workload metadata.

    One operation body serves every route; only the declared route and Target come from
    the device. Neither task names an instruction contract, so this family is portable
    without a per-device spelling.
    """
    validate_rowwise_contract(workload.document)
    operator = workload.document["operator"]
    device = BACKENDS[backend_for_target(workload.target)]
    args = workload.tensor_abi(case_id)
    primary = args[0].name
    width = next(arg.shape[-1] for arg in args if len(arg.shape) == 2)
    body, result = _body(operator, width)
    target = "out[row]" if len(OUTPUTS[operator_key(operator)][1]) == 1 else "out[row, :]"
    body.append(f'lm.store({target}, {result}, coalesced=False, id="store_out")')
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
                    + (', mode="output")' if arg.mode == "output" else ')') for arg in args]
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}", target="{workload.target}",\n'
            f'               backend="{device["route"]}", entry_point="cake_{operator}",\n'
            f'               metadata={{"workload_contract_sha256": "{workload.canonical_sha256}"}})\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            '    compute = lm.role(warps=[0])\n'
            f'    row = lm.program({primary}, axis=0, dimension=0, tile=1)\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')
