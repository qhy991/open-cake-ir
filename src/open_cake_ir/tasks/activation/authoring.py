"""Readable baseline Schedule generation, bound to the task-owned Workload ABI."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, backend_for_target
from .workload import (
    GELU_CUBIC_SCALE,
    GELU_INNER_SCALE,
    SELU_ALPHA,
    SELU_LAMBDA,
    validate_activation_contract,
)


def _body(operator: str, primary: str, tanh_contract: str) -> tuple[list[str], str]:
    """Return one task's operation statements and the expression its store writes.

    Two compositions here are the ones the AKA v6 expressibility review recorded, and
    both exist because the IR admits no conditional: absolute value is the sum of the
    operand's ReLU and its negation's, and SELU's positive arm cancels to zero because
    `x - relu(x)` is zero there and `alpha * exp(0) - alpha` therefore vanishes.
    """
    load = f'values = lm.load({primary}[row, :], id="load_{primary}")'
    if operator == "gelu_tanh_fp32":
        return ([load, 'squares = lm.square(values, id="square")',
                 f'inner = (values + squares * values * {GELU_CUBIC_SCALE!r}) * {GELU_INNER_SCALE!r}',
                 f'saturated = lm.tanh(inner, instruction={{"contract": "{tanh_contract}"}}, id="tanh")'],
                'values * 0.5 * (saturated + 1.0)')
    if operator == "gelu_tanh_backward_fp32":
        return ([load, 'gradients = lm.load(dy[row, :], id="load_dy")',
                 'squares = lm.square(values, id="square")',
                 f'inner = (values + squares * values * {GELU_CUBIC_SCALE!r}) * {GELU_INNER_SCALE!r}',
                 f'saturated = lm.tanh(inner, instruction={{"contract": "{tanh_contract}"}}, id="tanh")',
                 'tangent_squared = lm.square(saturated, id="square_tanh")',
                 # The parent's own mechanism: reuse the tangent for sech^2 instead of
                 # forming a hyperbolic cosine out of two exponentials.
                 'sech_squared = tangent_squared * -1.0 + 1.0',
                 f'cubic = squares * {3.0 * GELU_CUBIC_SCALE!r} + 1.0',
                 f'slope = values * 0.5 * sech_squared * cubic * {GELU_INNER_SCALE!r}',
                 'derivative = saturated * 0.5 + 0.5 + slope'],
                'gradients * derivative')
    if operator == "prelu_fp32":
        return ([load, 'slopes = lm.load(slope[:], id="load_slope")',
                 'positive = lm.relu(values, id="relu")',
                 'nonpositive = values - positive'],
                'positive + nonpositive * slopes')
    if operator == "softsign_fp32":
        return ([load, 'positive = lm.relu(values, id="relu_positive")',
                 'negated = values * -1.0',
                 'negative = lm.relu(negated, id="relu_negative")',
                 'magnitude = positive + negative'],
                'values / (magnitude + 1.0)')
    if operator == "selu_fp32":
        return ([load, 'positive = lm.relu(values, id="relu")',
                 'nonpositive = values - positive',
                 'decayed = lm.exp(nonpositive, id="exp")',
                 f'shifted = decayed * {SELU_ALPHA!r} - {SELU_ALPHA!r}'],
                f'(positive + shifted) * {SELU_LAMBDA!r}')
    negated = [load, 'negated = values * -1.0', 'decayed = lm.exp(negated, id="exp")']
    if operator == "softplus_gradient_fp32":
        return (negated + ['gradients = lm.load(dy[row, :], id="load_dy")'],
                'gradients * (decayed * -1.0 + 1.0)')
    body = negated + ['gate = lm.reciprocal(decayed + 1.0, id="reciprocal")']
    if operator == "swiglu_fp32":
        return (body + ['activated = values * gate',
                        'projected = lm.load(up[row, :], id="load_up")'],
                'activated * projected')
    return (body, 'values * gate')


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    """Emit high-level Python for the selected shape, with frozen Workload metadata.

    No task in this family reads a second coordinate, so no starter declares a reduction.
    Negation is spelled as a multiply because a literal is admitted only as the second
    binary operand.

    One operation body serves every route. Only the declared route, the Target and the
    tanh contract come from the device, which is the whole portability surface for this
    family: `coalesced=False` and the row-per-program map are admitted by Metal and Triton
    alike.
    """
    validate_activation_contract(workload.document)
    operator = workload.document["operator"]
    device = BACKENDS[backend_for_target(workload.target)]
    args = workload.tensor_abi(case_id)
    primary = args[0].name
    body, result = _body(operator, primary, device["tanh_contract"])
    body.append(f'lm.store(out[row, :], {result}, coalesced=False, id="store_out")')
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
