"""Readable baseline Schedule generation, bound to the task-owned Workload ABI."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, backend_for_target
from . import workload as math_module
from .workload import validate_optimizers_contract


def _body(operator: str) -> list[str]:
    """Return one step's statements including its own stores.

    Every task writes each accumulator it read, so the store schedule is part of the body.
    Adadelta needs a square root the vocabulary does not have, and takes it as the operand
    times its own reciprocal square root; the added epsilon keeps that operand strictly
    positive, so the product is never zero times an infinity.
    """
    if operator == "momentum_sgd_fp32":
        return ['parameters = lm.load(param[row, :], id="load_param")',
                'gradients = lm.load(grad[row, :], id="load_grad")',
                'moments = lm.load(moment[row, :], id="load_moment")',
                f'updated = moments * {math_module.MOMENTUM!r} + gradients * {math_module.LEARNING_RATE!r}',
                'lm.store(moment_out[row, :], updated, coalesced=False, id="store_moment")',
                'lm.store(param_out[row, :], parameters - updated, coalesced=False, id="store_param")']
    if operator == "adamw_fp32":
        return ['parameters = lm.load(param[row, :], id="load_param")',
                'gradients = lm.load(grad[row, :], id="load_grad")',
                'moments = lm.load(first[row, :], id="load_first")',
                'squares = lm.load(second[row, :], id="load_second")',
                'gradient_squares = lm.square(gradients, id="square_grad")',
                f'first_next = moments * {math_module.ADAM_BETA1!r} + gradients * {1.0 - math_module.ADAM_BETA1!r}',
                f'second_next = squares * {math_module.ADAM_BETA2!r} + gradient_squares * {1.0 - math_module.ADAM_BETA2!r}',
                f'corrected = second_next / {math_module.ADAM_BIAS2!r} + {math_module.ADAM_EPSILON!r}',
                'inverse = lm.rsqrt(corrected, id="rsqrt")',
                f'ratio = first_next / {math_module.ADAM_BIAS1!r} * inverse',
                f'decayed = parameters * {1.0 - math_module.ADAM_LEARNING_RATE * math_module.ADAM_WEIGHT_DECAY!r}',
                'lm.store(first_out[row, :], first_next, coalesced=False, id="store_first")',
                'lm.store(second_out[row, :], second_next, coalesced=False, id="store_second")',
                f'lm.store(param_out[row, :], decayed - ratio * {math_module.ADAM_LEARNING_RATE!r},'
                ' coalesced=False, id="store_param")']
    return ['parameters = lm.load(param[row, :], id="load_param")',
            'gradients = lm.load(grad[row, :], id="load_grad")',
            'squares = lm.load(square_accumulator[row, :], id="load_square")',
            'updates = lm.load(update_accumulator[row, :], id="load_update")',
            'gradient_squares = lm.square(gradients, id="square_grad")',
            f'square_next = squares * {math_module.ADADELTA_DECAY!r} + '
            f'gradient_squares * {1.0 - math_module.ADADELTA_DECAY!r}',
            f'shifted_update = updates + {math_module.ADADELTA_EPSILON!r}',
            f'shifted_square = square_next + {math_module.ADADELTA_EPSILON!r}',
            'update_inverse = lm.rsqrt(shifted_update, id="rsqrt_update")',
            'square_inverse = lm.rsqrt(shifted_square, id="rsqrt_square")',
            # sqrt(z) is z * rsqrt(z); the epsilon keeps z strictly positive.
            'update_root = shifted_update * update_inverse',
            'step = update_root * square_inverse * gradients',
            'step_squares = lm.square(step, id="square_step")',
            f'update_next = updates * {math_module.ADADELTA_DECAY!r} + '
            f'step_squares * {1.0 - math_module.ADADELTA_DECAY!r}',
            'lm.store(square_out[row, :], square_next, coalesced=False, id="store_square")',
            'lm.store(update_out[row, :], update_next, coalesced=False, id="store_update")',
            f'lm.store(param_out[row, :], parameters + step * {math_module.ADADELTA_LEARNING_RATE!r},'
            ' coalesced=False, id="store_param")']


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    """Emit high-level Python for the selected shape, with frozen Workload metadata.

    No task here names an instruction contract, so one body serves every route; only the
    declared route and Target come from the device.
    """
    validate_optimizers_contract(workload.document)
    operator = workload.document["operator"]
    device = BACKENDS[backend_for_target(workload.target)]
    args = workload.tensor_abi(case_id)
    primary = args[0].name
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
                    + (', mode="output")' if arg.mode == "output" else ')') for arg in args]
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}", target="{workload.target}",\n'
            f'               backend="{device["route"]}", entry_point="cake_{operator}",\n'
            f'               metadata={{"workload_contract_sha256": "{workload.canonical_sha256}"}})\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            '    compute = lm.role(warps=[0])\n'
            f'    row = lm.program({primary}, axis=0, dimension=0, tile=1)\n'
            '    with compute:\n        ' + '\n        '.join(_body(operator)) + '\n')
