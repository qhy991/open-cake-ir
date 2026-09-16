"""Readable baseline Schedule generation, bound to the task-owned Workload ABI."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, backend_for_target
from .workload import validate_solx_fib_contract


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    """Emit high-level Python for the selected shape, with frozen Workload metadata.

    The BF16 ABI is explicit on both sides: every operand widens on load, the reduction
    and the affine scale happen in FP32, and one narrowing cast produces the result. That
    is what the upstream definition's reference does and what the tolerance assumes.
    """
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
    device = BACKENDS[backend_for_target(workload.target)]
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
                    + (', mode="output")' if arg.mode == "output" else ')') for arg in args]
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}", target="{workload.target}",\n'
            f'               backend="{device["route"]}", entry_point="cake_{operator}",\n'
            f'               metadata={{"workload_contract_sha256": "{workload.canonical_sha256}"}})\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            '    compute = lm.role(warps=[0])\n'
            '    row = lm.program(x, axis=0, dimension=0, tile=1)\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')
