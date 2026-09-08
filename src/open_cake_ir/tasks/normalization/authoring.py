"""Readable baseline Schedule generation, bound to the task-owned Workload ABI."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract
from .workload import validate_normalization_contract


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    """Emit high-level Python for the selected shape, with frozen Workload metadata."""
    validate_normalization_contract(workload.document)
    operator = workload.document["operator"]
    args = workload.tensor_abi(case_id)
    width = args[0].shape[-1]
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
                    + (', mode="output")' if arg.mode == "output" else ')') for arg in args]
    body = ['values = lm.load(x[row, :], id="load_x")',
            'weights = lm.load(weight[:], id="load_weight")']
    operand = "values"
    if operator == "residual_rmsnorm_fp32":
        body += ['residual_values = lm.load(residual[row, :], id="load_residual")',
                 'combined = values + residual_values']
        operand = "combined"
    elif operator == "layernorm_fp32":
        body += ['biases = lm.load(bias[:], id="load_bias")',
                 'row_sum = lm.reduce(values, op="sum", axis=0, scope="cta", across_loop=False, id="sum_input")',
                 f'row_mean = row_sum / {float(width)!r}', 'centered = values - row_mean']
        operand = "centered"
    body += [f'squares = lm.square({operand}, id="square")',
             'square_sum = lm.reduce(squares, op="sum", axis=0, scope="cta", across_loop=False, id="sum_square")',
             f'mean_square = square_sum / {float(width)!r}',
             f'inverse = lm.rsqrt(mean_square + {workload.document["semantics"]["epsilon"]!r}, id="inverse")',
             f'normalized = {operand} * inverse', 'weighted = normalized * weights']
    result = "weighted"
    if operator == "layernorm_fp32":
        body.append('result = weighted + biases')
        result = "result"
    body.append(f'lm.store(out[row, :], {result}, coalesced=False, id="store_out")')
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}", target="{workload.target}",\n'
            f'               backend="metal", entry_point="cake_{operator}",\n'
            f'               metadata={{"workload_contract_sha256": "{workload.canonical_sha256}"}})\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            '    compute = lm.role(warps=[0])\n'
            '    row = lm.program(x, axis=0, dimension=0, tile=1)\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')
