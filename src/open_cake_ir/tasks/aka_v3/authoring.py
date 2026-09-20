"""Complete starters for the AKA contracts the existing tensor path can express."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, backend_for_target
from .workload import _admit_starter, _name_for, validate_aka_v3_contract


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    validate_aka_v3_contract(workload.document)
    name = _name_for(workload.document)
    shape = workload.case(case_id)["shape"]
    backend = backend_for_target(workload.target)
    _admit_starter(name, backend, columns=shape.get("C", shape.get("N", 1)),
                   depth=shape.get("K", 1), elements=shape.get("E", 1))
    args = workload.tensor_abi(case_id)
    if name == "residual_layernorm":
        program = 'row = lm.program(input_a, axis=0, dimension=0, tile=1)'
        body = [
            'left = lm.load(input_a[row, :], id="load_a")',
            'right = lm.load(input_b[row, :], id="load_b")',
            'weights = lm.load(weight[:], id="load_weight")',
            'biases = lm.load(bias[:], id="load_bias")',
            'combined = left + right',
            'total = lm.reduce(combined, op="sum", axis=0, scope="cta", across_loop=False, id="sum_input")',
            f'average = total / {float(shape["C"])!r}',
            'centered = combined - average',
            'squares = lm.square(centered, id="square")',
            'square_sum = lm.reduce(squares, op="sum", axis=0, scope="cta", across_loop=False, id="sum_square")',
            f'variance = square_sum / {float(shape["C"])!r}',
            f'inverse = lm.rsqrt(variance + {workload.document["semantics"]["epsilon"]!r}, id="rsqrt")',
            'result = centered * inverse * weights + biases',
            'lm.store(residual[row, :], combined, coalesced=False, id="store_residual")',
            'lm.store(normalized[row, :], result, coalesced=False, id="store_normalized")',
            'lm.store(mean[row], average, coalesced=False, id="store_mean")',
            'lm.store(reciprocal_stddev[row], inverse, coalesced=False, id="store_inverse")',
        ]
    elif name == "gemm_nt_bias":
        program = 'row = lm.program(input, axis=0, dimension=0, tile=1)'
        body = [
            'left = lm.load(input[row, :], id="load_input")',
            'right = lm.load(weight_nt[:, :], id="load_weight")',
            'biases = lm.load(bias[:], id="load_bias")',
            'products = right * lm.broadcast(left, axis=1)',
            'totals = lm.reduce(products, op="sum", axis=1, scope="cta", across_loop=False, id="sum_k")',
            'lm.store(output[row, :], totals + biases, coalesced=False, id="store_output")',
        ]
    elif name == "row_gather":
        program = 'row = lm.program(output, axis=0, dimension=0, tile=1)'
        body = [
            'indices = lm.load(output_row_to_input_row[row], id="load_index")',
            'selected = lm.scalar_index(indices)',
            'values = lm.load(input[selected, :], id="load_selected_row")',
            'lm.store(output[row, :], values, coalesced=False, id="store_output")',
        ]
    else:  # _admit_starter has closed this branch to momentum_sgd.
        tile = min(256, 1 << (shape["E"].bit_length() - 1))
        program = f'block = lm.program(param, axis=0, dimension=0, tile={tile})'
        body = [
            'parameters = lm.load(param[block], id="load_param")',
            'gradients = lm.load(grad[block], id="load_grad")',
            'moments = lm.load(moment[block], id="load_moment")',
            'momentum = lm.load(mu[:], id="load_mu")',
            'rate = lm.load(lr[:], id="load_lr")',
            'flag_int = lm.load(nesterov[:], id="load_nesterov")',
            'flag = lm.cast(flag_int, to="fp32", id="cast_nesterov")',
            'updated = moments * momentum + gradients * rate',
            'ordinary = parameters - updated',
            'accelerated = parameters - updated * (momentum + 1.0) + moments * momentum',
            # The Workload bounds the flag to 0/1 and every intermediate is finite.
            # Multiplication by those exact values preserves the chosen branch without
            # asking a Target for an unadmitted compare/select operation.
            'off = flag * -1.0 + 1.0',
            'result = ordinary * off + accelerated * flag',
            'lm.store(param_out[block], result, coalesced=False, id="store_param")',
            'lm.store(moment_out[block], updated, coalesced=False, id="store_moment")',
        ]
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
                    + (', mode="output")' if arg.mode == "output" else ')') for arg in args]
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}", target="{workload.target}",\n'
            f'               backend="{BACKENDS[backend]["route"]}", entry_point="cake_{workload.document["operator"]}",\n'
            f'               metadata={{"workload_contract_sha256": "{workload.canonical_sha256}"}})\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            '    compute = lm.role(execution_groups=[0])\n'
            f'    {program}\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')
