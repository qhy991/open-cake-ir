"""Readable baseline Schedule generation, bound to the task-owned Workload ABI."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract
from .workload import validate_gemm_contract


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    """Emit high-level Python for the selected shape, with frozen Workload metadata.

    The baseline keeps the whole B tile and its products resident, which is the plain
    reading of the definition and not a claim that it is the cheapest arrangement.
    """
    validate_gemm_contract(workload.document)
    operator = workload.document["operator"]
    args = workload.tensor_abi(case_id)
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
                    + (', mode="output")' if arg.mode == "output" else ')') for arg in args]
    body = ['a_row = lm.load(a[row, :], id="load_a")',
            'b_tile = lm.load(b[:, :], id="load_b")',
            'products = b_tile * lm.broadcast(a_row, axis=0)',
            'totals = lm.reduce(products, op="sum", axis=0, scope="cta", across_loop=False, id="sum_k")',
            'biases = lm.load(bias[:], id="load_bias")',
            'lm.store(out[row, :], totals + biases, coalesced=False, id="store_out")']
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}", target="{workload.target}",\n'
            f'               backend="metal", entry_point="cake_{operator}",\n'
            f'               metadata={{"workload_contract_sha256": "{workload.canonical_sha256}"}})\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            '    compute = lm.role(warps=[0])\n'
            '    row = lm.program(a, axis=0, dimension=0, tile=1)\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')
