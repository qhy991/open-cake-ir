"""Readable baseline Schedule generation, bound to the task-owned Workload ABI.

Each baseline keeps the whole contracted operand register-resident. That is the plain
reading of the definition and not a claim that it is the cheapest arrangement; it is the
arrangement whose cost a tiled contraction would improve on, which is the search.
"""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, backend_for_target
from .workload import validate_contraction_contract


def _body(operator: str, depth: int) -> tuple[list[str], str]:
    """Return one task's operation statements and the expression its store writes.

    `lm.broadcast(v, axis=a)` names the axis of the result the narrower operand *spans*,
    so a row of `a` spans the contracted axis of a `[K, N]` tile while a per-key weight
    spans the key axis of an `[N, K]` tile.
    """
    if operator == "contraction_pairwise_sqdist_fp32":
        return (['point = lm.load(x[row, :], id="load_x")',
                 'centroids = lm.load(c[:, :], id="load_c")',
                 # The contracted operand is a difference, so no fused multiply-accumulate
                 # instruction can serve this contraction the way it serves a GEMM.
                 'deltas = centroids - lm.broadcast(point, axis=1)',
                 'squares = lm.square(deltas, id="square")'],
                'lm.reduce(squares, op="sum", axis=1, scope="cta",'
                ' across_loop=False, id="sum_k")')
    if operator == "contraction_attention_decode_fp32":
        return (['query = lm.load(q[row, :], id="load_q")',
                 'keys = lm.load(k[:, :], id="load_k")',
                 'values = lm.load(v[:, :], id="load_v")',
                 'scores = keys * lm.broadcast(query, axis=1)',
                 'raw = lm.reduce(scores, op="sum", axis=1, scope="cta",'
                 ' across_loop=False, id="sum_qk")',
                 f'logits = raw * {depth ** -0.5!r}',
                 'peak = lm.reduce(logits, op="max", axis=0, scope="cta",'
                 ' across_loop=False, id="max_logit")',
                 'shifted = logits - peak',
                 'weights = lm.exp(shifted, id="exp")',
                 'total = lm.reduce(weights, op="sum", axis=0, scope="cta",'
                 ' across_loop=False, id="sum_exp")',
                 'weighted = values * lm.broadcast(weights, axis=0)',
                 'context = lm.reduce(weighted, op="sum", axis=0, scope="cta",'
                 ' across_loop=False, id="sum_av")'],
                'context / total')
    body = ['a_row = lm.load(a[row, :], id="load_a")',
            'b_tile = lm.load(b[:, :], id="load_b")',
            'biases = lm.load(bias[:], id="load_bias")',
            'products = b_tile * lm.broadcast(a_row, axis=0)',
            'totals = lm.reduce(products, op="sum", axis=0, scope="cta",'
            ' across_loop=False, id="sum_k")',
            'shifted = totals + biases']
    if operator == "contraction_gemm_silu_fp32":
        # The epilogue reads the accumulator before it leaves the kernel, which is the
        # scheduling question this task adds to the plain contraction.
        return (body + ['negated = shifted * -1.0',
                        'decayed = lm.exp(negated, id="exp")',
                        'gate = lm.reciprocal(decayed + 1.0, id="reciprocal")'],
                'shifted * gate')
    return (body, 'shifted')


def starter_source(workload: WorkloadContract, case_id: str = "primary") -> str:
    """Emit high-level Python for the selected shape, with frozen Workload metadata.

    No task here names an instruction contract, so one operation body serves every route;
    only the declared route and Target come from the device.
    """
    validate_contraction_contract(workload.document)
    operator = workload.document["operator"]
    device = BACKENDS[backend_for_target(workload.target)]
    args = workload.tensor_abi(case_id)
    primary = args[0].name
    depth = workload.case(case_id)["shape"]["K"]
    body, result = _body(operator, depth)
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
