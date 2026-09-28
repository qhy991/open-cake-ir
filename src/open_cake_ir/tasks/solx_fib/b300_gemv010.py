"""B300 column-tiled GEMV candidates for FlashInfer GEMM task 010."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract

from .gemm import TASKS, validate_contract


def column_tiled_source(workload: WorkloadContract, case_id: str = "primary", *,
                        columns_per_cta: int = 2, k_tile: int = 1024,
                        execution_groups: int = 4) -> str:
    """Reuse each A segment across adjacent B rows and reduce in FP32.

    K segments bound live register demand for the longer 010 reduction. Each
    output is rounded once, after all FP32 partial sums. This mapping only
    admits the official M=1 B300 workload; larger M needs its own search.
    """
    validate_contract(workload.document)
    if (workload.target != "sm_103a"
            or workload.document["operator"] != TASKS["fib_gemm_n6144_k4096"][0]):
        raise ValueError("B300 GEMV mapping requires FIB 010 on exact B300")
    args = workload.tensor_abi(case_id)
    if (args[0].shape != (1, 4096) or args[1].shape != (6144, 4096)
            or args[2].shape != (1, 6144)):
        raise ValueError("B300 GEMV mapping requires official M=1, N=6144, K=4096")
    if type(columns_per_cta) is not int or columns_per_cta not in (2, 4):
        raise ValueError("column-tiled GEMV admits 2 or 4 columns per CTA")
    if type(k_tile) is not int or k_tile not in (512, 1024, 2048):
        raise ValueError("column-tiled GEMV admits K tiles 512, 1024 or 2048")
    if type(execution_groups) is not int or execution_groups not in (4, 8):
        raise ValueError("column-tiled GEMV admits 4 or 8 execution groups")
    declarations = [f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
                    + (', mode="output")' if arg.mode == "output" else ')')
                    for arg in args]
    body = []
    partials = []
    for index, start in enumerate(range(0, 4096, k_tile)):
        stop = start + k_tile
        body.extend((
            f'a_{index} = lm.load(a[row, {start}:{stop}], id="load_a_{index}")',
            f'b_{index} = lm.load(b[column, {start}:{stop}], id="load_b_{index}")',
            f'a32_{index} = lm.cast(a_{index}, to="fp32", id="widen_a_{index}")',
            f'b32_{index} = lm.cast(b_{index}, to="fp32", id="widen_b_{index}")',
            f'products_{index} = b32_{index} * lm.broadcast(a32_{index}, axis=1)',
            f'partial_{index} = lm.reduce(products_{index}, op="sum", axis=1, '
            f'scope="cta", across_loop=False, id="sum_k_{index}")',
        ))
        partials.append(f'partial_{index}')
    body.extend((
        'total = ' + ' + '.join(partials),
        'rounded = lm.cast(total, to="fp16", id="round_out")',
        'lm.store(out[row, column], rounded, coalesced=False, id="store_out")',
    ))
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            f'@cake.schedule(name="{workload.workload_id}-gemv-n{columns_per_cta}'
            f'-k{k_tile}-w{execution_groups}", target="{workload.target}", '
            f'backend="triton", entry_point="cake_fib010_gemv_n{columns_per_cta}'
            f'_k{k_tile}_w{execution_groups}")\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            f'    compute = lm.role(execution_groups={list(range(execution_groups))!r})\n'
            '    row = lm.program(a, axis=0, dimension=0, tile=1)\n'
            f'    column = lm.program(b, axis=1, dimension=0, tile={columns_per_cta})\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')
