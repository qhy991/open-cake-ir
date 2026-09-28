"""B300 column-tiled GEMV candidates for FlashInfer GEMM task 011."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract

from .gemm import TASKS, validate_contract


def column_tiled_source(workload: WorkloadContract, case_id: str = "primary", *,
                        columns_per_cta: int = 2, k_tile: int = 1024,
                        execution_groups: int = 4) -> str:
    """Reuse each A segment across adjacent B rows and reduce in FP32.

    K segments bound live register demand for the 011 reduction. Each
    output is rounded once, after all FP32 partial sums. This mapping only
    admits the official M=1 B300 workload; larger M needs its own search.
    """
    validate_contract(workload.document)
    if (workload.target != "sm_103a"
            or workload.document["operator"] != TASKS["fib_gemm_n28672_k4096"][0]):
        raise ValueError("B300 GEMV mapping requires FIB 011 on exact B300")
    args = workload.tensor_abi(case_id)
    if (args[0].shape != (1, 4096) or args[1].shape != (28672, 4096)
            or args[2].shape != (1, 28672)):
        raise ValueError("B300 GEMV mapping requires official M=1, N=28672, K=4096")
    if type(columns_per_cta) is not int or columns_per_cta not in (2, 4):
        raise ValueError("column-tiled GEMV admits 2 or 4 columns per CTA")
    if type(k_tile) is not int or k_tile not in (512, 1024, 2048):
        raise ValueError("column-tiled GEMV admits K tiles 512, 1024 or 2048")
    if type(execution_groups) is not int or execution_groups not in (1, 4, 8):
        raise ValueError("column-tiled GEMV admits 1, 4 or 8 execution groups")
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
            f'backend="triton", entry_point="cake_fib011_gemv_n{columns_per_cta}'
            f'_k{k_tile}_w{execution_groups}")\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            f'    compute = lm.role(execution_groups={list(range(execution_groups))!r})\n'
            '    row = lm.program(a, axis=0, dimension=0, tile=1)\n'
            f'    column = lm.program(b, axis=1, dimension=0, tile={columns_per_cta})\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')


def row_tiled_source(workload: WorkloadContract, case_id: str = "primary", *,
                     rows_per_cta: int = 2, k_tile: int = 1024,
                     execution_groups: int = 4) -> str:
    """Reuse one B row across two or four independent A-row reductions.

    This is a separate M=2/M=4 mapping. K segmentation bounds live vectors;
    each output accumulates in FP32 and rounds once. A row tile larger than M
    is legal only because the lowering masks the tail loads and stores.
    """
    validate_contract(workload.document)
    if (workload.target != "sm_103a"
            or workload.document["operator"] != TASKS["fib_gemm_n28672_k4096"][0]):
        raise ValueError("B300 row-tiled GEMV requires FIB 011 on exact B300")
    args = workload.tensor_abi(case_id)
    rows = args[0].shape[0]
    if (rows not in (2, 4) or args[0].shape != (rows, 4096)
            or args[1].shape != (28672, 4096)
            or args[2].shape != (rows, 28672)):
        raise ValueError("row-tiled GEMV requires official M=2/4, N=28672, K=4096")
    if type(rows_per_cta) is not int or rows_per_cta not in (2, 4):
        raise ValueError("row-tiled GEMV admits 2 or 4 rows per CTA")
    if type(k_tile) is not int or k_tile not in (512, 1024, 2048):
        raise ValueError("row-tiled GEMV admits K tiles 512, 1024 or 2048")
    if type(execution_groups) is not int or execution_groups not in (1, 4):
        raise ValueError("row-tiled GEMV admits 1 or 4 execution groups")
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
            f'products_{index} = a32_{index} * lm.broadcast(b32_{index}, axis=1)',
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
            f'@cake.schedule(name="{workload.workload_id}-gemv-m{rows_per_cta}'
            f'-k{k_tile}-w{execution_groups}", target="{workload.target}", '
            f'backend="triton", entry_point="cake_fib011_gemv_m{rows_per_cta}'
            f'_k{k_tile}_w{execution_groups}")\n'
            f'def candidate(lm, {", ".join(declarations)}):\n'
            f'    compute = lm.role(execution_groups={list(range(execution_groups))!r})\n'
            f'    row = lm.program(a, axis=0, dimension=0, tile={rows_per_cta})\n'
            '    column = lm.program(b, axis=1, dimension=0, tile=1)\n'
            '    with compute:\n        ' + '\n        '.join(body) + '\n')
