"""B300 FP16 tensor-core Cake candidates for FlashInfer GEMM task 004."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract

from .gemm import TASKS, validate_contract


_TENSOR_CORE_M = frozenset((16, 17, 25, 32, 34, 63, 64, 93, 128, 172,
                            289, 492, 952, 8828, 11006, 12251, 12853,
                            14915, 16294))


def tensorcore_source(workload: WorkloadContract, case_id: str = "primary", *,
                      block_m: int = 16, block_n: int = 32,
                      block_k: int = 64, num_stages: int = 2,
                      num_warps: int = 4) -> str:
    """Tile the full FP16 contraction through the exact B300 Triton dot contract.

    The Workload fixes N=128, K=2048 and the final FP16 rounding. The Compiler
    owns the operation graph and hardware admission; device evidence decides
    whether this mapping is faster than the selected task implementation.
    """
    validate_contract(workload.document)
    if (workload.target != "sm_103a"
            or workload.document["operator"] != TASKS["fib_gemm_n128_k2048"][0]):
        raise ValueError("tensor-core mapping requires FIB 004 on exact B300")
    args = workload.tensor_abi(case_id)
    rows = args[0].shape[0]
    if (rows not in _TENSOR_CORE_M or args[0].shape != (rows, 2048)
            or args[1].shape != (128, 2048)
            or args[2].shape != (rows, 128)):
        raise ValueError("tensor-core mapping requires an official M>=16, N=128, K=2048")
    if (block_m, block_n, block_k, num_stages, num_warps) not in (
            (16, 32, 64, 2, 4), (16, 32, 256, 2, 4),
            (16, 32, 512, 2, 4), (16, 32, 1024, 1, 4),
            (16, 16, 512, 2, 4), (16, 64, 512, 2, 4),
            (32, 64, 256, 2, 4), (64, 64, 256, 2, 4),
            (64, 128, 256, 2, 4), (64, 64, 256, 2, 8),
            (64, 128, 256, 2, 8)):
        raise ValueError("tensor-core tile/stage choice is outside the bounded study")
    warp_suffix = f"-w{num_warps}" if num_warps != 4 else ""
    declarations = [
        f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
        + (', mode="output")' if arg.mode == "output" else ')')
        for arg in args
    ]
    return (
        'from open_cake_ir.compiler import frontend as cake\n\n'
        f'@cake.schedule(name="{workload.workload_id}-tensorcore-{block_m}x{block_n}x{block_k}-s{num_stages}{warp_suffix}", '
        f'target="{workload.target}", backend="triton", '
        'entry_point="cake_fib004_tensorcore")\n'
        f'def candidate(lm, {", ".join(declarations)}):\n'
        f'    compute = lm.role(execution_groups={list(range(num_warps))!r})\n'
        f'    row = lm.program(a, axis=0, dimension=0, tile={block_m})\n'
        f'    column = lm.program(b, axis=1, dimension=0, tile={block_n})\n'
        f'    for k in lm.range(a, name="k_loop", dimension=1, tile={block_k}, '
        f'num_stages={num_stages}, disallow_acc_multi_buffer=True):\n'
        '        with compute:\n'
        '            a_tile = lm.load(a[row, k], id="load_a")\n'
        '            b_tile = lm.load(b[column, k], id="load_b")\n'
        '            acc = lm.mma(a_tile, b_tile, '
        'instruction={"contract": "triton.dot.fp16_fp32"}, '
        f'tile_shape=({block_m}, {block_n}, {block_k}), id="dot")\n'
        '    with compute:\n'
        '        rounded = lm.cast(acc, to="fp16", id="round_out")\n'
        '        lm.store(out[row, column], rounded, id="store_out")\n'
    )


def n32_k512_development_lead_source(
        workload: WorkloadContract, case_id: str = "primary") -> str:
    """Apply the measured M128/M172 recipe; reject unqualified shape transfer.

    Both shapes passed the task-pack workload, all five frozen project input
    distributions and two independent quality-passed paired assays on B300.
    This is a shape-level authoring recipe, not a complete-task dispatcher.
    """
    validate_contract(workload.document)
    rows = workload.tensor_abi(case_id)[0].shape[0]
    if rows not in (128, 172):
        raise ValueError("N32 K512 development lead is qualified only at M128 and M172")
    return tensorcore_source(workload, case_id, block_n=32, block_k=512)
