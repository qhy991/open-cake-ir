"""Bounded B300 FP16 tensor-core Cake candidates for FlashInfer GEMM 009."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract

from .gemm import SPECS, TASKS, validate_contract


TASK = "fib_gemm_n5120_k2048"
TILES = frozenset(((16, 64, 256), (32, 64, 256), (64, 128, 256),
                   (64, 64, 64), (64, 64, 128), (64, 64, 256)))


def tensorcore_source(workload: WorkloadContract, case_id: str = "primary", *,
                      block_m: int = 16, block_n: int = 64,
                      block_k: int = 256) -> str:
    """Map a complete FP16 GEMM contraction to the admitted B300 dot contract.

    The grid owns disjoint output tiles. The target owns FP16 dot admission,
    while the Workload owns N, K, the tensor ABI and final FP16 rounding.
    """
    validate_contract(workload.document)
    if (workload.target != "sm_103a"
            or workload.document["operator"] != TASKS[TASK][0]):
        raise ValueError("FIB 009 tensor-core mapping requires its exact B300 Workload")
    if ((type(block_m), type(block_n), type(block_k)) != (int, int, int)
            or (block_m, block_n, block_k) not in TILES):
        raise ValueError("FIB 009 tile is outside the bounded tensor-core study")
    a, b, out = workload.tensor_abi(case_id)
    rows = a.shape[0]
    if (rows not in SPECS[TASK]["batches"] or rows < 16
            or a.shape != (rows, 2048) or b.shape != (5120, 2048)
            or out.shape != (rows, 5120)
            or (a.dtype, b.dtype, out.dtype) != ("fp16", "fp16", "fp16")):
        raise ValueError("FIB 009 tensor-core mapping requires an official FP16 M>=16 ABI")
    declarations = [
        f'{arg.name}: cake.Tensor({arg.shape!r}, "{arg.dtype}"'
        + (', mode="output")' if arg.mode == "output" else ')')
        for arg in (a, b, out)
    ]
    return (
        'from open_cake_ir.compiler import frontend as cake\n\n'
        f'@cake.schedule(name="{workload.workload_id}-tensorcore-{block_m}x{block_n}x{block_k}", '
        f'target="{workload.target}", backend="triton", '
        f'entry_point="cake_fib009_tensorcore_{block_m}x{block_n}x{block_k}")\n'
        f'def candidate(lm, {", ".join(declarations)}):\n'
        '    compute = lm.role(execution_groups=[0, 1, 2, 3])\n'
        f'    row = lm.program(a, axis=0, dimension=0, tile={block_m})\n'
        f'    column = lm.program(b, axis=1, dimension=0, tile={block_n})\n'
        f'    for k in lm.range(a, name="k_loop", dimension=1, tile={block_k}, '
        'num_stages=2, disallow_acc_multi_buffer=True):\n'
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
