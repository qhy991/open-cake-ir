"""One-pass B300 Cake mapping for the exact FIB 023 RMSNorm workload."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract

from .workload import SPECS, TASKS, validate_solx_fib_contract

TASK = "fib_rmsnorm_h1536"
PADDED_WIDTH = 2048


def padded_one_pass_source(workload: WorkloadContract,
                           case_id: str = "primary") -> str:
    """Keep one live row and mask its 512 padded lanes in Triton.

    A program column tile of 2048 is explicit because Triton arange requires
    a power-of-two span. Masked loads contribute zero to the reduction, its
    divisor remains the Workload's 1536, and the masked store writes only valid
    columns. The starter uses two exact spans and a second output read.
    """
    validate_solx_fib_contract(workload.document)
    if (workload.target != "sm_103a"
            or workload.document["operator"] != TASKS[TASK][0]):
        raise ValueError("padded FIB 023 mapping requires the exact B300 task")
    x, weight, out = workload.tensor_abi(case_id)
    rows = x.shape[0]
    if (rows not in SPECS[TASK]["batches"]
            or x.shape != (rows, 1536) or weight.shape != (1536,)
            or out.shape != (rows, 1536)
            or (x.dtype, weight.dtype, out.dtype) != ("bf16", "bf16", "bf16")):
        raise ValueError("padded FIB 023 mapping requires the official BF16 ABI")
    epsilon = workload.document["semantics"]["epsilon"]
    return (
        'from open_cake_ir.compiler import frontend as cake\n\n'
        f'@cake.schedule(name="{workload.workload_id}-padded-one-pass", '
        f'target="{workload.target}", backend="triton", '
        'entry_point="cake_fib023_padded_one_pass")\n'
        'def candidate(lm, '
        f'x: cake.Tensor(({rows}, 1536), "bf16"), '
        'weight: cake.Tensor((1536,), "bf16"), '
        f'out: cake.Tensor(({rows}, 1536), "bf16", mode="output")):\n'
        '    compute = lm.role(execution_groups=[0, 1, 2, 3])\n'
        '    row = lm.program(x, axis=0, dimension=0, tile=1)\n'
        f'    column = lm.program(x, axis=1, dimension=1, tile={PADDED_WIDTH})\n'
        '    with compute:\n'
        '        stored = lm.load(x[row, column], id="load_x")\n'
        '        values = lm.cast(stored, to="fp32", id="widen_x")\n'
        '        squares = lm.square(values, id="square")\n'
        '        square_sum = lm.reduce(squares, op="sum", axis=0, '
        'scope="cta", across_loop=False, id="sum_square")\n'
        '        mean_square = square_sum / 1536.0\n'
        f'        inverse = lm.rsqrt(mean_square + {epsilon!r}, id="inverse")\n'
        '        stored_weight = lm.load(weight[column], id="load_weight")\n'
        '        weights = lm.cast(stored_weight, to="fp32", id="widen_weight")\n'
        '        normalized = values * inverse\n'
        '        weighted = normalized * weights\n'
        '        narrowed = lm.cast(weighted, to="bf16", id="narrow_out")\n'
        '        lm.store(out[row, column], narrowed, '
        'coalesced=False, id="store_out")\n'
    )
