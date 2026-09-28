"""Exact NVIDIA Cake row-group candidates for FlashInfer RMSNorm task 021."""
from __future__ import annotations

from open_cake_ir.evaluation.workload import WorkloadContract

from .workload import SPECS, TASKS, validate_solx_fib_contract

TASK = "fib_rmsnorm_h128"
ROW_GROUPS = frozenset((4, 8, 16, 32, 64))
LARGE_ROWS = frozenset((49532, 65016, 396256, 520128))
TARGETS = frozenset(("sm_100a", "sm_103a"))
QUALIFIED_T16_ROWS = frozenset((49532, 65016, 520128))


def row_group_source(workload: WorkloadContract, case_id: str = "primary", *,
                     rows_per_cta: int = 4) -> str:
    """Reuse one 128-element weight vector across several independent rows.

    The row tile is explicit, including a masked tail at batch boundaries.
    Each row retains its own FP32 sum and rsqrt before one BF16 output cast.
    This mapping changes CTA count and reuse, not Workload semantics.
    """
    validate_solx_fib_contract(workload.document)
    if (workload.target not in TARGETS
            or workload.document["operator"] != TASKS[TASK][0]):
        raise ValueError("row-group FIB 021 mapping requires the exact B200 or B300 task")
    if type(rows_per_cta) is not int or rows_per_cta not in ROW_GROUPS:
        raise ValueError("FIB 021 row group must be 4, 8, 16, 32 or 64")
    x, weight, out = workload.tensor_abi(case_id)
    rows = x.shape[0]
    if (rows not in SPECS[TASK]["batches"]
            or x.shape != (rows, 128) or weight.shape != (128,)
            or out.shape != (rows, 128)
            or (x.dtype, weight.dtype, out.dtype) != ("bf16", "bf16", "bf16")):
        raise ValueError("row-group FIB 021 mapping requires the official BF16 ABI")
    if rows_per_cta >= 32 and rows not in LARGE_ROWS:
        raise ValueError("FIB 021 T32/T64 mapping is bounded to four official large batches")
    epsilon = workload.document["semantics"]["epsilon"]
    return (
        'from open_cake_ir.compiler import frontend as cake\n\n'
        f'@cake.schedule(name="{workload.workload_id}-row-group-{rows_per_cta}", '
        f'target="{workload.target}", backend="triton", '
        f'entry_point="cake_fib021_rowgroup_{rows_per_cta}")\n'
        'def candidate(lm, '
        f'x: cake.Tensor(({rows}, 128), "bf16"), '
        'weight: cake.Tensor((128,), "bf16"), '
        f'out: cake.Tensor(({rows}, 128), "bf16", mode="output")):\n'
        '    compute = lm.role(execution_groups=[0, 1, 2, 3])\n'
        f'    row = lm.program(x, axis=0, dimension=0, tile={rows_per_cta})\n'
        '    column = lm.program(x, axis=1, dimension=1, tile=128)\n'
        '    with compute:\n'
        '        stored = lm.load(x[row, column], id="load_x")\n'
        '        values = lm.cast(stored, to="fp32", id="widen_x")\n'
        '        squares = lm.square(values, id="square")\n'
        '        totals = lm.reduce(squares, op="sum", axis=1, '
        'scope="cta", across_loop=False, id="sum_square")\n'
        '        mean_square = totals / 128.0\n'
        f'        inverse = lm.rsqrt(mean_square + {epsilon!r}, id="inverse")\n'
        '        stored_weight = lm.load(weight[column], id="load_weight")\n'
        '        weights = lm.cast(stored_weight, to="fp32", id="widen_weight")\n'
        '        normalized = values * lm.broadcast(inverse, axis=0)\n'
        '        weighted = normalized * lm.broadcast(weights, axis=1)\n'
        '        narrowed = lm.cast(weighted, to="bf16", id="narrow_out")\n'
        '        lm.store(out[row, column], narrowed, '
        'coalesced=False, id="store_out")\n'
    )


def qualified_rowgroup16_source(workload: WorkloadContract,
                                 case_id: str = "primary") -> str:
    """Select only the three measured large-batch T16 development leads.

    Each passed task-pack correctness, five input distributions with all
    elements checked, and two independent quality-passed 10/10 paired assays.
    This is a shape-level Lab recipe, not a complete-task dispatcher.
    """
    validate_solx_fib_contract(workload.document)
    if workload.target != "sm_103a":
        raise ValueError("FIB 021 T16 measured development leads require B300")
    rows = workload.tensor_abi(case_id)[0].shape[0]
    if rows not in QUALIFIED_T16_ROWS:
        raise ValueError("FIB 021 T16 development lead is qualified only at R49532, R65016 and R520128")
    return row_group_source(workload, case_id, rows_per_cta=16)
