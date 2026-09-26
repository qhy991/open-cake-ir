"""One-stage Cake candidate for the exact FlashInfer MLA paged-decode task."""
from __future__ import annotations

import math

from open_cake_ir.evaluation.workload import WorkloadContract

from . import attention
from .plan_authoring import PlanAuthor


TASK = "fib_mla_paged_decode_h16_ckv512_kpe64_ps1"


def author_plan(workload: WorkloadContract, case_id: str = "primary") -> PlanAuthor:
    """Fuse metadata, QK, softmax and V for one-query-per-batch MLA decode.

    The exact Workload and Target remain authoritative. The key axis is padded to
    a power of two, with invalid scores set to negative infinity; an empty segment
    writes zero output and negative-infinity LSE.
    """
    attention.validate_contract(workload.document)
    if (workload.target != "sm_103a"
            or workload.document["semantics"]["task"] != TASK):
        raise ValueError("fused MLA decode requires its exact B300 Workload")
    spec = attention.SPECS[TASK]
    axes = workload.case(case_id)["shape"]
    length = attention._power_two(axes["num_kv_indices"])
    plan = PlanAuthor(workload, case_id)
    body = [
        "with compute:",
        '    qi = lm.coordinate(source="program", name="q_row")',
        "    next_batch = qi + 1",
        "    start = lm.load(kv_indptr[lm.scalar_index(qi)])",
        "    end = lm.load(kv_indptr[lm.scalar_index(next_batch)])",
        "    valid = end - start",
        f'    positions = lm.coordinate(source="range", start=0, extent={length})',
        "    absolute = positions + start",
        "    pages = lm.load(kv_indices[absolute])",
        '    zero = lm.coordinate(source="range", start=0, extent=1)',
        "    qc = lm.load(q_nope[q_row, h_head, :])",
        "    qp = lm.load(q_pe[q_row, h_head, :])",
        "    kc = lm.load(ckv_cache[pages, lm.scalar_index(zero), :])",
        "    kp = lm.load(kpe_cache[pages, lm.scalar_index(zero), :])",
        "    qc32 = lm.cast(qc, to=\"fp32\")",
        "    qp32 = lm.cast(qp, to=\"fp32\")",
        "    kc32 = lm.cast(kc, to=\"fp32\")",
        "    kp32 = lm.cast(kp, to=\"fp32\")",
        "    products_c = kc32 * lm.broadcast(qc32, axis=1)",
        "    products_p = kp32 * lm.broadcast(qp32, axis=1)",
        '    dot_c = lm.reduce(products_c, op="sum", axis=1, across_loop=False)',
        '    dot_p = lm.reduce(products_p, op="sum", axis=1, across_loop=False)',
        "    dot = dot_c + dot_p",
        "    scale = lm.load(sm_scale[:])",
        "    scaled = dot * scale",
        '    keep = lm.compare(positions, valid, op="lt")',
        '    masked = lm.select(keep, scaled, "negative_infinity")',
        '    valid_row = lm.compare(valid, 0, op="gt")',
        '    maximum = lm.reduce(masked, op="max", axis=0, across_loop=False)',
        "    safe_maximum = lm.select(valid_row, maximum, 0.0)",
        "    shifted = masked - safe_maximum",
        "    exponentials = lm.exp(shifted)",
        '    total = lm.reduce(exponentials, op="sum", axis=0, across_loop=False)',
        "    safe_total = lm.select(valid_row, total, 1.0)",
        "    weights = exponentials / safe_total",
        "    log_total = lm.log2(safe_total)",
        f"    log_max = safe_maximum * {math.log2(math.e)!r}",
        "    logarithm = log_total + log_max",
        '    final_lse = lm.select(valid_row, logarithm, "negative_infinity")',
        "    weighted_values = kc32 * lm.broadcast(weights, axis=0)",
        '    accum = lm.reduce(weighted_values, op="sum", axis=0, across_loop=False)',
        '    rounded = lm.cast(accum, to="bf16")',
        "    lm.store(output[q_row, h_head, :], rounded, coalesced=False)",
        "    lm.store(lse[q_row, h_head], final_lse, coalesced=False)",
    ]
    plan.stage(
        "mla_fused_decode", list(spec["inputs"]), list(spec["outputs"]),
        [("q_row", "output", 0, 1), ("h_head", "output", 1, 1)], body,
    )
    return plan


def launch_plan(workload: WorkloadContract, case_id: str = "primary"):
    return author_plan(workload, case_id).finish()
