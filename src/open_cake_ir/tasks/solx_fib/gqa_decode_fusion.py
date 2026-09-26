"""One-stage Cake candidates for the two FlashInfer GQA paged-decode tasks."""
from __future__ import annotations

import math

from open_cake_ir.evaluation.workload import WorkloadContract

from . import attention
from .plan_authoring import PlanAuthor


TASK = "fib_gqa_paged_decode_h32_kv4_d128_ps1"
FUSED_TASKS = (TASK, "fib_gqa_paged_decode_h32_kv8_d128_ps1")


def author_plan(workload: WorkloadContract, case_id: str = "primary") -> PlanAuthor:
    """One CTA owns a query/head and its padded KV axis on exact B300 GQA decode.

    Scores, softmax and weighted values share the same loaded page data. Invalid
    keys contribute zero; an empty segment writes zero output and -inf LSE.
    """
    attention.validate_contract(workload.document)
    task = workload.document["semantics"]["task"]
    if workload.target != "sm_103a" or task not in FUSED_TASKS:
        raise ValueError("fused GQA decode requires an admitted B300 Workload")
    spec = attention.SPECS[task]
    axes = workload.case(case_id)["shape"]
    length = attention._power_two(axes["num_kv_indices"])
    group = spec["constants"]["num_qo_heads"] // spec["constants"]["num_kv_heads"]
    plan = PlanAuthor(workload, case_id)
    body = [
        "with compute:",
        '    qi = lm.coordinate(source="program", name="q_row")',
        '    head = lm.coordinate(source="program", name="h_head")',
        "    next_batch = qi + 1",
        "    start = lm.load(kv_indptr[lm.scalar_index(qi)])",
        "    end = lm.load(kv_indptr[lm.scalar_index(next_batch)])",
        "    valid = end - start",
        f'    positions = lm.coordinate(source="range", start=0, extent={length})',
        "    absolute = positions + start",
        "    pages = lm.load(kv_indices[absolute])",
        '    zero = lm.coordinate(source="range", start=0, extent=1)',
        f"    kv_head = head // {group}",
        "    query = lm.load(q[q_row, h_head, :])",
        "    key = lm.load(k_cache[pages, lm.scalar_index(zero), "
        "lm.scalar_index(kv_head), :])",
        "    value = lm.load(v_cache[pages, lm.scalar_index(zero), "
        "lm.scalar_index(kv_head), :])",
        '    query32 = lm.cast(query, to="fp32")',
        '    key32 = lm.cast(key, to="fp32")',
        '    value32 = lm.cast(value, to="fp32")',
        "    products = key32 * lm.broadcast(query32, axis=1)",
        '    dot = lm.reduce(products, op="sum", axis=1, across_loop=False)',
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
        "    weighted_values = value32 * lm.broadcast(weights, axis=0)",
        '    accum = lm.reduce(weighted_values, op="sum", axis=0, across_loop=False)',
        '    rounded = lm.cast(accum, to="bf16")',
        "    lm.store(output[q_row, h_head, :], rounded, coalesced=False)",
        "    lm.store(lse[q_row, h_head], final_lse, coalesced=False)",
    ]
    plan.stage(
        "gqa_fused_decode", list(spec["inputs"]), list(spec["outputs"]),
        [("q_row", "output", 0, 1), ("h_head", "output", 1, 1)], body,
    )
    return plan


def launch_plan(workload: WorkloadContract, case_id: str = "primary"):
    return author_plan(workload, case_id).finish()
