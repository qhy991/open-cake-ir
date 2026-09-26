"""One-stage Cake candidates for exact B300 FlashInfer ragged GQA prefill."""
from __future__ import annotations

import math

from open_cake_ir.evaluation.workload import WorkloadContract

from . import attention
from .plan_authoring import PlanAuthor


TASK = "fib_gqa_ragged_prefill_causal_h32_kv4_d128"
FUSED_TASKS = (TASK, "fib_gqa_ragged_prefill_causal_h32_kv8_d128")


def author_plan(workload: WorkloadContract, case_id: str = "primary") -> PlanAuthor:
    """One CTA owns a query/head and the valid part of its causal KV segment.

    A nonempty segment with no visible key produces NaN output and -inf LSE;
    an empty segment produces zero output and -inf LSE, as the frozen oracle does.
    """
    attention.validate_contract(workload.document)
    task = workload.document["semantics"]["task"]
    if (workload.target != "sm_103a"
            or task not in FUSED_TASKS):
        raise ValueError("fused ragged GQA prefill requires an admitted B300 Workload")
    spec = attention.SPECS[task]
    axes = workload.case(case_id)["shape"]
    length = attention._power_two(axes["total_kv"])
    segments = attention._power_two(axes["len_indptr"] - 1)
    group = spec["constants"]["num_qo_heads"] // spec["constants"]["num_kv_heads"]
    plan = PlanAuthor(workload, case_id)
    body = [
        "with compute:",
        '    qi = lm.coordinate(source="program", name="q_row")',
        '    head = lm.coordinate(source="program", name="h_head")',
        f'    segments = lm.coordinate(source="range", start=0, extent={segments})',
        '    starts = lm.load(qo_indptr[segments])',
        '    preceding = lm.compare(starts, qi, op="le")',
        f'    inside = lm.compare(segments, {axes["len_indptr"] - 1}, op="lt")',
        '    flags = preceding * inside',
        '    count = lm.reduce(flags, op="sum", axis=0, across_loop=False)',
        '    batch_index = count - 1',
        '    next_batch = batch_index + 1',
        '    start = lm.load(kv_indptr[lm.scalar_index(batch_index)])',
        '    end = lm.load(kv_indptr[lm.scalar_index(next_batch)])',
        '    kv_length = end - start',
        '    query_start = lm.load(qo_indptr[lm.scalar_index(batch_index)])',
        '    query_end = lm.load(qo_indptr[lm.scalar_index(next_batch)])',
        '    query_length = query_end - query_start',
        '    local_query = qi - query_start',
        '    prefix = kv_length - query_length',
        '    causal_count = local_query + prefix + 1',
        '    shorter = lm.compare(causal_count, kv_length, op="lt")',
        '    bounded = lm.select(shorter, causal_count, kv_length)',
        '    positive = lm.compare(bounded, 0, op="gt")',
        '    valid = lm.select(positive, bounded, 0)',
        f'    positions = lm.coordinate(source="range", start=0, extent={length})',
        '    absolute = positions + start',
        '    keep = lm.compare(positions, valid, op="lt")',
        '    masked_absolute = lm.select(keep, absolute, -1)',
        f'    kv_head = head // {group}',
        '    query = lm.load(q[q_row, h_head, :])',
        '    key = lm.load(k[masked_absolute, lm.scalar_index(kv_head), :])',
        '    value = lm.load(v[masked_absolute, lm.scalar_index(kv_head), :])',
        '    query32 = lm.cast(query, to="fp32")',
        '    key32 = lm.cast(key, to="fp32")',
        '    value32 = lm.cast(value, to="fp32")',
        '    products = key32 * lm.broadcast(query32, axis=1)',
        '    dot = lm.reduce(products, op="sum", axis=1, across_loop=False)',
        '    scale = lm.load(sm_scale[:])',
        '    scaled = dot * scale',
        '    masked = lm.select(keep, scaled, "negative_infinity")',
        '    valid_row = lm.compare(valid, 0, op="gt")',
        '    maximum = lm.reduce(masked, op="max", axis=0, across_loop=False)',
        '    safe_maximum = lm.select(valid_row, maximum, 0.0)',
        '    shifted = masked - safe_maximum',
        '    exponentials = lm.exp(shifted)',
        '    total = lm.reduce(exponentials, op="sum", axis=0, across_loop=False)',
        '    safe_total = lm.select(valid_row, total, 1.0)',
        '    nonempty = lm.compare(kv_length, 0, op="gt")',
        '    denominator = lm.select(nonempty, total, 1.0)',
        '    weights = exponentials / denominator',
        '    log_total = lm.log2(safe_total)',
        f'    log_max = safe_maximum * {math.log2(math.e)!r}',
        '    logarithm = log_total + log_max',
        '    final_lse = lm.select(valid_row, logarithm, "negative_infinity")',
        '    weighted_values = value32 * lm.broadcast(weights, axis=0)',
        '    accum = lm.reduce(weighted_values, op="sum", axis=0, across_loop=False)',
        '    rounded = lm.cast(accum, to="bf16")',
        '    lm.store(output[q_row, h_head, :], rounded, coalesced=False)',
        '    lm.store(lse[q_row, h_head], final_lse, coalesced=False)',
    ]
    plan.stage(
        "gqa_fused_ragged_prefill", list(spec["inputs"]), list(spec["outputs"]),
        [("q_row", "output", 0, 1), ("h_head", "output", 1, 1)], body,
    )
    return plan


def launch_plan(workload: WorkloadContract, case_id: str = "primary"):
    return author_plan(workload, case_id).finish()
