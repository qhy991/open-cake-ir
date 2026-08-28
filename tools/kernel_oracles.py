"""Current correctness oracles for emitted-kernel observations.

`kernel_cases.ORACLES` predates the ranking calibration and its exact file bytes are now
part of frozen calibration authority. Rewriting those authorities would invalidate
historical replay. This module is the current extension point: it projects the retained
base registry and owns only oracles added after that freeze.
"""

from __future__ import annotations

from kernel_cases import ORACLES as _RETAINED_ORACLES


def _chunk_cumsum_oracle(inputs, torch):
    """The chunk-local inclusive prefix sum along the token axis."""

    gate, _ = inputs
    return gate.float().cumsum(dim=1), None


def _relu_oracle(inputs, torch):
    """Elementwise one-sided agreement used by the QSA score reduction."""

    x, _ = inputs
    return torch.relu(x), None


def _online_softmax_oracle(inputs, torch):
    """Dense FP32 softmax-weighted reduction, independent of the tiled recurrence."""

    logits, values, _ = inputs
    return torch.einsum(
        "bn,nd->bd",
        torch.softmax(logits, dim=-1),
        values.float(),
    ), None


def _index_expand_oracle(inputs, torch):
    """Expand each valid block id into its four consecutive token ids."""

    block_indices, _ = inputs
    offsets = torch.arange(4, dtype=torch.int32, device=block_indices.device)
    expanded = torch.where(
        block_indices[:, :, None] >= 0,
        block_indices[:, :, None] * 4 + offsets,
        -1,
    )
    return expanded.reshape(block_indices.shape[0], -1), None


def _cast_oracle(inputs, torch):
    """The explicit BF16-to-FP32 value-preserving conversion."""

    x, _ = inputs
    return x.float(), None


def _swiglu_oracle(inputs, torch):
    """The tanh identity used by the KDA v12 SwiGLU delta.

    Written from the operator definition rather than from the generated decomposition.
    It proves the standalone arithmetic slice, not grouped GEMMs or routed scatter.
    """

    up, gate, _ = inputs
    return up * gate * (0.5 * (torch.tanh(0.5 * gate) + 1.0)), None


def _top_k_oracle(inputs, torch):
    """Stable descending source indices for the standalone selection slice.

    The fixed prefix makes ties and negative infinity part of every observation rather
    than hoping random input happens to cover them. Stable sort is independent of the
    lowering's repeated reductions and defines the declared lowest-index tie break.
    """

    scores, _ = inputs
    scores.fill_(float("-inf"))
    prefix = torch.tensor(
        [10.0, 10.0, 9.0, 9.0, 8.0, 8.0, float("-inf"), float("-inf")],
        dtype=scores.dtype,
        device=scores.device,
    )
    scores[:, : prefix.numel()] = prefix
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :8].to(
        torch.int32
    ), None


def _block_scaled_gemm_oracle(inputs, torch):
    """Dequantize both operands, then contract them in float32.

    This is deliberately not the lowering's partial-dot decomposition. The profile
    relation says A scales each row/128-wide K block and B has one 128-row group with
    the same K granularity; expanding that relation produces ordinary dense tensors
    whose matmul is an independent answer.
    """

    a, b, scale_a, scale_b, _ = inputs
    a_dequant = a.to(torch.float32) * scale_a.t().repeat_interleave(128, dim=1)
    b_dequant = b.to(torch.float32) * (
        scale_b.expand(b.shape[0], -1).repeat_interleave(128, dim=1)
    )
    return a_dequant @ b_dequant.t(), None


def _ragged_zero_pad_oracle(inputs, torch):
    """Dense materialization from the declared prefix lengths."""

    ragged, lengths, _ = inputs
    rows = torch.arange(ragged.shape[1], device=ragged.device)
    valid = rows[None, :] < lengths[:, None]
    return torch.where(valid[:, :, None], ragged, torch.zeros_like(ragged)), None


def _ragged_grouped_gemm_oracle(inputs, torch):
    """Mask routed rows, then evaluate one independent dense GEMM per group."""

    a, b, lengths, _ = inputs
    rows = torch.arange(a.shape[1], device=a.device)
    masked_a = torch.where(
        (rows[None, :] < lengths[:, None])[:, :, None],
        a,
        torch.zeros_like(a),
    )
    return torch.bmm(masked_a.float(), b.float().transpose(1, 2)), None


def _indexed_gather_oracle(inputs, torch):
    """KDA-style expert/position tuples select rows; invalid routes contribute zero.

    Advanced indexing states the mathematical answer independently of the lowering's
    flattened pointer arithmetic and broadcasting. Clamping is only for memory safety;
    the explicit validity predicate owns the result for KDA's -1 sentinel.
    """

    expert_rows, expert_ids, row_ids, _ = inputs
    valid = (
        (expert_ids >= 0)
        & (expert_ids < expert_rows.shape[0])
        & (row_ids >= 0)
        & (row_ids < expert_rows.shape[1])
    )
    selected = expert_rows[
        expert_ids.clamp(0, expert_rows.shape[0] - 1).long(),
        row_ids.clamp(0, expert_rows.shape[1] - 1).long(),
    ]
    return torch.where(valid[:, :, None], selected, torch.zeros_like(selected)), None


def _kda_weighted_combine_oracle(inputs, torch):
    """Gather routed rows, apply route weights, then reduce per token in FP32."""

    expert_rows, expert_ids, row_ids, route_weights, _ = inputs
    valid = (
        (expert_ids >= 0)
        & (expert_ids < expert_rows.shape[0])
        & (row_ids >= 0)
        & (row_ids < expert_rows.shape[1])
    )
    selected = expert_rows[
        expert_ids.clamp(0, expert_rows.shape[0] - 1).long(),
        row_ids.clamp(0, expert_rows.shape[1] - 1).long(),
    ].float()
    weighted = selected * route_weights[:, :, None]
    return torch.where(valid[:, :, None], weighted, torch.zeros_like(weighted)).sum(
        dim=1
    ).to(torch.bfloat16), None


def _atomic_reservation_oracle(inputs, torch):
    """Snapshot the two facts an unordered atomic result must preserve."""

    expert_ids, counts, _ = inputs
    return expert_ids.clone(), counts.clone()


def _measure_atomic_reservation(
    observed, expert_ids, initial_counts, inputs, torch, tolerance
):
    """Judge contention by ownership invariants, never an invented lane order."""

    del tolerance
    current_counts = inputs[1]
    valid = (expert_ids >= 0) & (expert_ids < initial_counts.numel())
    invalid_mismatch = int((observed[~valid] != 0).sum().item())

    reservations_mismatch = 0
    expected_counts = initial_counts.clone()
    for expert in range(initial_counts.numel()):
        selected = expert_ids == expert
        count = int(selected.sum().item())
        expected_counts[expert] += count
        actual = torch.sort(observed[selected]).values
        start = int(initial_counts[expert].item())
        expected = torch.arange(
            start,
            start + count,
            dtype=observed.dtype,
            device=observed.device,
        )
        reservations_mismatch += int((actual != expected).sum().item())

    count_mismatch = int((current_counts != expected_counts).sum().item())
    mismatch = invalid_mismatch + reservations_mismatch + count_mismatch
    measured = {
        "unique_old_values": reservations_mismatch == 0,
        "masked_zero": invalid_mismatch == 0,
        "final_counts_match": count_mismatch == 0,
    }
    return mismatch, measured, mismatch == 0


def _reservation_owned_store_oracle(inputs, torch):
    """Snapshot the route identities and counters before the stateful launch."""

    expert_ids, payloads, counts, output = inputs
    for expert in range(counts.numel()):
        start = int(counts[expert].item())
        count = int((expert_ids == expert).sum().item())
        if start < 0 or start + count > output.shape[1]:
            raise ValueError(
                "reservation-owned-store observation requires every reserved "
                "interval to fit the output capacity"
            )
    return (expert_ids.clone(), payloads.clone()), counts.clone()


def _measure_reservation_owned_store(
    observed, route_snapshot, initial_counts, inputs, torch, tolerance
):
    """Judge each reserved interval as a set and require untouched capacity to stay zero."""

    del tolerance
    expert_ids, payloads = route_snapshot
    current_counts = inputs[2]
    expected_counts = initial_counts.clone()
    touched = torch.zeros_like(observed, dtype=torch.bool)
    payload_mismatch = 0

    for expert in range(initial_counts.numel()):
        selected = expert_ids == expert
        expected = torch.sort(payloads[selected]).values
        count = expected.numel()
        start = int(initial_counts[expert].item())
        stop = start + count
        expected_counts[expert] += count
        actual = torch.sort(observed[expert, start:stop]).values
        payload_mismatch += int((actual != expected).sum().item())
        touched[expert, start:stop] = True

    untouched_mismatch = int((observed[~touched] != 0).sum().item())
    count_mismatch = int((current_counts != expected_counts).sum().item())
    mismatch = payload_mismatch + untouched_mismatch + count_mismatch
    measured = {
        "reserved_payloads_match": payload_mismatch == 0,
        "untouched_slots_zero": untouched_mismatch == 0,
        "final_counts_match": count_mismatch == 0,
    }
    return mismatch, measured, mismatch == 0


def _masked_gemm_bias_oracle(inputs, torch):
    """Dense contraction plus the AccessMap's masked-zero bias semantics.

    A program coordinate may index a shorter Buffer.  `mask_tiled_axes` bounds that
    Buffer by its own declared extent and a masked load contributes zero, so pad the bias
    independently instead of assuming the Workload's output width.
    """

    a, b, bias, _ = inputs
    product = a.to(torch.float32) @ b.to(torch.float32).t()
    width = product.shape[-1]
    padded_bias = torch.nn.functional.pad(bias, (0, max(0, width - bias.shape[0])))[:width]
    return product + padded_bias[None, :], None


ORACLES = {
    **_RETAINED_ORACLES,
    "chunk_cumsum_b8_smoke": _chunk_cumsum_oracle,
    "relu_b8_smoke": _relu_oracle,
    "online_softmax_b8_smoke": _online_softmax_oracle,
    "index_expand_b8_smoke": _index_expand_oracle,
    "cast_b8_smoke": _cast_oracle,
    "gemm_bias_b1_smoke": _masked_gemm_bias_oracle,
    "swiglu_b8_smoke": _swiglu_oracle,
    "top_k_b8_smoke": _top_k_oracle,
    "block_scaled_gemm_b1_smoke": _block_scaled_gemm_oracle,
    "ragged_zero_pad_b1_smoke": _ragged_zero_pad_oracle,
    "ragged_grouped_gemm_b1_smoke": _ragged_grouped_gemm_oracle,
    "indexed_gather_b8_smoke": _indexed_gather_oracle,
    "kda_weighted_combine_b8_smoke": _kda_weighted_combine_oracle,
    "atomic_reservation_b8_smoke": _atomic_reservation_oracle,
    "reservation_owned_store_b8_smoke": _reservation_owned_store_oracle,
}

# Observation tools select an oracle by the executable interface the Compiler returns.
# The retained registry remains keyed by its historical calibration vocabulary because
# its bytes are frozen evidence; this is the sole current projection from lowering routes.
_WORKLOAD_BY_ENTRY_POINT = {
    "cake_chunk_cumsum_b8_smoke": "chunk_cumsum_b8_smoke",
    "cake_relu_b8_smoke": "relu_b8_smoke",
    "cake_online_softmax_b8_smoke": "online_softmax_b8_smoke",
    "cake_index_expand_b8_smoke": "index_expand_b8_smoke",
    "cake_cast_b8_smoke": "cast_b8_smoke",
    "cake_flash_kmeans_assign": "flash_kmeans_b32_smoke",
    "cake_flash_kmeans_assignment_full": "flash_kmeans_assignment_full",
    "cake_softmax_b8_smoke": "softmax_b8_smoke",
    "cake_rmsnorm_b8_smoke": "rmsnorm_b8_smoke",
    "cake_layernorm_b8_smoke": "layernorm_b8_smoke",
    "cake_gemm_bias_b1_smoke": "gemm_bias_b1_smoke",
    "cake_block_scaled_gemm_b1_smoke": "block_scaled_gemm_b1_smoke",
    "cake_ragged_zero_pad_b1_smoke": "ragged_zero_pad_b1_smoke",
    "cake_ragged_grouped_gemm_b1_smoke": "ragged_grouped_gemm_b1_smoke",
    "cake_indexed_gather_b8_smoke": "indexed_gather_b8_smoke",
    "cake_kda_weighted_combine_b8_smoke": "kda_weighted_combine_b8_smoke",
    "cake_atomic_reservation_b8_smoke": "atomic_reservation_b8_smoke",
    "cake_reservation_owned_store_b8_smoke": "reservation_owned_store_b8_smoke",
    "cake_swiglu_b8_smoke": "swiglu_b8_smoke",
    "cake_top_k_b8_smoke": "top_k_b8_smoke",
}
ORACLE_BY_ENTRY_POINT = {
    entry_point: ORACLES[workload]
    for entry_point, workload in _WORKLOAD_BY_ENTRY_POINT.items()
}

MEASURE_BY_ENTRY_POINT = {
    "cake_atomic_reservation_b8_smoke": _measure_atomic_reservation,
    "cake_reservation_owned_store_b8_smoke": _measure_reservation_owned_store,
}
