"""Pure Triton selector geometry shared by emission and performance reports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopKSelectionStructure:
    """Triton 3.7.1 frontend comparison work for one merge selection.

    ``comparison_lane_work`` counts input lanes participating in compare/select work in
    the pinned ``standard.py`` network: both lanes of compare-exchange and both inputs
    of its pairwise top-k reduction.  It is a toolchain-specific structural model, not a
    public API guarantee, cycle estimate, instruction count, or physical resource fact.
    The optimized algorithm is admitted only when public ``sort`` plus
    ``bitonic_merge`` removes at least one third of that baseline work.
    """

    algorithm: str
    comparison_model: str
    baseline_comparison_lane_work: int | None
    selected_comparison_lane_work: int | None
    comparison_lane_reduction_fraction: float | None


def top_k_selection_structure(
    k: int,
    merge_width: int,
    source_tiles_per_merge: int,
) -> TopKSelectionStructure:
    """Choose the exact batched-source selector whose structural saving is material."""

    if k <= 0 or k & (k - 1) or merge_width != 2 * k:
        return TopKSelectionStructure(
            "triton_topk", "triton_3_7_1_standard_py", None, None, None
        )
    log_k = k.bit_length() - 1
    baseline = k * (log_k**2 + 2 * log_k + 2)
    half_selection = k * (log_k + 1) * (log_k + 4) // 2
    if source_tiles_per_merge != 2 or 3 * half_selection > 2 * baseline:
        return TopKSelectionStructure(
            "triton_topk",
            "triton_3_7_1_standard_py",
            baseline,
            baseline,
            0.0,
        )
    return TopKSelectionStructure(
        "sorted_source_half_bitonic_merge",
        "triton_3_7_1_standard_py",
        baseline,
        half_selection,
        (baseline - half_selection) / baseline,
    )
