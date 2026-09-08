"""Pure advisory ordering and qualification decisions shared by independent consumers."""

from __future__ import annotations

import json
import math
from typing import Mapping, cast

from open_cake_ir.compiler.performance.empirical_cost import EmpiricalCostModel
from open_cake_ir.evaluation import EvaluationReceipt

from ._documents import _object
from .executor import ExecutorRevision
from .routing import CANDIDATE, COST_MODEL


class _EmpiricalSelection:
    """One Study-bound advisory model, shared by execution and CPU replay."""

    def __init__(
        self, binding: Mapping[str, object], *, context: Mapping[str, object],
        compiler_revision_id: str, compiler_revision_sha256: str, target: str,
    ) -> None:
        if set(binding) != {"kind", "model"} or binding.get("kind") != _EMPIRICAL_SELECTION:
            raise ValueError("empirical selection binding differs")
        self.model = EmpiricalCostModel(binding["model"])
        self._context_differences = tuple(
            key for key in ("timer", "cache_protocol", "runtime", "input_scope")
            if binding["model"]["context"].get(key) != context.get(key)
        ) + (("fields",) if set(binding["model"]["context"]) != set(context) else ())
        self._compiler_revision_id = compiler_revision_id
        self._compiler_revision_sha256 = compiler_revision_sha256
        self._target = target

    def estimate(self, schedule: dict) -> dict[str, object]:
        result = self.model.estimate(
            schedule, compiler_revision_id=self._compiler_revision_id,
            compiler_revision_sha256=self._compiler_revision_sha256,
            target=self._target,
        )
        reason = None
        if self._context_differences:
            reason = "model context differs in " + ", ".join(self._context_differences) + "; exact Workload/case/assay/Executor binding required"
        elif result["covered"] and (
            not math.isfinite(result["predicted_kernel_us"])
            or result["predicted_kernel_us"] <= 0
            or any(not math.isfinite(value) for value in result["empirical_range_us"])
        ):
            reason = "model prediction or empirical range is not finite and positive"
        if reason is not None:
            result.update(covered=False, predicted_kernel_us=None, empirical_range_us=None, reason=reason)
        # The full supplier document is already frozen in the CampaignLock. Arbitrary
        # context/provenance maps (including raw observations) are not agent feedback.
        return {key: result[key] for key in (
            "kind", "model_id", "model_compiler_revision_id",
            "model_compiler_revision_sha256", "target", "covered",
            "predicted_kernel_us", "empirical_range_us", "reason",
        )}


def _empirical_context(
    executor: ExecutorRevision, *, workload_sha256: str, case_id: str
) -> dict[str, object]:
    """Reference the shared CUPTI assay and its admitted, frozen runtime owner.

    The Executor content identity binds the helper, assay source and host closure;
    this projection does not infer equivalence between suppliers' free-text contexts.
    """
    packages = executor.document["host_environment"]["packages"]
    return {
        "timer": "flashinfer.testing.utils.bench_gpu_time_with_cupti;use_cuda_graph=false",
        "cache_protocol": "cold_l2_cache=true",
        "runtime": {
            "compiler_version": packages.get("triton"),
            "executor_revision": executor.canonical_sha256,
        },
        "input_scope": json.dumps(
            {"workload_contract_sha256": workload_sha256, "case_id": case_id},
            sort_keys=True, separators=(",", ":"),
        ),
    }


_EMPIRICAL_SELECTION = "external_empirical_advisory_v1"


def _matched_endpoint_from_checkpoint(
    checkpoint: object,
    protocol_adherence: str,
) -> tuple[str, Mapping[str, object] | None]:
    state = getattr(checkpoint, "state")
    if protocol_adherence != "adhered" or state == "unreached":
        return "missing", None
    if state == "reached_with_best":
        return (
            "qualified",
            {
                "qualified_by_budget": True,
                "budget": getattr(checkpoint, "provider_tokens"),
                "best_candidate_sha256": getattr(checkpoint, "best_candidate_sha256"),
                "best_confirmed_latency_ms": getattr(
                    checkpoint, "best_confirmed_latency_ms"
                ),
            },
        )
    return (
        "no_qualified_candidate",
        {"qualified_by_budget": False, "budget": getattr(checkpoint, "provider_tokens")},
    )


def _receipt_qualifies(receipt: EvaluationReceipt) -> bool:
    return (
        receipt.correctness_passed
        and receipt.kernel_calls == 1
        and receipt.fallback_calls == 0
        and receipt.timing is not None
        and receipt.timing.get("measurement_quality_passed") is True
        and _receipt_latency_ms(receipt) is not None
    )


def _receipt_latency_ms(receipt: EvaluationReceipt | None) -> float | None:
    if receipt is None or receipt.timing is None:
        return None
    value = receipt.timing.get("pooled_median_ms")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        return None
    return float(value)


def _matched_search_plan(
    order: list[Mapping[str, object]], searches_per_turn: int
) -> tuple[list[str], list[dict[str, str]]]:
    """Derive the unique programs to measure from the retained filter order."""

    first_by_semantic: dict[str, str] = {}
    searched: list[str] = []
    collapsed: list[dict[str, str]] = []
    for row in order:
        if len(searched) >= searches_per_turn or row["disposition"] != "launchable":
            break
        candidate = cast(str, row["candidate_sha256"])
        semantic = cast(str | None, row.get("semantic_sha256"))
        if semantic is not None and semantic in first_by_semantic:
            collapsed.append(
                {
                    "candidate_sha256": candidate,
                    "same_program_as": first_by_semantic[semantic],
                    "semantic_sha256": semantic,
                }
            )
        else:
            searched.append(candidate)
            if semantic is not None:
                first_by_semantic[semantic] = candidate
    return searched, collapsed


def _collapse_diagnosis(
    turn: int, collapsed: list[dict[str, str]]
) -> dict[str, object] | None:
    if not collapsed:
        return None
    return {
        "turn": turn,
        "routed_to": CANDIDATE,
        "routing_reason": (
            f"{len(collapsed)} of the ranked candidates this Turn are the same "
            "program as an earlier one under a different name"
        ),
        "collapsed": collapsed,
    }


def _matched_search_decision(
    turn: int,
    searched: list[tuple[str, EvaluationReceipt]],
    *,
    cost_order_applied: bool,
    materiality_ratio: float,
) -> tuple[list[int], int, dict[str, object] | None]:
    """Choose the measured winner and derive the sole cost-order diagnosis."""

    qualified = [
        index for index, (_, receipt) in enumerate(searched) if _receipt_qualifies(receipt)
    ]
    measured = sorted(
        qualified,
        key=lambda index: _receipt_latency_ms(searched[index][1]) or float("inf"),
    )
    best = measured[0] if measured else 0
    if not cost_order_applied or len(qualified) < 2 or best == qualified[0]:
        return qualified, best, None
    ranked_index = qualified[0]
    ranked = _receipt_latency_ms(searched[ranked_index][1])
    fastest = _receipt_latency_ms(searched[best][1])
    ratio = ranked / fastest if ranked is not None and fastest else None
    if ratio is None or ratio < materiality_ratio:
        return qualified, best, None
    ranked_sha = searched[ranked_index][0]
    fastest_sha = searched[best][0]
    return qualified, best, {
        "turn": turn,
        "routed_to": COST_MODEL,
        "routing_reason": (
            f"the filter ranked {ranked_sha} first and measurement put "
            f"{fastest_sha} {ratio:.3f}x ahead of it, which the Study counts as "
            f"material at {materiality_ratio}x"
        ),
        "ranked_first": ranked_sha,
        "measured_first": fastest_sha,
        "observed_ratio": round(ratio, 6),
        "materiality_ratio": materiality_ratio,
    }


def _empirical_filter(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Apply one complete advisory order; callers supply original provider order."""
    launchable = [row for row in rows if row["disposition"] == "launchable"]
    for row in launchable:
        estimate = _object(row.get("empirical_cost"), "candidate empirical cost")
        if estimate.get("covered") is True:
            value = estimate.get("predicted_kernel_us")
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError("candidate empirical prediction differs")
        elif estimate.get("covered") is not False:
            raise ValueError("candidate empirical coverage differs")
    applied = bool(launchable) and all(row["empirical_cost"]["covered"] for row in launchable)
    if applied:
        launchable.sort(key=lambda row: row["empirical_cost"]["predicted_kernel_us"])
    return (
        launchable + [row for row in rows if row["disposition"] != "launchable"],
        {
            "kind": _EMPIRICAL_SELECTION,
            "order_applied": applied,
            "reason": (
                "complete comparable point estimates; advisory only"
                if applied else "incomplete coverage or no launchable candidates; provider order retained"
            ),
        },
    )
