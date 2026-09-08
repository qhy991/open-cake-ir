"""Audited Campaign aggregation and descriptive claim projections."""

from __future__ import annotations

import json
import math
import statistics
from hashlib import sha256
from types import MappingProxyType
from typing import Callable, Mapping, cast

from open_cake_ir.evidence import EvidenceStore, RunAudit

from ._documents import _DIGEST, _object
from ._policies import _scientific_analysis_plan_version
from .contracts import AnalysisInclusion, CampaignRef, StudyReport
from .pairing import comparison_arm


def _promoted_artifact(
    evidence: EvidenceStore,
    audit: RunAudit,
) -> Mapping[str, object] | None:
    """Select one per-Run confirmed artifact without constructing a treatment contrast."""

    if (
        not audit.archive_integrity
        or not audit.filesystem_custody_verified
        or audit.protocol_adherence != "adhered"
    ):
        return None
    eligible: list[tuple[float, int, str, str]] = []
    for event in evidence.replay_events(audit.run_id):
        if event.get("kind") != "candidate_evaluated":
            continue
        payload = _object(event.get("payload"), "candidate_evaluated.payload")
        if payload.get("purpose") != "confirmatory":
            continue
        turn = payload.get("turn")
        candidate_sha256 = payload.get("candidate_sha256")
        objects = payload.get("objects")
        if (
            not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn <= 0
            or not isinstance(candidate_sha256, str)
            or _DIGEST.fullmatch(candidate_sha256) is None
            or not isinstance(objects, list)
        ):
            raise ValueError("artifact promotion Candidate evidence differs")
        receipt_refs = [
            cast(Mapping[str, object], item)
            for item in objects
            if isinstance(item, Mapping) and item.get("role") == "evaluation_receipt"
        ]
        if len(receipt_refs) != 1:
            raise ValueError("artifact promotion receipt evidence differs")
        receipt_bytes = evidence.read_object(receipt_refs[0])
        receipt = _object(json.loads(receipt_bytes), "artifact promotion receipt")
        timing = receipt.get("timing")
        latency = timing.get("pooled_median_ms") if isinstance(timing, Mapping) else None
        if (
            receipt.get("candidate_sha256") != candidate_sha256
            or receipt.get("correctness_passed") is not True
            or receipt.get("kernel_calls") != 1
            or receipt.get("fallback_calls") != 0
            or not isinstance(timing, Mapping)
            or timing.get("measurement_quality_passed") is not True
            or not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(float(latency))
            or float(latency) <= 0
        ):
            continue
        eligible.append(
            (
                float(latency),
                turn,
                candidate_sha256,
                sha256(receipt_bytes).hexdigest(),
            )
        )
    if not eligible:
        return None
    latency, turn, candidate_sha256, receipt_sha256 = min(eligible)
    return MappingProxyType(
        {
            "turn": turn,
            "candidate_sha256": candidate_sha256,
            "confirmed_latency_ms": latency,
            "evaluation_receipt_sha256": receipt_sha256,
        }
    )


def audit_campaign(
    campaign: CampaignRef,
    *,
    replay_run: Callable,
) -> StudyReport:
    """Audit terminal Runs, then apply the preregistered availability rule."""

    evidence = EvidenceStore.open(campaign.evidence_root)
    audits: list[RunAudit] = []
    campaign_complete = True
    for run_id in campaign.lock.run_order:
        try:
            audit = evidence.audit_run(run_id)
        except (OSError, ValueError, json.JSONDecodeError):
            campaign_complete = False
            continue
        if audit.authority_sha256 != campaign.lock.canonical_sha256:
            campaign_complete = False
        audits.append(audit)
    campaign_complete = campaign_complete and len(audits) == len(campaign.lock.run_order)
    archive_integrity_passed = campaign_complete and all(
        audit.archive_integrity for audit in audits
    )
    filesystem_custody_verified = campaign_complete and all(
        audit.filesystem_custody_verified for audit in audits
    )
    semantic_replay_passed = archive_integrity_passed
    evaluation_receipt_counts: dict[str, int] = {}
    if semantic_replay_passed:
        for audit in audits:
            try:
                if not replay_run(evidence, audit, campaign.lock):
                    semantic_replay_passed = False
                    break
                evaluation_receipt_counts[audit.run_id] = sum(
                    event.get("kind") == "candidate_evaluated"
                    for event in evidence.replay_events(audit.run_id)
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                semantic_replay_passed = False
                break
    missing_run_count = len(campaign.lock.run_order) - len(audits) + sum(
        not audit.archive_integrity
        or not audit.filesystem_custody_verified
        or audit.authority_sha256 != campaign.lock.canonical_sha256
        or audit.endpoint_observation == "missing"
        or audit.protocol_adherence != "adhered"
        for audit in audits
    )
    if campaign.lock.claim_scope == "artifact_optimization_only":
        promoted_artifacts = (
            {
                audit.run_id: _promoted_artifact(evidence, audit)
                for audit in audits
            }
            if semantic_replay_passed
            else {audit.run_id: None for audit in audits}
        )
        artifact_optimization_complete = (
            campaign_complete
            and archive_integrity_passed
            and filesystem_custody_verified
            and semantic_replay_passed
            and all(audit.protocol_adherence == "adhered" for audit in audits)
            and all(
                promoted_artifacts.get(run_id) is not None
                for run_id in campaign.lock.run_order
            )
        )
        inclusions = tuple(
            AnalysisInclusion(
                audit.run_id,
                False,
                False,
                "artifact_optimization_not_scientific_data",
            )
            for audit in audits
        )
        return StudyReport(
            study_id=campaign.lock.study_id,
            claim_scope=campaign.lock.claim_scope,
            system_qualification_passed=None,
            estimand=None,
            campaign_complete=campaign_complete,
            archive_integrity_passed=archive_integrity_passed,
            filesystem_custody_verified=filesystem_custody_verified,
            semantic_replay_passed=semantic_replay_passed,
            estimand_available=False,
            missing_run_count=sum(
                not audit.archive_integrity
                or not audit.filesystem_custody_verified
                or audit.protocol_adherence != "adhered"
                for audit in audits
            ),
            estimate=None,
            uncertainty=None,
            descriptive={
                "artifact_optimization_complete": artifact_optimization_complete,
                "promoted_artifacts": promoted_artifacts,
            },
            run_inclusion=inclusions,
            run_audits=tuple(audits),
        )
    if campaign.lock.claim_scope == "system_qualification_only":
        inclusions = tuple(
            AnalysisInclusion(
                audit.run_id,
                False,
                False,
                "system_qualification_not_scientific_data",
            )
            for audit in audits
        )
        system_qualification_passed = (
            campaign_complete
            and archive_integrity_passed
            and filesystem_custody_verified
            and semantic_replay_passed
            and all(audit.protocol_adherence == "adhered" for audit in audits)
            and all(
                evaluation_receipt_counts.get(run_id, 0) >= 1
                for run_id in campaign.lock.run_order
            )
        )
        descriptive: Mapping[str, object] = {
            "prescheduled_run_count": len(campaign.lock.run_order),
            "completed_run_count": len(audits),
            "runs": [
                {
                    "run_id": audit.run_id,
                    "archive_integrity": audit.archive_integrity,
                    "filesystem_custody_verified": audit.filesystem_custody_verified,
                    "protocol_adherence": audit.protocol_adherence,
                    "endpoint_observation": audit.endpoint_observation,
                    "evaluation_receipt_count": evaluation_receipt_counts.get(
                        audit.run_id, 0
                    ),
                }
                for audit in audits
            ],
        }
        return StudyReport(
            study_id=campaign.lock.study_id,
            claim_scope=campaign.lock.claim_scope,
            system_qualification_passed=system_qualification_passed,
            estimand=None,
            campaign_complete=campaign_complete,
            archive_integrity_passed=archive_integrity_passed,
            filesystem_custody_verified=filesystem_custody_verified,
            semantic_replay_passed=semantic_replay_passed,
            estimand_available=False,
            missing_run_count=missing_run_count,
            estimate=None,
            uncertainty=None,
            descriptive=descriptive,
            run_inclusion=inclusions,
            run_audits=tuple(audits),
        )
    assigned_arms = campaign.lock.document["resolved_inputs"]["arm_environments"]
    comparison = comparison_arm(assigned_arms)
    arm_runs: dict[str, list[RunAudit]] = {name: [] for name in assigned_arms}
    inclusions: list[AnalysisInclusion] = []
    for audit in audits:
        arm = audit.run_id.rsplit("-", 1)[0]
        if arm not in arm_runs:
            raise ValueError(f"Run {audit.run_id!r} has an unknown assigned arm")
        arm_runs[arm].append(audit)
        if not audit.archive_integrity:
            inclusions.append(
                AnalysisInclusion(audit.run_id, False, False, "archive_integrity")
            )
        elif audit.authority_sha256 != campaign.lock.canonical_sha256:
            inclusions.append(
                AnalysisInclusion(audit.run_id, False, False, "campaign_authority")
            )
        elif not audit.filesystem_custody_verified:
            inclusions.append(
                AnalysisInclusion(
                    audit.run_id,
                    False,
                    False,
                    "filesystem_custody_not_verified",
                )
            )
        elif audit.protocol_adherence != "adhered":
            inclusions.append(
                AnalysisInclusion(audit.run_id, False, False, "protocol_deviation")
            )
        elif audit.endpoint_observation == "missing":
            inclusions.append(
                AnalysisInclusion(audit.run_id, False, False, "endpoint_missing")
            )
        elif audit.endpoint_observation == "no_qualified_candidate":
            inclusions.append(
                AnalysisInclusion(
                    audit.run_id,
                    True,
                    False,
                    "observed_no_qualified_candidate_at_checkpoint",
                )
            )
        else:
            inclusions.append(AnalysisInclusion(audit.run_id, True, True, "included"))
    analysis_version = _scientific_analysis_plan_version(
        campaign.lock.analysis_plan, "campaign_lock.analysis_plan"
    )
    inclusion_by_run = {item.run_id: item for item in inclusions}
    qualification_rate: dict[str, float | None] = {}
    endpoint_counts: dict[str, dict[str, int]] = {}
    medians: dict[str, float | None] = {}
    ranges: dict[str, list[float] | None] = {}
    latencies: dict[str, dict[str, float]] = {}
    for arm, arm_audits in arm_runs.items():
        prescheduled = sum(
            run_id.rsplit("-", 1)[0] == arm
            for run_id in campaign.lock.run_order
        )
        observed = [
            audit
            for audit in arm_audits
            if inclusion_by_run[audit.run_id].qualification_endpoint_included
        ]
        qualified = [
            audit
            for audit in observed
            if audit.endpoint_observation == "qualified"
        ]
        denominator = len(observed)
        qualification_rate[arm] = (
            len(qualified) / denominator
            if denominator
            else (None)
        )
        endpoint_counts[arm] = {
            "prescheduled": prescheduled,
            "observed": len(observed),
            "qualified": len(qualified),
            "missing": prescheduled - len(observed),
        }
        values: list[float] = []
        latencies[arm] = {}
        for audit in qualified:
            endpoint = audit.endpoint
            if endpoint is None or endpoint.get("qualified_by_budget") is not True:
                raise ValueError(f"qualified Run {audit.run_id!r} lacks endpoint evidence")
            latency = endpoint.get("best_confirmed_latency_ms")
            if (
                not isinstance(latency, (int, float))
                or isinstance(latency, bool)
                or not math.isfinite(float(latency))
                or float(latency) <= 0
            ):
                raise ValueError(f"qualified Run {audit.run_id!r} has an invalid latency")
            values.append(float(latency))
            latencies[arm][audit.run_id.rsplit("-", 1)[1]] = float(latency)
        medians[arm] = statistics.median(values) if values else None
        ranges[arm] = [min(values), max(values)] if values else None
    estimand_available = (
        archive_integrity_passed
        and filesystem_custody_verified
        and semantic_replay_passed
        and missing_run_count == 0
        and (
            all(medians[arm] is not None for arm in arm_runs)
        )
    )
    paired: list[dict[str, object]] = []
    for repetition in sorted({name.rsplit("-", 1)[1] for name in campaign.lock.run_order}):
        open_latency = latencies["open_cake"].get(repetition)
        cuda_latency = latencies[comparison].get(repetition)
        paired.append(
            {
                "repetition": int(repetition) if repetition.isdigit() else repetition,
                "open_cake_latency_ms": open_latency,
                f"{comparison}_latency_ms": cuda_latency,
                "open_cake_outcome": next((a.endpoint_observation for a in audits if a.run_id == f"open_cake-{repetition}"), "missing"),
                f"{comparison}_outcome": next((a.endpoint_observation for a in audits if a.run_id == f"{comparison}-{repetition}"), "missing"),
                "open_cake_speedup": (
                    cuda_latency / open_latency
                    if open_latency is not None and cuda_latency is not None
                    else None
                ),
            }
        )
    descriptive = {
        "endpoint_counts": endpoint_counts,
        "qualification_rate_among_observed": qualification_rate,
        "qualification_rate_difference_among_observed": (
            cast(float, qualification_rate["open_cake"])
            - cast(float, qualification_rate[comparison])
            if all(value is not None for value in qualification_rate.values())
            else None
        ),
        "median_confirmed_latency_ms": medians,
        "paired_runs": paired,
    }
    estimate: Mapping[str, object] | None = None
    uncertainty: Mapping[str, object] | None = None
    if estimand_available:
        estimate = {
            "qualification_rate": qualification_rate,
            "median_confirmed_latency_ms": medians,
            "ratio_of_arm_medians": cast(float, medians[comparison])
            / cast(float, medians["open_cake"]),
        }
        if analysis_version in {"two_part_v2", "triton_optimization_v1", "cute_optimization_v1"}:
            estimate = {
                **estimate,
                "qualification_rate_difference": (
                    cast(float, qualification_rate["open_cake"])
                    - cast(float, qualification_rate[comparison])
                ),
            }
        uncertainty = {"latency_range_ms": ranges}
    return StudyReport(
        study_id=campaign.lock.study_id,
        claim_scope=campaign.lock.claim_scope,
        system_qualification_passed=None,
        estimand=campaign.lock.estimand,
        campaign_complete=campaign_complete,
        archive_integrity_passed=archive_integrity_passed,
        filesystem_custody_verified=filesystem_custody_verified,
        semantic_replay_passed=semantic_replay_passed,
        estimand_available=estimand_available,
        missing_run_count=missing_run_count,
        estimate=estimate,
        uncertainty=uncertainty,
        descriptive=descriptive,
        run_inclusion=tuple(inclusions),
        run_audits=tuple(audits),
    )


def threshold_view(
    campaign: CampaignRef,
    latency_threshold_ms: float,
    *,
    audit_campaign: Callable,
) -> Mapping[str, object]:
    """Describe first fresh confirmations from audited records; retain every Run.

    A caller-selected threshold is descriptive, never a new scientific estimand.
    Historical events without confirmation time retain unknown wall time.
    """
    if type(latency_threshold_ms) not in {int, float}:
        raise ValueError("latency threshold must be finite and positive")
    try:
        latency_threshold_ms = float(latency_threshold_ms)
    except (ValueError, OverflowError) as error:
        raise ValueError("latency threshold must be finite and positive") from error
    if not math.isfinite(latency_threshold_ms) or latency_threshold_ms <= 0:
        raise ValueError("latency threshold must be finite and positive")
    if campaign.lock.study_kind != "matched_search":
        raise ValueError("threshold view requires matched_search records")
    report = audit_campaign(campaign)
    evidence = EvidenceStore.open(campaign.evidence_root)
    audits = {audit.run_id: audit for audit in report.run_audits}
    limit = campaign.lock.document["resolved_inputs"]["budget"]["limit"]
    rows = []
    for run_id in campaign.lock.run_order:
        audit = audits.get(run_id)
        row = {"run_id": run_id, "endpoint": audit.endpoint_observation if audit else "missing",
               "first_confirmation_turn": None, "provider_tokens": None,
               "elapsed_wall_seconds": None, "candidate_sha256": None,
               "confirmed_latency_ms": None, "status": "missing"}
        eligible = (audit is not None and audit.archive_integrity and audit.filesystem_custody_verified
                    and audit.protocol_adherence == "adhered" and report.semantic_replay_passed)
        if audit is not None and not eligible:
            row["status"] = "unverified_archive_or_protocol"
        elif eligible:
            row["status"] = "threshold_not_reached"
            tokens = {}
            for event in evidence.replay_events(run_id):
                payload = event["payload"]
                if event["kind"] == "provider_turn_completed":
                    tokens[payload["turn"]] = payload["cumulative_provider_tokens"]
                if event["kind"] != "candidate_evaluated" or payload["purpose"] != "confirmatory":
                    continue
                reference = next(value for value in payload["objects"] if value["role"] == "evaluation_receipt")
                receipt = json.loads(evidence.read_object(reference))
                timing = receipt["timing"]
                if (tokens[payload["turn"]] <= limit and receipt["correctness_passed"] is True
                    and timing is not None and timing.get("measurement_quality_passed") is True
                    and timing["pooled_median_ms"] <= latency_threshold_ms):
                    row.update(status="reached_by_fresh_confirmation", first_confirmation_turn=payload["turn"],
                        provider_tokens=tokens[payload["turn"]], elapsed_wall_seconds=payload.get("elapsed_wall_seconds"),
                        candidate_sha256=payload["candidate_sha256"], confirmed_latency_ms=timing["pooled_median_ms"])
                    break
        rows.append(row)
    return {"audit": report, "scope": "descriptive_threshold_view", "latency_threshold_ms": latency_threshold_ms,
            "wall_time_definition": "run_start_to_archived_fresh_confirmation_including_authoring_build_and_evaluation",
            "missing_confirmation_timestamps": "unknown_never_inferred_from_search_or_terminal_time",
            "runs": rows}
