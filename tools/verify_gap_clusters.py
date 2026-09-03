#!/usr/bin/env python3
"""Verify that a model-proposed semantic clustering is an exact evidence partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_DECISIONS = {
    "repeated_candidate",
    "singleton_or_distinct",
    "conflicted_needs_review",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing text: {name}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or args.receipt.is_symlink():
        raise SystemExit(f"refusing existing receipt: {args.receipt}")

    source = load(args.input)
    proposal = load(args.proposal)
    input_names: dict[str, list[str]] = {}
    for group in source.get("exact_name_groups", []):
        name = nonempty_text(group.get("exact_candidate_name"), "input exact name")
        cases = sorted(member["case_id"] for member in group.get("members", []))
        if not cases or group.get("case_count") != len(cases) or name in input_names:
            raise ValueError(f"invalid input exact-name group: {name}")
        input_names[name] = cases
    input_cases = sorted(case for cases in input_names.values() for case in cases)
    if len(input_names) != 296 or len(input_cases) != 326 or len(set(input_cases)) != 326:
        raise ValueError("input counts differ")

    if (
        proposal.get("schema") != "open-cake.aka-ir-gap-clusters.proposal.v1"
        or proposal.get("status") != "proposal"
        or proposal.get("ir_change_status") != "proposal_only"
        or proposal.get("approval_granted") is not False
        or proposal.get("implementation_performed") is not False
        or proposal.get("release_performed") is not False
        or proposal.get("gpu_or_performance_claim") is not False
        or proposal.get("unclustered_case_ids") != []
        or proposal.get("duplicate_case_ids") != []
    ):
        raise ValueError("proposal authority boundary differs")
    proposal_source = proposal.get("source", {})
    source_counts = proposal_source.get("counts", proposal_source)
    if source_counts.get("gap_cases") != 326 or source_counts.get("exact_candidate_names") != 296:
        raise ValueError("proposal source counts differ")

    clusters = proposal.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("clusters are missing")
    cluster_ids: list[str] = []
    seen_names: list[str] = []
    seen_cases: list[str] = []
    repeated_clusters = repeated_cases = 0
    singleton_clusters = singleton_cases = 0
    conflicted_clusters = conflicted_cases = 0
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            raise ValueError(f"cluster {index} is not an object")
        cluster_id = nonempty_text(cluster.get("cluster_id"), f"cluster {index} id")
        nonempty_text(cluster.get("label"), f"cluster {cluster_id} label")
        nonempty_text(cluster.get("family"), f"cluster {cluster_id} family")
        nonempty_text(cluster.get("reason"), f"cluster {cluster_id} reason")
        boundary = cluster.get("evidence_boundary")
        if isinstance(boundary, str):
            nonempty_text(boundary, f"cluster {cluster_id} boundary")
        elif not (
            isinstance(boundary, dict)
            and boundary.get("basis") == "embedded_input_only"
            and boundary.get("authority") == "proposal_only"
        ):
            raise ValueError(f"cluster {cluster_id} boundary differs")
        decision = cluster.get("decision")
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"cluster {cluster_id} decision differs")
        names = cluster.get("member_exact_names")
        cases = cluster.get("case_ids")
        representatives = cluster.get("representative_cases")
        minimum = cluster.get("shared_minimum_semantics")
        differences = cluster.get("semantic_differences_preserved")
        if (
            not isinstance(names, list)
            or not names
            or names != sorted(names)
            or len(names) != len(set(names))
            or not isinstance(cases, list)
            or not cases
            or cases != sorted(cases)
            or len(cases) != len(set(cases))
            or not isinstance(representatives, list)
            or not 1 <= len(representatives) <= 3
            or not set(representatives) <= set(cases)
            or not isinstance(differences, list)
            or any(not isinstance(item, str) for item in differences)
        ):
            raise ValueError(f"cluster {cluster_id} list boundary differs")
        if isinstance(minimum, str):
            nonempty_text(minimum, f"cluster {cluster_id} minimum semantics")
        elif (
            not isinstance(minimum, list)
            or not minimum
            or any(not isinstance(item, str) or not item.strip() for item in minimum)
        ):
            raise ValueError(f"cluster {cluster_id} minimum semantics differ")
        try:
            expected_cases = sorted(
                case for name in names for case in input_names[name]
            )
        except KeyError as error:
            raise ValueError(f"cluster {cluster_id} invented exact name: {error}") from error
        if cases != expected_cases:
            raise ValueError(f"cluster {cluster_id} case membership differs")
        if cluster.get("exact_name_count") != len(names) or cluster.get("case_count") != len(cases):
            raise ValueError(f"cluster {cluster_id} declared count differs")
        candidate = cluster.get("candidate_primitive")
        if decision == "repeated_candidate":
            if len(cases) < 2 or not isinstance(candidate, dict):
                raise ValueError(f"cluster {cluster_id} invalid repeated candidate")
            nonempty_text(candidate.get("name"), f"cluster {cluster_id} candidate name")
            nonempty_text(candidate.get("semantics"), f"cluster {cluster_id} candidate semantics")
            repeated_clusters += 1
            repeated_cases += len(cases)
        elif candidate is not None:
            raise ValueError(f"cluster {cluster_id} non-proposal candidate differs")
        elif decision == "singleton_or_distinct":
            singleton_clusters += 1
            singleton_cases += len(cases)
        else:
            conflicted_clusters += 1
            conflicted_cases += len(cases)
        cluster_ids.append(cluster_id)
        seen_names.extend(names)
        seen_cases.extend(cases)

    if cluster_ids != sorted(cluster_ids) or len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("cluster ids are not unique and sorted")
    if sorted(seen_names) != sorted(input_names) or len(seen_names) != len(set(seen_names)):
        raise ValueError("exact-name partition differs")
    if sorted(seen_cases) != input_cases or len(seen_cases) != len(set(seen_cases)):
        raise ValueError("case partition differs")
    summary = proposal.get("summary")
    expected_summary = {
        "semantic_clusters": len(clusters),
        "repeated_candidate_clusters": repeated_clusters,
        "repeated_candidate_cases": repeated_cases,
        "singleton_or_distinct_clusters": singleton_clusters,
        "singleton_or_distinct_cases": singleton_cases,
        "conflicted_needs_review_clusters": conflicted_clusters,
        "conflicted_needs_review_cases": conflicted_cases,
        "covered_gap_cases": 326,
        "covered_exact_candidate_names": 296,
    }
    expected_nested_summary = {
        "semantic_cluster_count": len(clusters),
        "repeated_candidate_cluster_count": repeated_clusters,
        "repeated_candidate_case_count": repeated_cases,
        "singleton_or_distinct_cluster_count": singleton_clusters,
        "singleton_or_distinct_case_count": singleton_cases,
        "conflicted_needs_review_cluster_count": conflicted_clusters,
        "conflicted_needs_review_case_count": conflicted_cases,
        "coverage": {
            "case_count": 326,
            "exact_name_count": 296,
            "case_ids_exactly_once": True,
            "exact_names_exactly_once": True,
        },
    }
    if summary not in (expected_summary, expected_nested_summary):
        raise ValueError(
            f"summary differs: expected {expected_summary} or {expected_nested_summary}, got {summary}"
        )

    receipt = {
        "schema": "open-cake.aka-ir-gap-cluster-verification.v1",
        "status": "accepted",
        "deterministic": True,
        "source_gap_cases": 326,
        "source_exact_candidate_names": 296,
        "semantic_clusters": len(clusters),
        "partition_exact": True,
        "unclustered_case_ids": [],
        "duplicate_case_ids": [],
        "ir_change_status": "proposal_only",
        "approval_granted": False,
        "implementation_performed": False,
        "release_performed": False,
        "gpu_or_performance_claim": False,
        "summary": expected_summary,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
