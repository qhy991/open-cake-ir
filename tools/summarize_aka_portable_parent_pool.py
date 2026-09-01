#!/usr/bin/env python3
"""Create a publishable derived report from one terminal portable-parent pool."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any, Sequence


SUMMARY_SCHEMA = "open-cake.aka-portable-parent-ir-review-summary.v2"
POOL_FINAL_SCHEMA = "open-cake.aka-portable-parent-pool-final.v1"
CHECKED_SCHEMAS = {
    "open-cake.aka-expressibility-checked.v2",
    "open-cake.aka-expressibility-checked.v3",
}
ASSESSMENT_SCOPES = {
    "contract",
    "fixed_instance",
    "legacy_unspecified",
}
SIGNALS = {
    "runtime_parameterization": (
        "runtime",
        "symbolic",
        "scalar argument",
        "scalar input",
        "dynamic shape",
        "dynamic extent",
    ),
    "indexed_addressing_or_scatter": (
        "indexed",
        "indirect",
        "scatter",
        "gather",
        "index buffer",
        "data-dependent address",
    ),
    "control_or_predicate": ("predicate", "conditional", "branch", "select", "mask"),
    "reduction_or_scan": ("reduce", "reduction", "scan", "histogram", "pool"),
    "numeric_operation_or_conversion": (
        "log",
        "atan",
        "ceil",
        "floor",
        "round",
        "conversion",
        "cast",
        "fma",
        "sigmoid",
        "remainder",
    ),
    "vector_or_lane_mapping": ("vector", "lane", "thread mapping", "local-thread"),
    "multi_launch_or_program": (
        "multi-launch",
        "multiple launch",
        "program composition",
        "ordered",
    ),
}


class SummaryError(ValueError):
    """The external pool does not support a complete derived report."""


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SummaryError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise SummaryError(f"{label} must be one JSON object")
    return value


def one_file(paths: Sequence[Path], label: str) -> Path:
    if len(paths) != 1:
        raise SummaryError(f"{label} requires exactly one file, observed {len(paths)}")
    return paths[0]


def summarize(pool_root: Path, *, expected_count: int) -> dict[str, object]:
    pool_root = pool_root.resolve(strict=True)
    final = load_object(pool_root / "pool.final.json", "pool final")
    plan = load_object(pool_root / "pool.plan.json", "pool plan")
    pool = load_object(pool_root / "pool.json", "pool contract")
    if final.get("schema") != POOL_FINAL_SCHEMA:
        raise SummaryError("unsupported pool final schema")
    if final.get("selected") != expected_count or plan.get("selected") != expected_count:
        raise SummaryError("pool selected count differs from the expected report domain")
    if final.get("counts") != {"completed": expected_count}:
        raise SummaryError("pool is not completely successful")
    entries = plan.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise SummaryError("pool plan entries differ from the expected report domain")
    if len({entry.get("queue_case_id") for entry in entries if isinstance(entry, dict)}) != expected_count:
        raise SummaryError("pool plan queue case ids are not unique")

    rows: list[dict[str, object]] = []
    primary_counts: Counter[str] = Counter()
    complete_counts: Counter[str] = Counter()
    delta_counts: Counter[str] = Counter()
    owner_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    compiler_counts: Counter[str] = Counter()
    signal_counts: Counter[str] = Counter()

    for entry in entries:
        if not isinstance(entry, dict):
            raise SummaryError("pool plan entry is not an object")
        queue_case_id = entry.get("queue_case_id")
        if not isinstance(queue_case_id, str) or not queue_case_id:
            raise SummaryError("pool plan entry has no queue case id")
        case_root = pool_root / "cases" / queue_case_id
        worker = load_object(case_root / "worker.finished.json", f"worker {queue_case_id}")
        if worker.get("status") != "completed" or worker.get("exit_code") != 0:
            raise SummaryError(f"worker {queue_case_id} is not completed successfully")
        checked_path = one_file(
            list(case_root.glob("reviews/cases/*/checked.json")),
            f"checked result {queue_case_id}",
        )
        receipt_path = one_file(
            list(case_root.glob("reviews/runs/*/receipt.json")),
            f"model receipt {queue_case_id}",
        )
        marker_path = one_file(
            list(case_root.glob("parents/cases/*/PARENT_DONE.json")),
            f"parent marker {queue_case_id}",
        )
        checked = load_object(checked_path, f"checked result {queue_case_id}")
        receipt = load_object(receipt_path, f"model receipt {queue_case_id}")
        marker = load_object(marker_path, f"parent marker {queue_case_id}")
        if (
            receipt.get("exit_code") != 0
            or receipt.get("timed_out") is not False
            or receipt.get("model") != "gpt-5.6-sol"
            or receipt.get("reasoning_effort") != "max"
        ):
            raise SummaryError(f"model treatment differs for {queue_case_id}")
        if (
            marker.get("outcome") != "runnable_unqualified"
            or marker.get("counts_as_runnable_bundle") is not True
            or marker.get("counts_as_executable_parent") is not False
        ):
            raise SummaryError(f"parent boundary differs for {queue_case_id}")
        if (
            checked.get("semantic_binding") != "reviewer_claimed"
            or checked.get("gpu_test") != "not_run"
            or checked.get("review_state") != "checked"
            or checked.get("parent_contract", {}).get("status")
            != "runnable_by_parent_validator"
        ):
            raise SummaryError(f"checked claim boundary differs for {queue_case_id}")

        complete = checked.get("complete_parent_expressibility")
        delta = checked.get("delta_expressibility")
        if not isinstance(complete, dict) or not isinstance(delta, dict):
            raise SummaryError(f"checked lanes are invalid for {queue_case_id}")
        primary_class = str(checked.get("primary_class"))
        complete_class = str(complete.get("classification"))
        delta_class = str(delta.get("classification"))
        owner_scope = str(complete.get("owner_scope"))
        checked_schema = checked.get("schema")
        if checked_schema not in CHECKED_SCHEMAS:
            raise SummaryError(f"checked schema differs for {queue_case_id}")
        assessment_scope = complete.get("assessment_scope")
        if checked_schema == "open-cake.aka-expressibility-checked.v2":
            if assessment_scope is not None:
                raise SummaryError(
                    f"legacy checked scope is unexpectedly present for {queue_case_id}"
                )
            assessment_scope = "legacy_unspecified"
        if assessment_scope not in ASSESSMENT_SCOPES:
            raise SummaryError(f"checked assessment scope differs for {queue_case_id}")
        if primary_class != complete_class:
            raise SummaryError(f"primary class differs from complete lane for {queue_case_id}")
        if (
            assessment_scope == "fixed_instance"
            and not complete_class.startswith("fixed_instance_candidate_")
        ):
            raise SummaryError(f"fixed instance class differs for {queue_case_id}")
        if (
            assessment_scope == "contract"
            and complete_class.startswith("fixed_instance_candidate_")
        ):
            raise SummaryError(f"complete parent class differs for {queue_case_id}")
        compiler_check = str(complete.get("compiler_check"))
        primary_counts[primary_class] += 1
        complete_counts[complete_class] += 1
        delta_counts[delta_class] += 1
        owner_counts[owner_scope] += 1
        scope_counts[str(assessment_scope)] += 1
        compiler_counts[compiler_check] += 1
        gap = complete.get("missing_ir")
        capability = gap.get("capability") if isinstance(gap, dict) else None
        gap_text = " ".join(
            str(value)
            for value in (
                capability or "",
                gap.get("irreducible_semantics", "") if isinstance(gap, dict) else "",
                complete.get("reason", ""),
            )
        ).lower()
        matched_signals = [
            name for name, markers in SIGNALS.items() if any(marker in gap_text for marker in markers)
        ]
        signal_counts.update(matched_signals)
        rows.append(
            {
                "queue_case_id": queue_case_id,
                "portable_case_id": entry.get("portable_case_id"),
                "derived_parent_id": entry.get("derived_parent_id"),
                "source_identity": entry.get("source_identity"),
                "primary_class": primary_class,
                "complete_parent_class": complete_class,
                "assessment_scope": assessment_scope,
                "owner_scope": owner_scope,
                "owner_relation": complete.get("owner_relation"),
                "compiler_check": compiler_check,
                "delta_class": delta_class,
                "gap_capability": capability,
                "gap_signals": matched_signals,
                "reason": complete.get("reason"),
            }
        )

    if len(rows) != expected_count:
        raise SummaryError("derived report row count differs")
    implementation = pool.get("implementation")
    if not isinstance(implementation, dict):
        raise SummaryError("pool implementation identity is absent")
    return {
        "schema": SUMMARY_SCHEMA,
        "campaign_id": "aka-portable-parent-v2-sol-max-w30-20260829",
        "source_revision": final.get("source_revision"),
        "open_cake_revision": implementation.get("git_commit"),
        "model": pool.get("model"),
        "reasoning_effort": pool.get("reasoning_effort"),
        "codex_version": pool.get("codex", {}).get("version"),
        "max_workers": final.get("max_workers"),
        "finished_at": final.get("finished_at"),
        "records": len(rows),
        "worker_statuses": final.get("counts"),
        "primary_class_counts": dict(sorted(primary_counts.items())),
        "complete_parent_class_counts": dict(sorted(complete_counts.items())),
        "delta_class_counts": dict(sorted(delta_counts.items())),
        "owner_scope_counts": dict(sorted(owner_counts.items())),
        "assessment_scope_counts": dict(sorted(scope_counts.items())),
        "compiler_check_counts": dict(sorted(compiler_counts.items())),
        "retrieval_signal_counts": dict(sorted(signal_counts.items())),
        "claim_boundary": {
            "parent_status": "runnable_by_parent_validator",
            "portable_qualification": "projection_not_node_custody",
            "semantic_binding": "reviewer_claimed",
            "gpu_test": "not_run",
            "compiler_maturity": "draft",
            "allowed_claim": "provisional_ir_expressibility_review_only",
        },
        "rows": rows,
    }


def markdown(summary: dict[str, object]) -> str:
    primary = summary["primary_class_counts"]
    signals = summary["retrieval_signal_counts"]
    lines = [
        "# AKA portable parent v2 IR review",
        "",
        "Status: terminal derived report, 2026-08-29.",
        "",
        "## Outcome",
        "",
        f"The bounded Sol/max pool completed {summary['records']}/{summary['records']} portable parent reviews with no worker failure. The results are provisional expressibility assessments, not Compiler Corpus acceptance or GPU correctness of generated Cake code.",
        "",
        "| Primary class | Count |",
        "| --- | ---: |",
    ]
    for name, count in primary.items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## Frozen treatment",
            "",
            f"- AKA source revision: `{summary['source_revision']}`",
            f"- Open Cake revision: `{summary['open_cake_revision']}`",
            f"- Model: `{summary['model']}` with `{summary['reasoning_effort']}` reasoning",
            f"- Codex: `{summary['codex_version']}`",
            f"- Maximum concurrent workers: `{summary['max_workers']}`",
            "- Execution: independent cases, no retry and no sibling rollback",
            "- GPU execution: not run",
            "",
            "## Claim boundary",
            "",
            "Every portable record was materialized as a canonical `runnable_unqualified` parent. The fixed parent validator accepted the runnable source bundle, but the portable qualification projection was not re-established as node custody. Every checked result remains `semantic_binding=reviewer_claimed`, `gpu_test=not_run`, and uses the draft Compiler.",
            "",
            "The 100 parents were selected as the source-identity difference between the 150-record v2 snapshot and the 50-record v1 snapshot. This is not a representative IR coverage rate over all AKA rows; it is a challenge set reconstructed from historical parent-invalid records.",
            "",
            "## Retrieval-only signals",
            "",
            "These deterministic lexical projections exist only to retrieve cases for human review. They overlap, do not measure gaps, and cannot rank or authorize an IR primitive.",
            "",
            "| Signal | Cases |",
            "| --- | ---: |",
        ]
    )
    for name, count in signals.items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `fixed_instance_candidate_lowerable` means one static case-local Schedule passed Compiler assessment and lowering; it says nothing about the complete runtime parent domain.",
            "- `schedule_candidate_lowerable` is reserved for a reviewer claim that the case-local Schedule covers the whole narrowed parent contract. It still does not prove semantic equivalence or GPU correctness.",
            "- `schedule_gap_candidate` identifies a runnable derived single-kernel parent whose reviewer proposed an irreducible missing capability. Human review and independent recurrence are required before changing the IR.",
            "- `program_redirect_candidate` and `portfolio_redirect_candidate` preserve the one-Schedule/one-kernel boundary rather than encoding multi-launch relations or runtime dispatch as Schedule flags.",
            "",
            "## Per-parent results",
            "",
            "| Queue case | Derived parent | Source | Primary class | Scope | Owner | Compiler | Gap capability |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["rows"]:
        source = row["source_identity"]
        source_text = f"`{source[0]}:{source[1]}:{source[2]}`"
        capability = str(row.get("gap_capability") or "").replace("|", "\\|")
        lines.append(
            f"| `{row['queue_case_id']}` | `{row['derived_parent_id']}` | {source_text} | `{row['primary_class']}` | `{row['assessment_scope']}` | `{row['owner_scope']}` | `{row['compiler_check']}` | {capability} |"
        )
    lines.extend(
        [
            "",
            "## Promotion boundary",
            "",
            "Governance policy, not a finding of this corpus: a recurring gap may enter an IR proposal only after at least two independent source-complete cases require the same irreducible commitment. A proposal must update typed IR, authoring Schema, verifier, analysis and lowering together; add a minimal positive and near-miss falsifier; pass the full Corpus Gate; receive external human release approval; and obtain independent target correctness before any coverage claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_new(path: Path, payload: bytes) -> None:
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise SummaryError(f"cannot create report {path}: {error}") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_root", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        summary = summarize(arguments.pool_root, expected_count=arguments.expected_count)
        write_new(
            arguments.output_json,
            (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        write_new(arguments.output_markdown, markdown(summary).encode("utf-8"))
    except (OSError, SummaryError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "records": summary["records"],
                "primary_class_counts": summary["primary_class_counts"],
                "output_json": str(arguments.output_json),
                "output_markdown": str(arguments.output_markdown),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
