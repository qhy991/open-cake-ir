#!/usr/bin/env python3
"""Build a fail-closed execution order for the admitted AKA v1 review corpus.

The immutable AKA commit owns source rows.  A mechanism-augmentation campaign may
prioritize rows, but it cannot make them Compiler Corpus cases or canonical parent
completions.  This planner therefore keeps three facts separate: historical parent
qualification, terminal augmentation validity, and syntactic single-kernel scope.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_aka_corpus import (  # noqa: E402
    SourceRecord,
    load_records,
    projected_case,
    verify_git_snapshot,
)


PLAN_SCHEMA = "open-cake.aka-expressibility-execution-plan.v1"
RESULT_SCHEMA = "aka.mechanism-augmentation-result.v1"
TERMINAL_VALIDITIES = frozenset(
    {"valid", "valid_neutral", "valid_regression", "valid_improvement_unsealed"}
)
PRIORITIES = (
    "parent_qualified_terminal_valid_schedule",
    "parent_qualified_terminal_valid_program",
    "parent_qualified_other_schedule",
    "parent_qualified_other_program",
    "parent_invalid",
)


class PlanError(ValueError):
    """The source/campaign relation is incomplete or ambiguous."""


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise PlanError(f"{label} is not a regular non-symlink file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PlanError(f"{label} must be one JSON object")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise PlanError(f"{label} is not a regular non-symlink file: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PlanError(f"cannot read {label} {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise PlanError(f"{label}:{line_number} is malformed JSON") from error
        if not isinstance(value, dict):
            raise PlanError(f"{label}:{line_number} is not an object")
        rows.append(value)
    return rows


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _augmentation_ids(record: SourceRecord) -> tuple[str, str]:
    stem = (
        f"{record.category}__{record.operator}__{record.task}"
        f"__l{record.line_number:06d}"
    )
    return f"{stem}_b200_v1", f"{stem}__e2eb200v1"


def _primary_scope(record: SourceRecord) -> str:
    matches = [
        signals["scope_signal"]
        for role, signals in record.artifact_signals.items()
        if record.artifact_fields[role] == record.primary_field
    ]
    if len(matches) != 1 or matches[0] not in {
        "single_kernel",
        "multi_kernel_or_launch",
        "fragment_or_library",
    }:
        raise PlanError(f"source row has no unique primary scope: {record.relative_path}")
    return str(matches[0])


def _priority(parent_status: str, validity: str, scope: str) -> str:
    if parent_status == "invalid":
        return "parent_invalid"
    if parent_status != "qualified":
        raise PlanError(f"unsupported historical parent_status: {parent_status}")
    lane = "schedule" if scope == "single_kernel" else "program"
    terminal = "terminal_valid" if validity in TERMINAL_VALIDITIES else "other"
    return f"parent_qualified_{terminal}_{lane}"


def build_plan(
    *,
    dataset_root: Path,
    source_revision: str,
    campaign_root: Path,
    record_format: str = "aka_v1_operator_sft",
) -> dict[str, object]:
    if record_format != "aka_v1_operator_sft":
        raise PlanError("the historical B200 campaign maps only aka_v1_operator_sft")
    dataset_root = dataset_root.resolve(strict=True)
    campaign_root = campaign_root.resolve(strict=True)
    snapshot = verify_git_snapshot(dataset_root, source_revision)
    records = load_records(snapshot, record_format=record_format)
    ledger_path = campaign_root / "final-verified-ledger.jsonl"
    ledger = _load_jsonl(ledger_path, "final verified augmentation ledger")
    if len(ledger) != len(records):
        raise PlanError(
            f"campaign/source cardinality differs: {len(ledger)} != {len(records)}"
        )

    record_by_augmentation_id: dict[str, tuple[int, SourceRecord]] = {}
    for index, record in enumerate(records, 1):
        for augmentation_id in _augmentation_ids(record):
            if augmentation_id in record_by_augmentation_id:
                raise PlanError(f"ambiguous augmentation case id: {augmentation_id}")
            record_by_augmentation_id[augmentation_id] = (index, record)

    entries: list[dict[str, object]] = []
    observed_ids: set[str] = set()
    for row in ledger:
        augmentation_id = row.get("case_id")
        if not isinstance(augmentation_id, str) or not augmentation_id:
            raise PlanError("ledger row has no case_id")
        if augmentation_id in observed_ids:
            raise PlanError(f"duplicate ledger case_id: {augmentation_id}")
        observed_ids.add(augmentation_id)
        mapped = record_by_augmentation_id.get(augmentation_id)
        if mapped is None:
            raise PlanError(f"ledger case does not map to admitted source: {augmentation_id}")
        source_index, record = mapped

        relative_result = row.get("result_path")
        if not isinstance(relative_result, str) or not relative_result:
            raise PlanError(f"ledger result_path is missing: {augmentation_id}")
        raw_result = campaign_root / relative_result
        result_path = raw_result.resolve(strict=True)
        if raw_result.is_symlink() or not _inside(result_path, campaign_root):
            raise PlanError(f"campaign result escapes its root: {augmentation_id}")
        result = _load_object(result_path, f"augmentation result {augmentation_id}")
        if result.get("schema") != RESULT_SCHEMA:
            raise PlanError(f"unsupported augmentation result schema: {augmentation_id}")
        parent_status = result.get("parent_status")
        validity = result.get("validity")
        if parent_status not in {"qualified", "invalid"} or not isinstance(
            validity, str
        ):
            raise PlanError(f"augmentation result status is invalid: {augmentation_id}")
        if row.get("validity") != validity:
            raise PlanError(f"ledger/result validity differs: {augmentation_id}")

        scope = _primary_scope(record)
        priority = _priority(parent_status, validity, scope)
        source_case = projected_case(
            record,
            snapshot=snapshot,
            reported_dataset_label=dataset_root.name,
        )
        source_ref = source_case["source_ref"]
        entries.append(
            {
                "case_id": f"case-{source_index:06d}",
                "augmentation_case_id": augmentation_id,
                "augmentation_result": relative_result,
                "parent_status": parent_status,
                "augmentation_validity": validity,
                "primary_scope": scope,
                "priority": priority,
                "source_ref": source_ref,
                "parent_coordinate": (
                    f"{source_ref['revision']}:{source_ref['path']}:{source_ref['line']}"
                ),
                "record_field": source_ref["primary_field"],
                "canonical_parent_completion": None,
                "qualified_for_ir_claim": False,
                "next_gate": (
                    "canonical_complete_kernel_parent_bridge"
                    if parent_status == "qualified"
                    else "parent_contract_reconstruction"
                ),
            }
        )

    if len(observed_ids) != len(records):
        raise PlanError("campaign does not map one-to-one to the admitted source rows")
    entries.sort(key=lambda item: (PRIORITIES.index(str(item["priority"])), item["case_id"]))
    counts = Counter(str(entry["priority"]) for entry in entries)
    return {
        "schema": PLAN_SCHEMA,
        "source": {
            "revision": snapshot.revision,
            "dataset_path": snapshot.dataset_path,
            "record_format": record_format,
            "record_count": len(records),
        },
        "campaign": {
            "root": str(campaign_root),
            "ledger": str(ledger_path.relative_to(campaign_root)),
        },
        "claim_boundary": {
            "historical_parent_qualified_is_not_canonical_parent_completion": True,
            "syntactic_single_kernel_is_not_runtime_executability": True,
            "qualified_for_ir_claim": 0,
        },
        "counts": {priority: counts.get(priority, 0) for priority in PRIORITIES},
        "entries": entries,
    }


def write_plan(path: Path, plan: Mapping[str, object]) -> None:
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise PlanError(f"cannot create execution plan {path}: {error}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def selected_entries(
    plan: Mapping[str, object], priorities: Sequence[str]
) -> list[dict[str, object]]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise PlanError("unsupported execution plan schema")
    unknown = set(priorities) - set(PRIORITIES)
    if unknown:
        raise PlanError(f"unknown priorities: {sorted(unknown)}")
    entries = plan.get("entries")
    if not isinstance(entries, list) or any(not isinstance(row, dict) for row in entries):
        raise PlanError("execution plan entries are invalid")
    return [dict(row) for row in entries if row.get("priority") in priorities]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--record-format", default="aka_v1_operator_sft", choices=("aka_v1_operator_sft",)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        plan = build_plan(
            dataset_root=arguments.dataset_root,
            source_revision=arguments.source_revision,
            campaign_root=arguments.campaign_root,
            record_format=arguments.record_format,
        )
        write_plan(arguments.output, plan)
    except (OSError, PlanError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(arguments.output), "counts": plan["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
