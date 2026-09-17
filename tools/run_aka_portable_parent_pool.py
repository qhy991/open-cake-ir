#!/usr/bin/env python3
"""Run a commit-bound delta of portable AKA parent reviews with bounded concurrency."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.review_aka_expressibility import _git_closure_identity  # noqa: E402
from tools.run_aka_expressibility_codex import (  # noqa: E402
    MODEL,
    REASONING_EFFORT,
    _codex_identity,
)
from tools.run_aka_portable_parents_codex import (  # noqa: E402
    PortableEntry,
    PortableParentError,
    _json_bytes,
    _repository_root,
    _write_new,
    build_portable_plan,
)


POOL_SCHEMA = "open-cake.aka-portable-parent-pool.v1"
WORKER_SCHEMA = "open-cake.aka-portable-parent-pool-worker.v1"
FINAL_SCHEMA = "open-cake.aka-portable-parent-pool-final.v1"
MAX_WORKERS = 30


class PortablePoolError(ValueError):
    """The portable delta or bounded pool contract is invalid."""


def load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PortablePoolError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PortablePoolError(f"{label} must be one JSON object")
    return value


def recovery_basis(
    predecessor: Mapping[str, object],
    *,
    predecessor_path: Path,
    source_revision: str,
    requested_queue_case_ids: Sequence[str] | None,
) -> dict[str, object]:
    """Bind an explicit recovery subset to failed workers in one terminal pool."""

    if requested_queue_case_ids is None:
        raise PortablePoolError("recovery requires explicit queue-case ids")
    if predecessor.get("schema") != FINAL_SCHEMA:
        raise PortablePoolError("recovery predecessor has an unsupported schema")
    if predecessor.get("source_revision") != source_revision:
        raise PortablePoolError("recovery predecessor source revision differs")
    results = predecessor.get("results")
    if not isinstance(results, list):
        raise PortablePoolError("recovery predecessor has no worker results")
    failed_ids = {
        result.get("queue_case_id")
        for result in results
        if isinstance(result, dict) and result.get("status") == "failed"
    }
    requested = set(requested_queue_case_ids)
    outside = sorted(requested - failed_ids)
    if outside:
        raise PortablePoolError(
            "recovery selection was not failed in the predecessor: "
            + ", ".join(outside)
        )
    return {
        "pool_final": str(predecessor_path.resolve(strict=True)),
        "finished_at": predecessor.get("finished_at"),
        "failed_domain_count": len(failed_ids),
        "selected_failed_count": len(requested),
    }


def portable_identity(entry: PortableEntry) -> tuple[str, int, str]:
    source = entry.portable_record.get("provenance", {}).get("source_selection")
    if not isinstance(source, dict):
        raise PortablePoolError("portable entry has no source selection")
    path = source.get("path")
    line = source.get("line")
    field = source.get("parent_field")
    if not isinstance(path, str) or not isinstance(line, int) or not isinstance(field, str):
        raise PortablePoolError("portable source identity is invalid")
    return path, line, field


def select_new_entries(
    current: Sequence[PortableEntry],
    prior: Sequence[PortableEntry],
    *,
    expected_count: int,
    expected_review_ready_count: int | None = None,
) -> tuple[list[PortableEntry], list[PortableEntry]]:
    prior_ids = {portable_identity(entry) for entry in prior}
    current_ids = {portable_identity(entry) for entry in current}
    if not prior_ids <= current_ids:
        raise PortablePoolError("prior portable identities are not a subset of current")
    new_entries = [
        entry for entry in current if portable_identity(entry) not in prior_ids
    ]
    if len(new_entries) != expected_count:
        raise PortablePoolError(
            f"expected {expected_count} new portable parents, observed {len(new_entries)}"
        )
    selected = [entry for entry in new_entries if entry.review_ready]
    blocked = [entry for entry in new_entries if not entry.review_ready]
    if expected_review_ready_count is None and blocked:
        raise PortablePoolError(
            "new portable parents contain source-field blockers: "
            + ", ".join(entry.queue_case_id for entry in blocked)
        )
    expected_ready = (
        expected_count
        if expected_review_ready_count is None
        else expected_review_ready_count
    )
    if len(selected) != expected_ready:
        raise PortablePoolError(
            f"expected {expected_ready} review-ready portable parents, "
            f"observed {len(selected)}"
        )
    return selected, blocked


def select_requested_entries(
    entries: Sequence[PortableEntry],
    requested_queue_case_ids: Sequence[str] | None,
) -> list[PortableEntry]:
    """Select an explicit recovery subset while preserving the frozen delta order."""

    if requested_queue_case_ids is None:
        return list(entries)
    requested = tuple(requested_queue_case_ids)
    if not requested:
        raise PortablePoolError("explicit queue-case selection is empty")
    if len(set(requested)) != len(requested):
        raise PortablePoolError("explicit queue-case selection contains duplicates")
    available = {entry.queue_case_id for entry in entries}
    missing = sorted(set(requested) - available)
    if missing:
        raise PortablePoolError(
            "explicit queue-case selection is outside the review-ready delta: "
            + ", ".join(missing)
        )
    selected = [entry for entry in entries if entry.queue_case_id in set(requested)]
    if len(selected) != len(requested):
        raise PortablePoolError("explicit queue-case selection is ambiguous")
    return selected


def _worker_command(
    *,
    source_dataset_root: Path,
    portable_dataset_root: Path,
    source_revision: str,
    portable_record_count: int,
    case_root: Path,
    entry: PortableEntry,
    timeout_seconds: int,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools/run_aka_portable_parents_codex.py"),
        "--source-dataset-root",
        str(source_dataset_root),
        "--portable-dataset-root",
        str(portable_dataset_root),
        "--source-revision",
        source_revision,
        "--completion-root",
        str(case_root / "parents"),
        "--review-root",
        str(case_root / "reviews"),
        "--expected-count",
        str(portable_record_count),
        "--only-portable-case-id",
        str(entry.portable_record["case_id"]),
        "--limit",
        "1",
        "--timeout-seconds",
        str(timeout_seconds),
    ]


def _run_worker(
    *,
    batch_root: Path,
    source_dataset_root: Path,
    portable_dataset_root: Path,
    source_revision: str,
    portable_record_count: int,
    entry: PortableEntry,
    timeout_seconds: int,
) -> dict[str, object]:
    case_root = batch_root / "cases" / entry.queue_case_id
    case_root.mkdir(parents=True, exist_ok=False)
    command = _worker_command(
        source_dataset_root=source_dataset_root,
        portable_dataset_root=portable_dataset_root,
        source_revision=source_revision,
        portable_record_count=portable_record_count,
        case_root=case_root,
        entry=entry,
        timeout_seconds=timeout_seconds,
    )
    started = {
        "schema": WORKER_SCHEMA,
        "status": "running",
        "queue_case_id": entry.queue_case_id,
        "portable_case_id": entry.portable_record["case_id"],
        "derived_parent_id": entry.portable_record["derived_parent_id"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
    }
    _write_new(case_root / "worker.started.json", _json_bytes(started))
    stdout_path = case_root / "child.stdout.json"
    stderr_path = case_root / "child.stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    finished = {
        "schema": WORKER_SCHEMA,
        "status": "completed" if completed.returncode == 0 else "failed",
        "queue_case_id": entry.queue_case_id,
        "portable_case_id": entry.portable_record["case_id"],
        "derived_parent_id": entry.portable_record["derived_parent_id"],
        "started_at": started["started_at"],
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": completed.returncode,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
    }
    _write_new(case_root / "worker.finished.json", _json_bytes(finished))
    return finished


def build_delta_plan(
    *,
    source_dataset_root: Path,
    portable_dataset_root: Path,
    prior_portable_dataset_root: Path,
    source_revision: str,
    portable_record_count: int,
    prior_record_count: int,
    expected_new_count: int,
    expected_review_ready_count: int | None = None,
    only_queue_case_ids: Sequence[str] | None = None,
) -> tuple[dict[str, object], list[PortableEntry]]:
    current_plan, current = build_portable_plan(
        source_dataset_root=source_dataset_root,
        portable_dataset_root=portable_dataset_root,
        source_revision=source_revision,
        expected_count=portable_record_count,
    )
    prior_plan, prior = build_portable_plan(
        source_dataset_root=source_dataset_root,
        portable_dataset_root=prior_portable_dataset_root,
        source_revision=source_revision,
        expected_count=prior_record_count,
    )
    selected, blocked = select_new_entries(
        current,
        prior,
        expected_count=expected_new_count,
        expected_review_ready_count=expected_review_ready_count,
    )
    review_ready_domain_count = len(selected)
    selected = select_requested_entries(selected, only_queue_case_ids)
    plan = {
        "schema": POOL_SCHEMA,
        "source_revision": source_revision,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "portable_dataset": current_plan["portable_dataset"],
        "prior_portable_dataset": prior_plan["portable_dataset"],
        "new_identity_count": expected_new_count,
        "review_ready_domain_count": review_ready_domain_count,
        "selected": len(selected),
        "selection_mode": (
            "explicit_queue_case_ids"
            if only_queue_case_ids is not None
            else "full_review_ready_delta"
        ),
        "requested_queue_case_ids": (
            list(only_queue_case_ids) if only_queue_case_ids is not None else []
        ),
        "deterministically_blocked": [
            {
                "queue_case_id": entry.queue_case_id,
                "portable_case_id": entry.portable_record["case_id"],
                "source_identity": list(portable_identity(entry)),
                "blocker": entry.blocker,
            }
            for entry in blocked
        ],
        "entries": [
            {
                "queue_case_id": entry.queue_case_id,
                "portable_case_id": entry.portable_record["case_id"],
                "derived_parent_id": entry.portable_record["derived_parent_id"],
                "source_identity": list(portable_identity(entry)),
            }
            for entry in selected
        ],
        "claim_boundary": (
            "portable_runnable_parent_provisional_ir_review_only"
        ),
    }
    return plan, selected


def run_pool(
    *,
    source_dataset_root: Path,
    portable_dataset_root: Path,
    prior_portable_dataset_root: Path,
    source_revision: str,
    portable_record_count: int,
    prior_record_count: int,
    expected_new_count: int,
    expected_review_ready_count: int | None,
    only_queue_case_ids: Sequence[str] | None,
    recovery_from_pool_final: Path | None,
    batch_root: Path,
    max_workers: int,
    timeout_seconds: int,
) -> dict[str, object]:
    if max_workers < 1 or max_workers > MAX_WORKERS:
        raise PortablePoolError(f"max-workers must be between 1 and {MAX_WORKERS}")
    if timeout_seconds <= 0:
        raise PortablePoolError("timeout must be positive")
    source_dataset_root = source_dataset_root.resolve(strict=True)
    portable_dataset_root = portable_dataset_root.resolve(strict=True)
    prior_portable_dataset_root = prior_portable_dataset_root.resolve(strict=True)
    batch_root = batch_root.resolve(strict=False)
    repository = _repository_root(source_dataset_root)
    if os.path.lexists(batch_root):
        raise PortablePoolError(f"create-only batch root already exists: {batch_root}")
    try:
        batch_root.relative_to(ROOT)
    except ValueError:
        pass
    else:
        raise PortablePoolError("batch root must be outside the Open Cake checkout")
    try:
        batch_root.relative_to(repository)
    except ValueError:
        pass
    else:
        raise PortablePoolError("batch root must be outside the AKA repository")

    codex = shutil.which("codex")
    if codex is None:
        raise PortablePoolError("codex executable is unavailable")
    plan, selected = build_delta_plan(
        source_dataset_root=source_dataset_root,
        portable_dataset_root=portable_dataset_root,
        prior_portable_dataset_root=prior_portable_dataset_root,
        source_revision=source_revision,
        portable_record_count=portable_record_count,
        prior_record_count=prior_record_count,
        expected_new_count=expected_new_count,
        expected_review_ready_count=expected_review_ready_count,
        only_queue_case_ids=only_queue_case_ids,
    )
    if only_queue_case_ids is not None and recovery_from_pool_final is None:
        raise PortablePoolError(
            "explicit recovery selection requires --recovery-from-pool-final"
        )
    if recovery_from_pool_final is not None:
        predecessor_path = recovery_from_pool_final.resolve(strict=True)
        plan["recovery_from"] = recovery_basis(
            load_json_object(predecessor_path, "recovery predecessor"),
            predecessor_path=predecessor_path,
            source_revision=source_revision,
            requested_queue_case_ids=only_queue_case_ids,
        )
    batch_root.mkdir(parents=True, exist_ok=False)
    (batch_root / "cases").mkdir()
    _write_new(batch_root / "pool.plan.json", _json_bytes(plan))
    _write_new(
        batch_root / "pool.json",
        _json_bytes(
            {
                "schema": POOL_SCHEMA,
                "source_revision": source_revision,
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "codex": _codex_identity(Path(codex)),
                "max_workers": max_workers,
                "new_identity_count": plan["new_identity_count"],
                "review_ready_domain_count": plan["review_ready_domain_count"],
                "selected": len(selected),
                "selection_mode": plan["selection_mode"],
                "recovery_from": plan.get("recovery_from"),
                "deterministically_blocked": len(
                    plan["deterministically_blocked"]
                ),
                "execution": "independent_cases_no_retry_no_sibling_rollback",
                "gpu": "not_used",
                "implementation": _git_closure_identity(
                    ROOT,
                    (
                        "tools/run_aka_portable_parent_pool.py",
                        "tools/run_aka_portable_parents_codex.py",
                        "tools/run_aka_expressibility_codex.py",
                        "tools/review_aka_expressibility.py",
                    ),
                    "portable parent pool",
                ),
            }
        ),
    )

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_worker,
                batch_root=batch_root,
                source_dataset_root=source_dataset_root,
                portable_dataset_root=portable_dataset_root,
                source_revision=source_revision,
                portable_record_count=portable_record_count,
                entry=entry,
                timeout_seconds=timeout_seconds,
            ): entry
            for entry in selected
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # Preserve one controller-level failure per case.
                failure = {
                    "schema": WORKER_SCHEMA,
                    "status": "controller_failed",
                    "queue_case_id": entry.queue_case_id,
                    "portable_case_id": entry.portable_record["case_id"],
                    "derived_parent_id": entry.portable_record["derived_parent_id"],
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(error),
                }
                case_root = batch_root / "cases" / entry.queue_case_id
                case_root.mkdir(parents=True, exist_ok=True)
                _write_new(case_root / "worker.controller-failure.json", _json_bytes(failure))
                results.append(failure)

    counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
    final = {
        "schema": FINAL_SCHEMA,
        "source_revision": source_revision,
        "new_identity_count": plan["new_identity_count"],
        "review_ready_domain_count": plan["review_ready_domain_count"],
        "selected": len(selected),
        "selection_mode": plan["selection_mode"],
        "recovery_from": plan.get("recovery_from"),
        "deterministically_blocked": plan["deterministically_blocked"],
        "max_workers": max_workers,
        "counts": dict(sorted(counts.items())),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "results": sorted(results, key=lambda result: str(result["queue_case_id"])),
    }
    _write_new(batch_root / "pool.final.json", _json_bytes(final))
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--portable-dataset-root", type=Path, required=True)
    parser.add_argument("--prior-portable-dataset-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--portable-record-count", type=int, required=True)
    parser.add_argument("--prior-record-count", type=int, required=True)
    parser.add_argument("--expected-new-count", type=int, required=True)
    parser.add_argument("--expected-review-ready-count", type=int)
    parser.add_argument("--only-queue-case-id", action="append")
    parser.add_argument("--recovery-from-pool-final", type=Path)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.dry_run:
            plan, selected = build_delta_plan(
                source_dataset_root=arguments.source_dataset_root,
                portable_dataset_root=arguments.portable_dataset_root,
                prior_portable_dataset_root=arguments.prior_portable_dataset_root,
                source_revision=arguments.source_revision,
                portable_record_count=arguments.portable_record_count,
                prior_record_count=arguments.prior_record_count,
                expected_new_count=arguments.expected_new_count,
                expected_review_ready_count=arguments.expected_review_ready_count,
                only_queue_case_ids=arguments.only_queue_case_id,
            )
            print(json.dumps({"selected": len(selected), "plan": plan}, indent=2, sort_keys=True))
            return 0
        if arguments.batch_root is None:
            raise PortablePoolError("batch-root is required unless dry-run is used")
        result = run_pool(
            source_dataset_root=arguments.source_dataset_root,
            portable_dataset_root=arguments.portable_dataset_root,
            prior_portable_dataset_root=arguments.prior_portable_dataset_root,
            source_revision=arguments.source_revision,
            portable_record_count=arguments.portable_record_count,
            prior_record_count=arguments.prior_record_count,
            expected_new_count=arguments.expected_new_count,
            expected_review_ready_count=arguments.expected_review_ready_count,
            only_queue_case_ids=arguments.only_queue_case_id,
            recovery_from_pool_final=arguments.recovery_from_pool_final,
            batch_root=arguments.batch_root,
            max_workers=arguments.max_workers,
            timeout_seconds=arguments.timeout_seconds,
        )
    except (OSError, PortableParentError, PortablePoolError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return (
        0
        if result["counts"].get("completed", 0) == result["selected"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
