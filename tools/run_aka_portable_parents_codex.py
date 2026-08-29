#!/usr/bin/env python3
"""Review commit-bound portable AKA parents sequentially with GPT-5.6 Sol max.

The portable dataset contains complete narrowed semantics and source bundles, but its
qualification stages are projections rather than node-owned evidence.  This adapter
therefore creates canonical ``runnable_unqualified`` parent completions, validates them,
and permits only provisional Cake expressibility review.  It never upgrades the portable
projection into GPU custody, launches a GPU, retries a case, or skips a failed case.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_aka_corpus import (  # noqa: E402
    SourceRecord,
    load_records,
    verify_git_snapshot,
)
from tools.review_aka_expressibility import (  # noqa: E402
    DEFAULT_PARENT_VALIDATOR,
    _git_closure_identity,
)
from tools.run_aka_expressibility_codex import (  # noqa: E402
    MODEL,
    REASONING_EFFORT,
    RunnerError,
    _codex_identity,
    run_queue,
)


PORTABLE_SCHEMA = "aka.portable-kernel-parent.v1"
PLAN_SCHEMA = "open-cake.aka-portable-parent-plan.v1"
ADAPTER_SCHEMA = "open-cake.aka-portable-parent-adapter.v1"
VALIDATOR_RECEIPT_SCHEMA = "open-cake.aka-portable-parent-validation.v1"
COMMIT = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_TIMEOUT_SECONDS = 7200


class PortableParentError(ValueError):
    """The commit-bound portable parent or adapter state is not admissible."""


@dataclass(frozen=True)
class PortableEntry:
    queue_case_id: str
    source_record: SourceRecord
    portable_record: Mapping[str, Any]
    review_ready: bool
    blocker: str | None


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as error:
        raise PortableParentError(f"cannot create {path}: {error}") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git(
    repository: Path, arguments: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "-C", str(repository), *arguments],
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.decode("utf-8", errors="replace").strip()
            if isinstance(error, subprocess.CalledProcessError) and error.stderr
            else str(error)
        )
        raise PortableParentError(f"cannot read portable Git snapshot: {detail}") from error


def _repository_root(path: Path) -> Path:
    result = _git(path, ["rev-parse", "--show-toplevel"])
    return Path(result.stdout.decode("utf-8").strip()).resolve(strict=True)


def _repository_relative(repository: Path, path: Path, label: str) -> PurePosixPath:
    try:
        relative = path.resolve(strict=True).relative_to(repository)
    except ValueError as error:
        raise PortableParentError(f"{label} is outside its Git repository") from error
    return PurePosixPath(relative.as_posix())


def _git_blob(repository: Path, revision: str, relative: PurePosixPath) -> tuple[bytes, int]:
    listing = _git(repository, ["ls-tree", revision, "--", relative.as_posix()])
    entries = [line for line in listing.stdout.decode("utf-8").splitlines() if line]
    if len(entries) != 1:
        raise PortableParentError(f"portable path is not one Git blob: {relative}")
    try:
        metadata, observed_path = entries[0].split("\t", 1)
        mode, object_type, object_id = metadata.split()
    except ValueError as error:
        raise PortableParentError(f"cannot parse Git entry for {relative}") from error
    if observed_path != relative.as_posix() or object_type != "blob" or mode not in {
        "100644",
        "100755",
    }:
        raise PortableParentError(f"portable path is not a regular Git blob: {relative}")
    payload = _git(repository, ["cat-file", "blob", object_id]).stdout
    return payload, 0o755 if mode == "100755" else 0o644


def _object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PortableParentError(f"{label} is not one UTF-8 JSON object") from error
    if not isinstance(value, dict):
        raise PortableParentError(f"{label} is not one JSON object")
    return value


def _records(payload: bytes) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise PortableParentError("portable records are not UTF-8") from error
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise PortableParentError(f"portable records line {index} is malformed") from error
        if not isinstance(value, dict):
            raise PortableParentError(f"portable records line {index} is not an object")
        rows.append(value)
    return rows


def _portable_relative(
    portable_dataset: PurePosixPath, value: object, label: str
) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise PortableParentError(f"{label} is not a non-empty path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise PortableParentError(f"{label} escapes the portable dataset")
    return portable_dataset / relative


def build_portable_plan(
    *,
    source_dataset_root: Path,
    portable_dataset_root: Path,
    source_revision: str,
) -> tuple[dict[str, object], list[PortableEntry]]:
    if COMMIT.fullmatch(source_revision) is None:
        raise PortableParentError("source revision must be one exact lowercase commit")
    source_dataset_root = source_dataset_root.resolve(strict=True)
    portable_dataset_root = portable_dataset_root.resolve(strict=True)
    repository = _repository_root(source_dataset_root)
    if _repository_root(portable_dataset_root) != repository:
        raise PortableParentError("source and portable datasets belong to different repositories")
    resolved = _git(repository, ["rev-parse", "--verify", f"{source_revision}^{{commit}}"])
    if resolved.stdout.decode("utf-8").strip() != source_revision:
        raise PortableParentError("source revision does not resolve exactly")

    snapshot = verify_git_snapshot(source_dataset_root, source_revision)
    source_records = load_records(snapshot, record_format="aka_v1_operator_sft")
    source_by_coordinate = {
        (record.relative_path, record.line_number): (index, record)
        for index, record in enumerate(source_records, 1)
    }
    portable_relative = _repository_relative(
        repository, portable_dataset_root, "portable dataset"
    )
    payload, _ = _git_blob(
        repository, source_revision, portable_relative / "records.jsonl"
    )
    portable_records = _records(payload)
    if len(portable_records) != 50:
        raise PortableParentError(
            f"portable review requires exactly 50 records, observed {len(portable_records)}"
        )

    entries: list[PortableEntry] = []
    seen_queue: set[str] = set()
    seen_portable: set[str] = set()
    seen_derived: set[str] = set()
    seen_locators: set[tuple[str, str]] = set()
    for row_index, row in enumerate(portable_records, 1):
        label = f"portable record {row_index}"
        if (
            row.get("schema") != PORTABLE_SCHEMA
            or row.get("outcome") != "qualified"
            or row.get("recovery_mode") != "contract_narrowed"
            or row.get("training_eligibility") is not False
            or row.get("missing_facts") != []
        ):
            raise PortableParentError(f"{label} terminal contract is invalid")
        portable_case_id = row.get("case_id")
        if not isinstance(portable_case_id, str) or not portable_case_id:
            raise PortableParentError(f"{label} has no case id")
        if portable_case_id in seen_portable:
            raise PortableParentError(f"duplicate portable case id: {portable_case_id}")
        seen_portable.add(portable_case_id)
        derived_parent_id = row.get("derived_parent_id")
        if not isinstance(derived_parent_id, str) or not derived_parent_id:
            raise PortableParentError(f"{label} has no derived parent id")
        if derived_parent_id in seen_derived:
            raise PortableParentError(f"duplicate derived parent id: {derived_parent_id}")
        seen_derived.add(derived_parent_id)

        selection = row.get("provenance", {}).get("source_selection")
        if not isinstance(selection, dict):
            raise PortableParentError(f"{label} has no source selection")
        path = selection.get("path")
        line = selection.get("line")
        field = selection.get("parent_field")
        if (
            not isinstance(path, str)
            or not isinstance(line, int)
            or isinstance(line, bool)
            or line < 1
            or field not in {"input", "output"}
        ):
            raise PortableParentError(f"{label} source selection is invalid")
        mapped = source_by_coordinate.get((path, line))
        if mapped is None:
            raise PortableParentError(f"{label} does not map to the admitted source commit")
        source_index, source_record = mapped
        queue_case_id = f"case-{source_index:06d}"
        if queue_case_id in seen_queue:
            raise PortableParentError(f"portable records repeat source row {queue_case_id}")
        seen_queue.add(queue_case_id)
        if row.get("original_parent") != {
            "case_path": f"{path}:{line}",
            "record_field": field,
        }:
            raise PortableParentError(f"{label} original parent is not canonical")

        bundle = row.get("bundle")
        if not isinstance(bundle, dict):
            raise PortableParentError(f"{label} bundle is invalid")
        source_files = bundle.get("source_files")
        if not isinstance(source_files, dict) or "input.json" not in source_files:
            raise PortableParentError(f"{label} has no bundled source record")
        input_path = _portable_relative(
            portable_relative, source_files["input.json"], f"{label} input"
        )
        bundled_input = _object(
            _git_blob(repository, source_revision, input_path)[0], f"{label} input"
        )
        if bundled_input.get("record") != dict(source_record.fields):
            raise PortableParentError(f"{label} bundled source differs from source commit")

        qualification = row.get("qualification")
        if not isinstance(qualification, dict) or qualification.get("lifecycle") != "completed" or qualification.get("validity") != "valid":
            raise PortableParentError(f"{label} portable qualification projection is invalid")
        locator = qualification.get("locator")
        if not isinstance(locator, dict) or any(
            not isinstance(locator.get(key), str) or not locator[key]
            for key in ("node_id", "run_id")
        ):
            raise PortableParentError(f"{label} portable locator is invalid")
        locator_key = (locator["node_id"], locator["run_id"])
        if locator_key in seen_locators:
            raise PortableParentError(f"duplicate portable locator: {locator_key}")
        seen_locators.add(locator_key)
        for stage in ("compile", "correctness", "sanitize"):
            result = qualification.get("stages", {}).get(stage)
            if not isinstance(result, dict) or result.get("status") != "passed" or result.get("validity") != "valid":
                raise PortableParentError(f"{label} {stage} projection is invalid")
        correctness = qualification["stages"]["correctness"].get("workloads")
        if (
            not isinstance(correctness, list)
            or len(correctness) < 2
            or any(
                not isinstance(workload, dict) or workload.get("correct") is not True
                for workload in correctness
            )
        ):
            raise PortableParentError(
                f"{label} has no two complete-output correctness projections"
            )
        sanitizer = json.dumps(
            qualification["stages"]["sanitize"], ensure_ascii=False
        ).lower()
        if "memcheck" not in sanitizer or "racecheck" not in sanitizer:
            raise PortableParentError(f"{label} sanitizer projection is incomplete")
        review_ready = field == source_record.primary_field
        blocker = (
            None
            if review_ready
            else "source_field_override_required"
        )
        entries.append(
            PortableEntry(
                queue_case_id,
                source_record,
                row,
                review_ready,
                blocker,
            )
        )

    plan = {
        "schema": PLAN_SCHEMA,
        "source": {
            "repository": str(repository),
            "revision": source_revision,
            "dataset_path": snapshot.dataset_path,
            "record_format": "aka_v1_operator_sft",
        },
        "portable_dataset": {
            "path": portable_relative.as_posix(),
            "records": len(entries),
            "review_ready": sum(entry.review_ready for entry in entries),
            "blocked": sum(not entry.review_ready for entry in entries),
            "qualification_authority": "projection_not_node_custody",
        },
        "claim_boundary": {
            "parent_completion_outcome": "runnable_unqualified",
            "semantic_binding": "reviewer_claimed",
            "gpu_test": "not_run",
        },
        "entries": [
            {
                "queue_case_id": entry.queue_case_id,
                "portable_case_id": entry.portable_record["case_id"],
                "derived_parent_id": entry.portable_record["derived_parent_id"],
                "source_path": entry.source_record.relative_path,
                "source_line": entry.source_record.line_number,
                "source_field": entry.source_record.primary_field,
                "portable_parent_field": entry.portable_record["original_parent"][
                    "record_field"
                ],
                "review_ready": entry.review_ready,
                "blocker": entry.blocker,
            }
            for entry in entries
        ],
    }
    return plan, entries


def _copy_artifacts(
    *,
    repository: Path,
    revision: str,
    portable_dataset: PurePosixPath,
    record: Mapping[str, Any],
    case_root: Path,
) -> dict[str, str]:
    artifacts = record.get("artifacts")
    bundle = record.get("bundle")
    if not isinstance(artifacts, dict) or not isinstance(bundle, dict):
        raise PortableParentError("portable artifact declaration is invalid")
    bundle_path = PurePosixPath(str(bundle.get("path")))
    copied_roles: dict[str, str] = {}
    for role, values in sorted(artifacts.items()):
        if not isinstance(role, str) or not isinstance(values, list) or not values:
            raise PortableParentError(f"portable artifact role {role!r} is invalid")
        prefix = bundle_path / "sources" / role
        for index, value in enumerate(values):
            repository_path = _portable_relative(
                portable_dataset, value, f"artifact {role}[{index}]"
            )
            try:
                relative = PurePosixPath(str(value)).relative_to(prefix)
            except ValueError as error:
                raise PortableParentError(
                    f"artifact {role}[{index}] escapes its bundle role"
                ) from error
            payload, mode = _git_blob(repository, revision, repository_path)
            _write_new(case_root / "artifacts" / role / relative, payload, mode=mode)
        copied_roles[role] = f"artifacts/{role}"
    for required in ("baseline", "reference", "harness"):
        if required not in copied_roles:
            raise PortableParentError(f"portable record lacks {required} artifacts")
    return copied_roles


def materialize_portable_completion(
    *,
    completion_root: Path,
    repository: Path,
    revision: str,
    portable_dataset: PurePosixPath,
    source_dataset: PurePosixPath,
    entry: PortableEntry,
) -> Path:
    if not entry.review_ready:
        raise PortableParentError(
            f"portable parent is not review-ready: {entry.queue_case_id}: {entry.blocker}"
        )
    case_root = completion_root / "cases" / entry.queue_case_id
    case_root.mkdir(parents=True, exist_ok=False)
    record = dict(entry.portable_record)
    copied_roles = _copy_artifacts(
        repository=repository,
        revision=revision,
        portable_dataset=portable_dataset,
        record=record,
        case_root=case_root,
    )
    source_path = (
        source_dataset / PurePosixPath(entry.source_record.relative_path)
    ).as_posix()
    parent_coordinate = f"{revision}:{source_path}:{entry.source_record.line_number}"
    _write_new(case_root / "portable_record.json", _json_bytes(record))
    task = {
        "schema": "open-cake.aka-portable-parent-review-task.v1",
        "purpose": "provisional_ir_expressibility_review",
        "source_coordinate": parent_coordinate,
        "portable_case_id": record["case_id"],
        "portable_qualification": "projection_not_node_custody",
        "gpu_execution": "not_authorized_or_run",
    }
    _write_new(case_root / "task.json", _json_bytes(task))
    completion = {
        "schema": "aka.kernel-parent-completion.v1",
        "case_id": record["case_id"],
        "original_parent": {
            "case_path": parent_coordinate,
            "record_field": entry.source_record.primary_field,
        },
        "recovery_mode": "contract_narrowed",
        "source": record["source"],
        "taxonomy": record["taxonomy"],
        "derived_parent_id": record["derived_parent_id"],
        "semantics": record["semantics"],
        "contract": record["contract"],
        "optimization_handoff": record["optimization_handoff"],
        "artifacts": {
            "baseline": copied_roles["baseline"],
            "reference": copied_roles["reference"],
            "harness": copied_roles["harness"],
            "task": "task.json",
        },
        "qualification": None,
        "outcome": "runnable_unqualified",
        "missing_facts": [],
        "evidence": ["portable_record.json", "task.json"],
        "training_route": "augmentation_parent_pending_qualification",
        "training_eligibility": False,
        "next_action": (
            "Perform provisional IR expressibility review without claiming portable "
            "qualification as node custody."
        ),
    }
    completion_path = case_root / "parent_completion.json"
    _write_new(completion_path, _json_bytes(completion))
    run_root = completion_root / "runs" / entry.queue_case_id
    run_root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    validated = subprocess.run(
        [
            sys.executable,
            str(DEFAULT_PARENT_VALIDATOR),
            str(completion_path),
            "--finalize",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    receipt = {
        "schema": VALIDATOR_RECEIPT_SCHEMA,
        "case_id": entry.queue_case_id,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "exit_code": validated.returncode,
        "outcome": "runnable_unqualified",
        "portable_qualification": "projection_not_node_custody",
    }
    _write_new(run_root / "receipt.json", _json_bytes(receipt))
    _write_new(run_root / "validator.stdout.json", validated.stdout)
    _write_new(run_root / "validator.stderr.log", validated.stderr)
    if validated.returncode != 0:
        detail = (validated.stderr or validated.stdout).decode(
            "utf-8", errors="replace"
        ).strip()
        raise PortableParentError(
            f"canonical runnable parent validation failed for {entry.queue_case_id}: {detail}"
        )
    return completion_path


def run_portable_queue(
    *,
    source_dataset_root: Path,
    portable_dataset_root: Path,
    source_revision: str,
    completion_root: Path,
    review_root: Path,
    limit: int,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    codex_bin: Path | None = None,
) -> dict[str, object]:
    if limit <= 0 or timeout_seconds <= 0:
        raise PortableParentError("limit and timeout must be positive")
    source_dataset_root = source_dataset_root.resolve(strict=True)
    portable_dataset_root = portable_dataset_root.resolve(strict=True)
    completion_root = completion_root.resolve(strict=False)
    review_root = review_root.resolve(strict=False)
    repository = _repository_root(source_dataset_root)
    for path, label in ((completion_root, "completion root"), (review_root, "review root")):
        if os.path.lexists(path):
            raise PortableParentError(f"create-only {label} already exists: {path}")
        if any(
            _inside(path, root)
            for root in (ROOT, repository, completion_root if label == "review root" else review_root)
        ):
            raise PortableParentError(f"{label} must be outside source and review roots")
    if not DEFAULT_PARENT_VALIDATOR.is_file():
        raise PortableParentError(
            f"complete-kernel-parent validator is unavailable: {DEFAULT_PARENT_VALIDATOR}"
        )
    discovered = shutil.which("codex") if codex_bin is None else str(codex_bin)
    if not discovered:
        raise PortableParentError("codex executable is unavailable")
    command_path = Path(discovered).resolve(strict=True)

    plan, entries = build_portable_plan(
        source_dataset_root=source_dataset_root,
        portable_dataset_root=portable_dataset_root,
        source_revision=source_revision,
    )
    selected = [entry for entry in entries if entry.review_ready][:limit]
    completion_root.mkdir(parents=True, exist_ok=False)
    (completion_root / "cases").mkdir()
    (completion_root / "runs").mkdir()
    _write_new(completion_root / "execution-plan.json", _json_bytes(plan))
    _write_new(
        completion_root / "adapter.json",
        _json_bytes(
            {
                "schema": ADAPTER_SCHEMA,
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "codex": _codex_identity(command_path),
                "execution": "sequential_stop_on_first_failure_no_retry",
                "portable_qualification": "projection_not_node_custody",
                "gpu": "not_used",
                "implementation": _git_closure_identity(
                    ROOT,
                    (
                        "tools/run_aka_portable_parents_codex.py",
                        "tools/run_aka_expressibility_codex.py",
                        "tools/review_aka_expressibility.py",
                    ),
                    "portable parent adapter",
                ),
            }
        ),
    )
    portable_relative = _repository_relative(
        repository, portable_dataset_root, "portable dataset"
    )
    source_relative = _repository_relative(
        repository, source_dataset_root, "source dataset"
    )

    processed: list[dict[str, object]] = []
    for entry in selected:
        completion = materialize_portable_completion(
            completion_root=completion_root,
            repository=repository,
            revision=source_revision,
            portable_dataset=portable_relative,
            source_dataset=source_relative,
            entry=entry,
        )
        review = run_queue(
            work_root=review_root,
            dataset_root=source_dataset_root,
            source_revision=source_revision,
            record_format="aka_v1_operator_sft",
            dataset_label=source_dataset_root.name,
            limit=1,
            timeout_seconds=timeout_seconds,
            codex_bin=command_path,
            case_ids=[entry.queue_case_id],
            parent_completions={entry.queue_case_id: completion},
        )
        processed.append(
            {
                "queue_case_id": entry.queue_case_id,
                "portable_case_id": entry.portable_record["case_id"],
                "derived_parent_id": entry.portable_record["derived_parent_id"],
                "parent_completion": str(completion),
                "review": review["processed"][0],
            }
        )
    return {
        "schema": "open-cake.aka-portable-parent-batch.v1",
        "source_revision": source_revision,
        "requested_limit": limit,
        "deterministically_blocked": [
            {
                "queue_case_id": entry.queue_case_id,
                "portable_case_id": entry.portable_record["case_id"],
                "blocker": entry.blocker,
            }
            for entry in entries
            if not entry.review_ready
        ],
        "processed": processed,
        "completion_root": str(completion_root),
        "review_root": str(review_root),
        "claim_boundary": "portable_runnable_parent_provisional_ir_review_only",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument("--portable-dataset-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--completion-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = run_portable_queue(
            source_dataset_root=arguments.source_dataset_root,
            portable_dataset_root=arguments.portable_dataset_root,
            source_revision=arguments.source_revision,
            completion_root=arguments.completion_root,
            review_root=arguments.review_root,
            limit=arguments.limit,
            timeout_seconds=arguments.timeout_seconds,
        )
    except (OSError, PortableParentError, RunnerError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
