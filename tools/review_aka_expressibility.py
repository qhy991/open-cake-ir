#!/usr/bin/env python3
"""Check one AKA challenge row at a time against the current Cake Compiler.

The bridge materializes external review work and deterministically checks a review. It is
not a CUDA-to-Schedule translator, a semantic equivalence prover, a Compiler Corpus
builder, or an evaluation runner. It never invokes a model, provider, remote host, or GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Assessment, Compiler, CompilerError  # noqa: E402
from open_cake_ir.compiler.schema import schedule_schema  # noqa: E402
from tools.audit_aka_corpus import (  # noqa: E402
    CorpusAuditError,
    GitSnapshot,
    RECORD_FORMATS,
    SourceRecord,
    load_records,
    projected_case,
    verify_git_snapshot,
)


WORK_SCHEMA = "open-cake.aka-expressibility-work.v2"
INPUT_SCHEMA = "open-cake.aka-expressibility-input.v2"
REVIEW_SCHEMA = "open-cake.aka-expressibility-review.v2"
CHECKED_SCHEMA = "open-cake.aka-expressibility-checked.v2"
STATUS_SCHEMA = "open-cake.aka-expressibility-status.v2"

PARENT_STATUSES = frozenset({"unknown", "missing", "completion"})
OWNER_SCOPES = frozenset({"unknown", "schedule", "program", "portfolio"})
DISPOSITIONS = frozenset(
    {"unknown", "not_applicable", "schedule_gap", "schedule", "owner_redirect"}
)
GAP_FIELDS = frozenset(
    {
        "capability",
        "irreducible_semantics",
        "hardware_commitment",
        "verification_rule",
        "analysis_requirement",
        "lowering_requirement",
        "acceptance_case",
    }
)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
PARENT_VALIDATOR_TIMEOUT_SECONDS = 30

DEFAULT_PARENT_VALIDATOR = (
    Path.home()
    / ".codex/skills/complete-kernel-parent/scripts/validate_completion.py"
)


class ReviewError(ValueError):
    """The external work or submitted review violates this bridge contract."""


@dataclass(frozen=True)
class Context:
    work_root: Path
    manifest: Mapping[str, Any]
    snapshot: GitSnapshot
    records: tuple[SourceRecord, ...]
    compiler: Compiler


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Create one regular file without following or replacing a final symlink."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise ReviewError(f"create-only output already exists: {path}") from error
    except OSError as error:
        raise ReviewError(f"cannot create {path}: {error}") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_object(path: Path, label: str) -> dict[str, Any]:
    _require_regular(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReviewError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReviewError(f"{label} must be one JSON object")
    return value


def _require_regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ReviewError(f"cannot inspect {label} {path}: {error}") from error
    if not stat.S_ISREG(mode):
        raise ReviewError(f"{label} is not a regular non-symlink file: {path}")


def _require_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ReviewError(f"cannot inspect {label} {path}: {error}") from error
    if not stat.S_ISDIR(mode):
        raise ReviewError(f"{label} is not a non-symlink directory: {path}")


def _expect_fields(value: Mapping[str, object], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise ReviewError(
            f"{label} fields are {sorted(value)}; expected {sorted(fields)}"
        )


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewError(f"{label} must be a non-empty string")
    return value


def _strings(value: object, label: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ReviewError(f"{label} must be an array of non-empty strings")
    if required and not value:
        raise ReviewError(f"{label} must not be empty")
    return value


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _git(root: Path, arguments: Sequence[str], *, check: bool = True) -> str:
    try:
        completed = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(root), *arguments],
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise ReviewError(f"cannot inspect Git checkout: {str(detail).strip()}") from error
    if not check and completed.returncode not in {0, 1}:
        raise ReviewError(f"cannot inspect Git checkout: {completed.stderr.strip()}")
    return completed.stdout


def _git_bytes(root: Path, arguments: Sequence[str]) -> bytes:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise ReviewError(f"cannot inspect Git bytes: {str(detail).strip()}") from error


def _git_head(root: Path) -> str:
    top = Path(_git(root, ["rev-parse", "--show-toplevel"]).strip()).resolve()
    if top != root:
        raise ReviewError("Compiler project root is not its Git worktree root")
    commit = _git(root, ["rev-parse", "HEAD"]).strip()
    if _COMMIT.fullmatch(commit) is None:
        raise ReviewError("Compiler Git HEAD is not one exact commit")
    return commit


def _repository_root(path: Path) -> Path:
    return Path(_git(path, ["rev-parse", "--show-toplevel"]).strip()).resolve(
        strict=True
    )


def _git_closure_identity(
    root: Path, paths: Sequence[str], label: str
) -> dict[str, object]:
    """Bind tracked raw worktree bytes to one exact Git commit.

    The explicit byte comparison is required even when status is clean because Git index
    flags such as assume-unchanged and skip-worktree may suppress ordinary drift reports.
    """

    root = root.resolve(strict=True)
    commit = _git_head(root)
    normalized: list[str] = []
    for index, value in enumerate(paths):
        relative = PurePosixPath(_nonempty(value, f"{label}.paths[{index}]"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ReviewError(f"{label}.paths[{index}] is not project-relative")
        path = root / relative
        if path.is_symlink():
            raise ReviewError(f"{label}.paths[{index}] must not be a symlink")
        _require_regular(path, f"{label}.paths[{index}]")
        normalized.append(relative.as_posix())
    if not normalized or len(set(normalized)) != len(normalized):
        raise ReviewError(f"{label} paths are empty or duplicated")
    _git(root, ["ls-files", "--error-unmatch", "--", *normalized])
    dirty = _git(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all", "--", *normalized],
    ).strip()
    if dirty:
        raise ReviewError(f"{label} closure is dirty: {dirty}")
    for relative in normalized:
        if (root / relative).read_bytes() != _git_bytes(
            root, ["show", f"{commit}:{relative}"]
        ):
            raise ReviewError(f"{label} raw worktree bytes differ from Git HEAD: {relative}")
    return {"git_commit": commit, "paths": normalized}


def _checker_identity(root: Path = ROOT) -> dict[str, object]:
    return _git_closure_identity(
        root,
        (
            "tools/review_aka_expressibility.py",
            "tools/run_aka_expressibility_codex.py",
            "tools/audit_aka_corpus.py",
        ),
        "checker",
    )


def _relative_file(root: Path, path: Path, label: str) -> str:
    resolved = path.resolve(strict=True)
    if not _inside(resolved, root):
        raise ReviewError(f"{label} escapes the Compiler checkout")
    relative = resolved.relative_to(root).as_posix()
    if path.is_symlink():
        raise ReviewError(f"{label} must not be a symlink")
    _require_regular(resolved, label)
    return relative


def _source_set_paths(root: Path, source_set_path: Path) -> tuple[str, ...]:
    document = _load_object(source_set_path, "Compiler source set")
    _expect_fields(document, {"schema_version", "paths"}, "Compiler source set")
    if document.get("schema_version") != 1:
        raise ReviewError("Compiler source set schema is unsupported")
    values = document.get("paths")
    if not isinstance(values, list) or not values:
        raise ReviewError("Compiler source set paths must be a non-empty array")
    paths: list[str] = []
    for index, value in enumerate(values):
        relative = PurePosixPath(_nonempty(value, f"source_set.paths[{index}]"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ReviewError(f"source_set.paths[{index}] is not project-relative")
        resolved = (root / relative).resolve(strict=True)
        if not _inside(resolved, root) or (root / relative).is_symlink():
            raise ReviewError(f"source_set.paths[{index}] escapes or is a symlink")
        _require_regular(resolved, f"source_set.paths[{index}]")
        paths.append(relative.as_posix())
    if len(set(paths)) != len(paths):
        raise ReviewError("Compiler source set contains duplicate paths")
    return tuple(paths)


def _compiler_identity(
    root: Path, revision_path: Path, source_set_path: Path
) -> tuple[dict[str, object], Compiler]:
    root = root.resolve(strict=True)
    if root != ROOT:
        raise ReviewError("this bridge checks only the Compiler in its own checkout")
    revision_relative = _relative_file(root, revision_path, "Compiler Revision")
    source_set_relative = _relative_file(root, source_set_path, "Compiler source set")
    closure = tuple(
        dict.fromkeys(
            (revision_relative, source_set_relative)
            + _source_set_paths(root, source_set_path)
        )
    )
    closure_identity = _git_closure_identity(root, closure, "Compiler")

    compiler = Compiler.load(root, revision_path)
    revision = _load_object(revision_path, "Compiler Revision")
    revision_id = _nonempty(revision.get("revision_id"), "revision_id")
    state = revision.get("state")
    if state not in {"draft", "released"} or state != compiler.state:
        raise ReviewError("Compiler Revision state is invalid")
    return (
        {
            "project_root": str(root),
            "git_commit": closure_identity["git_commit"],
            "revision_path": revision_relative,
            "source_set_path": source_set_relative,
            "revision_id": revision_id,
            "state": state,
        },
        compiler,
    )


def _validator_identity() -> dict[str, str]:
    path = DEFAULT_PARENT_VALIDATOR.resolve(strict=True)
    _require_regular(path, "fixed parent completion validator")
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def _gap_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(GAP_FIELDS),
        "properties": {
            field: {"type": "string", "minLength": 1}
            for field in sorted(GAP_FIELDS)
        },
        "description": (
            "A provisional minimal IR capability proposal. It does not authorize a "
            "Compiler change or claim semantic coverage."
        ),
    }


def _aspect_schema() -> dict[str, object]:
    missing_ir = _gap_schema()
    # OpenAI structured outputs admit a nullable object type but not this schema's former
    # oneOf(null, object) spelling. Object-only constraints remain inactive for null.
    missing_ir["type"] = ["object", "null"]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "owner_scope",
            "owner_relation",
            "disposition",
            "schedule",
            "missing_ir",
            "reason",
        ],
        "properties": {
            "owner_scope": {"type": "string", "enum": sorted(OWNER_SCOPES)},
            "owner_relation": {"type": ["string", "null"]},
            "disposition": {"type": "string", "enum": sorted(DISPOSITIONS)},
            "schedule": {"type": ["string", "null"]},
            "missing_ir": missing_ir,
            "reason": {"type": "string", "minLength": 1},
        },
    }


def review_schema() -> dict[str, object]:
    """Return the one work-level model-facing review schema."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "source_ref",
            "parent_contract",
            "complete_parent",
            "delta",
        ],
        "properties": {
            "schema": {"type": "string", "const": REVIEW_SCHEMA},
            "source_ref": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "repository_remote_observed_sanitized",
                    "revision",
                    "dataset_path",
                    "reported_dataset_label",
                    "path",
                    "line",
                    "primary_field",
                ],
                "properties": {
                    "repository_remote_observed_sanitized": {
                        "type": ["string", "null"]
                    },
                    "revision": {"type": "string", "minLength": 1},
                    "dataset_path": {"type": "string", "minLength": 1},
                    "reported_dataset_label": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "minLength": 1},
                    "line": {"type": "integer", "minimum": 1},
                    "primary_field": {"type": "string", "minLength": 1},
                },
            },
            "parent_contract": {
                "type": "object",
                "additionalProperties": False,
                "required": ["status", "missing_facts", "completion_path"],
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": sorted(PARENT_STATUSES),
                    },
                    "missing_facts": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "completion_path": {"type": ["string", "null"]},
                },
            },
            "complete_parent": _aspect_schema(),
            "delta": _aspect_schema(),
        },
    }


TASK_TEXT = """# AKA to Open Cake provisional expressibility check

Review exactly one commit-addressed row from `cases/<case-id>/input.json`. The source
reference owns the row. Folder labels, historical reasoning, and positive or neutral
filenames are not semantic, correctness, or performance truth.

Before making any non-unknown expressibility or owner claim, reference an absolute
external `parent_completion.json` whose `original_parent.case_path` and `record_field`
match the input. The fixed complete-kernel-parent validator and its `PARENT_DONE.json`
must accept that completion. This check does not independently establish node custody.

Keep the complete parent and delta questions separate. The input states which reviews
apply. Analysis and generation have no delta lane; v2 neutral/negative reviewer rows have
neither a code parent nor a code delta. A v2 positive baseline/candidate pair retains both.
For an applicable delta, a mechanism match never establishes parent coverage.

One Schedule owns one kernel. Put any submitted Schedule inside the case and reference it
with a case-relative path. Do not use a Compiler Corpus Schedule as proof. Program graphs
and multi-launch relations belong to program composition; runtime specialist selection
belongs to a portfolio. Those redirects require an explicit owner relation and no
Schedule. A Schedule vocabulary gap requires every `missing_ir` field.

Copy `review.template.json` to `review.json`. The checker derives classifications; do not
add one. It runs case-local Schedules through the exact Compiler assess/lower interface.
The result remains a provisional compiler check with `semantic_binding=reviewer_claimed`,
`gpu_test=not_run`, and explicit draft/released Compiler maturity.

Do not edit AKA, Compiler sources, Corpus expectations, release files, existing evidence,
or the work-level reference bundle. Do not invoke a provider, remote host, GPU, profiler,
or benchmark. Write all review prose in English.
"""


REFERENCE_TEXT = """# Frozen work-level reference bundle

`schedule.schema.json` is generated from the typed IR in the Git-bound Compiler checkout.
`AUTHORING_CONTRACT.md` defines Schedule authoring. `compiler.json` identifies the exact
Git commit, Revision, and source-set closure checked before every command.

The checker proves only how that Compiler assesses and lowers a submitted case-local
Schedule. Semantic binding remains reviewer-claimed. GPU correctness, sanitizer,
profiling, performance, end-to-end behavior, promotion, and independent evidence custody
are outside this workflow.
"""


def _reference_files(
    compiler_ref: Mapping[str, object], checker_ref: Mapping[str, object]
) -> dict[str, bytes]:
    return {
        "README.md": REFERENCE_TEXT.encode("utf-8"),
        "TASK.md": TASK_TEXT.encode("utf-8"),
        "review.schema.json": _json_bytes(review_schema()),
        "schedule.schema.json": _json_bytes(schedule_schema()),
        "AUTHORING_CONTRACT.md": (ROOT / "compiler/AUTHORING_CONTRACT.md").read_bytes(),
        "compiler.json": _json_bytes(
            {
                "schema": "open-cake.aka-expressibility-compiler-reference.v2",
                **compiler_ref,
                "checker": dict(checker_ref),
                "claim_boundary": "provisional_compiler_check_only",
            }
        ),
    }


def _check_reference_bundle(
    work_root: Path,
    compiler_ref: Mapping[str, object],
    checker_ref: Mapping[str, object],
) -> None:
    reference_root = work_root / "reference"
    _require_directory(reference_root, "reference root")
    expected = _reference_files(compiler_ref, checker_ref)
    observed_names = {path.name for path in reference_root.iterdir()}
    if observed_names != set(expected):
        raise ReviewError("work-level reference bundle files differ")
    for name, payload in expected.items():
        path = reference_root / name
        _require_regular(path, f"reference bundle {name}")
        if path.read_bytes() != payload:
            raise ReviewError(f"work-level reference bundle changed: {name}")


def initialize(
    *,
    dataset_root: Path,
    work_root: Path,
    source_revision: str,
    record_format: str,
    compiler_revision: Path | None = None,
    dataset_label: str | None = None,
) -> dict[str, object]:
    """Create a work manifest and one frozen reference bundle outside both repositories."""

    dataset_root = dataset_root.resolve(strict=True)
    source_repository = _repository_root(dataset_root)
    work_root = work_root.resolve(strict=False)
    if _inside(work_root, ROOT):
        raise ReviewError("work root must be outside the Open Cake checkout")
    if _inside(work_root, source_repository):
        raise ReviewError("work root must be outside the AKA source repository")
    if os.path.lexists(work_root):
        raise ReviewError(f"create-only work root already exists: {work_root}")
    if record_format not in RECORD_FORMATS:
        raise ReviewError(f"unsupported record format: {record_format}")

    revision_path = (
        ROOT / "compiler/revision.json"
        if compiler_revision is None
        else (
            compiler_revision
            if compiler_revision.is_absolute()
            else ROOT / compiler_revision
        )
    ).resolve(strict=True)
    compiler_ref, _ = _compiler_identity(
        ROOT, revision_path, ROOT / "compiler/source_set.json"
    )
    checker_ref = _checker_identity()
    validator_ref = _validator_identity()
    snapshot = verify_git_snapshot(dataset_root, source_revision)
    records = load_records(snapshot, record_format=record_format)
    label = dataset_label or dataset_root.name
    _nonempty(label, "reported dataset label")
    manifest: dict[str, object] = {
        "schema": WORK_SCHEMA,
        "source": {
            "dataset_root": str(dataset_root),
            "source_repository": str(source_repository),
            "revision": snapshot.revision,
            "record_format": record_format,
            "reported_dataset_label": label,
            "record_count": len(records),
        },
        "compiler": compiler_ref,
        "checker": checker_ref,
        "parent_completion_validator": validator_ref,
    }
    references = _reference_files(compiler_ref, checker_ref)

    work_root.mkdir(parents=True, exist_ok=False)
    _require_directory(work_root, "work root")
    reference_root = work_root / "reference"
    reference_root.mkdir(exist_ok=False)
    _require_directory(reference_root, "reference root")
    _write_exclusive(work_root / "manifest.json", _json_bytes(manifest))
    for name, payload in sorted(references.items()):
        _write_exclusive(reference_root / name, payload)
    return manifest


def _load_manifest(work_root: Path) -> dict[str, Any]:
    _require_directory(work_root, "work root")
    manifest = _load_object(work_root / "manifest.json", "work manifest")
    _expect_fields(
        manifest,
        {
            "schema",
            "source",
            "compiler",
            "checker",
            "parent_completion_validator",
        },
        "work manifest",
    )
    if manifest.get("schema") != WORK_SCHEMA:
        raise ReviewError("unsupported work manifest schema")
    source = manifest.get("source")
    compiler = manifest.get("compiler")
    checker = manifest.get("checker")
    validator = manifest.get("parent_completion_validator")
    if not all(
        isinstance(value, dict) for value in (source, compiler, checker, validator)
    ):
        raise ReviewError("work manifest authority references must be objects")
    _expect_fields(
        source,
        {
            "dataset_root",
            "source_repository",
            "revision",
            "record_format",
            "reported_dataset_label",
            "record_count",
        },
        "work manifest source",
    )
    _expect_fields(
        compiler,
        {
            "project_root",
            "git_commit",
            "revision_path",
            "source_set_path",
            "revision_id",
            "state",
        },
        "work manifest compiler",
    )
    _expect_fields(checker, {"git_commit", "paths"}, "work manifest checker")
    _expect_fields(validator, {"path", "sha256"}, "parent completion validator")
    return manifest


def _load_context(work_root: Path) -> Context:
    work_root = work_root.resolve(strict=True)
    manifest = _load_manifest(work_root)
    source = manifest["source"]
    compiler_stored = manifest["compiler"]
    checker_stored = manifest["checker"]

    checker_current = _checker_identity()
    if checker_current != checker_stored:
        raise ReviewError("checker Git/raw-byte identity drifted")
    validator_current = _validator_identity()
    if validator_current != manifest["parent_completion_validator"]:
        raise ReviewError("fixed parent completion validator identity drifted")
    if _DIGEST.fullmatch(validator_current["sha256"]) is None:
        raise ReviewError("fixed parent completion validator digest is invalid")

    dataset_root = Path(_nonempty(source["dataset_root"], "source.dataset_root"))
    source_repository = Path(
        _nonempty(source["source_repository"], "source.source_repository")
    ).resolve(strict=True)
    if _repository_root(dataset_root) != source_repository:
        raise ReviewError("AKA dataset root now resolves to a different repository")
    snapshot = verify_git_snapshot(
        dataset_root, _nonempty(source["revision"], "source.revision")
    )
    records = tuple(
        load_records(
            snapshot,
            record_format=_nonempty(source["record_format"], "source.record_format"),
        )
    )
    count = source.get("record_count")
    if not isinstance(count, int) or isinstance(count, bool) or count != len(records):
        raise ReviewError("source record count differs from the initialized queue")
    _nonempty(source["reported_dataset_label"], "source.reported_dataset_label")

    project_root = Path(
        _nonempty(compiler_stored["project_root"], "compiler.project_root")
    ).resolve(strict=True)
    if project_root != ROOT:
        raise ReviewError("Compiler project root differs from this checker checkout")
    revision_path = project_root / PurePosixPath(
        _nonempty(compiler_stored["revision_path"], "compiler.revision_path")
    )
    source_set_path = project_root / PurePosixPath(
        _nonempty(compiler_stored["source_set_path"], "compiler.source_set_path")
    )
    compiler_current, compiler = _compiler_identity(
        project_root, revision_path, source_set_path
    )
    if compiler_current != compiler_stored:
        raise ReviewError("Compiler Git/Revision/source-set identity drifted")
    _check_reference_bundle(work_root, compiler_stored, checker_stored)
    return Context(work_root, manifest, snapshot, records, compiler)


def _case_id(index: int) -> str:
    return f"case-{index + 1:06d}"


def _case_index(case_id: str, record_count: int) -> int:
    match = re.fullmatch(r"case-([0-9]{6})", case_id)
    if match is None:
        raise ReviewError(f"invalid case id: {case_id}")
    index = int(match.group(1)) - 1
    if index < 0 or index >= record_count:
        raise ReviewError(f"case id is outside this queue: {case_id}")
    return index


def _review_applicability(record: SourceRecord) -> tuple[bool, bool]:
    roles = set(record.artifact_fields)
    code_roles = {"source", "generated", "broken", "repaired", "baseline", "candidate"}
    complete_parent = bool(roles & code_roles)
    delta = {"baseline", "candidate"} <= roles or {"broken", "repaired"} <= roles
    return complete_parent, delta


def _unknown_aspect(reason: str) -> dict[str, object]:
    return {
        "owner_scope": "unknown",
        "owner_relation": None,
        "disposition": "unknown",
        "schedule": None,
        "missing_ir": None,
        "reason": reason,
    }


def _not_applicable_aspect() -> dict[str, object]:
    return {
        "owner_scope": "unknown",
        "owner_relation": None,
        "disposition": "not_applicable",
        "schedule": None,
        "missing_ir": None,
        "reason": "This source role has no admitted delta expressibility lane.",
    }


def _case_documents(context: Context, index: int) -> tuple[dict[str, object], dict[str, object]]:
    case_id = _case_id(index)
    record = context.records[index]
    challenge = projected_case(
        record,
        snapshot=context.snapshot,
        reported_dataset_label=context.manifest["source"]["reported_dataset_label"],
    )
    locator = challenge["source_ref"]
    complete_applicable, delta_applicable = _review_applicability(record)
    case_input = {
        "schema": INPUT_SCHEMA,
        "case_id": case_id,
        "parent_coordinate": (
            f"{locator['revision']}:{locator['path']}:{locator['line']}"
        ),
        "reference_bundle": "../../reference",
        "complete_parent_review": {
            "applicable": complete_applicable,
            "basis": (
                "record_has_code_artifact_role"
                if complete_applicable
                else "record_has_no_code_artifact_role"
            ),
        },
        "delta_review": {
            "applicable": delta_applicable,
            "basis": (
                "record_has_baseline_candidate_or_broken_repaired_roles"
                if delta_applicable
                else "record_has_no_paired_code_delta_roles"
            ),
        },
        "challenge_case": challenge,
        "source_record": dict(record.fields),
    }
    template = {
        "schema": REVIEW_SCHEMA,
        "source_ref": locator,
        "parent_contract": {
            "status": "unknown",
            "missing_facts": [],
            "completion_path": None,
        },
        "complete_parent": (
            _unknown_aspect(
                "The available evidence is insufficient to classify the complete parent."
            )
            if complete_applicable
            else _not_applicable_aspect()
        ),
        "delta": (
            _unknown_aspect(
                "The available evidence is insufficient to classify the delta."
            )
            if delta_applicable
            else _not_applicable_aspect()
        ),
    }
    return case_input, template


def _cases_root(work_root: Path, *, create: bool) -> Path:
    path = work_root / "cases"
    if os.path.lexists(path):
        _require_directory(path, "cases root")
    elif create:
        path.mkdir(exist_ok=False)
        _require_directory(path, "cases root")
    return path


def _case_root(cases_root: Path, case_id: str, *, required: bool) -> Path:
    path = cases_root / case_id
    if not os.path.lexists(path):
        if required:
            raise ReviewError(f"case is not materialized: {case_id}")
        return path
    _require_directory(path, f"case {case_id}")
    if path.resolve(strict=True).parent != cases_root.resolve(strict=True):
        raise ReviewError(f"case {case_id} escapes the cases root")
    return path


def _materialized(context: Context, index: int) -> bool:
    cases_root = _cases_root(context.work_root, create=False)
    case_id = _case_id(index)
    case_root = cases_root / case_id
    if not os.path.lexists(case_root):
        return False
    _case_root(cases_root, case_id, required=True)
    expected_input, expected_template = _case_documents(context, index)
    for name, expected in (
        ("input.json", expected_input),
        ("review.template.json", expected_template),
    ):
        path = case_root / name
        if not os.path.lexists(path):
            return False
        _require_regular(path, f"case {case_id} {name}")
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        if observed != expected:
            return False
    return True


def materialize(work_root: Path, case_id: str | None = None) -> dict[str, object]:
    """Atomically create exactly one absent case with input and review template."""

    context = _load_context(work_root)
    cases_root = _cases_root(context.work_root, create=True)
    if case_id is None:
        index = next(
            (
                candidate
                for candidate in range(len(context.records))
                if not os.path.lexists(cases_root / _case_id(candidate))
            ),
            None,
        )
        if index is None:
            incomplete = sum(
                not _materialized(context, candidate)
                for candidate in range(len(context.records))
            )
            if incomplete:
                raise ReviewError(
                    f"queue contains {incomplete} incomplete create-only case directories"
                )
            return {"done": True, "record_count": len(context.records)}
        case_id = _case_id(index)
    else:
        index = _case_index(case_id, len(context.records))
    final_root = _case_root(cases_root, case_id, required=False)
    if os.path.lexists(final_root):
        raise ReviewError(f"create-only case already exists: {final_root}")

    case_input, template = _case_documents(context, index)
    temporary = Path(tempfile.mkdtemp(prefix=f".{case_id}.tmp-", dir=cases_root))
    try:
        _require_directory(temporary, "temporary case")
        _write_exclusive(temporary / "input.json", _json_bytes(case_input))
        _write_exclusive(temporary / "review.template.json", _json_bytes(template))
        if os.path.lexists(final_root):
            raise ReviewError(f"create-only case already exists: {final_root}")
        os.rename(temporary, final_root)
    finally:
        if os.path.lexists(temporary):
            _require_directory(temporary, "temporary case cleanup")
            shutil.rmtree(temporary)
    return {"done": False, "case_id": case_id, "case_root": str(final_root)}


def _validate_parent_completion(
    context: Context,
    completion_value: str,
    expected_input: Mapping[str, Any],
) -> tuple[dict[str, object], Mapping[str, Any]]:
    raw = Path(completion_value)
    if not raw.is_absolute():
        raise ReviewError("parent completion path must be absolute")
    completion_path = raw.resolve(strict=True)
    if raw.is_symlink() or completion_path.name != "parent_completion.json":
        raise ReviewError("parent completion must be a non-symlink parent_completion.json")
    excluded_roots = (
        context.work_root,
        Path(context.manifest["source"]["source_repository"]).resolve(strict=True),
        ROOT,
    )
    if any(_inside(completion_path, root) for root in excluded_roots):
        raise ReviewError("parent completion must be outside work, source, and Compiler roots")
    _require_regular(completion_path, "parent completion")
    marker_path = completion_path.parent / "PARENT_DONE.json"
    _require_regular(marker_path, "parent completion marker")

    validator = context.manifest["parent_completion_validator"]
    try:
        completed = subprocess.run(
            [sys.executable, validator["path"], str(completion_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=PARENT_VALIDATOR_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise ReviewError(
            f"fixed parent validator exceeded {PARENT_VALIDATOR_TIMEOUT_SECONDS}s"
        ) from error
    except (OSError, UnicodeError) as error:
        raise ReviewError(f"cannot run fixed parent validator: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ReviewError(f"canonical parent completion is invalid: {detail}")
    try:
        validation = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReviewError("fixed parent validator returned malformed JSON") from error
    if not isinstance(validation, dict) or validation.get("valid") is not True:
        raise ReviewError("fixed parent validator did not return valid=true")

    completion = _load_object(completion_path, "parent completion")
    marker = _load_object(marker_path, "parent completion marker")
    original = completion.get("original_parent")
    if not isinstance(original, dict) or original.get("case_path") != expected_input[
        "parent_coordinate"
    ]:
        raise ReviewError("parent completion belongs to a different source coordinate")
    expected_field = expected_input["challenge_case"]["source_ref"]["primary_field"]
    if original.get("record_field") != expected_field:
        raise ReviewError("parent completion belongs to a different source field")
    expected_marker = {
        "schema": "aka.kernel-parent-completion-done.v1",
        "case_id": completion.get("case_id"),
        "outcome": completion.get("outcome"),
        "recovery_mode": completion.get("recovery_mode"),
        "verified_by": "complete-kernel-parent-validator",
        "completion": completion_path.name,
        "counts_as_executable_parent": completion.get("outcome") == "qualified",
        "training_eligibility": False,
    }
    for field, expected in expected_marker.items():
        if marker.get(field) != expected:
            raise ReviewError(f"PARENT_DONE.json field {field!r} is invalid")
    _nonempty(marker.get("verified_at"), "PARENT_DONE.json verified_at")
    if set(marker) != set(expected_marker) | {"verified_at"}:
        raise ReviewError("PARENT_DONE.json fields differ from the canonical marker")
    if validation.get("case_id") != completion.get("case_id") or validation.get(
        "outcome"
    ) != completion.get("outcome"):
        raise ReviewError("parent validator output disagrees with parent completion")
    reference = {
        "path": str(completion_path),
        "case_id": completion["case_id"],
        "outcome": completion["outcome"],
        "recovery_mode": completion["recovery_mode"],
    }
    return reference, completion


def _parent_contract(
    context: Context, value: object, expected_input: Mapping[str, Any]
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReviewError("parent_contract must be an object")
    _expect_fields(
        value,
        {"status", "missing_facts", "completion_path"},
        "parent_contract",
    )
    status_value = value.get("status")
    if status_value not in PARENT_STATUSES:
        raise ReviewError("parent_contract.status is invalid")
    missing = _strings(value.get("missing_facts"), "parent_contract.missing_facts")
    completion_value = value.get("completion_path")
    if completion_value is not None and not isinstance(completion_value, str):
        raise ReviewError("parent_contract.completion_path must be a string or null")

    completion_ref = None
    if status_value == "unknown":
        if missing or completion_value is not None:
            raise ReviewError("unknown parent contract carries no missing facts or completion")
        derived_status = "unknown"
    elif status_value == "missing":
        if completion_value is not None:
            raise ReviewError("reviewer-reported missing contract carries no completion")
        _strings(missing, "parent_contract.missing_facts", required=True)
        derived_status = "missing_by_reviewer"
    else:
        if missing or not completion_value:
            raise ReviewError("completion status requires one path and no copied missing facts")
        completion_ref, completion = _validate_parent_completion(
            context, completion_value, expected_input
        )
        if completion_ref["outcome"] == "qualified":
            derived_status = "qualified_by_parent_validator"
        else:
            derived_status = "missing_by_parent_validator"
            missing = _strings(
                completion.get("missing_facts"),
                "parent_completion.missing_facts",
                required=True,
            )
    return {
        "status": derived_status,
        "missing_facts": missing,
        "parent_completion": completion_ref,
        "independent_node_custody": "not_assessed",
    }


def _missing_ir(value: object, label: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ReviewError(f"{label}.missing_ir must be an object or null")
    _expect_fields(value, set(GAP_FIELDS), f"{label}.missing_ir")
    return {
        field: _nonempty(value.get(field), f"{label}.missing_ir.{field}")
        for field in sorted(GAP_FIELDS)
    }


def _aspect(value: object, label: str, *, applicable: bool) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReviewError(f"{label} must be an object")
    _expect_fields(
        value,
        {
            "owner_scope",
            "owner_relation",
            "disposition",
            "schedule",
            "missing_ir",
            "reason",
        },
        label,
    )
    owner = value.get("owner_scope")
    disposition = value.get("disposition")
    if owner not in OWNER_SCOPES or disposition not in DISPOSITIONS:
        raise ReviewError(f"{label} owner_scope or disposition is invalid")
    relation = value.get("owner_relation")
    if relation is not None and (not isinstance(relation, str) or not relation.strip()):
        raise ReviewError(f"{label}.owner_relation must be a non-empty string or null")
    schedule = value.get("schedule")
    if schedule is not None and (not isinstance(schedule, str) or not schedule.strip()):
        raise ReviewError(f"{label}.schedule must be a non-empty string or null")
    gap = _missing_ir(value.get("missing_ir"), label)
    reason = _nonempty(value.get("reason"), f"{label}.reason")

    empty_claims = owner == "unknown" and relation is None and schedule is None and gap is None
    if disposition == "not_applicable":
        if applicable or not empty_claims:
            raise ReviewError(f"{label} not_applicable conflicts with the source role")
    elif not applicable:
        raise ReviewError(f"{label} must be not_applicable for this source role")
    elif disposition == "unknown":
        if not empty_claims:
            raise ReviewError(f"{label} unknown disposition carries no owner claim")
    elif disposition == "schedule_gap":
        if owner != "schedule" or relation is not None or schedule is not None or gap is None:
            raise ReviewError(
                f"{label} schedule_gap requires detailed missing_ir and no Schedule"
            )
    elif disposition == "schedule":
        if owner != "schedule" or relation is not None or schedule is None or gap is not None:
            raise ReviewError(f"{label} Schedule disposition is inconsistent")
    elif (
        owner not in {"program", "portfolio"}
        or relation is None
        or schedule is not None
        or gap is not None
    ):
        raise ReviewError(
            f"{label} redirect requires a program/portfolio relation and no Schedule"
        )
    return {
        "owner_scope": owner,
        "owner_relation": relation,
        "disposition": disposition,
        "schedule": schedule,
        "missing_ir": gap,
        "reason": reason,
    }


def _case_local_file(case_root: Path, relative_value: str, label: str) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReviewError(f"{label} must be case-relative")
    current = case_root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise ReviewError(f"cannot inspect {label}: {error}") from error
        if stat.S_ISLNK(mode):
            raise ReviewError(f"{label} must not traverse a symlink")
        if index + 1 < len(relative.parts) and not stat.S_ISDIR(mode):
            raise ReviewError(f"{label} parent is not a directory")
    _require_regular(current, label)
    if not _inside(current.resolve(strict=True), case_root.resolve(strict=True)):
        raise ReviewError(f"{label} escapes its case")
    return current


def _assessment(assessment: Assessment) -> dict[str, object]:
    return {
        "compiler_revision_id": assessment.compiler_revision_id,
        "compiler_revision_sha256": assessment.compiler_revision_sha256,
        "schedule_id": assessment.schedule_id,
        "schedule_sha256": assessment.schedule_sha256,
        "target": assessment.target,
        "accepted": assessment.accepted,
        "lowering_eligible": assessment.lowering_eligible,
        "calibration_available": assessment.calibration_available,
        "findings": [
            {
                "code": finding.code,
                "path": finding.path,
                "message": finding.message,
                "blocks_acceptance": finding.blocks_acceptance,
                "blocks_lowering": finding.blocks_lowering,
            }
            for finding in assessment.findings
        ],
    }


def _classify(
    context: Context,
    aspect: Mapping[str, object],
    *,
    label: str,
    parent_status: str,
    case_root: Path,
) -> dict[str, object]:
    disposition = aspect["disposition"]
    if disposition == "not_applicable":
        return {
            **aspect,
            "classification": "not_applicable",
            "compiler_check": "not_applicable",
            "compiler_evidence": None,
        }
    if disposition == "unknown":
        classification = (
            "contract_missing"
            if label == "complete_parent" and parent_status.startswith("missing_by_")
            else "unknown"
        )
        return {
            **aspect,
            "classification": classification,
            "compiler_check": "unknown",
            "compiler_evidence": None,
        }
    if parent_status != "qualified_by_parent_validator":
        raise ReviewError(
            f"{label} non-unknown claim requires a qualified canonical parent completion"
        )
    if disposition == "schedule_gap":
        return {
            **aspect,
            "classification": "schedule_gap_candidate",
            "compiler_check": "not_run",
            "compiler_evidence": None,
        }
    if disposition == "owner_redirect":
        owner = aspect["owner_scope"]
        return {
            **aspect,
            "classification": f"{owner}_redirect_candidate",
            "compiler_check": "not_run",
            "compiler_evidence": None,
        }

    schedule_path = _case_local_file(
        case_root, str(aspect["schedule"]), f"{label} Schedule"
    )
    try:
        assessment = context.compiler.assess_file(schedule_path)
    except (OSError, UnicodeError, json.JSONDecodeError, CompilerError) as error:
        raise ReviewError(f"{label} Schedule cannot be assessed: {error}") from error
    if not assessment.accepted:
        codes = ", ".join(finding.code for finding in assessment.findings) or "none"
        raise ReviewError(
            f"{label} Schedule is invalid authoring, not an IR gap; findings: {codes}"
        )
    lowering_record = None
    if assessment.lowering_eligible:
        try:
            lowering = context.compiler.lower(assessment)
        except CompilerError as error:
            raise ReviewError(
                f"{label} Schedule was eligible but exact lowering failed: {error}"
            ) from error
        lowering_record = {
            "generated": lowering.generated,
            "route": {
                "backend": lowering.route.backend.value,
                "entry_point": lowering.route.entry_point,
            },
            "source_sha256": lowering.source_sha256,
            "toolchain_requirements": dict(lowering.toolchain_requirements),
        }
        classification = "schedule_candidate_lowerable"
        compiler_check = "lowerable"
    else:
        classification = "schedule_candidate_backend_blocked"
        compiler_check = "backend_blocked"
    return {
        **aspect,
        "classification": classification,
        "compiler_check": compiler_check,
        "compiler_evidence": {
            "schedule": aspect["schedule"],
            "assessment": _assessment(assessment),
            "lowering": lowering_record,
        },
    }


def _review_path(case_root: Path, value: Path | None) -> Path:
    path = case_root / "review.json" if value is None else value
    if not path.is_absolute():
        path = case_root / path
    try:
        relative = path.relative_to(case_root)
    except ValueError as error:
        raise ReviewError("review path must remain inside its case") from error
    return _case_local_file(case_root, relative.as_posix(), "review")


def _derive_review(
    context: Context, case_id: str, *, review_path: Path | None = None
) -> dict[str, object]:
    index = _case_index(case_id, len(context.records))
    if not _materialized(context, index):
        raise ReviewError(f"case bundle is incomplete or differs: {case_id}")
    cases_root = _cases_root(context.work_root, create=False)
    case_root = _case_root(cases_root, case_id, required=True)
    expected_input, _ = _case_documents(context, index)
    review = _load_object(_review_path(case_root, review_path), "review")
    _expect_fields(
        review,
        {"schema", "source_ref", "parent_contract", "complete_parent", "delta"},
        "review",
    )
    if review.get("schema") != REVIEW_SCHEMA:
        raise ReviewError("unsupported review schema")
    expected_ref = expected_input["challenge_case"]["source_ref"]
    if review.get("source_ref") != expected_ref:
        raise ReviewError("review source_ref differs from the materialized source row")

    parent = _parent_contract(context, review.get("parent_contract"), expected_input)
    complete = _aspect(
        review.get("complete_parent"),
        "complete_parent",
        applicable=bool(expected_input["complete_parent_review"]["applicable"]),
    )
    delta = _aspect(
        review.get("delta"),
        "delta",
        applicable=bool(expected_input["delta_review"]["applicable"]),
    )
    prose = [
        complete["reason"],
        delta["reason"],
        *parent["missing_facts"],
        *([complete["owner_relation"]] if complete["owner_relation"] else []),
        *([delta["owner_relation"]] if delta["owner_relation"] else []),
        *(complete["missing_ir"].values() if complete["missing_ir"] else []),
        *(delta["missing_ir"].values() if delta["missing_ir"] else []),
    ]
    if any(_CJK.search(str(value)) for value in prose):
        raise ReviewError("review prose must be English")

    complete_result = _classify(
        context,
        complete,
        label="complete_parent",
        parent_status=str(parent["status"]),
        case_root=case_root,
    )
    delta_result = _classify(
        context,
        delta,
        label="delta",
        parent_status=str(parent["status"]),
        case_root=case_root,
    )
    compiler_ref = context.manifest["compiler"]
    return {
        "schema": CHECKED_SCHEMA,
        "case_id": case_id,
        "source_ref": expected_ref,
        "reported_axes": expected_input["challenge_case"]["reported_axes"],
        "relation": expected_input["challenge_case"]["relation"],
        "compiler": dict(compiler_ref),
        "compiler_maturity": compiler_ref["state"],
        "semantic_binding": "reviewer_claimed",
        "parent_contract": parent,
        "complete_parent_expressibility": complete_result,
        "delta_expressibility": delta_result,
        "primary_class": complete_result["classification"],
        "gpu_test": "not_run",
        "review_state": "checked",
        "claim_boundary": "external_challenge_and_provisional_compiler_check_only",
    }


def verify_review(
    work_root: Path,
    case_id: str,
    *,
    review_path: Path | None = None,
    finalize: bool = False,
) -> dict[str, object]:
    """Check one review and optionally create its deterministic checked cache."""

    if finalize and review_path is not None:
        raise ReviewError("finalized checks require the canonical case review.json")
    context = _load_context(work_root)
    result = _derive_review(context, case_id, review_path=review_path)
    if finalize:
        case_root = _case_root(
            _cases_root(context.work_root, create=False), case_id, required=True
        )
        _write_exclusive(case_root / "checked.json", _json_bytes(result))
    return result


def status(work_root: Path) -> dict[str, object]:
    """Rederive every checked review under one authority context and aggregate it."""

    context = _load_context(work_root)
    cases_root = _cases_root(context.work_root, create=False)
    materialized = 0
    incomplete = 0
    review_ready = 0
    checked = 0
    primary: Counter[str] = Counter()
    complete_checks: Counter[str] = Counter()
    delta_classes: Counter[str] = Counter()
    delta_checks: Counter[str] = Counter()
    for index in range(len(context.records)):
        case_id = _case_id(index)
        case_root = cases_root / case_id
        if not os.path.lexists(case_root):
            continue
        _case_root(cases_root, case_id, required=True)
        if not _materialized(context, index):
            incomplete += 1
            continue
        materialized += 1
        review_file = case_root / "review.json"
        if os.path.lexists(review_file):
            _require_regular(review_file, f"case {case_id} review")
            review_ready += 1
        checked_file = case_root / "checked.json"
        if not os.path.lexists(checked_file):
            continue
        _require_regular(checked_file, f"case {case_id} checked cache")
        expected = _derive_review(context, case_id)
        cached = _load_object(checked_file, "checked cache")
        if cached != expected:
            raise ReviewError(f"checked cache differs from rederived review: {case_id}")
        checked += 1
        primary[str(expected["primary_class"])] += 1
        complete = expected["complete_parent_expressibility"]
        delta = expected["delta_expressibility"]
        complete_checks[str(complete["compiler_check"])] += 1
        delta_classes[str(delta["classification"])] += 1
        delta_checks[str(delta["compiler_check"])] += 1
    return {
        "schema": STATUS_SCHEMA,
        "record_count": len(context.records),
        "materialized": materialized,
        "incomplete": incomplete,
        "review_ready": review_ready,
        "checked": checked,
        "unmaterialized": len(context.records) - materialized,
        "unchecked": len(context.records) - checked,
        "primary_class_counts": dict(sorted(primary.items())),
        "complete_parent_compiler_check_counts": dict(sorted(complete_checks.items())),
        "delta_class_counts": dict(sorted(delta_classes.items())),
        "delta_compiler_check_counts": dict(sorted(delta_checks.items())),
        "compiler": dict(context.manifest["compiler"]),
        "semantic_binding": "reviewer_claimed",
        "gpu_test": "not_run",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create external work and references")
    init.add_argument("--dataset-root", type=Path, required=True)
    init.add_argument("--work-root", type=Path, required=True)
    init.add_argument("--source-revision", required=True)
    init.add_argument("--record-format", required=True, choices=RECORD_FORMATS)
    init.add_argument("--dataset-label")
    init.add_argument(
        "--compiler-revision", type=Path, default=Path("compiler/revision.json")
    )

    for name in ("next", "materialize"):
        command = commands.add_parser(name, help="atomically materialize one case")
        command.add_argument("--work-root", type=Path, required=True)
        if name == "materialize":
            command.add_argument("--case-id", required=True)

    verify = commands.add_parser("verify", help="check one review sidecar")
    verify.add_argument("--work-root", type=Path, required=True)
    verify.add_argument("--case-id", required=True)
    verify.add_argument("--review", type=Path)
    verify.add_argument("--finalize", action="store_true")

    aggregate = commands.add_parser("status", help="rederive and aggregate checked work")
    aggregate.add_argument("--work-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "init":
            result = initialize(
                dataset_root=arguments.dataset_root,
                work_root=arguments.work_root,
                source_revision=arguments.source_revision,
                record_format=arguments.record_format,
                compiler_revision=arguments.compiler_revision,
                dataset_label=arguments.dataset_label,
            )
        elif arguments.command in {"next", "materialize"}:
            result = materialize(
                arguments.work_root, getattr(arguments, "case_id", None)
            )
        elif arguments.command == "verify":
            result = verify_review(
                arguments.work_root,
                arguments.case_id,
                review_path=arguments.review,
                finalize=arguments.finalize,
            )
        else:
            result = status(arguments.work_root)
    except (CorpusAuditError, CompilerError, ReviewError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
