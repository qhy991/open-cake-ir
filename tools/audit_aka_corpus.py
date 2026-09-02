#!/usr/bin/env python3
"""Project AKA-shaped operator shards into a read-only Cake challenge-corpus view.

The selected Git commit owns the bytes; the observed remote and reported dataset label do
not establish source authority.  This tool does not translate CUDA into Schedules and does
not accept the records as correctness, performance, or provenance evidence.  It validates
the narrow four-string storage contract, preserves commit-bound path-and-line locators, and
reports only syntactic scope/signals that help a reviewer route a record to Schedule,
program, portfolio, or contract work.

The two output modes deliberately answer different questions:

* ``summary`` describes the source collection and its audit limitations.
* ``cases`` emits one reference-only projection per source row for later human assessment.

Neither output is a Compiler Corpus manifest.  A record becomes a Compiler Corpus case
only after an independently authored Schedule and expected Assessment/Lowering exist.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit


RECORD_FIELDS = frozenset({"instruction", "input", "reasoning", "output"})

# The source field is a task-storage convention, not a claim that the field is a complete
# translation unit or the factual baseline.  In particular, generation stores its produced
# implementation in output while the other tasks put source, baseline or review context in
# input. The declared record format decides which of those roles applies.
TASKS: Mapping[str, tuple[str, str]] = {
    "analysis": ("input", "analyze"),
    "generation": ("output", "generate"),
    "debug": ("input", "repair"),
    "optimization_positive": ("input", "optimize"),
    "optimization_neutral": ("input", "optimize"),
    "optimization_negative": ("input", "optimize"),
}

_V1_ARTIFACT_FIELDS: Mapping[str, Mapping[str, str]] = {
    "analysis": {"source": "input"},
    "generation": {"contract": "instruction", "generated": "output"},
    "debug": {"broken": "input", "repaired": "output"},
    "optimization_positive": {"baseline": "input", "candidate": "output"},
    "optimization_neutral": {"baseline": "input", "candidate": "output"},
    "optimization_negative": {"baseline": "input", "candidate": "output"},
}

_V2_ARTIFACT_FIELDS: Mapping[str, Mapping[str, str]] = {
    **_V1_ARTIFACT_FIELDS,
    "optimization_neutral": {"review_context": "input", "review": "output"},
    "optimization_negative": {"review_context": "input", "review": "output"},
}

ARTIFACT_FIELDS_BY_FORMAT: Mapping[str, Mapping[str, Mapping[str, str]]] = {
    "aka_v1_operator_sft": _V1_ARTIFACT_FIELDS,
    "aka_v2_review_projection": _V2_ARTIFACT_FIELDS,
}

# A generation instruction is a contract locator, not code.  In the v2 review projection,
# neutral and negative rows are reviewer/context pairs rather than candidate artifacts, so
# neither field is scanned as CUDA.  Positive rows retain the v1 baseline/candidate roles.
_V1_CODE_ARTIFACT_FIELDS: Mapping[str, Mapping[str, str]] = {
    task: {role: field for role, field in fields.items() if role != "contract"}
    for task, fields in _V1_ARTIFACT_FIELDS.items()
}
_V2_CODE_ARTIFACT_FIELDS: Mapping[str, Mapping[str, str]] = {
    **_V1_CODE_ARTIFACT_FIELDS,
    "optimization_neutral": {},
    "optimization_negative": {},
}
CODE_ARTIFACT_FIELDS_BY_FORMAT: Mapping[str, Mapping[str, Mapping[str, str]]] = {
    "aka_v1_operator_sft": _V1_CODE_ARTIFACT_FIELDS,
    "aka_v2_review_projection": _V2_CODE_ARTIFACT_FIELDS,
}
RECORD_FORMATS = tuple(ARTIFACT_FIELDS_BY_FORMAT)

# These are overlapping lexical incidence signals, never semantic labels.  Every pattern
# has a concrete Cake ownership question; broad operator-name classifiers are intentionally
# absent because an AKA folder or prose label is not an IR operation oracle.
LEXICAL_SIGNALS: Mapping[str, re.Pattern[str]] = {
    "thread_mapping": re.compile(r"\b(?:threadIdx|blockIdx|blockDim|gridDim)\b"),
    "shared_memory": re.compile(r"\b__shared__\b"),
    "dynamic_shared_memory": re.compile(r"\bextern\s+__shared__\b"),
    "barrier": re.compile(
        r"\b(?:__syncthreads|__syncwarp|cuda::barrier|cooperative_groups)\b"
    ),
    "warp_collective": re.compile(
        r"\b(?:__shfl\w*|__ballot\w*|__activemask|cooperative_groups)\b"
    ),
    "atomic": re.compile(r"\batomic[A-Za-z0-9_]*\s*\("),
    "vectorized_access": re.compile(
        r"\b(?:float[234]|double[24]|half2|__half2|int[234]|uint[234]|uchar[234])\b"
    ),
    "tensor_core": re.compile(r"\b(?:wmma|wgmma|mma\.sync|tcgen05)\b", re.IGNORECASE),
    "inline_ptx": re.compile(r"\b(?:__asm__|asm)\s*(?:volatile\s*)?\("),
    "library_or_framework": re.compile(
        r"\b(?:ATen|c10|Caffe2|cub::|thrust::|cudaLaunchKernel)\b"
    ),
    "collective_communication": re.compile(
        r"\b(?:nccl|all[_ ]?reduce|reduce[_ ]?scatter|all[_ ]?gather)\b",
        re.IGNORECASE,
    ),
}

KERNEL_DEFINITION = re.compile(r"\b__global__\b")
LAUNCH_SITE = re.compile(r"<<<")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
PORTABLE_PARENT_SCHEMA = "aka.portable-kernel-parent.v1"
PORTABLE_QUALIFICATION_FIELDS = frozenset(
    {"authority", "lifecycle", "locator", "stages", "validity"}
)
PORTABLE_QUALIFICATION_STAGES = frozenset({"compile", "correctness", "sanitize"})
PORTABLE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CorpusAuditError(ValueError):
    """The source collection violates the bounded AKA storage contract."""


@dataclass(frozen=True)
class SnapshotShard:
    relative_path: str
    repository_path: str
    text: str


@dataclass(frozen=True)
class GitSnapshot:
    repository_remote_observed_sanitized: str | None
    dataset_path: str
    revision: str
    shards: tuple[SnapshotShard, ...]


@dataclass(frozen=True)
class PortableQualifiedSnapshot:
    """One commit-bound, fully admitted portable qualified-parent snapshot.

    ``records`` and ``record_lines`` retain source order.  Qualification fields are a
    portable projection of node-owned evidence; this type deliberately does not convert
    that projection into node custody or an executable-parent claim.
    """

    dataset_root: Path
    dataset_path: str
    source_dataset_path: str
    revision: str
    records: tuple[Mapping[str, Any], ...]
    record_lines: Mapping[str, int]
    artifact_paths: Mapping[str, tuple[str, ...]]
    validation: Mapping[str, Any]


@dataclass(frozen=True)
class SourceRecord:
    relative_path: str
    line_number: int
    category: str
    operator: str
    task: str
    primary_field: str
    relation: str
    fields: Mapping[str, str]
    record_format: str

    @property
    def source(self) -> str:
        return self.fields[self.primary_field]

    @property
    def exact_record(self) -> tuple[str, str, str, str]:
        return tuple(self.fields[field] for field in sorted(RECORD_FIELDS))  # type: ignore[return-value]

    @property
    def artifact_fields(self) -> Mapping[str, str]:
        return ARTIFACT_FIELDS_BY_FORMAT[self.record_format][self.task]

    @property
    def artifact_signals(self) -> dict[str, dict[str, object]]:
        return {
            role: {
                "field": field,
                "scope_signal": _scope_signal(self.fields[field]),
                "lexical_signals": list(_lexical_signals(self.fields[field])),
            }
            for role, field in CODE_ARTIFACT_FIELDS_BY_FORMAT[self.record_format][
                self.task
            ].items()
        }


def _scope_signal(source: str) -> str:
    definitions = len(KERNEL_DEFINITION.findall(source))
    launches = len(LAUNCH_SITE.findall(source))
    if definitions >= 2 or launches >= 2:
        return "multi_kernel_or_launch"
    if definitions == 1:
        return "single_kernel"
    return "fragment_or_library"


def _lexical_signals(source: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in LEXICAL_SIGNALS.items() if pattern.search(source))


def _git(
    directory: Path, arguments: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", "-C", str(directory), *arguments],
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as error:
        detail = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError) and error.stderr
            else str(error)
        )
        raise CorpusAuditError(f"cannot read source Git snapshot: {detail}") from error


def _sanitize_remote(remote: str) -> str | None:
    """Retain a useful observed repository hint without exposing URL credentials."""

    remote = remote.strip()
    if not remote:
        return None
    if "://" in remote:
        try:
            parsed = urlsplit(remote)
            host = parsed.hostname
            if host is None:
                return None
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            return urlunsplit((parsed.scheme.lower(), host, parsed.path, "", ""))
        except ValueError:
            return None
    scp = re.fullmatch(r"(?:[^@/:]+@)?([^:]+):(.+)", remote)
    if scp is not None:
        return f"{scp.group(1)}:{scp.group(2)}"
    # A local filesystem remote can identify a checkout but not a portable repository,
    # and printing it would disclose an unrelated host path.
    return None


def verify_git_snapshot(dataset_root: Path, source_revision: str) -> GitSnapshot:
    """Enumerate and read the exact regular shard blobs named by ``source_revision``.

    The working tree is used only to locate the repository and dataset root.  Shard paths
    and bytes come from the selected commit tree, so ignored files, checkout filters,
    symlinks and concurrent worktree changes cannot enter the projection.
    """

    if GIT_COMMIT.fullmatch(source_revision) is None:
        raise CorpusAuditError("source revision must be one exact lowercase 40-hex Git commit")
    repository = Path(
        _git(dataset_root, ["rev-parse", "--show-toplevel"]).stdout.strip()
    ).resolve()
    try:
        relative = dataset_root.resolve().relative_to(repository)
    except ValueError as error:
        raise CorpusAuditError(f"{dataset_root} is outside its reported Git repository") from error

    resolved = _git(repository, ["rev-parse", "--verify", f"{source_revision}^{{commit}}"])
    if resolved.stdout.strip() != source_revision:
        raise CorpusAuditError(f"source revision {source_revision} does not resolve exactly")
    dataset_path = PurePosixPath(relative.as_posix())
    categories = dataset_path / "categories"
    _git(repository, ["cat-file", "-e", f"{source_revision}:{categories.as_posix()}"])

    listing = _git(
        repository,
        [
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            source_revision,
            "--",
            categories.as_posix(),
        ],
    ).stdout
    entries = [entry for entry in listing.split("\0") if entry]
    if not entries:
        raise CorpusAuditError(f"{categories} contains no operator JSONL shards")

    shards: list[SnapshotShard] = []
    for entry in entries:
        try:
            metadata, repository_path = entry.split("\t", 1)
            mode, object_type, object_id = metadata.split()
        except ValueError as error:
            raise CorpusAuditError(f"cannot parse Git tree entry {entry!r}") from error
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise CorpusAuditError(
                f"{repository_path} is not a regular Git blob (mode={mode}, type={object_type})"
            )
        try:
            relative_path = PurePosixPath(repository_path).relative_to(dataset_path)
        except ValueError as error:
            raise CorpusAuditError(
                f"Git tree path {repository_path} escapes dataset {dataset_path}"
            ) from error
        if (
            len(relative_path.parts) != 4
            or relative_path.parts[0] != "categories"
            or relative_path.suffix != ".jsonl"
        ):
            raise CorpusAuditError(
                f"{repository_path} is not "
                "<dataset>/categories/<category>/<operator>/<task>.jsonl"
            )
        text = _git(repository, ["cat-file", "blob", object_id]).stdout
        shards.append(
            SnapshotShard(relative_path.as_posix(), repository_path, text)
        )

    remote = _git(repository, ["config", "--get", "remote.origin.url"], check=False)
    if remote.returncode not in {0, 1}:
        raise CorpusAuditError(
            f"cannot observe repository origin remote: {remote.stderr.strip()}"
        )
    return GitSnapshot(
        repository_remote_observed_sanitized=_sanitize_remote(remote.stdout),
        dataset_path=dataset_path.as_posix(),
        revision=source_revision,
        shards=tuple(sorted(shards, key=lambda shard: shard.relative_path)),
    )


def load_records(snapshot: GitSnapshot, *, record_format: str) -> list[SourceRecord]:
    if record_format not in RECORD_FORMATS:
        raise CorpusAuditError(
            f"record format {record_format!r} is unsupported; admitted: {RECORD_FORMATS}"
        )
    if not snapshot.shards:
        raise CorpusAuditError(
            f"{snapshot.dataset_path}/categories contains no operator JSONL shards"
        )

    records: list[SourceRecord] = []
    for shard in snapshot.shards:
        relative = PurePosixPath(shard.relative_path)
        if len(relative.parts) != 4 or relative.parts[0] != "categories":
            raise CorpusAuditError(
                f"{relative} is not categories/<category>/<operator>/<task>.jsonl"
            )
        _, category, operator, filename = relative.parts
        task = Path(filename).stem
        if task not in TASKS:
            admitted = ", ".join(sorted(TASKS))
            raise CorpusAuditError(
                f"{relative} has unsupported task {task!r}; admitted: {admitted}"
            )
        primary_field, relation = TASKS[task]

        lines = shard.text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if not lines:
            raise CorpusAuditError(f"{relative} contains no records")
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                raise CorpusAuditError(f"{relative}:{line_number} is blank")
            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise CorpusAuditError(
                    f"{relative}:{line_number} is not JSON: {error.msg}"
                ) from error
            if not isinstance(document, dict) or set(document) != RECORD_FIELDS:
                actual = (
                    sorted(document)
                    if isinstance(document, dict)
                    else type(document).__name__
                )
                raise CorpusAuditError(
                    f"{relative}:{line_number} fields are {actual}; "
                    f"expected {sorted(RECORD_FIELDS)}"
                )
            non_strings = sorted(
                field for field, value in document.items() if not isinstance(value, str)
            )
            if non_strings:
                raise CorpusAuditError(
                    f"{relative}:{line_number} non-string fields: {non_strings}"
                )
            if not document[primary_field].strip():
                raise CorpusAuditError(
                    f"{relative}:{line_number} has empty primary field {primary_field!r}"
                )
            records.append(
                SourceRecord(
                    relative.as_posix(),
                    line_number,
                    category,
                    operator,
                    task,
                    primary_field,
                    relation,
                    document,
                    record_format,
                )
            )
    return records


def _git_blob_bytes(repository: Path, object_id: str) -> bytes:
    try:
        return subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repository),
                "cat-file",
                "blob",
                object_id,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise CorpusAuditError(
            f"cannot read portable Git blob: {str(detail).strip()}"
        ) from error


def _plain_dataset_root(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise CorpusAuditError(f"cannot inspect {label} {path}: {error}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise CorpusAuditError(f"{label} must be a regular non-symlink directory")
    return path.resolve(strict=True)


def _repository_relative(repository: Path, path: Path, label: str) -> PurePosixPath:
    try:
        relative = path.resolve(strict=True).relative_to(repository)
    except ValueError as error:
        raise CorpusAuditError(f"{label} is outside its Git repository") from error
    return PurePosixPath(relative.as_posix())


def _portable_tree(
    repository: Path, revision: str, dataset_path: PurePosixPath
) -> dict[str, str]:
    listing = _git(
        repository,
        [
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            revision,
            "--",
            dataset_path.as_posix(),
        ],
    ).stdout
    tree: dict[str, str] = {}
    for entry in (value for value in listing.split("\0") if value):
        try:
            metadata, repository_path = entry.split("\t", 1)
            mode, object_type, object_id = metadata.split()
            relative = PurePosixPath(repository_path).relative_to(dataset_path)
        except ValueError as error:
            raise CorpusAuditError(
                f"cannot parse portable Git tree entry {entry!r}"
            ) from error
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise CorpusAuditError(
                f"portable path {repository_path} is not a regular Git blob"
            )
        key = relative.as_posix()
        if key in tree:
            raise CorpusAuditError(f"portable Git tree repeats {key}")
        tree[key] = object_id
    if not tree:
        raise CorpusAuditError(f"portable dataset {dataset_path} is empty at {revision}")
    return tree


def _portable_relative(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise CorpusAuditError(f"{label} is not a non-empty dataset-relative path")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise CorpusAuditError(f"{label} contains an inadmissible path character")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise CorpusAuditError(f"{label} escapes the portable dataset")
    return relative


def _plain_worktree_file(root: Path, relative: PurePosixPath, label: str) -> Path:
    current = root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise CorpusAuditError(f"cannot inspect {label} {current}: {error}") from error
        if stat.S_ISLNK(mode):
            raise CorpusAuditError(f"{label} traverses a symlink: {current}")
        if index + 1 < len(relative.parts):
            if not stat.S_ISDIR(mode):
                raise CorpusAuditError(f"{label} parent is not a directory: {current}")
        elif not stat.S_ISREG(mode):
            raise CorpusAuditError(f"{label} is not a regular file: {current}")
    try:
        current.resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise CorpusAuditError(f"{label} resolves outside the portable dataset") from error
    return current


def _portable_object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CorpusAuditError(f"{label} must be one JSON object")
    return value


def _portable_nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusAuditError(f"{label} must be a non-empty string")
    return value


def _portable_identifier(value: object, label: str) -> str:
    identifier = _portable_nonempty(value, label)
    if PORTABLE_IDENTIFIER.fullmatch(identifier) is None:
        raise CorpusAuditError(f"{label} is not a path-safe identifier")
    return identifier


def _portable_strings(
    value: object, label: str, *, allow_empty: bool = False
) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise CorpusAuditError(f"{label} must be an array of non-empty strings")
    if not allow_empty and not value:
        raise CorpusAuditError(f"{label} must not be empty")
    return value


def _parse_portable_payload(
    payload: bytes, relative: PurePosixPath, label: str
) -> str:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise CorpusAuditError(f"{label} is not UTF-8: {relative}") from error
    if relative.suffix.lower() == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as error:
            raise CorpusAuditError(
                f"{label} is malformed JSON: {relative}: {error.msg}"
            ) from error
        return "json"
    if relative.suffix.lower() == ".jsonl":
        lines = text.splitlines()
        if not lines:
            raise CorpusAuditError(f"{label} JSONL is empty: {relative}")
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                raise CorpusAuditError(
                    f"{label} JSONL has a blank line: {relative}:{line_number}"
                )
            try:
                json.loads(line)
            except json.JSONDecodeError as error:
                raise CorpusAuditError(
                    f"{label} has malformed JSONL: "
                    f"{relative}:{line_number}: {error.msg}"
                ) from error
        return "jsonl"
    return "utf8"


def verify_portable_qualified_snapshot(
    *,
    source_dataset_root: Path,
    portable_dataset_root: Path,
    source_revision: str,
    expected_count: int | None = None,
) -> PortableQualifiedSnapshot:
    """Admit the portable corpus at Cake's independent handoff boundary.

    AKA's ``export_parent_completion_dataset.py`` owns how producer records and bundles
    are constructed. Importing that mutable cross-repository script would couple Cake to
    producer execution. This adapter instead checks only the facts Cake consumes: exact
    commit/path custody, source coordinates, qualification gates, frozen contract fields,
    uniqueness, and every declared artifact. Portable qualification remains review
    admission; it is not current GPU custody or an IR, optimization, or training result.
    """

    if GIT_COMMIT.fullmatch(source_revision) is None:
        raise CorpusAuditError(
            "source revision must be one exact lowercase 40-hex Git commit"
        )
    source_dataset_root = _plain_dataset_root(source_dataset_root, "source dataset")
    portable_dataset_root = _plain_dataset_root(
        portable_dataset_root, "portable dataset"
    )
    repository = Path(
        _git(portable_dataset_root, ["rev-parse", "--show-toplevel"]).stdout.strip()
    ).resolve(strict=True)
    source_repository = Path(
        _git(source_dataset_root, ["rev-parse", "--show-toplevel"]).stdout.strip()
    ).resolve(strict=True)
    if repository != source_repository:
        raise CorpusAuditError(
            "source and portable datasets belong to different repositories"
        )
    resolved = _git(
        repository, ["rev-parse", "--verify", f"{source_revision}^{{commit}}"]
    ).stdout.strip()
    if resolved != source_revision:
        raise CorpusAuditError(f"source revision {source_revision} does not resolve exactly")

    portable_path = _repository_relative(
        repository, portable_dataset_root, "portable dataset"
    )
    source_path = _repository_relative(repository, source_dataset_root, "source dataset")
    tree = _portable_tree(repository, source_revision, portable_path)
    blob_cache: dict[str, bytes] = {}

    def committed_file(value: object, label: str) -> tuple[PurePosixPath, bytes]:
        relative = _portable_relative(value, label)
        object_id = tree.get(relative.as_posix())
        if object_id is None:
            raise CorpusAuditError(
                f"{label} is not a regular file in the source commit: {relative}"
            )
        working = _plain_worktree_file(portable_dataset_root, relative, label)
        payload = blob_cache.get(relative.as_posix())
        if payload is None:
            payload = _git_blob_bytes(repository, object_id)
            blob_cache[relative.as_posix()] = payload
        try:
            working_payload = working.read_bytes()
        except OSError as error:
            raise CorpusAuditError(f"cannot read {label} {working}: {error}") from error
        if working_payload != payload:
            raise CorpusAuditError(
                f"{label} working bytes differ from {source_revision}: {relative}"
            )
        return relative, payload

    records_relative, records_payload = committed_file(
        "records.jsonl", "portable records"
    )
    _parse_portable_payload(records_payload, records_relative, "portable records")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(records_payload.decode("utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CorpusAuditError(
                f"portable records line {line_number} is not an object"
            )
        records.append(value)
    if not records:
        raise CorpusAuditError("portable records contain no rows")
    if expected_count is not None and len(records) != expected_count:
        raise CorpusAuditError(
            f"portable snapshot expected {expected_count} rows, observed {len(records)}"
        )

    source_snapshot = verify_git_snapshot(source_dataset_root, source_revision)
    source_records = {
        (record.relative_path, record.line_number): record
        for record in load_records(
            source_snapshot, record_format="aka_v1_operator_sft"
        )
    }
    case_ids: set[str] = set()
    derived_ids: set[str] = set()
    source_identities: set[tuple[str, int, str]] = set()
    locators: set[tuple[str, str]] = set()
    bundle_paths: set[str] = set()
    declared_paths: set[str] = set()
    record_lines: dict[str, int] = {}
    artifact_paths: dict[str, tuple[str, ...]] = {}

    for line_number, row in enumerate(records, 1):
        label = f"portable record {line_number}"
        if row.get("schema") != PORTABLE_PARENT_SCHEMA:
            raise CorpusAuditError(f"{label} has unsupported schema")
        if (
            row.get("outcome") != "qualified"
            or row.get("missing_facts") != []
            or row.get("training_eligibility") is not False
        ):
            raise CorpusAuditError(f"{label} is not a qualified, complete parent")

        case_id = _portable_identifier(row.get("case_id"), f"{label}.case_id")
        derived_id = _portable_identifier(
            row.get("derived_parent_id"), f"{label}.derived_parent_id"
        )
        if case_id in case_ids:
            raise CorpusAuditError(f"duplicate portable case_id: {case_id}")
        if derived_id in derived_ids:
            raise CorpusAuditError(f"duplicate portable derived_parent_id: {derived_id}")
        case_ids.add(case_id)
        derived_ids.add(derived_id)
        record_lines[case_id] = line_number

        provenance = _portable_object(row.get("provenance"), f"{label}.provenance")
        selection = _portable_object(
            provenance.get("source_selection"),
            f"{label}.provenance.source_selection",
        )
        if set(selection) != {"path", "line", "parent_field"}:
            raise CorpusAuditError(f"{label} source selection fields are invalid")
        selected_path = _portable_nonempty(
            selection.get("path"), f"{label}.source_selection.path"
        )
        selected_line = selection.get("line")
        selected_field = selection.get("parent_field")
        if (
            not isinstance(selected_line, int)
            or isinstance(selected_line, bool)
            or selected_line < 1
            or selected_field not in {"input", "output"}
        ):
            raise CorpusAuditError(f"{label} source selection coordinate is invalid")
        identity = (selected_path, selected_line, str(selected_field))
        if identity in source_identities:
            raise CorpusAuditError(
                f"duplicate portable original source identity: {identity}"
            )
        source_identities.add(identity)
        source_record = source_records.get((selected_path, selected_line))
        if source_record is None or not source_record.fields[str(selected_field)].strip():
            raise CorpusAuditError(
                f"{label} source identity is absent or empty in the source commit"
            )
        original = _portable_object(
            row.get("original_parent"), f"{label}.original_parent"
        )
        if dict(original) != {
            "case_path": f"{selected_path}:{selected_line}",
            "record_field": selected_field,
        }:
            raise CorpusAuditError(
                f"{label} original_parent differs from source_selection"
            )

        taxonomy = _portable_object(row.get("taxonomy"), f"{label}.taxonomy")
        for field in ("category", "operator"):
            _portable_nonempty(taxonomy.get(field), f"{label}.taxonomy.{field}")
        semantics = _portable_object(row.get("semantics"), f"{label}.semantics")
        for field in ("inputs", "outputs", "valid_domain"):
            _portable_strings(semantics.get(field), f"{label}.semantics.{field}")
        _portable_nonempty(
            semantics.get("computation"), f"{label}.semantics.computation"
        )
        contract = _portable_object(row.get("contract"), f"{label}.contract")
        for field in (
            "dtypes",
            "index_types",
            "layouts",
            "invariants",
            "exclusions",
        ):
            _portable_strings(contract.get(field), f"{label}.contract.{field}")
        _portable_strings(
            contract.get("optional_inputs"),
            f"{label}.contract.optional_inputs",
            allow_empty=True,
        )
        for field in ("api", "launch_policy"):
            _portable_nonempty(contract.get(field), f"{label}.contract.{field}")

        qualification = _portable_object(
            row.get("qualification"), f"{label}.qualification"
        )
        if set(qualification) != PORTABLE_QUALIFICATION_FIELDS:
            raise CorpusAuditError(f"{label} qualification schema is invalid")
        if (
            qualification.get("authority")
            != "node-owned evidence summarized by portable export"
            or qualification.get("lifecycle") != "completed"
            or qualification.get("validity") != "valid"
        ):
            raise CorpusAuditError(f"{label} qualification is not completed and valid")
        locator = _portable_object(
            qualification.get("locator"), f"{label}.qualification.locator"
        )
        if set(locator) != {"node_id", "run_id"}:
            raise CorpusAuditError(f"{label} fixed locator schema is invalid")
        locator_key = (
            _portable_nonempty(locator.get("node_id"), f"{label}.locator.node_id"),
            _portable_nonempty(locator.get("run_id"), f"{label}.locator.run_id"),
        )
        if locator_key in locators:
            raise CorpusAuditError(f"duplicate portable fixed locator: {locator_key}")
        locators.add(locator_key)
        stages = _portable_object(
            qualification.get("stages"), f"{label}.qualification.stages"
        )
        if set(stages) != PORTABLE_QUALIFICATION_STAGES:
            raise CorpusAuditError(f"{label} qualification stages are incomplete")
        for stage_name in sorted(PORTABLE_QUALIFICATION_STAGES):
            stage = _portable_object(stages.get(stage_name), f"{label}.{stage_name}")
            if stage.get("status") != "passed" or stage.get("validity") != "valid":
                raise CorpusAuditError(
                    f"{label} {stage_name} stage did not pass validly"
                )
        correctness = _portable_object(stages["correctness"], f"{label}.correctness")
        summary = _portable_nonempty(
            correctness.get("summary"), f"{label}.correctness.summary"
        )
        if re.search(r"complete[- ]output", summary, re.IGNORECASE) is None:
            raise CorpusAuditError(
                f"{label} correctness is not complete-output evidence"
            )
        workloads = correctness.get("workloads")
        if (
            not isinstance(workloads, list)
            or len(workloads) < 2
            or any(
                not isinstance(workload, dict)
                or workload.get("correct") is not True
                or not isinstance(workload.get("id", workload.get("name")), str)
                or not workload.get("id", workload.get("name"))
                for workload in workloads
            )
        ):
            raise CorpusAuditError(
                f"{label} lacks two correct complete-output workloads"
            )
        workload_ids = [
            str(value.get("id", value.get("name"))) for value in workloads
        ]
        if len(set(workload_ids)) != len(workload_ids):
            raise CorpusAuditError(f"{label} repeats a correctness workload id")
        sanitize_summary = _portable_nonempty(
            stages["sanitize"].get("summary"), f"{label}.sanitize.summary"
        ).lower()
        if "memcheck" not in sanitize_summary or "racecheck" not in sanitize_summary:
            raise CorpusAuditError(
                f"{label} sanitizer evidence lacks memcheck or racecheck"
            )

        bundle = _portable_object(row.get("bundle"), f"{label}.bundle")
        bundle_path = _portable_relative(bundle.get("path"), f"{label}.bundle.path")
        if bundle_path.parts[0] != "bundles":
            raise CorpusAuditError(f"{label}.bundle.path is outside bundles/")
        if bundle_path.as_posix() in bundle_paths:
            raise CorpusAuditError(f"duplicate portable bundle path: {bundle_path}")
        bundle_paths.add(bundle_path.as_posix())
        artifacts = _portable_object(row.get("artifacts"), f"{label}.artifacts")
        if not {"baseline", "reference", "harness"} <= set(artifacts):
            raise CorpusAuditError(f"{label} lacks required artifact roles")
        row_paths: list[str] = []

        def declare(
            value: object, artifact_label: str, prefix: PurePosixPath
        ) -> tuple[PurePosixPath, bytes]:
            relative = _portable_relative(value, artifact_label)
            if not relative.is_relative_to(prefix):
                raise CorpusAuditError(
                    f"{artifact_label} escapes its declared bundle location"
                )
            if relative.as_posix() in declared_paths:
                raise CorpusAuditError(
                    f"portable corpus repeats declared artifact {relative}"
                )
            relative, payload = committed_file(relative.as_posix(), artifact_label)
            _parse_portable_payload(payload, relative, artifact_label)
            declared_paths.add(relative.as_posix())
            row_paths.append(relative.as_posix())
            return relative, payload

        for role in sorted(artifacts):
            _portable_identifier(role, f"{label}.artifacts role")
            values = artifacts[role]
            if not isinstance(values, list) or not values:
                raise CorpusAuditError(f"{label}.artifacts.{role} must not be empty")
            role_prefix = bundle_path / "sources" / role
            for index, value in enumerate(values):
                declare(value, f"{label}.artifacts.{role}[{index}]", role_prefix)

        documents = _portable_object(
            bundle.get("documents"), f"{label}.bundle.documents"
        )
        for name, value in sorted(documents.items()):
            declare(
                value,
                f"{label}.bundle.documents.{name}",
                bundle_path / "documents",
            )
        source_files = _portable_object(
            bundle.get("source_files"), f"{label}.bundle.source_files"
        )
        if set(source_files) != {"ORIGINAL_RESULT.json", "input.json"}:
            raise CorpusAuditError(f"{label} source-file declaration is incomplete")
        bundled_input: Mapping[str, Any] | None = None
        for name, value in sorted(source_files.items()):
            _, payload = declare(
                value,
                f"{label}.bundle.source_files.{name}",
                bundle_path / "source",
            )
            if name == "input.json":
                parsed = json.loads(payload.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise CorpusAuditError(f"{label} bundled input is not an object")
                bundled_input = parsed
        if bundled_input is None or bundled_input.get("record") != dict(
            source_record.fields
        ):
            raise CorpusAuditError(
                f"{label} bundled source differs from the source commit"
            )
        artifact_paths[case_id] = tuple(row_paths)

    _, merge_payload = committed_file("MERGE_SUMMARY.json", "portable merge summary")
    _parse_portable_payload(
        merge_payload, PurePosixPath("MERGE_SUMMARY.json"), "portable merge summary"
    )
    merge_summary = json.loads(merge_payload.decode("utf-8"))
    if not isinstance(merge_summary, dict) or any(
        merge_summary.get(field) != expected
        for field, expected in (
            ("schema", "aka.portable-kernel-parent-merge.v1"),
            ("records", len(records)),
            ("outcomes", {"qualified": len(records)}),
            ("training_eligible", 0),
            ("unique_locators", len(records)),
        )
    ):
        raise CorpusAuditError("portable merge summary disagrees with admitted rows")

    validation: dict[str, Any] = {
        "records_file": records_relative.as_posix(),
        "records": len(records),
        "declared_artifacts": len(declared_paths),
        "source_revision": source_revision,
        "allowed_claim": "qualified_parent_corpus_review_admission_only",
    }
    return PortableQualifiedSnapshot(
        dataset_root=portable_dataset_root,
        dataset_path=portable_path.as_posix(),
        source_dataset_path=source_path.as_posix(),
        revision=source_revision,
        records=tuple(records),
        record_lines=record_lines,
        artifact_paths=artifact_paths,
        validation=validation,
    )


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_summary(
    records: list[SourceRecord], *, snapshot: GitSnapshot, reported_dataset_label: str
) -> dict[str, object]:
    record_formats = {record.record_format for record in records}
    if len(record_formats) != 1:
        raise CorpusAuditError(
            f"summary requires one record format; observed: {sorted(record_formats)}"
        )
    record_format = next(iter(record_formats))
    exact_records = Counter(record.exact_record for record in records)
    source_groups: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        source_groups[record.source].append(record)

    repeated_sources = [group for source, group in source_groups.items() if source and len(group) > 1]
    cross_category_groups = [
        group for group in repeated_sources if len({record.category for record in group}) > 1
    ]
    artifact_signals: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        for role, signals in record.artifact_signals.items():
            artifact_signals[role].append(signals)

    return {
        "schema": "open-cake.aka-challenge-corpus-audit.v2",
        "record_format": record_format,
        "source": {
            "repository_remote_observed_sanitized": (
                snapshot.repository_remote_observed_sanitized
            ),
            "revision": snapshot.revision,
            "dataset_path": snapshot.dataset_path,
            "reported_dataset_label": reported_dataset_label,
        },
        "records": len(records),
        "tasks": _sorted_counts(record.task for record in records),
        "relations": _sorted_counts(record.relation for record in records),
        "categories": _sorted_counts(record.category for record in records),
        "operators": _sorted_counts(record.operator for record in records),
        "artifact_scope_signals": {
            role: _sorted_counts(str(signals["scope_signal"]) for signals in values)
            for role, values in sorted(artifact_signals.items())
        },
        "artifact_lexical_signals": {
            role: {
                name: sum(
                    name in signals["lexical_signals"]  # type: ignore[operator]
                    for signals in values
                )
                for name in LEXICAL_SIGNALS
            }
            for role, values in sorted(artifact_signals.items())
        },
        "empty_input_records": sum(not record.fields["input"] for record in records),
        "duplicate_full_record_groups": sum(count > 1 for count in exact_records.values()),
        "duplicate_primary_source_groups": len(repeated_sources),
        "duplicate_primary_source_records": sum(len(group) for group in repeated_sources),
        "cross_category_primary_source_groups": len(cross_category_groups),
        "authority_boundary": {
            "provenance_status": "absent_from_four_field_record",
            "evidence_status": "absent_from_four_field_record",
            "split_group_status": "unknown",
            "compiler_corpus_eligible": False,
            "allowed_claim": "external_challenge_and_gap_discovery_only",
        },
    }


def projected_case(
    record: SourceRecord, *, snapshot: GitSnapshot, reported_dataset_label: str
) -> dict[str, object]:
    repository_path = (PurePosixPath(snapshot.dataset_path) / record.relative_path).as_posix()
    return {
        "schema": "open-cake.aka-challenge-corpus-case.v2",
        "record_format": record.record_format,
        "source_ref": {
            "repository_remote_observed_sanitized": (
                snapshot.repository_remote_observed_sanitized
            ),
            "revision": snapshot.revision,
            "dataset_path": snapshot.dataset_path,
            "reported_dataset_label": reported_dataset_label,
            "path": repository_path,
            "line": record.line_number,
            "primary_field": record.primary_field,
        },
        "reported_axes": {
            "category": record.category,
            "operator": record.operator,
            "task": record.task,
        },
        "relation": record.relation,
        "artifact_fields": dict(record.artifact_fields),
        "artifact_signals": record.artifact_signals,
        "owner_scope": "unknown",
        "complete_parent_expressibility": "unknown",
        "delta_expressibility": "unknown",
        "provenance_status": "absent_from_four_field_record",
        "evidence_status": "absent_from_four_field_record",
        "split_group": None,
        "review_state": "unreviewed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--source-revision",
        required=True,
        help="exact lowercase 40-hex source Git commit owning the operator shards",
    )
    parser.add_argument(
        "--dataset-label",
        "--dataset-id",
        dest="dataset_label",
        help=(
            "reported dataset label; defaults to the dataset directory name and is not "
            "source authority"
        ),
    )
    parser.add_argument(
        "--record-format",
        required=True,
        choices=RECORD_FORMATS,
        help="explicit contract assigning the four stored fields to review roles",
    )
    parser.add_argument("--emit", choices=("summary", "cases"), default="summary")
    arguments = parser.parse_args()

    dataset_root = arguments.dataset_root.resolve()
    dataset_label = arguments.dataset_label or dataset_root.name
    try:
        snapshot = verify_git_snapshot(dataset_root, arguments.source_revision)
        records = load_records(snapshot, record_format=arguments.record_format)
    except CorpusAuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if arguments.emit == "cases":
        for record in records:
            print(
                json.dumps(
                    projected_case(
                        record,
                        snapshot=snapshot,
                        reported_dataset_label=dataset_label,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return 0

    print(
        json.dumps(
            build_summary(
                records,
                snapshot=snapshot,
                reported_dataset_label=dataset_label,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    # `--emit cases` is intentionally streamable into jq/head. Let the ordinary Unix
    # broken-pipe status replace Python's traceback when the downstream reader stops early.
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(main())
