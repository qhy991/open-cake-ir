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
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping
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
