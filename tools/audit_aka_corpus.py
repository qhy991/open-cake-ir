#!/usr/bin/env python3
"""Project AKA operator shards into a read-only Cake challenge-corpus view.

AKA owns the source records.  This tool does not translate CUDA into Schedules and does
not accept the records as correctness, performance, or provenance evidence.  It validates
the narrow four-string storage contract, preserves path-and-line locators, and reports
only syntactic scope/signals that help a reviewer route a record to Schedule, program,
portfolio, or contract work.

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
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


RECORD_FIELDS = frozenset({"instruction", "input", "reasoning", "output"})

# The source field is a task-storage convention, not a claim that the field is a complete
# translation unit or the factual baseline.  In particular, generation stores its produced
# implementation in output while the other tasks put the source/baseline in input.
TASKS: Mapping[str, tuple[str, str]] = {
    "analysis": ("input", "analyze"),
    "generation": ("output", "generate"),
    "debug": ("input", "repair"),
    "optimization_positive": ("input", "optimize"),
    "optimization_neutral": ("input", "optimize"),
    "optimization_negative": ("input", "optimize"),
}

ARTIFACT_FIELDS: Mapping[str, Mapping[str, str]] = {
    "analysis": {"source": "input"},
    "generation": {"contract": "instruction", "generated": "output"},
    "debug": {"broken": "input", "repaired": "output"},
    "optimization_positive": {"baseline": "input", "candidate": "output"},
    "optimization_neutral": {"baseline": "input", "candidate": "output"},
    "optimization_negative": {"baseline": "input", "candidate": "output"},
}

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
class SourceRecord:
    relative_path: str
    line_number: int
    category: str
    operator: str
    task: str
    primary_field: str
    relation: str
    fields: Mapping[str, str]

    @property
    def source(self) -> str:
        return self.fields[self.primary_field]

    @property
    def scope_signal(self) -> str:
        definitions = len(KERNEL_DEFINITION.findall(self.source))
        launches = len(LAUNCH_SITE.findall(self.source))
        if definitions >= 2 or launches >= 2:
            return "multi_kernel_or_launch"
        if definitions == 1:
            return "single_kernel"
        return "fragment_or_library"

    @property
    def lexical_signals(self) -> tuple[str, ...]:
        return tuple(
            name for name, pattern in LEXICAL_SIGNALS.items() if pattern.search(self.source)
        )

    @property
    def exact_record(self) -> tuple[str, str, str, str]:
        return tuple(self.fields[field] for field in sorted(RECORD_FIELDS))  # type: ignore[return-value]


def _git(
    directory: Path, arguments: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", "-C", str(directory), *arguments],
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError) and error.stderr
            else str(error)
        )
        raise CorpusAuditError(f"cannot verify AKA Git snapshot: {detail}") from error


def verify_git_snapshot(dataset_root: Path, source_revision: str) -> None:
    """Prove that the read shards are exactly the tree named by ``source_revision``.

    A path-and-line locator paired with an unchecked commit string is false provenance.
    Tracked differences and untracked files are separate Git states, so both are refused.
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
    categories = relative / "categories"
    _git(repository, ["cat-file", "-e", f"{source_revision}:{categories.as_posix()}"])

    difference = _git(
        repository,
        ["diff", "--quiet", "--no-ext-diff", source_revision, "--", categories.as_posix()],
        check=False,
    )
    if difference.returncode == 1:
        raise CorpusAuditError(
            f"{categories} tracked bytes differ from source revision {source_revision}"
        )
    if difference.returncode != 0:
        raise CorpusAuditError(
            f"cannot compare {categories} with source revision {source_revision}: "
            f"{difference.stderr.strip()}"
        )

    untracked = _git(
        repository,
        ["ls-files", "--others", "--exclude-standard", "--", categories.as_posix()],
    ).stdout.splitlines()
    if untracked:
        raise CorpusAuditError(f"{categories} contains untracked files: {untracked[:3]}")


def load_records(dataset_root: Path) -> list[SourceRecord]:
    categories = dataset_root / "categories"
    if not categories.is_dir():
        raise CorpusAuditError(f"{dataset_root} has no categories/ directory")

    paths = sorted(categories.glob("*/*/*.jsonl"))
    if not paths:
        raise CorpusAuditError(f"{categories} contains no operator JSONL shards")

    records: list[SourceRecord] = []
    for path in paths:
        relative = path.relative_to(dataset_root)
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

        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
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
                    )
                )
    return records


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_summary(
    records: list[SourceRecord], *, dataset_id: str, source_revision: str
) -> dict[str, object]:
    exact_records = Counter(record.exact_record for record in records)
    source_groups: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        source_groups[record.source].append(record)

    repeated_sources = [group for source, group in source_groups.items() if source and len(group) > 1]
    cross_category_groups = [
        group for group in repeated_sources if len({record.category for record in group}) > 1
    ]
    # Some rows contain very large framework sources.  Evaluate each lexical pattern once
    # per row; recomputing the complete tuple once per signal turns this linear audit into
    # signals-squared work without changing an answer.
    record_signals = [record.lexical_signals for record in records]

    return {
        "schema": "open-cake.aka-challenge-corpus-audit.v1",
        "source": {
            "dataset_id": dataset_id,
            "revision": source_revision,
            "record_authority": "AKA operator shards",
        },
        "records": len(records),
        "tasks": _sorted_counts(record.task for record in records),
        "relations": _sorted_counts(record.relation for record in records),
        "categories": _sorted_counts(record.category for record in records),
        "operators": _sorted_counts(record.operator for record in records),
        "scope_signals": _sorted_counts(record.scope_signal for record in records),
        "lexical_signals": {
            name: sum(name in signals for signals in record_signals)
            for name in LEXICAL_SIGNALS
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
    record: SourceRecord, *, dataset_id: str, source_revision: str
) -> dict[str, object]:
    return {
        "schema": "open-cake.aka-challenge-corpus-case.v1",
        "source_ref": {
            "dataset_id": dataset_id,
            "revision": source_revision,
            "path": record.relative_path,
            "line": record.line_number,
            "primary_field": record.primary_field,
        },
        "reported_axes": {
            "category": record.category,
            "operator": record.operator,
            "task": record.task,
        },
        "relation": record.relation,
        "artifact_fields": dict(ARTIFACT_FIELDS[record.task]),
        "scope_signal": record.scope_signal,
        "lexical_signals": list(record.lexical_signals),
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
        help="exact lowercase 40-hex AKA Git commit owning the operator shards",
    )
    parser.add_argument(
        "--dataset-id",
        help="stable dataset version label; defaults to the dataset directory name",
    )
    parser.add_argument("--emit", choices=("summary", "cases"), default="summary")
    arguments = parser.parse_args()

    dataset_root = arguments.dataset_root.resolve()
    dataset_id = arguments.dataset_id or dataset_root.name
    try:
        verify_git_snapshot(dataset_root, arguments.source_revision)
        records = load_records(dataset_root)
    except CorpusAuditError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if arguments.emit == "cases":
        for record in records:
            print(
                json.dumps(
                    projected_case(
                        record,
                        dataset_id=dataset_id,
                        source_revision=arguments.source_revision,
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
                dataset_id=dataset_id,
                source_revision=arguments.source_revision,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
