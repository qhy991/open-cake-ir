#!/usr/bin/env python3
"""Discover immutable Executor references and derive a read-only release plan."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping


_REVISION_ID = re.compile(
    r"open-cake-ir-(?P<family>b200|gfx1151)-v(?P<ordinal>[1-9][0-9]*)"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIGEST_FIELDS = (
    "canonical_sha256",
    "descriptor_raw_sha256",
    "executor_descriptor_raw_sha256",
)
_DERIVED_INVENTORY = "inventory/EXECUTOR_REVISIONS.json"


@dataclass(frozen=True, order=True)
class ExecutorRevisionIdentity:
    revision_id: str
    family: str
    ordinal: int


@dataclass(frozen=True, order=True)
class ExecutorRevisionWitness:
    revision_id: str
    family: str
    ordinal: int
    path: str
    reference_path: str
    digests: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ExecutorRevisionCyclePlan:
    current_revision_id: str
    family: str
    current_witnessed: bool
    next_revision_id: str
    reclaimable_descriptors: tuple[str, ...]


def verify_executor_revision_commit(
    project_root: str | Path,
    executor_id: object,
    final_path: str | Path,
    initial_raw_sha256: str | None,
) -> ExecutorRevisionCyclePlan:
    """Recheck witness and working bytes at the destructive commit boundary."""

    root = Path(project_root).resolve(strict=True)
    path = Path(final_path)
    unresolved = path if path.is_absolute() else root / path
    final = unresolved.parent.resolve(strict=True) / unresolved.name
    runtime = (root / "runtime/executors").resolve(strict=True)
    if final.parent.resolve(strict=True) != runtime:
        raise ValueError("Executor commit path differs")
    if initial_raw_sha256 is not None and _SHA256.fullmatch(initial_raw_sha256) is None:
        raise ValueError("Executor initial descriptor digest differs")
    plan = plan_executor_revision_cycle(root, executor_id)
    if plan.current_witnessed or plan.next_revision_id != executor_id:
        raise ValueError("Executor witness state changed during release")
    relative = final.relative_to(root).as_posix()
    if initial_raw_sha256 is None:
        if final.exists() or final.is_symlink():
            raise ValueError("Executor appeared during release")
    elif (
        not final.is_file()
        or final.is_symlink()
        or relative not in plan.reclaimable_descriptors
        or sha256(final.read_bytes()).hexdigest() != initial_raw_sha256
    ):
        raise ValueError("working Executor changed during release")
    return plan


def parse_executor_revision_id(value: object) -> ExecutorRevisionIdentity | None:
    """Parse one released Executor identity from either independent lineage."""

    if not isinstance(value, str):
        return None
    match = _REVISION_ID.fullmatch(value)
    if match is None:
        return None
    return ExecutorRevisionIdentity(
        revision_id=value,
        family=match.group("family"),
        ordinal=int(match.group("ordinal")),
    )


def _is_executor_revision_id(value: object) -> bool:
    return parse_executor_revision_id(value) is not None


def _executor_revision_id(family: str, ordinal: int) -> str:
    if family not in {"b200", "gfx1151"} or ordinal <= 0:
        raise ValueError("Executor Revision family or ordinal differs")
    return f"open-cake-ir-{family}-v{ordinal}"


def _bound_digests(
    value: Mapping[str, object], *, source: str, reference_path: str
) -> tuple[tuple[str, str], ...]:
    """Return exact descriptor bindings carried by one reference object.

    An Executor id in prose, status, or a released descriptor is not a witness. The
    object that names it must also bind canonical descriptor bytes or raw descriptor
    bytes. A malformed attempted binding fails closed instead of making the id reusable.
    """

    result: list[tuple[str, str]] = []
    for field in _DIGEST_FIELDS:
        if field not in value:
            continue
        digest = value[field]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(
                f"Executor witness {source}:{reference_path}.{field} digest differs"
            )
        result.append((field, digest))
    return tuple(result)


def _walk_references(
    value: object,
    *,
    source: str,
    reference_path: str = "$",
) -> list[ExecutorRevisionWitness]:
    found: list[ExecutorRevisionWitness] = []
    if isinstance(value, Mapping):
        identity = parse_executor_revision_id(value.get("executor_id"))
        if identity is not None:
            digests = _bound_digests(
                value, source=source, reference_path=reference_path
            )
            if digests:
                found.append(
                    ExecutorRevisionWitness(
                        revision_id=identity.revision_id,
                        family=identity.family,
                        ordinal=identity.ordinal,
                        path=source,
                        reference_path=reference_path,
                        digests=digests,
                    )
                )
        for key, child in value.items():
            _walk_path = f"{reference_path}.{key}"
            found.extend(
                _walk_references(
                    child,
                    source=source,
                    reference_path=_walk_path,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _walk_references(
                    child,
                    source=source,
                    reference_path=f"{reference_path}[{index}]",
                )
            )
    return found


def _witness_documents(root: Path) -> tuple[tuple[Path, bool], ...]:
    """Return candidate documents and whether their top level must be frozen."""

    candidates: set[tuple[Path, bool]] = set()
    for pattern in ("contracts/studies/*.json", "contracts/calibrations/*.json"):
        candidates.update(
            (path, True)
            for path in root.glob(pattern)
            if path.is_file() and not path.is_symlink()
        )
    candidates.update(
        (path, False)
        for path in root.glob("evidence/**/*.json")
        if path.is_file() and not path.is_symlink()
    )
    candidates.update(
        (path, False)
        for path in root.glob("runtime/*.campaign.lock.json")
        if path.is_file() and not path.is_symlink()
    )
    candidates.update(
        (path, False)
        for path in root.glob("inventory/*.json")
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix() != _DERIVED_INVENTORY
    )
    return tuple(sorted(candidates, key=lambda item: item[0].as_posix()))


def executor_revision_witnesses(
    project_root: str | Path,
) -> tuple[ExecutorRevisionWitness, ...]:
    """Return every digest-bound Executor reference owned by frozen repository facts.

    Study templates and non-frozen calibration drafts are moving design inputs, so they
    do not witness. Frozen calibrations, historical repository Campaign Locks, Evidence
    and inventory observations freeze a reference by binding its descriptor digest. The
    derived Executor inventory and runtime descriptors are deliberately outside this
    discovery set and cannot witness their own working id. A new external Campaign Lock
    needs a repository-owned immutable registration before dispatch because a release
    process cannot discover files outside this custody root.
    """

    root = Path(project_root).resolve(strict=True)
    witnesses: set[ExecutorRevisionWitness] = set()
    for path, requires_frozen_state in _witness_documents(root):
        relative = path.relative_to(root).as_posix()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read Executor witness {relative}") from error
        if (
            requires_frozen_state
            and (
                not isinstance(document, Mapping)
                or document.get("state") != "frozen"
            )
        ):
            continue
        witnesses.update(_walk_references(document, source=relative))
        if (
            relative.startswith("evidence/executors/")
            and relative.endswith("/runtime/executor.json")
            and isinstance(document, Mapping)
            and document.get("state") == "released"
        ):
            identity = parse_executor_revision_id(document.get("executor_id"))
            if identity is None:
                raise ValueError(f"archived Executor witness {relative} identity differs")
            canonical = json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
            witnesses.add(
                ExecutorRevisionWitness(
                    revision_id=identity.revision_id,
                    family=identity.family,
                    ordinal=identity.ordinal,
                    path=relative,
                    reference_path="$",
                    digests=(
                        ("canonical_sha256", sha256(canonical).hexdigest()),
                        ("descriptor_raw_sha256", sha256(path.read_bytes()).hexdigest()),
                    ),
                )
            )
    return tuple(sorted(witnesses))


def plan_executor_revision_cycle(
    project_root: str | Path, current_revision_id: object
) -> ExecutorRevisionCyclePlan:
    """Derive the next id and reclaimable working tail without changing any file."""

    current = parse_executor_revision_id(current_revision_id)
    if current is None:
        raise ValueError("current Executor Revision identity differs")
    root = Path(project_root).resolve(strict=True)
    family_witnesses = tuple(
        item
        for item in executor_revision_witnesses(root)
        if item.family == current.family
    )
    witnessed_ids = {item.revision_id for item in family_witnesses}
    history = max((item.ordinal for item in family_witnesses), default=0)

    reclaimable: list[tuple[int, str]] = []
    for path in sorted((root / "runtime/executors").glob("*.json")):
        if path.name.startswith(".") or not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read Executor descriptor {relative}") from error
        if not isinstance(document, Mapping):
            raise ValueError(f"Executor descriptor {relative} must be an object")
        identity = parse_executor_revision_id(document.get("executor_id"))
        if identity is None:
            continue
        if path.name != f"{identity.revision_id}.json":
            raise ValueError(f"Executor descriptor {relative} identity differs")
        if (
            identity.family == current.family
            and identity.ordinal > history
            and identity.revision_id not in witnessed_ids
        ):
            reclaimable.append((identity.ordinal, relative))

    return ExecutorRevisionCyclePlan(
        current_revision_id=current.revision_id,
        family=current.family,
        current_witnessed=current.revision_id in witnessed_ids,
        next_revision_id=_executor_revision_id(current.family, history + 1),
        reclaimable_descriptors=tuple(
            relative for _, relative in sorted(reclaimable)
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--current-revision-id")
    arguments = parser.parse_args()
    if arguments.current_revision_id is None:
        value: object = [
            asdict(item)
            for item in executor_revision_witnesses(arguments.project_root)
        ]
    else:
        value = asdict(
            plan_executor_revision_cycle(
                arguments.project_root, arguments.current_revision_id
            )
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
