#!/usr/bin/env python3
"""List frozen artifacts that make a Compiler Revision identity immutable."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

_REVISION_ID = re.compile(r"open-cake-ir-(?:sm100a-)?v([1-9][0-9]*)")
_REVISION_LABEL = re.compile(r"v([1-9][0-9]*)")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FROZEN_GLOBS = (
    "compiler/releases/*/revision.lock.json",
    "contracts/studies/*.json",
    "evidence/**/*.json",
    "inventory/*.json",
)


def _is_compiler_revision_id(value: object) -> bool:
    """Admit the historical single-Target lineage and generic successors."""

    return isinstance(value, str) and _REVISION_ID.fullmatch(value) is not None


def compiler_revision_ordinal(value: object) -> int | None:
    """Return one released Compiler ordinal, independent of its valid lineage."""

    if not isinstance(value, str):
        return None
    match = _REVISION_ID.fullmatch(value)
    return int(match.group(1)) if match is not None else None


def generic_compiler_revision_id(label: object, *, draft: bool = False) -> str:
    """Name a multi-Target Compiler Revision from one derived ordinal label."""

    if not isinstance(label, str) or _REVISION_LABEL.fullmatch(label) is None:
        raise ValueError("Compiler Revision label differs")
    return f"open-cake-ir-{label}{'-draft' if draft else ''}"


def _canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, order=True)
class CompilerRevisionWitness:
    revision_id: str
    path: str
    revision_sha256: str | None = None


@dataclass(frozen=True)
class CompilerRevisionCyclePlan:
    """The release-cycle transition derived from frozen Revision witnesses."""

    next_label: str
    archive_label: str | None
    stale_labels: tuple[str, ...]
    collision_incident: str | None = None


def _registered_archive_collision(
    root: Path,
    *,
    revision_id: str,
    current: Mapping[str, object],
    archived: Mapping[str, object],
) -> str | None:
    """Return the incident that exactly retires two colliding frozen variants."""

    if (
        current.get("revision_id") != revision_id
        or archived.get("revision_id") != revision_id
    ):
        return None
    required = {_canonical_sha256(current), _canonical_sha256(archived)}
    if len(required) != 2:
        return None
    for path in sorted((root / "inventory").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(document, Mapping)
            or document.get("kind") != "compiler_revision_identity_incident"
            or document.get("affected_revision_id") != revision_id
            or document.get("status")
            != "identity_collision_historical_records_immutable"
        ):
            continue
        variants = document.get("known_variants")
        resolution = document.get("resolution")
        if not isinstance(variants, list) or not isinstance(resolution, Mapping):
            continue
        registered = {
            value.get("canonical_sha256")
            for value in variants
            if isinstance(value, Mapping)
            and value.get("revision_id") == revision_id
            and isinstance(value.get("canonical_sha256"), str)
            and _SHA256.fullmatch(str(value["canonical_sha256"])) is not None
        }
        retired = resolution.get("ambiguous_ids_retired")
        if (
            required <= registered
            and resolution.get("historical_records_changed") is False
            and isinstance(retired, list)
            and revision_id in retired
        ):
            return path.relative_to(root).as_posix()
    return None


def _walk_references(
    value: object,
    *,
    path: str,
    current_id: str | None,
    current_sha256: str | None,
    known_revision_by_sha256: Mapping[str, str],
) -> list[CompilerRevisionWitness]:
    found: list[CompilerRevisionWitness] = []
    if isinstance(value, Mapping):
        identity = value.get("compiler_revision_id")
        if _is_compiler_revision_id(identity):
            found.append(CompilerRevisionWitness(identity, path))
        identity = value.get("revision_id")
        if _is_compiler_revision_id(identity):
            digest = value.get("revision_sha256", value.get("canonical_sha256"))
            found.append(
                CompilerRevisionWitness(
                    identity,
                    path,
                    digest if isinstance(digest, str) else None,
                )
            )
        reference = value.get("compiler_revision")
        if isinstance(reference, Mapping):
            identity = reference.get("revision_id")
            digest = reference.get(
                "revision_sha256", reference.get("canonical_sha256")
            )
            # Old Study Contracts pinned the canonical bytes but omitted the id. They
            # remain readable only while those bytes are the current lock; successors
            # carry both fields and do not depend on this bounded adapter.
            if (
                not isinstance(identity, str)
                and isinstance(digest, str)
            ):
                identity = (
                    current_id
                    if digest == current_sha256
                    else known_revision_by_sha256.get(digest)
                )
            if _is_compiler_revision_id(identity):
                found.append(
                    CompilerRevisionWitness(
                        identity,
                        path,
                        digest if isinstance(digest, str) else None,
                    )
                )
        for child in value.values():
            found.extend(
                _walk_references(
                    child,
                    path=path,
                    current_id=current_id,
                    current_sha256=current_sha256,
                    known_revision_by_sha256=known_revision_by_sha256,
                )
            )
    elif isinstance(value, list):
        for child in value:
            found.extend(
                _walk_references(
                    child,
                    path=path,
                    current_id=current_id,
                    current_sha256=current_sha256,
                    known_revision_by_sha256=known_revision_by_sha256,
                )
            )
    return found


def compiler_revision_witnesses(root: Path) -> tuple[CompilerRevisionWitness, ...]:
    """Return the deduplicated union of every declared frozen-artifact source."""

    root = root.resolve(strict=True)
    lock_path = root / "compiler/revision.lock.json"
    current_id: str | None = None
    current_sha256: str | None = None
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        identity = lock.get("revision_id") if isinstance(lock, Mapping) else None
        if _is_compiler_revision_id(identity):
            current_id = identity
            current_sha256 = _canonical_sha256(lock)
    paths = {
        path
        for pattern in _FROZEN_GLOBS
        for path in root.glob(pattern)
        if path.is_file() and not path.is_symlink()
    }
    documents: list[tuple[Path, object]] = []
    for path in sorted(paths):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"could not read frozen Compiler witness {path.relative_to(root)}"
            ) from error
        documents.append((path, document))

    # First collect self-identifying references. Released locks need their canonical
    # digest derived because the digest is intentionally outside the document it seals.
    witnesses: set[CompilerRevisionWitness] = set()
    for path, document in documents:
        relative = path.relative_to(root).as_posix()
        witnesses.update(
            _walk_references(
                document,
                path=relative,
                current_id=current_id,
                current_sha256=current_sha256,
                known_revision_by_sha256={},
            )
        )
        if relative.startswith("compiler/releases/") and isinstance(document, Mapping):
            identity = document.get("revision_id")
            if _is_compiler_revision_id(identity):
                witnesses.add(
                    CompilerRevisionWitness(
                        identity,
                        relative,
                        _canonical_sha256(document),
                    )
                )

    identities_by_digest: dict[str, set[str]] = {}
    for witness in witnesses:
        if witness.revision_sha256 is not None:
            identities_by_digest.setdefault(witness.revision_sha256, set()).add(
                witness.revision_id
            )
    known_revision_by_sha256 = {
        digest: next(iter(identities))
        for digest, identities in identities_by_digest.items()
        if len(identities) == 1
    }
    for path, document in documents:
        witnesses.update(
            _walk_references(
                document,
                path=path.relative_to(root).as_posix(),
                current_id=current_id,
                current_sha256=current_sha256,
                known_revision_by_sha256=known_revision_by_sha256,
            )
        )
    resolved = {
        (item.revision_id, item.path)
        for item in witnesses
        if item.revision_sha256 is not None
    }
    witnesses = {
        item
        for item in witnesses
        if item.revision_sha256 is not None
        or (item.revision_id, item.path) not in resolved
    }
    return tuple(
        sorted(
            witnesses,
            key=lambda item: (
                item.revision_id,
                item.path,
                item.revision_sha256 or "",
            ),
        )
    )


def plan_compiler_revision_cycle(
    root: Path, current_revision_id: object
) -> CompilerRevisionCyclePlan:
    """Derive archive/reuse policy without mutating release state.

    Ordinals are shared across the historical ``sm100a`` lineage and the generic
    multi-Target lineage. A frozen reference to either spelling reserves that ordinal.
    """

    current_ordinal = compiler_revision_ordinal(current_revision_id)
    if current_ordinal is None:
        raise ValueError("current Compiler Revision identity differs")
    witnessed_ordinals = {
        ordinal
        for item in compiler_revision_witnesses(root)
        if (ordinal := compiler_revision_ordinal(item.revision_id)) is not None
    }
    history = max(witnessed_ordinals, default=0)
    next_label = f"v{history + 1}"
    if current_ordinal in witnessed_ordinals:
        archive_label = f"v{current_ordinal}"
        current_path = root / "compiler/revision.lock.json"
        archived_path = root / "compiler/releases" / archive_label / "revision.lock.json"
        incident = None
        if current_path.is_file() and archived_path.is_file():
            current = json.loads(current_path.read_text(encoding="utf-8"))
            archived = json.loads(archived_path.read_text(encoding="utf-8"))
            if isinstance(current, Mapping) and isinstance(archived, Mapping):
                incident = _registered_archive_collision(
                    root,
                    revision_id=str(current_revision_id),
                    current=current,
                    archived=archived,
                )
        return CompilerRevisionCyclePlan(
            next_label, archive_label, (), incident
        )

    root = root.resolve(strict=True)
    stale = tuple(
        sorted(
            path.name
            for path in (root / "compiler/releases").glob("v*")
            if (
                (match := _REVISION_LABEL.fullmatch(path.name)) is not None
                and int(match.group(1)) > history
            )
        )
    )
    return CompilerRevisionCyclePlan(next_label, None, stale)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            [asdict(item) for item in compiler_revision_witnesses(arguments.project_root)],
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
