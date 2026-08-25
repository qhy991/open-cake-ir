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

_REVISION_ID = re.compile(r"open-cake-ir-(?:sm100a-)?v[1-9][0-9]*")
_FROZEN_GLOBS = (
    "compiler/releases/*/revision.lock.json",
    "contracts/studies/*.json",
    "evidence/**/*.json",
    "inventory/*.json",
)


def _is_compiler_revision_id(value: object) -> bool:
    """Admit the historical single-Target lineage and the generic successor lineage."""

    return isinstance(value, str) and _REVISION_ID.fullmatch(value) is not None


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
