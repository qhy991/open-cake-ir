"""The Compiler manifest, its declared Targets and the source identity it runs from.

`compiler/revision.json` names the Corpus manifest and the calibration coverage. Every
document under `compiler/targets/` is a declared Target, so adding a target is adding a
document. Code identity is the clean git commit of the checkout (ADR 0065): the Compiler
records it when there is one and still assesses and lowers without it, while the Lab
refuses to run a Campaign without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, cast

from ..source_identity import checkout_commit_or_none, untracked_paths
from .errors import CompilerError
from .ir import ScheduleParseError
from .ir.instruction_contracts import CONTRACTS, ContractKind
from .target import Target, TargetParseError

TARGETS_DIRECTORY = "compiler/targets"


@dataclass(frozen=True)
class CompilerRevision:
    """Declared Targets, Corpus and calibration coverage at one source identity.

    The identity is the clean commit alone: `revision_id` is `open-cake-ir@<commit>`, and
    every document this loader read is a tracked file at that commit, so a second digest
    over those documents could only restate it.
    """

    project_root: Path
    revision_id: str
    commit: str | None
    targets: Mapping[str, Target]
    corpus_path: Path
    calibration_coverage: frozenset[str]


def _object(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CompilerError(f"{path} must be an object")
    return cast(Mapping[str, object], value)


def _objects(value: object, path: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise CompilerError(f"{path} must be a list")
    return tuple(_object(item, f"{path}[{index}]") for index, item in enumerate(value))


def _strings(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise CompilerError(f"{path} must be a list of non-empty strings")
    return tuple(cast(list[str], value))


def _name(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompilerError(f"{path} must be a non-empty string")
    return value


def _digest(value: object, path: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CompilerError(f"{path} must be a lowercase SHA256 digest")
    return value


def _project_path(root: Path, value: object, context: str) -> tuple[str, Path]:
    relative = _name(value, context)
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative:
        raise CompilerError(f"{context} is unsafe")
    path = (root / relative).resolve(strict=True)
    if root not in path.parents:
        raise CompilerError(f"{context} escapes project root")
    return relative, path


def _declared_contracts(target: Target) -> None:
    """Refuse a declared contract name that no record owns, or one of the wrong kind.

    The document's shape is `Target.from_dict`'s; this is membership in the registry,
    which the parser does not open. A name outside it would reach the verifier as a
    string it cannot type.
    """
    for label, names, synchronization in (
        ("instruction", target.instruction_contracts, False),
        ("synchronization", target.synchronization_contracts, True),
    ):
        for name in sorted(names):
            record = CONTRACTS.get(name)
            if record is None:
                raise CompilerError(
                    f"target definition {target.target_id!r} declares {label} contract "
                    f"{name!r} that no contract record owns"
                )
            if (record.kind is ContractKind.SYNCHRONIZATION) is not synchronization:
                raise CompilerError(
                    f"target definition {target.target_id!r} declares {label} contract "
                    f"{name!r} whose record is of kind {record.kind.value}"
                )


def _load_target(target_path: Path) -> Target:
    target_id = target_path.stem
    document = _object(
        json.loads(target_path.read_text(encoding="utf-8")),
        f"target_definition.{target_id}",
    )
    if document.get("target_id") != target_id:
        raise CompilerError(f"target definition {target_id!r} identity differs from its file name")
    try:
        typed_target = Target.from_dict(document)
    except (TargetParseError, ScheduleParseError) as error:
        raise CompilerError(f"target definition {target_id!r}: {error}") from error
    citations = _objects(document.get("citations"), f"target_definition.{target_id}.citations")
    if not citations:
        raise CompilerError(f"target definition {target_id!r} requires citations")
    _declared_contracts(typed_target)
    return typed_target


def load_revision(project_root: str | Path, revision_path: str | Path) -> CompilerRevision:
    """Load the Compiler manifest, every declared Target and the checkout's commit."""

    root = Path(project_root).resolve(strict=True)
    path = Path(revision_path).resolve(strict=True)
    if root not in path.parents:
        raise CompilerError(f"compiler revision {path} is outside the project root {root}")
    revision = _object(json.loads(path.read_text(encoding="utf-8")), "compiler_revision")
    if (set(revision) != {"schema_version", "corpus_manifest", "calibration_coverage"}
            or revision.get("schema_version") != 2):
        raise CompilerError("compiler revision fields differ")
    corpus_relative, corpus_path = _project_path(
        root, revision.get("corpus_manifest"), "compiler_revision.corpus_manifest"
    )
    calibration = revision.get("calibration_coverage")
    if not isinstance(calibration, list) or any(
        not isinstance(item, str) or not item for item in calibration
    ):
        raise CompilerError("compiler revision calibration_coverage must be a list")
    directory = root / TARGETS_DIRECTORY
    target_paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
    if not target_paths:
        raise CompilerError(f"{TARGETS_DIRECTORY} declares no Target")
    targets: dict[str, Target] = {}
    for target_path in target_paths:
        if target_path.is_symlink() or not target_path.is_file():
            raise CompilerError(f"target definition {target_path.name!r} custody differs")
        targets[target_path.stem] = _load_target(target_path)
    json.loads(corpus_path.read_text(encoding="utf-8"))
    commit = checkout_commit_or_none(root)
    if commit is not None:
        # The commit is the identity, so it must cover every document read above. A
        # clean tree can still hold a file that `git status` does not show (an excludes
        # entry); naming it here is the refusal that used to be an implicit
        # `open-cake-ir@uncommitted`.
        untracked = untracked_paths(root, (
            path.relative_to(root).as_posix(),
            corpus_relative,
            *(target_path.relative_to(root).as_posix() for target_path in target_paths),
        ))
        if untracked:
            raise CompilerError(
                f"compiler revision at commit {commit} reads documents the commit does not "
                f"track: {', '.join(untracked)}"
            )
    return CompilerRevision(
        project_root=root,
        revision_id=f"open-cake-ir@{commit or 'uncommitted'}",
        commit=commit,
        targets=MappingProxyType(targets),
        corpus_path=corpus_path,
        calibration_coverage=frozenset(cast(list[str], calibration)),
    )
