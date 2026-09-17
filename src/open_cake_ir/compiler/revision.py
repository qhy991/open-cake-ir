"""The Compiler manifest, its declared Targets and the source identity it runs from.

`compiler/revision.json` names the Corpus manifest and the calibration coverage. Every
document under `compiler/targets/` is a declared Target, so adding a target is adding a
document. Code identity is the clean git commit of the checkout (ADR 0065): the Compiler
records it when there is one and still assesses and lowers without it, while the Lab
refuses to run a Campaign without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, cast

from ..serialization import canonical_json_bytes as _canonical_json_bytes
from ..source_identity import checkout_commit_or_none
from .errors import CompilerError
from .ir import ScheduleParseError
from .target import Target, TargetParseError, TargetSource

TARGETS_DIRECTORY = "compiler/targets"

_TARGET_REQUIRED_FIELDS = frozenset({
    "schema_version",
    "target_id",
    "architecture",
    "device_names",
    "memory_spaces",
    "operation_kinds",
    # The role-slot width is required, not optional: a Target without one would leave a
    # fact for shared code to invent.
    "warp_size",
    # Vendor is required for the same reason: shared code used to infer it from the
    # presence of a CUDA field.
    "vendor",
    # And the code object for the same reason again: the Evaluation layer used to hold
    # three frozensets of target ids that no Target document could state.
    "code_object",
    "resource_limits",
    "instruction_contracts",
    "synchronization_contracts",
    "citations",
})
_TARGET_OPTIONAL_FIELDS = frozenset({
    "occupancy", "compute_capability", "peak", "warps_per_warpgroup",
})


@dataclass(frozen=True)
class CompilerRevision:
    """Declared Targets, Corpus and calibration coverage at one source identity."""

    project_root: Path
    revision_id: str
    canonical_sha256: str
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


def _load_target(target_path: Path) -> Target:
    target_id = target_path.stem
    document = _object(
        json.loads(target_path.read_text(encoding="utf-8")),
        f"target_definition.{target_id}",
    )
    if (
        not _TARGET_REQUIRED_FIELDS <= set(document)
        <= _TARGET_REQUIRED_FIELDS | _TARGET_OPTIONAL_FIELDS
        or document.get("schema_version") != 1
    ):
        raise CompilerError(f"target definition {target_id!r} fields differ")
    if document.get("target_id") != target_id:
        raise CompilerError(f"target definition {target_id!r} identity differs from its file name")
    try:
        typed_target = Target.from_dict(document)
    except (TargetParseError, ScheduleParseError) as error:
        raise CompilerError(f"target definition {target_id!r}: {error}") from error
    limits = _object(document.get("resource_limits"), f"target_definition.{target_id}.resource_limits")
    required_limits = {
        "maximum_threads_per_cta",
        "maximum_warps_per_cta",
        "maximum_shared_memory_bytes",
        "maximum_tensor_memory_bytes",
        "grid",
    }
    if not required_limits <= set(limits) <= required_limits | {"maximum_registers_per_thread"}:
        raise CompilerError(f"target definition {target_id!r} resource limits differ")
    citations = _objects(document.get("citations"), f"target_definition.{target_id}.citations")
    if not citations:
        raise CompilerError(f"target definition {target_id!r} requires citations")
    document_bytes = _canonical_json_bytes(document)
    return replace(
        typed_target,
        source=TargetSource(
            canonical_sha256=sha256(document_bytes).hexdigest(),
            document_bytes=document_bytes,
        ),
    )


def load_revision(project_root: str | Path, revision_path: str | Path) -> CompilerRevision:
    """Load the Compiler manifest, every declared Target and the checkout's commit."""

    root = Path(project_root).resolve(strict=True)
    path = Path(revision_path).resolve(strict=True)
    revision = _object(json.loads(path.read_text(encoding="utf-8")), "compiler_revision")
    if (set(revision) != {"schema_version", "corpus_manifest", "calibration_coverage"}
            or revision.get("schema_version") != 2):
        raise CompilerError("compiler revision fields differ")
    _, corpus_path = _project_path(
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
    corpus_document = json.loads(corpus_path.read_text(encoding="utf-8"))
    commit = checkout_commit_or_none(root)
    identity = {
        "commit": commit,
        "revision": revision,
        "targets": {
            target_id: cast(TargetSource, targets[target_id].source).canonical_sha256
            for target_id in sorted(targets)
        },
        "corpus_manifest": sha256(_canonical_json_bytes(corpus_document)).hexdigest(),
    }
    return CompilerRevision(
        project_root=root,
        revision_id=f"open-cake-ir@{commit or 'uncommitted'}",
        canonical_sha256=sha256(_canonical_json_bytes(identity)).hexdigest(),
        commit=commit,
        targets=MappingProxyType(targets),
        corpus_path=corpus_path,
        calibration_coverage=frozenset(cast(list[str], calibration)),
    )
