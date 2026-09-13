"""Strict admission of draft and released Compiler Revisions.

A Revision binds identities and parsed Targets. Hardware facts remain in Target;
source, Corpus Gate and approval checks retain the released admission boundary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, cast

from .errors import CompilerError
from .ir import ScheduleParseError
from .target import Target, TargetParseError, TargetSource


@dataclass(frozen=True)
class CompilerRevision:
    """Admitted Compiler identity and its exact typed Target bindings."""

    project_root: Path
    revision_id: str
    canonical_sha256: str
    state: str
    targets: Mapping[str, Target]
    corpus_path: Path
    calibration_coverage: frozenset[str]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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


def _load_target(
    root: Path,
    target_id: str,
    value: object,
    context: str,
) -> Target:
    reference = _object(value, context)
    if set(reference) != {"path", "canonical_sha256"}:
        raise CompilerError(f"{context} fields differ")
    _, path = _project_path(root, reference.get("path"), f"{context}.path")
    document = _object(
        json.loads(path.read_text(encoding="utf-8")),
        f"target_definition.{target_id}",
    )
    expected_fields = {
        "schema_version",
        "target_id",
        "architecture",
        "device_names",
        "memory_spaces",
        "operation_kinds",
        "resource_limits",
        "instruction_contracts",
        "synchronization_contracts",
        "citations",
    }
    optional_fields = {"occupancy", "compute_capability", "peak"}
    if (
        not expected_fields <= set(document) <= expected_fields | optional_fields
        or document.get("schema_version") != 1
    ):
        raise CompilerError(f"target definition {target_id!r} fields differ")
    if document.get("target_id") != target_id:
        raise CompilerError(f"target definition {target_id!r} identity differs")
    document_bytes = _canonical_json_bytes(document)
    canonical_sha256 = sha256(document_bytes).hexdigest()
    if reference.get("canonical_sha256") != canonical_sha256:
        raise CompilerError(f"target definition {target_id!r} bytes differ")
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
    return replace(
        typed_target,
        source=TargetSource(canonical_sha256=canonical_sha256, document_bytes=document_bytes),
    )


def load_revision(project_root: str | Path, revision_path: str | Path) -> CompilerRevision:
    """Load a draft or released manifest without relaxing its admission checks."""

    root = Path(project_root).resolve(strict=True)
    path = Path(revision_path).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    revision = _object(value, "compiler_revision")
    draft_fields = {
        "schema_version",
        "revision_id",
        "state",
        "target_definitions",
        "corpus_manifest",
        "calibration_coverage",
    }
    released_fields = {
        "schema_version",
        "revision_id",
        "state",
        "target_definitions",
        "corpus_manifest",
        "calibration_coverage",
        "corpus_gate",
        "release_approval",
        "sources",
    }
    state = revision.get("state")
    expected_fields = draft_fields if state == "draft" else released_fields
    if set(revision) != expected_fields or revision.get("schema_version") != 1:
        raise CompilerError("compiler revision fields differ")
    if state not in {"draft", "released"}:
        raise CompilerError("compiler revision state must be draft or released")
    revision_id = _name(revision.get("revision_id"), "compiler_revision.revision_id")
    target_references = _object(
        revision.get("target_definitions"),
        "compiler_revision.target_definitions",
    )
    if not target_references:
        raise CompilerError("compiler revision must bind at least one Target")
    targets = {
        _name(target_id, "compiler_revision.target_definitions key"): _load_target(
            root,
            target_id,
            reference,
            f"compiler_revision.target_definitions.{target_id}",
        )
        for target_id, reference in target_references.items()
    }
    calibration = revision.get("calibration_coverage")
    if not isinstance(calibration, list) or any(
        not isinstance(item, str) or not item for item in calibration
    ):
        raise CompilerError("compiler revision calibration_coverage must be a list")
    if state == "draft":
        _, corpus_path = _project_path(
            root, revision.get("corpus_manifest"), "compiler_revision.corpus_manifest"
        )
    else:
        corpus = _object(revision.get("corpus_manifest"), "compiler_revision.corpus_manifest")
        if set(corpus) != {"path", "canonical_sha256"}:
            raise CompilerError("released corpus_manifest fields differ")
        _, corpus_path = _project_path(
            root, corpus.get("path"), "compiler_revision.corpus_manifest.path"
        )
        corpus_document = json.loads(corpus_path.read_text(encoding="utf-8"))
        if corpus.get("canonical_sha256") != sha256(
            _canonical_json_bytes(corpus_document)
        ).hexdigest():
            raise CompilerError("released corpus manifest bytes differ")
        sources = revision.get("sources")
        if not isinstance(sources, list) or not sources:
            raise CompilerError("released Compiler Revision sources differ")
        observed_paths: set[str] = set()
        for index, source in enumerate(sources):
            item = _object(source, f"compiler_revision.sources[{index}]")
            if set(item) != {"path", "sha256", "size_bytes"}:
                raise CompilerError(f"compiler_revision.sources[{index}] fields differ")
            relative, source_path = _project_path(
                root, item.get("path"), f"compiler_revision.sources[{index}].path"
            )
            if relative in observed_paths:
                raise CompilerError(f"released compiler source {relative!r} is duplicated")
            observed_paths.add(relative)
            payload = source_path.read_bytes()
            if (
                item.get("sha256") != sha256(payload).hexdigest()
                or item.get("size_bytes") != len(payload)
            ):
                raise CompilerError(f"released compiler source {relative!r} differs")
        gate = _object(revision.get("corpus_gate"), "compiler_revision.corpus_gate")
        if set(gate) != {
            "path",
            "canonical_sha256",
            "case_count",
            "matched_case_count",
        }:
            raise CompilerError("released corpus_gate fields differ")
        _, gate_path = _project_path(
            root, gate.get("path"), "compiler_revision.corpus_gate.path"
        )
        gate_document = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("canonical_sha256") != sha256(
            _canonical_json_bytes(gate_document)
        ).hexdigest() or gate.get("case_count") != gate_document.get(
            "case_count"
        ) or gate.get("matched_case_count") != gate_document.get("matched_case_count"):
            raise CompilerError("released Corpus Gate report bytes differ")
        approval = _object(
            revision.get("release_approval"), "compiler_revision.release_approval"
        )
        if set(approval) != {"path", "canonical_sha256"}:
            raise CompilerError("released approval reference fields differ")
        _, approval_path = _project_path(
            root, approval.get("path"), "compiler_revision.release_approval.path"
        )
        approval_document = _object(
            json.loads(approval_path.read_text(encoding="utf-8")),
            "compiler_revision.release_approval.document",
        )
        approval_gate = _object(
            approval_document.get("gate_report"),
            "compiler_revision.release_approval.gate_report",
        )
        if (
            approval.get("canonical_sha256")
            != sha256(_canonical_json_bytes(approval_document)).hexdigest()
            or approval_document.get("decision") != "approved"
            or approval_gate.get("path") != gate.get("path")
            or approval_gate.get("canonical_sha256") != gate.get("canonical_sha256")
        ):
            raise CompilerError("released Compiler approval bytes differ")
    return CompilerRevision(
        project_root=root,
        revision_id=revision_id,
        canonical_sha256=sha256(_canonical_json_bytes(revision)).hexdigest(),
        state=cast(str, state),
        targets=MappingProxyType(targets),
        corpus_path=corpus_path,
        calibration_coverage=frozenset(cast(list[str], calibration)),
    )
