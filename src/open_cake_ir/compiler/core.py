"""Compiler assessment for canonical Open Cake schedules."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Mapping, Sequence, cast

from .emit_cutedsl import EmitError, emit as emit_cutedsl
from .emit_triton import emit as emit_triton
from .ir import (
    _SCHEDULE_OPTIONAL,
    _SCHEDULE_REQUIRED,
    OperationKind,
    Schedule,
    ScheduleParseError,
)
from .target import Target, TargetParseError
from .ranking import Cost, rank as rank_candidates
from .verifier import FindingSeverity, verify as verify_contracts


class CompilerError(ValueError):
    """Raised when compiler authority or Schedule syntax cannot be interpreted."""


@dataclass(frozen=True)
class Finding:
    """One localized compiler observation."""

    code: str
    path: str
    message: str
    blocks_acceptance: bool = True
    blocks_lowering: bool = True


@dataclass(frozen=True)
class Assessment:
    """Immutable assessment of one Schedule under one Compiler Revision."""

    compiler_revision_id: str
    compiler_revision_sha256: str
    schedule_id: str
    schedule_sha256: str
    target: str
    profile: str
    accepted: bool
    lowering_eligible: bool
    findings: tuple[Finding, ...]
    analysis: Mapping[str, object]
    lowering_parameters: Mapping[str, int]
    calibration_available: bool
    schedule_bytes: bytes


@dataclass(frozen=True)
class Lowering:
    """Inspectable target source derived from one eligible Assessment."""

    compiler_revision_id: str
    compiler_revision_sha256: str
    schedule_id: str
    schedule_sha256: str
    target: str
    profile: str
    entry_point: str
    source: str
    source_sha256: str
    source_map: Mapping[str, tuple[int, int]]
    toolchain_requirements: Mapping[str, object]


@dataclass(frozen=True)
class TargetDefinition:
    """Revision-bound exact target capabilities used by assessment/lowering."""

    target_id: str
    canonical_sha256: str
    device_names: tuple[str, ...]
    compute_capability: tuple[int, int]
    memory_spaces: frozenset[str]
    operation_kinds: frozenset[str]
    maximum_threads_per_cta: int
    maximum_warps_per_cta: int
    maximum_shared_memory_bytes: int
    maximum_tensor_memory_bytes: int
    maximum_grid: tuple[int, int, int]
    instruction_contracts: frozenset[str]
    synchronization_contracts: frozenset[str]
    citations: tuple[Mapping[str, object], ...]
    document: Mapping[str, object]
    """The Revision-bound source document, retained so the contract verifier can build
    its own typed Target from the same bytes this definition was parsed from."""


@dataclass(frozen=True)
class CorpusCaseReport:
    """Observed-versus-expected result for one Compiler Corpus case."""

    case_id: str
    schedule_path: str
    expected_accepted: bool
    expected_lowering_eligible: bool
    expected_finding_codes: tuple[str, ...]
    expected_schedule_sha256: str
    expected_lowering_source_sha256: str | None
    observed_accepted: bool
    observed_lowering_eligible: bool
    observed_finding_codes: tuple[str, ...]
    observed_schedule_sha256: str
    observed_lowering_source_sha256: str | None
    matched: bool


@dataclass(frozen=True)
class CorpusGateReport:
    """Release-gate projection for the full declared Compiler Corpus."""

    corpus_id: str
    compiler_revision_id: str
    compiler_revision_sha256: str
    passed: bool
    cases: tuple[CorpusCaseReport, ...]

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def accepted_case_count(self) -> int:
        return sum(case.expected_accepted for case in self.cases)

    @property
    def rejected_case_count(self) -> int:
        return self.case_count - self.accepted_case_count

    @property
    def lowerable_case_count(self) -> int:
        return sum(case.expected_lowering_eligible for case in self.cases)

    @property
    def nonlowerable_case_count(self) -> int:
        return self.case_count - self.lowerable_case_count


# The IR owns the Schedule's field set. Restating it here meant a new top-level field
# parsed cleanly and was then rejected as an unknown root field by this check.
_REQUIRED_TOP_LEVEL_FIELDS = set(_SCHEDULE_REQUIRED)
_OPTIONAL_TOP_LEVEL_FIELDS = set(_SCHEDULE_OPTIONAL)
_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "int32": 4, "int64": 8}
# The IR's enum is what the compiler knows how to parse, so restating the list here
# made a second authority that a new kind had to be added to as well -- and forgetting
# it rejected the Schedule as unsupported rather than saying anything about the gap.
# What a Target admits is a separate question, and stays with the Target.
_SUPPORTED_OPERATION_KINDS = {member.value for member in OperationKind}


def _flash_kmeans_b32_smoke_conformance(buffers, operations) -> list["Finding"]:
    tokens = _shape_of(buffers, "tokens")
    centroids = _shape_of(buffers, "centroids")
    coheres = (
        tokens is not None
        and len(tokens) == 3
        and centroids is not None
        and len(centroids) == 3
        and tokens[0] == centroids[0]
        and tokens[2] == centroids[2] == 128
        and _shape_of(buffers, "centroid_sq") == (tokens[0], centroids[1])
        and _shape_of(buffers, "assignments") == (tokens[0], tokens[1])
    )
    if coheres:
        return []
    return [
        Finding(
            "PROFILE_SHAPE_MISMATCH",
            "buffers.tokens.shape",
            "Flash-KMeans external tensor shapes are inconsistent",
            blocks_acceptance=False,
            blocks_lowering=True,
        )
    ]


def _flash_kmeans_assignment_full_conformance(buffers, operations) -> list["Finding"]:
    distance = buffers.get("distance_scratch")
    shape = distance.get("shape") if distance is not None else None
    if shape == [128, 1024]:
        return []
    return [
        Finding(
            "PROFILE_SHAPE_MISMATCH",
            "buffers.distance_scratch.shape",
            "full Flash-KMeans assignment requires distance_scratch shape [128, 1024]",
            blocks_acceptance=False,
            blocks_lowering=True,
        )
    ]


def _tinygemm2_stage4_split_k_conformance(buffers, operations) -> list["Finding"]:
    reduction = next(
        (operation for operation in operations if operation.get("id") == "reduce_partials"),
        None,
    )
    parameters = (
        _object(reduction.get("parameters"), "operations.reduce_partials.parameters")
        if reduction is not None
        else {}
    )
    # The part count is the extent of the axis the sum collapses, which the read buffer
    # already declares. Reading it there keeps this profile rule and the Schedule from
    # disagreeing about how many partials stage4 combines.
    source = buffers.get(str(reduction.get("reads", [""])[0])) if reduction else None
    shape = source.get("shape") if isinstance(source, Mapping) else None
    axis = parameters.get("axis")
    parts = (
        shape[axis]
        if isinstance(shape, list) and isinstance(axis, int) and 0 <= axis < len(shape)
        else None
    )
    if (
        reduction is not None
        and reduction.get("kind") == "reduce_sum"
        and parts == 4
        and parameters.get("scope") == "cta"
    ):
        return []
    return [
        Finding(
            "REDUCE_SUM_SEMANTICS",
            "operations.reduce_partials.parameters.axis",
            "TinyGEMM2 stage4 requires a four-part CTA reduction",
        )
    ]


def _rmsnorm_b8_smoke_conformance(buffers, operations) -> list["Finding"]:
    x = _shape_of(buffers, "x")
    coheres = (
        x is not None
        and len(x) == 3
        and _shape_of(buffers, "y") == x
        and _shape_of(buffers, "gamma") == (x[2],)
    )
    if coheres:
        return []
    return [
        Finding(
            "PROFILE_SHAPE_MISMATCH",
            "buffers.x.shape",
            "RMSNorm normalizes the last axis, so y matches x and gamma spans it",
            blocks_acceptance=False,
            blocks_lowering=True,
        )
    ]


@dataclass(frozen=True)
class _Profile:
    """One admitted lowering profile and every fact that follows from admitting it.

    These facts used to live in five dictionaries and an `elif` chain keyed by the same
    profile string, so adding an operator meant finding all six and keeping them in step.
    One record owns them, and a profile that omits one is a construction error rather than
    a lookup that silently returns nothing.

    `emitter` generates the source from the Schedule; `asset` is the older path that fills
    a digest into a checked-in template, and only that path needs `closed_semantics` --
    the file matches one Schedule, so a pin is what keeps a second one from reaching it.
    """

    toolchain: Mapping[str, object]
    conformance: "Callable[[Mapping[str, object], Sequence[Mapping[str, object]]], list[Finding]]"
    emitter: object | None = None
    asset: tuple[str, str, str] | None = None
    closed_semantics: str | None = None

    def __post_init__(self) -> None:
        if (self.emitter is None) == (self.asset is None):
            raise CompilerError("a profile lowers through exactly one of emitter or asset")
        if self.asset is None and self.closed_semantics is not None:
            raise CompilerError("an emitted profile pins no closed semantics digest")


_PROFILES: Mapping[str, _Profile] = {
    "flash_kmeans_b32_smoke": _Profile(
        toolchain={
            "source_language": "python",
            "compiler": "triton",
            "entry_point": "cake_flash_kmeans_assign",
            "target": "sm_100a",
            "entry_abi": "four_cuda_tensors_current_stream",
        },
        conformance=_flash_kmeans_b32_smoke_conformance,
        emitter=emit_triton,
    ),
    "flash_kmeans_assignment_full": _Profile(
        toolchain={
            "source_language": "python",
            "compiler": "cutlass_cute_dsl",
            "entry_point": "cake_flash_kmeans_assignment_full",
            "target": "sm_100a",
            "entry_abi": "four_cuda_tensors_current_stream",
        },
        conformance=_flash_kmeans_assignment_full_conformance,
        emitter=emit_cutedsl,
    ),
    # The first operator admitted after the registry became one record. It needed no
    # emitter change, no formula of its own and no entry anywhere else: an operator is
    # now one row plus the Schedules that claim it.
    "rmsnorm_b8_smoke": _Profile(
        toolchain={
            "source_language": "python",
            "compiler": "triton",
            "entry_point": "cake_rmsnorm_b8_smoke",
            "target": "sm_100a",
            "entry_abi": "three_cuda_tensors_current_stream",
        },
        conformance=_rmsnorm_b8_smoke_conformance,
        emitter=emit_triton,
    ),
    "tinygemm2_stage4_split_k": _Profile(
        toolchain={
            "source_language": "cuda_cpp",
            "compiler": "nvcc",
            "target": "sm_100a",
            "entry_abi": "tinygemm2_tensor_map_v1",
        },
        conformance=_tinygemm2_stage4_split_k_conformance,
        asset=(
            "src/open_cake_ir/compiler/assets/tinygemm2_stage4_split_k_sm100.cu.tmpl",
            "@@SCHEDULE_SHA256@@",
            "cake_tinygemm2_stage4_split_k",
        ),
        closed_semantics="70d5c3a06a9aa026a6da275accd79398e8a5e7909422cda6b5ffe091feacfece",
    ),
}


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


def _positive_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CompilerError(f"{path} must be a positive integer")
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


def _load_target_definition(
    root: Path,
    target_id: str,
    value: object,
    context: str,
) -> TargetDefinition:
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
        "compute_capability",
        "memory_spaces",
        "operation_kinds",
        "resource_limits",
        "instruction_contracts",
        "synchronization_contracts",
        "citations",
    }
    optional_fields = {"occupancy"}
    if (
        not expected_fields <= set(document) <= expected_fields | optional_fields
        or document.get("schema_version") != 1
    ):
        raise CompilerError(f"target definition {target_id!r} fields differ")
    if document.get("target_id") != target_id:
        raise CompilerError(f"target definition {target_id!r} identity differs")
    canonical_sha256 = sha256(_canonical_json_bytes(document)).hexdigest()
    if reference.get("canonical_sha256") != canonical_sha256:
        raise CompilerError(f"target definition {target_id!r} bytes differ")
    capability = document.get("compute_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in capability)
    ):
        raise CompilerError(f"target definition {target_id!r} compute capability differs")
    limits = _object(document.get("resource_limits"), f"target_definition.{target_id}.resource_limits")
    if set(limits) != {
        "maximum_threads_per_cta",
        "maximum_warps_per_cta",
        "maximum_shared_memory_bytes",
        "maximum_tensor_memory_bytes",
        "grid",
    }:
        raise CompilerError(f"target definition {target_id!r} resource limits differ")
    grid = _object(limits.get("grid"), f"target_definition.{target_id}.resource_limits.grid")
    if set(grid) != {"x", "y", "z"}:
        raise CompilerError(f"target definition {target_id!r} grid limits differ")
    citations = _objects(document.get("citations"), f"target_definition.{target_id}.citations")
    if not citations:
        raise CompilerError(f"target definition {target_id!r} requires citations")
    return TargetDefinition(
        target_id=target_id,
        canonical_sha256=canonical_sha256,
        device_names=_strings(document.get("device_names"), f"target_definition.{target_id}.device_names"),
        compute_capability=(cast(list[int], capability)[0], cast(list[int], capability)[1]),
        memory_spaces=frozenset(
            _strings(document.get("memory_spaces"), f"target_definition.{target_id}.memory_spaces")
        ),
        operation_kinds=frozenset(
            _strings(document.get("operation_kinds"), f"target_definition.{target_id}.operation_kinds")
        ),
        maximum_threads_per_cta=_positive_int(
            limits.get("maximum_threads_per_cta"),
            f"target_definition.{target_id}.maximum_threads_per_cta",
        ),
        maximum_warps_per_cta=_positive_int(
            limits.get("maximum_warps_per_cta"),
            f"target_definition.{target_id}.maximum_warps_per_cta",
        ),
        maximum_shared_memory_bytes=_positive_int(
            limits.get("maximum_shared_memory_bytes"),
            f"target_definition.{target_id}.maximum_shared_memory_bytes",
        ),
        maximum_tensor_memory_bytes=_positive_int(
            limits.get("maximum_tensor_memory_bytes"),
            f"target_definition.{target_id}.maximum_tensor_memory_bytes",
        ),
        maximum_grid=(
            _positive_int(grid.get("x"), f"target_definition.{target_id}.grid.x"),
            _positive_int(grid.get("y"), f"target_definition.{target_id}.grid.y"),
            _positive_int(grid.get("z"), f"target_definition.{target_id}.grid.z"),
        ),
        instruction_contracts=frozenset(
            _strings(
                document.get("instruction_contracts"),
                f"target_definition.{target_id}.instruction_contracts",
            )
        ),
        synchronization_contracts=frozenset(
            _strings(
                document.get("synchronization_contracts"),
                f"target_definition.{target_id}.synchronization_contracts",
            )
        ),
        citations=tuple(MappingProxyType(dict(item)) for item in citations),
        document=MappingProxyType(dict(document)),
    )


def _named(items: Sequence[Mapping[str, object]], path: str) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for index, item in enumerate(items):
        name = _name(item.get("name"), f"{path}[{index}].name")
        if name in result:
            raise CompilerError(f"{path}[{index}].name duplicates {name!r}")
        result[name] = item
    return result


def _resolve_grid(
    schedule: Mapping[str, object],
    buffers: Mapping[str, Mapping[str, object]],
) -> tuple[int, int, int]:
    grid = schedule.get("grid")
    if grid is not None:
        if not isinstance(grid, list) or len(grid) != 3:
            raise CompilerError("schedule.grid must contain exactly three dimensions")
        return cast(
            tuple[int, int, int],
            tuple(_positive_int(value, f"grid[{index}]") for index, value in enumerate(grid)),
        )

    program_map = _object(schedule.get("program_map"), "program_map")
    axes = _objects(program_map.get("axes"), "program_map.axes")
    resolved = [1, 1, 1]
    seen_axes: set[int] = set()
    for index, axis in enumerate(axes):
        axis_index = axis.get("axis")
        dimension = axis.get("dimension")
        if (
            not isinstance(axis_index, int)
            or isinstance(axis_index, bool)
            or not 0 <= axis_index < 3
            or axis_index in seen_axes
            or not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 0
        ):
            raise CompilerError(f"program_map.axes[{index}] has an invalid axis or dimension")
        buffer_name = _name(axis.get("buffer"), f"program_map.axes[{index}].buffer")
        buffer = buffers.get(buffer_name)
        shape = buffer.get("shape") if buffer is not None else None
        if not isinstance(shape, list) or dimension >= len(shape):
            raise CompilerError(f"program_map.axes[{index}] refers to an invalid buffer dimension")
        extent = _positive_int(shape[dimension], f"buffers.{buffer_name}.shape[{dimension}]")
        tile = _positive_int(axis.get("tile"), f"program_map.axes[{index}].tile")
        resolved[axis_index] = (extent + tile - 1) // tile
        seen_axes.add(axis_index)
    return cast(tuple[int, int, int], tuple(resolved))


def _semantic_schedule_sha256(schedule: Mapping[str, object]) -> str:
    semantic = dict(schedule)
    semantic.pop("schedule_id", None)
    metadata = dict(_object(semantic.get("metadata"), "metadata"))
    metadata.pop("legacy_source", None)
    semantic["metadata"] = metadata
    return sha256(_canonical_json_bytes(semantic)).hexdigest()


def _shape_of(
    buffers: Mapping[str, Mapping[str, object]],
    name: str,
) -> tuple[int, ...] | None:
    buffer = buffers.get(name)
    shape = buffer.get("shape") if buffer is not None else None
    if not isinstance(shape, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape
    ):
        return None
    return tuple(cast(list[int], shape))


class Compiler:
    """Assess and lower Schedules independently of the Research Lab."""

    def __init__(
        self,
        *,
        project_root: Path,
        revision_id: str,
        revision_sha256: str,
        state: str,
        target_definitions: Mapping[str, TargetDefinition],
        corpus_path: Path,
        calibration_coverage: frozenset[str],
    ) -> None:
        self._project_root = project_root
        self._revision_id = revision_id
        self._revision_sha256 = revision_sha256
        self._state = state
        self._target_definitions = MappingProxyType(dict(target_definitions))
        self._corpus_path = corpus_path
        self._calibration_coverage = calibration_coverage

    @property
    def state(self) -> str:
        """Return draft or released without exposing mutable manifest state."""

        return self._state

    @classmethod
    def load(cls, project_root: str | Path, revision_path: str | Path) -> "Compiler":
        """Load a draft or released Compiler Revision manifest."""

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
            _name(target_id, "compiler_revision.target_definitions key"): _load_target_definition(
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
        return cls(
            project_root=root,
            revision_id=revision_id,
            revision_sha256=sha256(_canonical_json_bytes(revision)).hexdigest(),
            state=cast(str, state),
            target_definitions=targets,
            corpus_path=corpus_path,
            calibration_coverage=frozenset(cast(list[str], calibration)),
        )

    def check_corpus(self) -> CorpusGateReport:
        """Assess every declared Corpus case through the public Compiler Interface."""

        manifest = _object(
            json.loads(self._corpus_path.read_text(encoding="utf-8")),
            "corpus",
        )
        if set(manifest) != {"schema_version", "corpus_id", "state", "cases"}:
            raise CompilerError("corpus manifest fields differ")
        if manifest.get("schema_version") != 1 or manifest.get("state") not in {"draft", "released"}:
            raise CompilerError("corpus manifest schema or state differs")
        corpus_id = _name(manifest.get("corpus_id"), "corpus.corpus_id")
        cases = _objects(manifest.get("cases"), "corpus.cases")
        if not cases:
            raise CompilerError("Compiler Corpus must contain at least one case")
        reports: list[CorpusCaseReport] = []
        observed_ids: set[str] = set()
        for index, case in enumerate(cases):
            if set(case) != {"case_id", "schedule", "expected"}:
                raise CompilerError(f"corpus.cases[{index}] fields differ")
            case_id = _name(case.get("case_id"), f"corpus.cases[{index}].case_id")
            if case_id in observed_ids:
                raise CompilerError(f"corpus case {case_id!r} is duplicated")
            observed_ids.add(case_id)
            relative = _name(case.get("schedule"), f"corpus.cases[{index}].schedule")
            schedule_path = (self._project_root / relative).resolve(strict=True)
            if self._project_root not in schedule_path.parents:
                raise CompilerError(f"corpus case {case_id!r} escapes project root")
            expected = _object(case.get("expected"), f"corpus.cases[{index}].expected")
            if set(expected) != {
                "accepted",
                "lowering_eligible",
                "finding_codes",
                "schedule_sha256",
                "lowering_source_sha256",
            }:
                raise CompilerError(f"corpus case {case_id!r} expected fields differ")
            expected_accepted = expected.get("accepted")
            expected_lowering = expected.get("lowering_eligible")
            if not isinstance(expected_accepted, bool) or not isinstance(expected_lowering, bool):
                raise CompilerError(f"corpus case {case_id!r} expected booleans differ")
            expected_codes = _strings(
                expected.get("finding_codes"), f"corpus.cases[{index}].expected.finding_codes"
            )
            expected_schedule_sha = _digest(
                expected.get("schedule_sha256"),
                f"corpus.cases[{index}].expected.schedule_sha256",
            )
            assert expected_schedule_sha is not None
            expected_source_sha = _digest(
                expected.get("lowering_source_sha256"),
                f"corpus.cases[{index}].expected.lowering_source_sha256",
                nullable=True,
            )
            assessment = self.assess_file(schedule_path)
            observed_codes = tuple(finding.code for finding in assessment.findings)
            observed_source_sha = (
                self.lower(assessment).source_sha256 if assessment.lowering_eligible else None
            )
            matched = (
                assessment.accepted is expected_accepted
                and assessment.lowering_eligible is expected_lowering
                and observed_codes == expected_codes
                and assessment.schedule_sha256 == expected_schedule_sha
                and observed_source_sha == expected_source_sha
            )
            reports.append(
                CorpusCaseReport(
                    case_id=case_id,
                    schedule_path=relative,
                    expected_accepted=expected_accepted,
                    expected_lowering_eligible=expected_lowering,
                    expected_finding_codes=expected_codes,
                    expected_schedule_sha256=expected_schedule_sha,
                    expected_lowering_source_sha256=expected_source_sha,
                    observed_accepted=assessment.accepted,
                    observed_lowering_eligible=assessment.lowering_eligible,
                    observed_finding_codes=observed_codes,
                    observed_schedule_sha256=assessment.schedule_sha256,
                    observed_lowering_source_sha256=observed_source_sha,
                    matched=matched,
                )
            )
        return CorpusGateReport(
            corpus_id=corpus_id,
            compiler_revision_id=self._revision_id,
            compiler_revision_sha256=self._revision_sha256,
            passed=all(report.matched for report in reports),
            cases=tuple(reports),
        )

    def assess_file(self, path: str | Path) -> Assessment:
        """Assess one Schedule JSON file without provider, GPU or evidence side effects."""

        schedule_path = Path(path).resolve(strict=True)
        value = json.loads(schedule_path.read_text(encoding="utf-8"))
        return self.assess(_object(value, "schedule"))

    def assess(self, schedule: Mapping[str, object]) -> Assessment:
        """Assess one parsed Schedule document."""

        missing = _REQUIRED_TOP_LEVEL_FIELDS - schedule.keys()
        extra = schedule.keys() - _REQUIRED_TOP_LEVEL_FIELDS - _OPTIONAL_TOP_LEVEL_FIELDS
        if missing or extra or schedule.get("schema_version") != 1:
            raise CompilerError("schedule root fields or schema_version differ")
        if ("grid" in schedule) == ("program_map" in schedule):
            raise CompilerError("schedule must define exactly one of grid or program_map")

        # Structural admissibility has one owner: the typed IR. It is stricter than the
        # checks below -- closed vocabularies are enums and unknown fields are refused --
        # so a document that reaches the rest of this method is known to be well formed.
        #
        # A structural violation is a Finding, not an exception. The agent needs a repair
        # target, and an exception crossing the Authoring Environment becomes a harness
        # fault rather than candidate feedback.
        try:
            typed_schedule = Schedule.from_dict(schedule)
        except ScheduleParseError as error:
            return self._structural_rejection(schedule, error)
        schedule_id = _name(schedule.get("schedule_id"), "schedule.schedule_id")
        target = _name(schedule.get("target"), "schedule.target")
        findings: list[Finding] = []
        target_definition = self._target_definitions.get(target)
        if target_definition is None:
            findings.append(
                Finding(
                    "TARGET_UNSUPPORTED",
                    "target",
                    f"target {target!r} is not defined by Compiler Revision {self._revision_id}",
                )
            )

        roles = _objects(schedule.get("roles"), "roles")
        allocations = _objects(schedule.get("allocations"), "allocations")
        buffers = _objects(schedule.get("buffers"), "buffers")
        pipelines = _objects(schedule.get("pipelines"), "pipelines")
        barriers = _objects(schedule.get("barriers"), "barriers")
        operations = _objects(schedule.get("operations"), "operations")
        role_by_name = _named(roles, "roles")
        allocation_by_name = _named(allocations, "allocations")
        buffer_by_name = _named(buffers, "buffers")
        pipeline_by_name = _named(pipelines, "pipelines")
        barrier_by_name = _named(barriers, "barriers")
        parsed_grid = _resolve_grid(schedule, buffer_by_name)
        if "program_map" in schedule:
            _objects(schedule.get("tile_loops", []), "tile_loops")
            _objects(schedule.get("access_maps", []), "access_maps")

        used_warps = {
            warp for role in typed_schedule.roles for warp in role.warps
        }

        if target_definition is not None:
            for axis, (observed, maximum) in enumerate(zip(parsed_grid, target_definition.maximum_grid)):
                if observed > maximum:
                    findings.append(
                        Finding(
                            "TARGET_GRID_LIMIT",
                            f"grid[{axis}]",
                            f"grid extent {observed} exceeds Target limit {maximum}",
                        )
                    )

        allocation_sizes = {
            name: _positive_int(item.get("size_bytes"), f"allocations.{name}.size_bytes")
            for name, item in allocation_by_name.items()
        }
        allocation_spaces: Counter[str] = Counter()
        for index, allocation in enumerate(allocations):
            space = _name(allocation.get("space"), f"allocations[{index}].space")
            if target_definition is not None and space not in target_definition.memory_spaces:
                findings.append(
                    Finding(
                        "TARGET_MEMORY_SPACE_UNSUPPORTED",
                        f"allocations[{index}].space",
                        f"memory space {space!r} is not supported by Target {target!r}",
                    )
                )
            allocation_spaces[space] += allocation_sizes[_name(allocation.get("name"), f"allocations[{index}].name")]
        if target_definition is not None:
            if allocation_spaces["shared"] > target_definition.maximum_shared_memory_bytes:
                findings.append(
                    Finding(
                        "TARGET_SHARED_MEMORY_LIMIT",
                        "allocations",
                        "Schedule exceeds the Target shared-memory limit",
                    )
                )
            if allocation_spaces["tensor"] > target_definition.maximum_tensor_memory_bytes:
                findings.append(
                    Finding(
                        "TARGET_TENSOR_MEMORY_LIMIT",
                        "allocations",
                        "Schedule exceeds the Target tensor-memory limit",
                    )
                )
        for index, buffer in enumerate(buffers):
            dtype = _name(buffer.get("dtype"), f"buffers[{index}].dtype")
            space = _name(buffer.get("space"), f"buffers[{index}].space")
            if target_definition is not None and space not in target_definition.memory_spaces:
                findings.append(
                    Finding(
                        "TARGET_MEMORY_SPACE_UNSUPPORTED",
                        f"buffers[{index}].space",
                        f"memory space {space!r} is not supported by Target {target!r}",
                    )
                )
            shape = buffer.get("shape")
            if dtype not in _DTYPE_BYTES or not isinstance(shape, list) or not shape:
                raise CompilerError(f"buffers[{index}] has unsupported dtype or shape")
            dimensions = tuple(
                _positive_int(value, f"buffers[{index}].shape[{axis}]")
                for axis, value in enumerate(shape)
            )
            stages = _positive_int(buffer.get("stages", 1), f"buffers[{index}].stages")
            byte_offset = buffer.get("byte_offset", 0)
            if not isinstance(byte_offset, int) or isinstance(byte_offset, bool) or byte_offset < 0:
                raise CompilerError(f"buffers[{index}].byte_offset must be non-negative")
            allocation = buffer.get("allocation")
            if allocation is not None:
                allocation_name = _name(allocation, f"buffers[{index}].allocation")
                if allocation_name not in allocation_sizes:
                    findings.append(
                        Finding(
                            "BUFFER_ALLOCATION_UNKNOWN",
                            f"buffers[{index}].allocation",
                            f"allocation {allocation_name!r} is not declared",
                        )
                    )
                else:
                    elements = 1
                    for dimension in dimensions:
                        elements *= dimension
                    extent = byte_offset + elements * _DTYPE_BYTES[dtype] * stages
                    if extent > allocation_sizes[allocation_name]:
                        findings.append(
                            Finding(
                                "BUFFER_ALLOCATION_OVERFLOW",
                                f"buffers[{index}]",
                                f"buffer extent {extent} exceeds allocation {allocation_name!r}",
                            )
                        )

        operation_ids: set[str] = set()
        operation_counts: Counter[str] = Counter()
        written_buffers: set[str] = set()
        if not operations:
            findings.append(
                Finding(
                    "SCHEDULE_OPERATIONS_EMPTY",
                    "operations",
                    "Schedule must contain at least one operation",
                )
            )
        for index, operation in enumerate(operations):
            op_id = _name(operation.get("id"), f"operations[{index}].id")
            kind = _name(operation.get("kind"), f"operations[{index}].kind")
            role = _name(operation.get("role"), f"operations[{index}].role")
            if op_id in operation_ids:
                raise CompilerError(f"operations[{index}].id duplicates {op_id!r}")
            if kind not in _SUPPORTED_OPERATION_KINDS:
                findings.append(
                    Finding("OPERATION_UNSUPPORTED", f"operations[{index}].kind", f"operation {kind!r} is unsupported")
                )
            elif target_definition is not None and kind not in target_definition.operation_kinds:
                findings.append(
                    Finding(
                        "TARGET_OPERATION_UNSUPPORTED",
                        f"operations[{index}].kind",
                        f"operation {kind!r} is not supported by Target {target!r}",
                    )
                )
            if role not in role_by_name:
                findings.append(
                    Finding("OPERATION_ROLE_UNKNOWN", f"operations[{index}].role", f"role {role!r} is not declared")
                )
            for field in ("reads", "writes"):
                for name in _strings(operation.get(field), f"operations[{index}].{field}"):
                    if name not in buffer_by_name:
                        findings.append(
                            Finding(
                                "OPERATION_BUFFER_UNKNOWN",
                                f"operations[{index}].{field}",
                                f"buffer {name!r} is not declared",
                            )
                        )
                    elif field == "writes":
                        written_buffers.add(name)
            for field in ("waits", "signals"):
                for name in _strings(operation.get(field, []), f"operations[{index}].{field}"):
                    if name not in barrier_by_name:
                        findings.append(
                            Finding(
                                "OPERATION_BARRIER_UNKNOWN",
                                f"operations[{index}].{field}",
                                f"barrier {name!r} is not declared",
                            )
                        )
            for dependency in _strings(
                operation.get("depends_on", []), f"operations[{index}].depends_on"
            ):
                if dependency not in operation_ids:
                    findings.append(
                        Finding(
                            "OPERATION_DEPENDENCY_ORDER",
                            f"operations[{index}].depends_on",
                            f"dependency {dependency!r} is missing or appears later",
                        )
                    )
            pipeline = operation.get("pipeline")
            if pipeline is not None and _name(pipeline, f"operations[{index}].pipeline") not in pipeline_by_name:
                findings.append(
                    Finding(
                        "OPERATION_PIPELINE_UNKNOWN",
                        f"operations[{index}].pipeline",
                        f"pipeline {pipeline!r} is not declared",
                    )
                )
            _object(operation.get("parameters"), f"operations[{index}].parameters")
            operation_ids.add(op_id)
            operation_counts[kind] += 1

        outputs = _strings(schedule.get("outputs"), "outputs")
        for index, output in enumerate(outputs):
            buffer = buffer_by_name.get(output)
            if buffer is None or buffer.get("mode") != "output":
                findings.append(
                    Finding("OUTPUT_INVALID", f"outputs[{index}]", f"output buffer {output!r} is not declared as output")
                )
            elif output not in written_buffers:
                findings.append(
                    Finding(
                        "OUTPUT_UNWRITTEN",
                        f"outputs[{index}]",
                        f"output buffer {output!r} has no writer",
                    )
                )

        metadata = _object(schedule.get("metadata"), "metadata")
        instruction = metadata.get("mma_instruction")
        if (
            instruction is not None
            and target_definition is not None
            and _name(instruction, "metadata.mma_instruction")
            not in target_definition.instruction_contracts
        ):
            findings.append(
                Finding(
                    "TARGET_INSTRUCTION_UNSUPPORTED",
                    "metadata.mma_instruction",
                    f"instruction {instruction!r} is not admitted by Target {target!r}",
                )
            )
        profile = _name(metadata.get("profile"), "metadata.profile")
        lowering_parameters: dict[str, int] = {}
        definition = _PROFILES.get(profile)
        if definition is None:
            findings.append(
                Finding("LOWERING_PROFILE_UNSUPPORTED", "metadata.profile", f"profile {profile!r} is unsupported")
            )
        else:
            findings.extend(definition.conformance(buffer_by_name, operations))
        if definition is not None and definition.closed_semantics is not None:
            semantic_sha = _semantic_schedule_sha256(schedule)
            known_delta = any(
                finding.code in {"PROFILE_SHAPE_MISMATCH", "REDUCE_SUM_SEMANTICS"}
                for finding in findings
            )
            if semantic_sha != definition.closed_semantics and not known_delta:
                findings.append(
                    Finding(
                        "PROFILE_SEMANTICS_MISMATCH",
                        "metadata.profile",
                        f"Schedule semantics are outside the closed lowering subset for {profile!r}",
                        blocks_acceptance=False,
                        blocks_lowering=True,
                    )
                )

        findings.extend(self._contract_findings(typed_schedule, target))

        accepted = not any(finding.blocks_acceptance for finding in findings)
        lowering_eligible = accepted and not any(
            finding.blocks_lowering for finding in findings
        )
        analysis = MappingProxyType(
            {
                "grid": parsed_grid,
                "operation_counts": dict(sorted(operation_counts.items())),
                "role_count": len(roles),
                "total_warps": len(used_warps),
                "semantic_sha256": _semantic_schedule_sha256(schedule),
            }
        )
        return Assessment(
            compiler_revision_id=self._revision_id,
            compiler_revision_sha256=self._revision_sha256,
            schedule_id=schedule_id,
            schedule_sha256=sha256(_canonical_json_bytes(schedule)).hexdigest(),
            target=target,
            profile=profile,
            accepted=accepted,
            lowering_eligible=lowering_eligible,
            findings=tuple(findings),
            analysis=analysis,
            lowering_parameters=MappingProxyType(dict(lowering_parameters)),
            calibration_available=profile in self._calibration_coverage,
            schedule_bytes=_canonical_json_bytes(schedule),
        )

    def _structural_rejection(
        self, schedule: Mapping[str, object], error: ScheduleParseError
    ) -> Assessment:
        """One Assessment carrying the localized reason a Schedule is not well formed."""

        message = str(error)
        path, _, detail = message.partition(" ")
        metadata = schedule.get("metadata")
        profile = ""
        if isinstance(metadata, Mapping) and isinstance(metadata.get("profile"), str):
            profile = cast(str, metadata["profile"])
        schedule_id = schedule.get("schedule_id")
        target = schedule.get("target")
        return Assessment(
            compiler_revision_id=self._revision_id,
            compiler_revision_sha256=self._revision_sha256,
            schedule_id=schedule_id if isinstance(schedule_id, str) else "",
            schedule_sha256=sha256(_canonical_json_bytes(schedule)).hexdigest(),
            target=target if isinstance(target, str) else "",
            profile=profile,
            accepted=False,
            lowering_eligible=False,
            findings=(
                Finding("SCHEDULE_STRUCTURE", path, detail.strip() or message),
            ),
            analysis=MappingProxyType({}),
            lowering_parameters=MappingProxyType({}),
            calibration_available=False,
            schedule_bytes=_canonical_json_bytes(schedule),
        )

    def _emit(self, assessment: Assessment, emitter) -> Lowering:
        """Generate the target source from the Schedule."""

        definition = self._target_definitions.get(assessment.target)
        if definition is None:
            raise CompilerError(f"Target {assessment.target!r} is not bound by this Revision")
        schedule = Schedule.from_dict(
            _object(json.loads(assessment.schedule_bytes), "assessment.schedule")
        )
        try:
            emission = emitter(
                schedule,
                Target.from_dict(dict(definition.document)),
                entry_point=_PROFILES[assessment.profile].toolchain["entry_point"],
            )
        except EmitError as error:
            raise CompilerError(f"Schedule does not determine its source: {error}") from error
        source = emission.source.replace("__SCHEDULE_SHA256__", assessment.schedule_sha256)
        return Lowering(
            compiler_revision_id=self._revision_id,
            compiler_revision_sha256=self._revision_sha256,
            schedule_id=assessment.schedule_id,
            schedule_sha256=assessment.schedule_sha256,
            target=assessment.target,
            profile=assessment.profile,
            entry_point=emission.entry_point,
            source=source,
            source_sha256=sha256(source.encode("utf-8")).hexdigest(),
            source_map=MappingProxyType(_source_map(source)),
            toolchain_requirements=MappingProxyType(
                {
                    **_PROFILES[assessment.profile].toolchain,
                    **(emission.toolchain or {}),
                }
            ),
        )

    def _contract_findings(
        self, schedule: Schedule, target: str
    ) -> list[Finding]:
        """Target-derived contract violations, as localized Findings.

        Blocking violations and reports both join the Assessment; a report carries
        bottleneck attribution and reaches the agent through the Study's feedback
        projection without affecting acceptance. Hints -- a commitment the Schedule
        declined to make, such as an undeclared swizzle -- describe its position rather
        than its behaviour, and stay out.
        """

        definition = self._target_definitions.get(target)
        if definition is None:
            return []
        try:
            typed_target = Target.from_dict(dict(definition.document))
        except TargetParseError:
            return []
        findings: list[Finding] = []
        for item in verify_contracts(schedule, typed_target):
            if item.blocks_lowering:
                findings.append(Finding(item.code, item.path, item.message))
            elif item.severity is FindingSeverity.REPORT:
                findings.append(
                    Finding(
                        item.code,
                        item.path,
                        item.message,
                        blocks_acceptance=False,
                        blocks_lowering=False,
                    )
                )
        return findings

    def rank(
        self, assessments: Sequence[Assessment]
    ) -> tuple[tuple["Cost", ...], tuple[str, ...]]:
        """Order eligible candidates best-first, before any of them reaches a GPU.

        This is the paper's pre-GPU filter stage, and the boundary it keeps is the point:
        an Assessment that the gates refused is not ranked at all. Ranking a rejected
        candidate would let a good score argue against a hard gate, and the gates are what
        the harness is for. Rejected and unscorable candidates come back named rather than
        dropped, so a caller cannot mistake the order for a complete view of its set.

        The order carries no predicted time. `compiler/ranking.py` says why, and
        `docs/ANALYSIS_CALIBRATION.md` records what measurement says about the order that
        is produced: it separates good from bad and does not resolve fine distinctions.
        """

        eligible: list[Schedule] = []
        withheld: list[str] = []
        for assessment in assessments:
            if assessment.compiler_revision_id != self._revision_id:
                raise CompilerError("assessment belongs to a different Compiler Revision")
            if not assessment.lowering_eligible:
                withheld.append(assessment.schedule_id)
                continue
            definition = self._target_definitions.get(assessment.target)
            if definition is None:
                withheld.append(assessment.schedule_id)
                continue
            eligible.append(
                Schedule.from_dict(
                    _object(json.loads(assessment.schedule_bytes), "assessment.schedule")
                )
            )
        if not eligible:
            return (), tuple(withheld)
        target = Target.from_dict(
            dict(self._target_definitions[assessments[0].target].document)
        )
        scored, unscored = rank_candidates(eligible, target)
        return scored, tuple(withheld) + unscored

    def lower(self, assessment: Assessment) -> Lowering:
        """Lower an eligible Assessment to deterministic inspectable target source."""

        if (
            assessment.compiler_revision_id != self._revision_id
            or assessment.compiler_revision_sha256 != self._revision_sha256
        ):
            raise CompilerError("assessment belongs to a different Compiler Revision")
        replayed = self.assess(
            _object(json.loads(assessment.schedule_bytes), "assessment.schedule")
        )
        if assessment != replayed:
            raise CompilerError("assessment fields differ from canonical Schedule replay")
        if not assessment.lowering_eligible:
            codes = ", ".join(finding.code for finding in assessment.findings)
            raise CompilerError(f"assessment is not lowering eligible: {codes}")
        definition = _PROFILES[assessment.profile]
        emitter = definition.emitter
        if emitter is not None:
            return self._emit(assessment, emitter)
        asset = definition.asset
        if asset is None:
            raise CompilerError(f"lowering profile {assessment.profile!r} is not implemented")
        relative_path, placeholder, entry_point = asset
        template_path = self._project_root / relative_path
        template = template_path.read_text(encoding="utf-8")
        if template.count(placeholder) != 1:
            raise CompilerError(f"lowering template for {assessment.profile!r} has an invalid placeholder")
        source = template.replace(placeholder, assessment.schedule_sha256)
        source_map = _source_map(source)
        requirements = dict(definition.toolchain)
        return Lowering(
            compiler_revision_id=self._revision_id,
            compiler_revision_sha256=self._revision_sha256,
            schedule_id=assessment.schedule_id,
            schedule_sha256=assessment.schedule_sha256,
            target=assessment.target,
            profile=assessment.profile,
            entry_point=entry_point,
            source=source,
            source_sha256=sha256(source.encode("utf-8")).hexdigest(),
            source_map=MappingProxyType(source_map),
            toolchain_requirements=MappingProxyType(requirements),
        )


def _source_map(source: str) -> dict[str, tuple[int, int]]:
    lines = source.splitlines()
    kernel_end = next(
        (index for index, line in enumerate(lines, start=1) if "CAKE_KERNEL_END" in line),
        None,
    )
    if kernel_end is None:
        raise CompilerError("lowered source is missing CAKE_KERNEL_END")
    markers: list[tuple[str, int]] = []
    for line_number, line in enumerate(lines, start=1):
        marker = "CAKE_OP:"
        if marker in line:
            markers.append((line.split(marker, 1)[1].strip(), line_number))
    if not markers or len({name for name, _ in markers}) != len(markers):
        raise CompilerError("lowered source operation markers are missing or duplicated")
    result: dict[str, tuple[int, int]] = {}
    for index, (name, start) in enumerate(markers):
        stop = markers[index + 1][1] - 1 if index + 1 < len(markers) else kernel_end
        result[name] = (start, stop)
    return result
