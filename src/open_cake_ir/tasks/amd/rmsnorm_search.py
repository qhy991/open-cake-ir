"""Pure contracts and decisions for the bounded gfx1151 llama RMSNorm search."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, cast

from open_cake_ir.compiler.ir import Schedule

from open_cake_ir.evaluation.timing import (
    PairedTimingObservation,
    PairedTimingProtocol,
    derive_paired_timing,
)
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.workloads import load_workload


TARGET = "gfx1151"
ROUTE_BACKEND = "triton"
ROW_TILES = (1,)
NUM_WARPS = (1, 2, 4, 8)
TEMPLATE_GEOMETRY = (1, 8)
BASELINE_GEOMETRY = (64, 4)

LEAF_TIMING_WIN = "LEAF_TIMING_WIN"
STOP_CLOSE_NULL = "STOP_CLOSE_NULL"
STOP_BASELINE_FASTER = "STOP_BASELINE_FASTER"
INCONCLUSIVE_MEASUREMENT_QUALITY = "INCONCLUSIVE_MEASUREMENT_QUALITY"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "search_id",
    "state",
    "target",
    "route",
    "compiler",
    "executor",
    "template",
    "workload",
    "geometry",
    "screening",
    "noise",
    "confirmatory",
    "attribution",
}
_REFERENCE_FIELDS = {"path", "canonical_sha256"}
_COMPILER_REFERENCE_FIELDS = {*_REFERENCE_FIELDS, "revision_id"}
_EXECUTOR_REFERENCE_FIELDS = {*_REFERENCE_FIELDS, "executor_id"}
_GEOMETRY = {
    "row_tiles": list(ROW_TILES),
    "num_warps": list(NUM_WARPS),
    "candidate_count": len(ROW_TILES) * len(NUM_WARPS),
    "baseline_geometry": {"row_tile": 64, "num_warps": 4},
    "prior_screening_winner": {
        "row_tile": 8,
        "num_warps": 8,
        "disposition": "STOP_CLOSE_NULL",
    },
}
_SCREENING = {
    "case_id": "seeded_random",
    "rounds_per_candidate": 5,
    "warmup_launches": 5,
    "samples_per_round": 10,
    "launches_per_sample": 50,
    "l2_flush_bytes": 268435456,
    "maximum_cv": 0.05,
}
_NOISE = {
    "case_id": "seeded_random",
    "arms": ["baseline_a", "baseline_b"],
    "pair_order": [
        ["baseline_a", "baseline_b"],
        ["baseline_b", "baseline_a"],
        ["baseline_b", "baseline_a"],
        ["baseline_a", "baseline_b"],
    ],
    "samples_per_cohort": 25,
    "warmup_launches_per_cohort": 5,
    "launches_per_sample": 50,
    "route_calls_per_cohort": 1255,
    "maximum_cv": 0.05,
    "materiality_ratio": 1.05,
    "required_pair_wins": 3,
    "required_classification": "close_null",
}
_CONFIRMATORY = {
    "arms": ["candidate", "baseline"],
    "pair_order": [
        ["candidate", "baseline"],
        ["baseline", "candidate"],
        ["baseline", "candidate"],
        ["candidate", "baseline"],
    ],
    "samples_per_cohort": 25,
    "warmup_launches_per_cohort": 5,
    "launches_per_sample": 50,
    "route_calls_per_cohort": 1255,
    "maximum_cv": 0.05,
    "materiality_ratio": 1.05,
    "required_pair_wins": 3,
}
_ATTRIBUTION = {
    "tool_kind": "rocprofv3",
    "trace": "kernel",
    "stats": True,
    "output_formats": ["csv", "json"],
    "arm_order": ["candidate", "baseline"],
    "profile_case_id": "seeded_random",
    "profile_launches": 1,
    "trigger_status": LEAF_TIMING_WIN,
    "timing": "none",
}
_ROW_MATRIX_BUFFERS = frozenset({"x_tile", "sq", "normed", "y_tile"})
_ROW_SCALAR_BUFFERS = frozenset({"sumsq", "meansq", "shifted", "inv_rms"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strict_json(payload: bytes, context: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> object:
        raise ValueError(f"{context} contains non-finite number {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _exact_fields(
    value: object, expected: set[str], context: str
) -> Mapping[str, object]:
    document = _object(value, context)
    if set(document) != expected:
        raise ValueError(f"{context} fields differ")
    return document


def _name(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _sha(value: object, context: str) -> str:
    digest = _name(value, context)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return digest


def _resolve_owned(project_root: Path, value: object, context: str) -> Path:
    relative = Path(_name(value, context))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} must be a project-relative path")
    resolved = (project_root / relative).resolve(strict=True)
    if resolved == project_root or project_root not in resolved.parents:
        raise ValueError(f"{context} escapes the project root")
    if not resolved.is_file():
        raise ValueError(f"{context} must name a file")
    return resolved


def _load_reference(
    project_root: Path,
    value: object,
    *,
    context: str,
    fields: set[str] = _REFERENCE_FIELDS,
) -> tuple[Mapping[str, object], Path, Mapping[str, object], bytes]:
    reference = _exact_fields(value, fields, context)
    path = _resolve_owned(project_root, reference.get("path"), f"{context}.path")
    document = _object(_strict_json(path.read_bytes(), str(path)), str(path))
    canonical = _canonical_json_bytes(document)
    expected = _sha(reference.get("canonical_sha256"), f"{context}.canonical_sha256")
    if sha256(canonical).hexdigest() != expected:
        raise ValueError(f"{context} canonical SHA-256 differs")
    return reference, path, document, canonical


def _row_tile(document: Mapping[str, object]) -> int:
    program_map = _object(document.get("program_map"), "template.program_map")
    axes = program_map.get("axes")
    if not isinstance(axes, list):
        raise ValueError("template.program_map.axes must be a list")
    matches = [
        _object(axis, "template.program_map.axes[]")
        for axis in axes
        if isinstance(axis, Mapping) and axis.get("name") == "row_block"
    ]
    if len(matches) != 1:
        raise ValueError("template must have one row_block program axis")
    tile = matches[0].get("tile")
    if not isinstance(tile, int) or isinstance(tile, bool):
        raise ValueError("template row tile must be an integer")
    return tile


def _num_warps(document: Mapping[str, object]) -> int:
    roles = document.get("roles")
    if not isinstance(roles, list) or len(roles) != 1:
        raise ValueError("template must have one role")
    role = _object(roles[0], "template.roles[0]")
    warps = role.get("warps")
    if not isinstance(warps, list) or warps != list(range(len(warps))):
        raise ValueError("template warps must be contiguous from zero")
    return len(warps)


def _validate_template(
    document: Mapping[str, object], workload_sha256: str
) -> None:
    schedule = Schedule.from_dict(document)
    if (
        schedule.target != TARGET
        or schedule.lowering.backend.value != ROUTE_BACKEND
    ):
        raise ValueError("template Target or lowering route differs")
    if (_row_tile(document), _num_warps(document)) != TEMPLATE_GEOMETRY:
        raise ValueError("template one-row geometry differs")
    metadata = _object(document.get("metadata"), "template.metadata")
    if metadata.get("workload_contract_sha256") != workload_sha256:
        raise ValueError("template is not bound to the frozen Workload")

    buffers = document.get("buffers")
    if not isinstance(buffers, list):
        raise ValueError("template.buffers must be a list")
    shapes = {
        _name(_object(item, "template.buffers[]").get("name"), "buffer.name"):
        _object(item, "template.buffers[]").get("shape")
        for item in buffers
    }
    for name in _ROW_MATRIX_BUFFERS:
        if shapes.get(name) != [1, 128]:
            raise ValueError(f"template buffer {name!r} one-row shape differs")
    for name in _ROW_SCALAR_BUFFERS:
        if shapes.get(name) != [1]:
            raise ValueError(f"template buffer {name!r} one-row shape differs")


@dataclass(frozen=True)
class AmdRmsNormScreeningProtocol:
    """Frozen, non-confirmatory ranking measurements for the complete domain."""

    case_id: str
    rounds_per_candidate: int
    warmup_launches: int
    samples_per_round: int
    launches_per_sample: int
    l2_flush_bytes: int
    maximum_cv: float


@dataclass(frozen=True)
class AmdRmsNormNoiseProtocol:
    """Baseline-to-baseline false-win gate before any candidate timing."""

    case_id: str
    timing: PairedTimingProtocol
    warmup_launches_per_cohort: int
    launches_per_sample: int
    required_classification: str


@dataclass(frozen=True)
class AmdRmsNormConfirmatoryProtocol:
    """Paired decision protocol plus the launch aggregation it does not model."""

    timing: PairedTimingProtocol
    warmup_launches_per_cohort: int
    launches_per_sample: int


@dataclass(frozen=True)
class AmdRmsNormAttributionProtocol:
    """Profiler-only replay performed after the no-profiler decision is fixed."""

    tool_kind: str
    trace: str
    stats: bool
    output_formats: tuple[str, str]
    arm_order: tuple[str, str]
    profile_case_id: str
    profile_launches: int
    trigger_status: str
    timing: str


@dataclass(frozen=True)
class AmdRmsNormCandidate:
    """One immutable materialized Schedule in the closed geometry domain."""

    candidate_id: str
    row_tile: int
    num_warps: int
    schedule_bytes: bytes

    @property
    def canonical_sha256(self) -> str:
        return sha256(self.schedule_bytes).hexdigest()

    @property
    def document(self) -> dict[str, object]:
        return cast(dict[str, object], _strict_json(self.schedule_bytes, self.candidate_id))

    @property
    def schedule(self) -> Schedule:
        return Schedule.from_dict(self.document)


def _paired_protocol(
    document: Mapping[str, object], context: str
) -> tuple[PairedTimingProtocol, int, int]:
    timing = PairedTimingProtocol(
        arms=cast(tuple[str, str], tuple(document["arms"])),
        pair_order=tuple(
            cast(tuple[str, str], tuple(pair))
            for pair in cast(list[list[str]], document["pair_order"])
        ),
        samples_per_cohort=cast(int, document["samples_per_cohort"]),
        route_calls_per_cohort=cast(int, document["route_calls_per_cohort"]),
        maximum_cv=cast(float, document["maximum_cv"]),
        materiality_ratio=cast(float, document["materiality_ratio"]),
        required_pair_wins=cast(int, document["required_pair_wins"]),
    )
    warmups = cast(int, document["warmup_launches_per_cohort"])
    launches_per_sample = cast(int, document["launches_per_sample"])
    expected_route_calls = warmups + timing.samples_per_cohort * launches_per_sample
    if timing.route_calls_per_cohort != expected_route_calls:
        raise ValueError(f"{context} route_calls_per_cohort differs")
    return timing, warmups, launches_per_sample


@dataclass(frozen=True)
class AmdRmsNormSearchContract:
    """Validated authority for the one bounded gfx1151 hardware search."""

    project_root: Path
    path: Path
    canonical_sha256: str
    search_id: str
    compiler_path: Path
    compiler_revision_id: str
    compiler_sha256: str
    executor_path: Path
    executor_id: str
    executor_sha256: str
    template_path: Path
    template_sha256: str
    workload_path: Path
    workload_sha256: str
    screening: AmdRmsNormScreeningProtocol
    noise: AmdRmsNormNoiseProtocol
    confirmatory: AmdRmsNormConfirmatoryProtocol
    attribution: AmdRmsNormAttributionProtocol
    _template_bytes: bytes

    @classmethod
    def load(
        cls, project_root: str | Path, path: str | Path
    ) -> "AmdRmsNormSearchContract":
        root = Path(project_root).resolve(strict=True)
        contract_path = Path(path)
        if not contract_path.is_absolute():
            contract_path = root / contract_path
        contract_path = contract_path.resolve(strict=True)
        if root not in contract_path.parents or not contract_path.is_file():
            raise ValueError("search contract must be a file inside the project root")
        raw = _object(
            _strict_json(contract_path.read_bytes(), str(contract_path)),
            "search contract",
        )
        document = _exact_fields(raw, _TOP_LEVEL_FIELDS, "search contract")
        if document.get("schema_version") != 1:
            raise ValueError("search contract schema_version differs")
        search_id = _name(document.get("search_id"), "search contract.search_id")
        if document.get("state") != "frozen":
            raise ValueError("search contract must be frozen")
        if document.get("target") != TARGET or document.get("route") != {
            "backend": ROUTE_BACKEND
        }:
            raise ValueError("search contract Target or route differs")
        if document.get("geometry") != _GEOMETRY:
            raise ValueError("search contract geometry differs")
        if document.get("screening") != _SCREENING:
            raise ValueError("search contract screening protocol differs")
        if document.get("noise") != _NOISE:
            raise ValueError("search contract baseline noise protocol differs")
        if document.get("confirmatory") != _CONFIRMATORY:
            raise ValueError("search contract confirmatory protocol differs")
        if document.get("attribution") != _ATTRIBUTION:
            raise ValueError("search contract attribution protocol differs")

        compiler_ref, compiler_path, compiler, _ = _load_reference(
            root,
            document.get("compiler"),
            context="search contract.compiler",
            fields=_COMPILER_REFERENCE_FIELDS,
        )
        compiler_id = _name(
            compiler_ref.get("revision_id"), "search contract.compiler.revision_id"
        )
        if compiler.get("revision_id") != compiler_id or compiler.get("state") != "released":
            raise ValueError("search contract Compiler identity or state differs")

        executor_ref, executor_path, executor, _ = _load_reference(
            root,
            document.get("executor"),
            context="search contract.executor",
            fields=_EXECUTOR_REFERENCE_FIELDS,
        )
        executor_id = _name(
            executor_ref.get("executor_id"), "search contract.executor.executor_id"
        )
        if executor.get("executor_id") != executor_id or executor.get("state") != "released":
            raise ValueError("search contract Executor identity or state differs")

        workload_ref, workload_path, _, _ = _load_reference(
            root, document.get("workload"), context="search contract.workload"
        )
        workload = load_workload(workload_path)
        workload_sha256 = _sha(
            workload_ref.get("canonical_sha256"),
            "search contract.workload.canonical_sha256",
        )
        if (
            workload.canonical_sha256 != workload_sha256
            or workload.workload_id != "llama-rmsnorm-mul-fp32-independent-v2"
        ):
            raise ValueError("search contract Workload identity differs")

        template_ref, template_path, template, template_bytes = _load_reference(
            root, document.get("template"), context="search contract.template"
        )
        _validate_template(template, workload_sha256)

        screening = cast(Mapping[str, object], document["screening"])
        noise = cast(Mapping[str, object], document["noise"])
        noise_timing, noise_warmups, noise_launches = _paired_protocol(
            noise, "baseline noise"
        )
        confirmatory = cast(Mapping[str, object], document["confirmatory"])
        timing, warmups, launches_per_sample = _paired_protocol(
            confirmatory, "confirmatory"
        )
        attribution = cast(Mapping[str, object], document["attribution"])
        return cls(
            project_root=root,
            path=contract_path,
            canonical_sha256=sha256(_canonical_json_bytes(document)).hexdigest(),
            search_id=search_id,
            compiler_path=compiler_path,
            compiler_revision_id=compiler_id,
            compiler_sha256=cast(str, compiler_ref["canonical_sha256"]),
            executor_path=executor_path,
            executor_id=executor_id,
            executor_sha256=cast(str, executor_ref["canonical_sha256"]),
            template_path=template_path,
            template_sha256=cast(str, template_ref["canonical_sha256"]),
            workload_path=workload_path,
            workload_sha256=workload_sha256,
            screening=AmdRmsNormScreeningProtocol(
                case_id=cast(str, screening["case_id"]),
                rounds_per_candidate=cast(int, screening["rounds_per_candidate"]),
                warmup_launches=cast(int, screening["warmup_launches"]),
                samples_per_round=cast(int, screening["samples_per_round"]),
                launches_per_sample=cast(int, screening["launches_per_sample"]),
                l2_flush_bytes=cast(int, screening["l2_flush_bytes"]),
                maximum_cv=cast(float, screening["maximum_cv"]),
            ),
            noise=AmdRmsNormNoiseProtocol(
                case_id=cast(str, noise["case_id"]),
                timing=noise_timing,
                warmup_launches_per_cohort=noise_warmups,
                launches_per_sample=noise_launches,
                required_classification=cast(
                    str, noise["required_classification"]
                ),
            ),
            confirmatory=AmdRmsNormConfirmatoryProtocol(
                timing=timing,
                warmup_launches_per_cohort=warmups,
                launches_per_sample=launches_per_sample,
            ),
            attribution=AmdRmsNormAttributionProtocol(
                tool_kind=cast(str, attribution["tool_kind"]),
                trace=cast(str, attribution["trace"]),
                stats=cast(bool, attribution["stats"]),
                output_formats=cast(
                    tuple[str, str], tuple(attribution["output_formats"])
                ),
                arm_order=cast(tuple[str, str], tuple(attribution["arm_order"])),
                profile_case_id=cast(str, attribution["profile_case_id"]),
                profile_launches=cast(int, attribution["profile_launches"]),
                trigger_status=cast(str, attribution["trigger_status"]),
                timing=cast(str, attribution["timing"]),
            ),
            _template_bytes=template_bytes,
        )


@dataclass(frozen=True)
class AmdRmsNormSearchDecision:
    """Terminal leaf-only interpretation of one retained paired measurement."""

    status: str
    observation: PairedTimingObservation


@dataclass(frozen=True)
class AmdRmsNormNoiseDecision:
    """Replayable decision for one baseline-to-baseline measurement."""

    passed: bool
    observation: PairedTimingObservation


@dataclass(frozen=True)
class AmdRmsNormDiagnosis:
    """Typed next owner for one terminal no-profiler RMSNorm decision."""

    route_owner: str
    disposition: str
    reason_code: str
    next_action: str
    complete: bool

    def document(self) -> dict[str, object]:
        return {
            "route_owner": self.route_owner,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
            "next_action": self.next_action,
            "complete": self.complete,
            "cost_model_evaluated": False,
            "cost_model_abstained_reason": "gfx1151_calibration_unavailable",
        }


def candidate_id(row_tile: int, num_warps: int) -> str:
    """Return the stable identity of one member of the closed geometry domain."""

    if (
        type(row_tile) is not int
        or type(num_warps) is not int
        or row_tile not in ROW_TILES
        or num_warps not in NUM_WARPS
    ):
        raise ValueError("candidate geometry is outside the frozen domain")
    return f"r{row_tile}-w{num_warps}"


def materialize_candidates(
    contract: AmdRmsNormSearchContract,
) -> tuple[AmdRmsNormCandidate, ...]:
    """Derive every candidate from fresh template bytes without mutating authority."""

    candidates: list[AmdRmsNormCandidate] = []
    for row_tile in ROW_TILES:
        for num_warps in NUM_WARPS:
            identity = candidate_id(row_tile, num_warps)
            document = cast(
                dict[str, object],
                _strict_json(contract._template_bytes, "search template"),
            )
            document["schedule_id"] = (
                "llama-rmsnorm-mul-b8-n512-d128-gfx1151-"
                f"{identity}-v1"
            )
            roles = cast(list[dict[str, object]], document["roles"])
            roles[0]["warps"] = list(range(num_warps))
            program_map = cast(dict[str, object], document["program_map"])
            for axis in cast(list[dict[str, object]], program_map["axes"]):
                if axis.get("name") == "row_block":
                    axis["tile"] = row_tile
            for buffer in cast(list[dict[str, object]], document["buffers"]):
                name = buffer.get("name")
                if name in _ROW_MATRIX_BUFFERS:
                    buffer["shape"] = [row_tile, 128]
                elif name in _ROW_SCALAR_BUFFERS:
                    buffer["shape"] = [row_tile]
            lowering = cast(dict[str, object], document["lowering"])
            lowering["entry_point"] = (
                f"cake_llama_rmsnorm_mul_gfx1151_r{row_tile}_w{num_warps}"
            )
            payload = _canonical_json_bytes(document)
            Schedule.from_dict(document)
            candidates.append(
                AmdRmsNormCandidate(
                    candidate_id=identity,
                    row_tile=row_tile,
                    num_warps=num_warps,
                    schedule_bytes=payload,
                )
            )
    return tuple(candidates)


def derive_confirmatory_decision(
    contract: AmdRmsNormSearchContract,
    measurements: object,
) -> AmdRmsNormSearchDecision:
    """Recompute the paired observation and map it to the frozen leaf-only status."""

    observation = derive_paired_timing(
        measurements,
        contract.confirmatory.timing,
    )
    statuses = {
        "first_arm_faster": LEAF_TIMING_WIN,
        "second_arm_faster": STOP_BASELINE_FASTER,
        "close_null": STOP_CLOSE_NULL,
        "measurement_quality_failed": INCONCLUSIVE_MEASUREMENT_QUALITY,
    }
    try:
        status = statuses[observation.classification]
    except KeyError as error:
        raise ValueError("paired timing classification is unsupported") from error
    return AmdRmsNormSearchDecision(status=status, observation=observation)


def derive_noise_decision(
    contract: AmdRmsNormSearchContract,
    measurements: object,
) -> AmdRmsNormNoiseDecision:
    """Reject timing when the same baseline produces a material false winner."""

    observation = derive_paired_timing(measurements, contract.noise.timing)
    return AmdRmsNormNoiseDecision(
        passed=(
            observation.measurement_quality_passed
            and observation.classification == contract.noise.required_classification
        ),
        observation=observation,
    )


def diagnose_terminal_decision(
    status: str, *, profiler_evidence_collected: bool
) -> AmdRmsNormDiagnosis:
    """Route retained timing evidence without treating an abstaining model as wrong."""

    if type(profiler_evidence_collected) is not bool:
        raise ValueError("profiler_evidence_collected must be boolean")
    if status == LEAF_TIMING_WIN:
        if profiler_evidence_collected:
            return AmdRmsNormDiagnosis(
                route_owner="candidate",
                disposition="continue",
                reason_code="LEAF_WIN_READY_FOR_AITER_COMPARISON",
                next_action="run_matched_aiter_abba",
                complete=True,
            )
        return AmdRmsNormDiagnosis(
            route_owner="candidate",
            disposition="continue",
            reason_code="LEAF_WIN_PROFILE_REQUIRED",
            next_action="collect_selected_and_baseline_rocprofv3",
            complete=False,
        )
    if status == STOP_CLOSE_NULL:
        return AmdRmsNormDiagnosis(
            route_owner="candidate",
            disposition="stop",
            reason_code="BELOW_MATERIALITY",
            next_action="close_one_row_search",
            complete=True,
        )
    if status == STOP_BASELINE_FASTER:
        return AmdRmsNormDiagnosis(
            route_owner="candidate",
            disposition="stop",
            reason_code="BASELINE_FASTER",
            next_action="close_one_row_search",
            complete=True,
        )
    if status == INCONCLUSIVE_MEASUREMENT_QUALITY:
        return AmdRmsNormDiagnosis(
            route_owner="candidate",
            disposition="inconclusive",
            reason_code="MEASUREMENT_QUALITY_FAILED",
            next_action="inspect_machine_health_without_promoting",
            complete=False,
        )
    raise ValueError("terminal RMSNorm status is unsupported")


def derive_profiled_diagnosis(
    contract: AmdRmsNormSearchContract,
    *,
    no_profiler_status: str,
    arm_receipts: object,
    checked_projections: object,
) -> AmdRmsNormDiagnosis:
    """Admit the profiled diagnosis only from a complete two-arm checked pair."""

    if (
        no_profiler_status != contract.attribution.trigger_status
        or no_profiler_status != LEAF_TIMING_WIN
    ):
        raise ValueError("profiler attribution requires the frozen leaf timing win")
    receipts = _object(arm_receipts, "profile arm receipts")
    projections = _object(checked_projections, "profile checked projections")
    arms = set(contract.attribution.arm_order)
    if set(receipts) != arms or set(projections) != arms:
        raise ValueError("profile attribution requires both frozen arms")
    for arm in contract.attribution.arm_order:
        receipt = _object(receipts[arm], f"profile receipt {arm}")
        correctness = _object(
            receipt.get("profiled_launch_correctness"),
            f"profile receipt {arm}.correctness",
        )
        identity = _object(
            receipt.get("artifact_identity"), f"profile receipt {arm}.identity"
        )
        if (
            receipt.get("kind")
            != "open_cake_gfx1151_rmsnorm_profile_replay_v1"
            or receipt.get("status") != "PROFILE_REPLAY_COMPLETE"
            or receipt.get("arm") != arm
            or receipt.get("search_contract_sha256") != contract.canonical_sha256
            or receipt.get("target_dispatch_count")
            != contract.attribution.profile_launches
            or receipt.get("performance_measured") is not False
            or receipt.get("timing_samples") != 0
            or receipt.get("timing_used_for_decision") is not False
            or receipt.get("performance_decision") is not None
            or receipt.get("promotion_authorized") is not False
            or receipt.get("fallback_calls") != 0
            or correctness.get("passed") is not True
            or correctness.get("inputs_unchanged") is not True
            or correctness.get("fallback_calls") != 0
            or not isinstance(identity.get("kernel_name"), str)
            or not identity["kernel_name"]
        ):
            raise ValueError(f"profile receipt {arm} differs")
        checked = _object(projections[arm], f"profile projection {arm}")
        if checked.get("cross_output_agreement") is not True:
            raise ValueError(f"profile projection {arm} lacks cross-output agreement")
        for name, count_field in (
            ("kernel_trace", "dispatch_count"),
            ("kernel_stats", "calls"),
            ("results_json", "dispatch_count"),
        ):
            projection = _object(
                checked.get(name), f"profile projection {arm}.{name}"
            )
            if (
                projection.get("kernel_name") != identity["kernel_name"]
                or projection.get(count_field)
                != contract.attribution.profile_launches
                or projection.get("duration_used_for_timing_or_promotion")
                is not False
            ):
                raise ValueError(f"profile projection {arm}.{name} differs")
        stats = _object(
            checked.get("kernel_stats"), f"profile projection {arm}.kernel_stats"
        )
        if stats.get("duration_values_projected") is not False:
            raise ValueError(f"profile projection {arm} exposes profiler duration")
    return diagnose_terminal_decision(
        no_profiler_status, profiler_evidence_collected=True
    )


__all__ = [
    "AmdRmsNormCandidate",
    "AmdRmsNormAttributionProtocol",
    "AmdRmsNormConfirmatoryProtocol",
    "AmdRmsNormNoiseDecision",
    "AmdRmsNormNoiseProtocol",
    "AmdRmsNormScreeningProtocol",
    "AmdRmsNormSearchContract",
    "AmdRmsNormSearchDecision",
    "AmdRmsNormDiagnosis",
    "INCONCLUSIVE_MEASUREMENT_QUALITY",
    "LEAF_TIMING_WIN",
    "NUM_WARPS",
    "BASELINE_GEOMETRY",
    "ROUTE_BACKEND",
    "TEMPLATE_GEOMETRY",
    "ROW_TILES",
    "STOP_BASELINE_FASTER",
    "STOP_CLOSE_NULL",
    "TARGET",
    "candidate_id",
    "derive_confirmatory_decision",
    "derive_noise_decision",
    "derive_profiled_diagnosis",
    "diagnose_terminal_decision",
    "materialize_candidates",
]
