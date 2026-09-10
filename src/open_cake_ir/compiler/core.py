"""Compiler assessment for canonical Open Cake schedules."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence, cast

from .backends import BACKENDS, Backend
from .frontend import read_schedule
from .passes import FusionResult, fuse_pointwise_epilogue
from .backends.common import EmitError
from .ir import (
    _SCHEDULE_OPTIONAL,
    _SCHEDULE_REQUIRED,
    LoweringBackend,
    LoweringRoute,
    Schedule,
    ScheduleParseError,
)
from .corpus import CorpusCaseReport, CorpusGateReport, check_corpus
from .errors import CompilerError, LoweringRefusedError
from .revision import CompilerRevision, load_revision
from .target import Target
from .performance.ranking import Cost, rank as rank_candidates
from .performance.compiled_resources import CompiledResources
from .performance.empirical_cost import EmpiricalCostModel
from .diagnostics import Finding, FindingCategory, FindingSeverity
from .verifier import name_conflicts, resolve_grid, verify as verify_contracts


@dataclass(frozen=True)
class Assessment:
    """Immutable assessment of one Schedule under one Compiler Revision.

    ``findings`` retains the Corpus Gate's blocking/report observations. ``guidance``
    carries typed non-blocking hints, independently of that existing gate contract.
    """

    compiler_revision_id: str
    compiler_revision_sha256: str
    schedule_id: str
    schedule_sha256: str
    target: str
    route: LoweringRoute | None
    accepted: bool
    lowering_eligible: bool
    findings: tuple[Finding, ...]
    analysis: Mapping[str, object]
    lowering_parameters: Mapping[str, int]
    calibration_available: bool
    schedule_bytes: bytes
    guidance: tuple[Finding, ...] = ()


@dataclass(frozen=True)
class Lowering:
    """Inspectable target source materialized from one eligible Assessment.

    Current lowerings are generated from Schedule operations. The `generated` field
    remains part of the public result so historical materialized-source observations
    retain their distinct meaning.
    """

    compiler_revision_id: str
    compiler_revision_sha256: str
    schedule_id: str
    schedule_sha256: str
    target: str
    route: LoweringRoute
    generated: bool
    source: str
    source_sha256: str
    source_map: Mapping[str, tuple[int, int]]
    toolchain_requirements: Mapping[str, object]


# The IR owns the Schedule's field set. Restating it here meant a new top-level field
# parsed cleanly and was then rejected as an unknown root field by this check.
_REQUIRED_TOP_LEVEL_FIELDS = set(_SCHEDULE_REQUIRED)
_OPTIONAL_TOP_LEVEL_FIELDS = set(_SCHEDULE_OPTIONAL)


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


# A Compiler Mapping historically required lists at these reference boundaries,
# while the typed IR admits general string Sequences. This adapter records only
# that container distinction; IR parsing still owns elements and all structure.
_InputError = tuple[tuple[int, int, int], str]


def _input_container_error(document: Mapping[str, object]) -> _InputError | None:
    """Observe the first legacy container error without preempting IR parsing."""

    operations = document.get("operations")
    if isinstance(operations, list):
        for index, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                continue
            for position, field in enumerate(("reads", "writes", "waits", "signals", "depends_on"), 1):
                if not isinstance(operation.get(field, []), list):
                    return ((1, index, position),
                            f"operations[{index}].{field} must be a list of non-empty strings")
    if not isinstance(document.get("outputs"), list):
        return ((2, 0, 0), "outputs must be a list of non-empty strings")
    route = document.get("lowering")
    loops = document.get("tile_loops", [])
    if isinstance(route, Mapping) and route.get("backend") == LoweringBackend.TRITON.value and isinstance(loops, list):
        for index, loop in enumerate(loops):
            if isinstance(loop, Mapping) and not isinstance(loop.get("body"), list):
                return ((3, index, 0),
                        f"tile_loops[{index}].body must be a list of non-empty strings")
    return None


def _check_input_boundary(schedule: Schedule, error: _InputError | None) -> None:
    """Project typed name facts and legacy container errors in their old order."""

    conflict = next((item for item in name_conflicts(schedule)
                     if item.collection != "tile_loops"), None)
    if conflict is not None:
        position = (1, conflict.index, 0) if conflict.collection == "operations" else (0, 0, 0)
        if error is None or position < error[0]:
            error = (position,
                     f"{conflict.collection}[{conflict.index}].{conflict.field} duplicates {conflict.name!r}")
    if error is not None:
        raise CompilerError(error[1])


def _semantic_schedule_sha256(schedule: Mapping[str, object]) -> str:
    semantic = dict(schedule)
    semantic.pop("schedule_id", None)
    metadata = dict(cast(Mapping[str, object], semantic["metadata"]))
    metadata.pop("legacy_source", None)
    semantic["metadata"] = metadata
    return sha256(_canonical_json_bytes(semantic)).hexdigest()


class Compiler:
    """Assess and lower Schedules independently of the Research Lab."""

    def __init__(
        self,
        *,
        project_root: Path,
        revision_id: str,
        revision_sha256: str,
        state: str,
        target_definitions: Mapping[str, Target],
        corpus_path: Path,
        calibration_coverage: frozenset[str],
    ) -> None:
        self._revision = CompilerRevision(
            project_root=project_root,
            revision_id=revision_id,
            canonical_sha256=revision_sha256,
            state=state,
            targets=MappingProxyType(dict(target_definitions)),
            corpus_path=corpus_path,
            calibration_coverage=calibration_coverage,
        )

    @property
    def state(self) -> str:
        """Return draft or released without exposing mutable manifest state."""

        return self._revision.state

    @classmethod
    def load(cls, project_root: str | Path, revision_path: str | Path) -> "Compiler":
        """Load a draft or released Compiler Revision manifest."""

        revision = load_revision(project_root, revision_path)
        return cls(
            project_root=revision.project_root,
            revision_id=revision.revision_id,
            revision_sha256=revision.canonical_sha256,
            state=revision.state,
            target_definitions=revision.targets,
            corpus_path=revision.corpus_path,
            calibration_coverage=revision.calibration_coverage,
        )

    def check_corpus(self) -> CorpusGateReport:
        """Assess the declared Corpus through its canonical report owner."""

        return check_corpus(self, self._revision.corpus_path)

    def fuse_pointwise_epilogue(self, producer: Mapping[str, object],
                                epilogue: Mapping[str, object], *, private_intermediate: str,
                                schedule_id: str, entry_point: str) -> FusionResult:
        """Explicit composition pass; it is never invoked by assess() or lower()."""
        return fuse_pointwise_epilogue(self, producer, epilogue,
            private_intermediate=private_intermediate, schedule_id=schedule_id, entry_point=entry_point)

    def assess_file(self, path: str | Path) -> Assessment:
        """Assess a JSON or Python Schedule without executing authored Python."""

        return self.assess(_object(read_schedule(path).document, "schedule"))

    def assess(self, schedule: Mapping[str, object]) -> Assessment:
        """Assess one Schedule through its input, typed-verifier and backend owners."""

        missing = _REQUIRED_TOP_LEVEL_FIELDS - schedule.keys()
        extra = schedule.keys() - _REQUIRED_TOP_LEVEL_FIELDS - _OPTIONAL_TOP_LEVEL_FIELDS
        if missing or extra or schedule.get("schema_version") != 1:
            raise CompilerError("schedule root fields or schema_version differ")
        if ("grid" in schedule) == ("program_map" in schedule):
            raise CompilerError("schedule must define exactly one of grid or program_map")

        input_error = _input_container_error(schedule)
        try:
            typed_schedule = Schedule.from_dict(schedule)
        except ScheduleParseError as error:
            return self._structural_rejection(schedule, error)
        _check_input_boundary(typed_schedule, input_error)
        schedule = typed_schedule.canonical_document(schedule)

        target = typed_schedule.target
        target_definition = self._revision.targets.get(target)
        findings: list[Finding] = []
        if target_definition is None:
            findings.append(Finding(
                "TARGET_UNSUPPORTED", "target",
                f"target {target!r} is not defined by Compiler Revision {self._revision.revision_id}",
                FindingCategory.HARDWARE_CONFORMANCE,
            ))

        route = typed_schedule.lowering
        backend = BACKENDS.get(route.backend)
        semantic_sha256 = _semantic_schedule_sha256(schedule)
        if backend is not None:
            findings.extend(backend.module.requirements(typed_schedule))

        if target_definition is not None:
            findings.extend(verify_contracts(typed_schedule, target_definition))
        if (backend is not None and target_definition is not None
                and not any(finding.blocks_lowering for finding in findings)):
            findings.extend(backend.module.preflight(typed_schedule, target_definition))

        accepted = not any(finding.blocks_acceptance for finding in findings)
        lowering_eligible = accepted and not any(finding.blocks_lowering for finding in findings)
        analysis = MappingProxyType({
            "grid": resolve_grid(typed_schedule),
            "operation_counts": dict(sorted(Counter(
                operation.kind.value for operation in typed_schedule.operations
            ).items())),
            "role_count": len(typed_schedule.roles),
            "total_warps": len({warp for role in typed_schedule.roles for warp in role.warps}),
            "semantic_sha256": semantic_sha256,
        })
        return Assessment(
            compiler_revision_id=self._revision.revision_id,
            compiler_revision_sha256=self._revision.canonical_sha256,
            schedule_id=typed_schedule.schedule_id,
            schedule_sha256=sha256(_canonical_json_bytes(schedule)).hexdigest(),
            target=target,
            route=route,
            accepted=accepted,
            lowering_eligible=lowering_eligible,
            findings=tuple(finding for finding in findings if finding.severity is not FindingSeverity.HINT),
            analysis=analysis,
            lowering_parameters=MappingProxyType({}),
            calibration_available=semantic_sha256 in self._revision.calibration_coverage,
            schedule_bytes=_canonical_json_bytes(schedule),
            guidance=tuple(finding for finding in findings if finding.severity is FindingSeverity.HINT),
        )

    def _structural_rejection(
        self, schedule: Mapping[str, object], error: ScheduleParseError
    ) -> Assessment:
        """One Assessment carrying the localized reason a Schedule is not well formed."""

        message = str(error)
        path, _, detail = message.partition(" ")
        route = None
        try:
            route = LoweringRoute.from_dict(schedule.get("lowering"))
        except ScheduleParseError:
            pass
        schedule_id = schedule.get("schedule_id")
        target = schedule.get("target")
        return Assessment(
            compiler_revision_id=self._revision.revision_id,
            compiler_revision_sha256=self._revision.canonical_sha256,
            schedule_id=schedule_id if isinstance(schedule_id, str) else "",
            schedule_sha256=sha256(_canonical_json_bytes(schedule)).hexdigest(),
            target=target if isinstance(target, str) else "",
            route=route,
            accepted=False,
            lowering_eligible=False,
            findings=(
                Finding("SCHEDULE_STRUCTURE", path, detail.strip() or message,
                        FindingCategory.SCHEDULE_SEMANTICS),
            ),
            analysis=MappingProxyType({}),
            lowering_parameters=MappingProxyType({}),
            calibration_available=False,
            schedule_bytes=_canonical_json_bytes(schedule),
        )

    def _emit(self, assessment: Assessment, backend: Backend) -> Lowering:
        """Generate the target source from the Schedule."""

        definition = self._revision.targets.get(assessment.target)
        if definition is None:
            raise CompilerError(f"Target {assessment.target!r} is not bound by this Revision")
        schedule = Schedule.from_dict(
            _object(json.loads(assessment.schedule_bytes), "assessment.schedule")
        )
        route = assessment.route
        if route is None:
            raise CompilerError("assessment has no lowering route")
        try:
            emission = backend.module.emit(
                schedule,
                definition,
                entry_point=route.entry_point,
            )
        except EmitError as error:
            raise LoweringRefusedError(f"Schedule does not determine its source: {error}") from error
        source = emission.source.replace("__SCHEDULE_SHA256__", assessment.schedule_sha256)
        return Lowering(
            compiler_revision_id=self._revision.revision_id,
            compiler_revision_sha256=self._revision.canonical_sha256,
            schedule_id=assessment.schedule_id,
            schedule_sha256=assessment.schedule_sha256,
            target=assessment.target,
            route=route,
            generated=True,
            source=source,
            source_sha256=sha256(source.encode("utf-8")).hexdigest(),
            source_map=MappingProxyType(_source_map(source)),
            toolchain_requirements=MappingProxyType(
                {
                    "source_language": backend.source_language,
                    "compiler": backend.compiler,
                    "target": assessment.target,
                    **(emission.toolchain or {}),
                }
            ),
        )


    def profile(self, assessment: Assessment, *, compiled_resources: CompiledResources | None = None,
                cost_model: EmpiricalCostModel | None = None):
        """Profile the canonical assessed program without executing a kernel.

        Reuse lower's full Assessment replay so compiled facts cannot be paired with a
        changed Schedule that happens to retain its display name.
        """
        from .performance.profile import profile_envelope

        lowering = self.lower(assessment)
        schedule = Schedule.from_dict(
            _object(json.loads(assessment.schedule_bytes), "assessment.schedule")
        )
        target = self._revision.targets[assessment.target]
        profile = profile_envelope(
            schedule, target, lowering=lowering, compiled_resources=compiled_resources,
        )
        if cost_model is not None:
            from dataclasses import replace
            profile = replace(profile, empirical_cost=cost_model.estimate(
                json.loads(assessment.schedule_bytes), compiler_revision_id=self._revision.revision_id,
                compiler_revision_sha256=self._revision.canonical_sha256,
                target=assessment.target,
                compiled_compiler_version=compiled_resources.compiler_version if compiled_resources else None,
            ))
        return profile

    def rank(
        self, assessments: Sequence[Assessment]
    ) -> tuple[tuple["Cost", ...], tuple[str, ...]]:
        """Order calibrated eligible candidates before any of them reaches a GPU.

        This is the paper's pre-GPU filter stage, and the boundary it keeps is the point:
        an Assessment that the gates refused is not ranked at all. Ranking a rejected
        candidate would let a good score argue against a hard gate, and the gates are what
        the harness is for. Rejected and unscorable candidates come back named rather than
        dropped, so a caller cannot mistake the order for a complete view of its set.

        The released Revision owns calibration coverage. An eligible candidate from an
        uncovered semantic domain is returned as withheld rather than being assigned precision
        that the Revision does not claim. The order carries no predicted time;
        `compiler/performance/ranking.py` defines the dormant structural primitive and
        `docs/ANALYSIS_CALIBRATION.md` records the measurements required to activate it.
        """

        eligible: list[Schedule] = []
        withheld: list[str] = []
        for assessment in assessments:
            if (
                assessment.compiler_revision_id != self._revision.revision_id
                or assessment.compiler_revision_sha256 != self._revision.canonical_sha256
            ):
                raise CompilerError("assessment belongs to a different Compiler Revision")
            replayed = self.assess(
                _object(json.loads(assessment.schedule_bytes), "assessment.schedule")
            )
            if assessment != replayed:
                raise CompilerError(
                    "assessment fields differ from canonical Schedule replay"
                )
            if not assessment.lowering_eligible:
                withheld.append(assessment.schedule_id)
                continue
            if not assessment.calibration_available:
                withheld.append(assessment.schedule_id)
                continue
            definition = self._revision.targets.get(assessment.target)
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
        target = self._revision.targets[assessments[0].target]
        scored, unscored = rank_candidates(eligible, target)
        return scored, tuple(withheld) + unscored

    def lower(self, assessment: Assessment) -> Lowering:
        """Lower an eligible Assessment to deterministic inspectable target source."""

        if (
            assessment.compiler_revision_id != self._revision.revision_id
            or assessment.compiler_revision_sha256 != self._revision.canonical_sha256
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
        route = assessment.route
        if route is None:
            raise CompilerError("assessment has no lowering route")
        backend = BACKENDS.get(route.backend)
        if backend is not None:
            return self._emit(assessment, backend)
        raise CompilerError(f"lowering route {route!r} is not implemented")


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
