"""Public Lab records and validation of persisted Study and Campaign documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, cast

from open_cake_ir.compiler.performance.empirical_cost import EmpiricalCostModel
from open_cake_ir.evaluation import LaunchableCandidate, LogicalEvaluationAttempt
from open_cake_ir.evaluation.paired import candidate_from_identity, paired_protocol
from open_cake_ir.evidence import RunAudit

from ._documents import _canonical_json_bytes, _digest, _name, _object
from ._policies import (
    _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN,
    _MATCHED_CLAIM_SCOPES,
    _ONE_RUN_PER_ARM_SCOPES,
    _PORTFOLIO_STUDY_FIELDS,
    _RALPH_STUDY_FIELDS,
    _SYSTEM_QUALIFICATION_ANALYSIS_PLAN,
    _matched_evidence_policy_version,
    _scientific_analysis_plan_version,
)
from .pairing import comparison_arm, native_backend, matched_run_arms
from .providers import ProviderTurn
from .ralph import RalphBudget
from .selection import _EMPIRICAL_SELECTION
from .task_package import TASK_AGENTS_RALPH_V1


@dataclass(frozen=True)
class StudyContract:
    """Frozen matched-search or Portfolio execution and data-use authority."""

    document: Mapping[str, object]
    source_path: Path
    study_id: str
    schema_version: int
    state: str
    canonical_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "StudyContract":
        """Load the currently supported closed Study Contract variant."""

        source = Path(path).resolve(strict=True)
        document = _object(json.loads(source.read_text(encoding="utf-8")), "study")
        kind = document.get("kind")
        schema_version = document.get("schema_version")
        fields = (
            _RALPH_STUDY_FIELDS
            if kind == "matched_search" and schema_version == 2
            else _PORTFOLIO_STUDY_FIELDS
            if kind == "portfolio" and schema_version == 1
            else set()
        )
        if set(document) != fields:
            raise ValueError("study root fields or schema_version differ")
        state = document.get("state")
        if state not in {"template", "frozen"} or kind not in {
            "matched_search",
            "portfolio",
        }:
            raise ValueError("Study state or kind differs")
        study_id = _name(document.get("study_id"), "study.study_id")
        if schema_version == 2:
            interface = _object(document.get("agent_interface"), "study.agent_interface")
            if interface != {"schema_version": 1, "kind": TASK_AGENTS_RALPH_V1}:
                raise ValueError("Study Ralph agent interface differs")
        claim_scope = _name(document.get("claim_scope"), "study.claim_scope")
        if kind == "matched_search" and claim_scope not in _MATCHED_CLAIM_SCOPES:
            raise ValueError("matched Study Contract claim scope differs")
        object_fields = (
            (
                "workload",
                "arms",
                "allocation",
                "budget",
                "run_protocol",
                "evaluation_protocol",
                "execution",
                "analysis_plan",
                "evidence",
            )
            if kind == "matched_search"
            else (
                "workload",
                "compiler_revision",
                "kernel_seed",
                "case_roles",
                "specialization_policy",
                "dispatch_policy",
                "evaluation_protocol",
                "execution",
                "analysis_plan",
                "evidence",
            )
        )
        for field in object_fields:
            _object(document.get(field), f"study.{field}")
        detached = cast(Mapping[str, object], json.loads(_canonical_json_bytes(document)))
        return cls(
            document=detached,
            source_path=source,
            study_id=study_id,
            schema_version=cast(int, schema_version),
            state=cast(str, state),
            canonical_sha256=sha256(_canonical_json_bytes(document)).hexdigest(),
        )


@dataclass(frozen=True)
class CampaignLock:
    """Resolved immutable closure authorizing one Campaign execution instance."""

    document: Mapping[str, object]
    canonical_sha256: str
    study_id: str
    study_kind: str
    claim_scope: str
    agent_interface: str
    workload_id: str
    compiler_revision_id: str
    run_order: tuple[str, ...]
    experimental_unit: str
    estimand: str | None
    analysis_plan: Mapping[str, object]

    @classmethod
    def from_dict(cls, value: object) -> "CampaignLock":
        """Validate and reconstruct one resolved Campaign Lock."""

        document = _object(value, "campaign_lock")
        fields = {
            "schema_version",
            "study",
            "workload",
            "compiler_revision",
            "resolved_inputs",
            "run_order",
            "evaluation_protocol",
            "execution",
            "analysis_plan",
            "analysis_plan_sha256",
        }
        if set(document) != fields or document.get("schema_version") != 1:
            raise ValueError("Campaign Lock fields or schema_version differ")
        study = _object(document.get("study"), "campaign_lock.study")
        workload = _object(document.get("workload"), "campaign_lock.workload")
        compiler = _object(
            document.get("compiler_revision"), "campaign_lock.compiler_revision"
        )
        resolved = _object(document.get("resolved_inputs"), "campaign_lock.resolved_inputs")
        if set(study) != {"study_id", "kind", "claim_scope", "canonical_sha256"}:
            raise ValueError("Campaign Lock study fields differ")
        if set(workload) != {"workload_id", "path", "canonical_sha256"}:
            raise ValueError("Campaign Lock workload fields differ")
        if set(compiler) != {"revision_id", "path", "canonical_sha256"}:
            raise ValueError("Campaign Lock Compiler Revision fields differ")
        for context, digest in (
            ("study", study.get("canonical_sha256")),
            ("workload", workload.get("canonical_sha256")),
            ("compiler", compiler.get("canonical_sha256")),
            ("analysis_plan", document.get("analysis_plan_sha256")),
        ):
            _digest(digest, f"campaign_lock.{context}.sha256")
        run_order_value = document.get("run_order")
        if not isinstance(run_order_value, list) or not run_order_value:
            raise ValueError("Campaign Lock run_order must be non-empty")
        run_order = tuple(_name(item, "campaign_lock.run_order[]") for item in run_order_value)
        if len(run_order) != len(set(run_order)):
            raise ValueError("Campaign Lock run_order contains duplicates")
        study_kind = _name(study.get("kind"), "campaign_lock.study.kind")
        claim_scope = _name(study.get("claim_scope"), "campaign_lock.study.claim_scope")
        if study_kind == "matched_search":
            if claim_scope not in _MATCHED_CLAIM_SCOPES:
                raise ValueError("matched Campaign Lock claim scope differs")
            if set(resolved) != {
                "arm_environments", "arm_environment_sha256", "budget",
                "run_protocol", "evidence_policy", "agent_interface",
            }:
                raise ValueError("matched Campaign Lock requires the Ralph interface")
            interface = _object(resolved["agent_interface"], "campaign_lock.agent_interface")
            if interface != {"schema_version": 1, "kind": TASK_AGENTS_RALPH_V1}:
                raise ValueError("Campaign Lock Ralph agent interface differs")
            agent_interface = TASK_AGENTS_RALPH_V1
            RalphBudget.from_mapping(_object(resolved["budget"], "campaign_lock.budget"))
            arms = _object(
                resolved.get("arm_environments"),
                "campaign_lock.resolved_inputs.arm_environments",
            )
            arm_hashes = _object(
                resolved.get("arm_environment_sha256"),
                "campaign_lock.resolved_inputs.arm_environment_sha256",
            )
            comparison = comparison_arm(arms)
            if set(arm_hashes) != set(arms):
                raise ValueError("Campaign Lock Authoring Environment set differs")
            for arm_name in arms:
                environment = _object(
                    arms.get(arm_name),
                    f"campaign_lock.resolved_inputs.arm_environments.{arm_name}",
                )
                if "prompt_template" in environment:
                    raise ValueError("Ralph arms cannot contain prompt_template")
                digest = _digest(
                    arm_hashes.get(arm_name),
                    f"campaign_lock.resolved_inputs.{arm_name}.sha256",
                )
                if digest != sha256(_canonical_json_bytes(environment)).hexdigest():
                    raise ValueError(f"Campaign Lock {arm_name} environment bytes differ")
            budget = _object(
                resolved.get("budget"), "campaign_lock.resolved_inputs.budget"
            )
            selection = arms["open_cake"].get("candidate_selection")
            if comparison is not None and "candidate_selection" in arms[comparison]:
                raise ValueError(f"{comparison} empirical selection is unsupported")
            if "candidate_selection" in arms["open_cake"]:
                if (
                    comparison != "direct_cuda"
                    or "input_format" in arms["open_cake"]
                ):
                    raise ValueError("empirical selection requires the complete-Schedule/direct-CUDA assay")
                if (
                    claim_scope != "artifact_optimization_only"
                    or "maximum_candidates_per_turn" not in budget
                    or not isinstance(selection, Mapping)
                    or set(selection) != {"kind", "model"}
                    or selection.get("kind") != _EMPIRICAL_SELECTION
                ):
                    raise ValueError("Campaign Lock empirical selection policy differs")
                EmpiricalCostModel(selection["model"])
            _object(resolved.get("run_protocol"), "campaign_lock.resolved_inputs.run_protocol")
            evidence_policy = _object(
                resolved.get("evidence_policy"),
                "campaign_lock.resolved_inputs.evidence_policy",
            )
            _matched_evidence_policy_version(
                evidence_policy,
                "campaign_lock.resolved_inputs.evidence_policy",
            )
            expected_arms = matched_run_arms(arms, claim_scope)
            if sorted(name.rsplit("-", 1)[0] for name in run_order) != expected_arms:
                raise ValueError("matched Campaign Lock Run allocation differs")
        elif study_kind == "portfolio":
            agent_interface = "portfolio_v1"
            if claim_scope != "bounded_local_b200_reconstruction":
                raise ValueError("portfolio Campaign Lock claim scope differs")
            if set(resolved) != {
                "kernel_seed",
                "case_roles",
                "specialization_policy",
                "dispatch_policy",
                "evidence_policy",
            }:
                raise ValueError("portfolio Campaign Lock inputs differ")
            seed = _object(resolved.get("kernel_seed"), "campaign_lock.resolved_inputs.kernel_seed")
            if set(seed) != {"seed_id", "path", "canonical_sha256"}:
                raise ValueError("portfolio Kernel Seed reference differs")
            _digest(seed.get("canonical_sha256"), "campaign_lock.kernel_seed.sha256")
            _object(resolved.get("case_roles"), "campaign_lock.resolved_inputs.case_roles")
            _object(
                resolved.get("specialization_policy"),
                "campaign_lock.resolved_inputs.specialization_policy",
            )
            _object(resolved.get("dispatch_policy"), "campaign_lock.resolved_inputs.dispatch_policy")
            _object(
                resolved.get("evidence_policy"),
                "campaign_lock.resolved_inputs.evidence_policy",
            )
        else:
            raise ValueError("Campaign Lock Study kind is unsupported")
        for field in ("evaluation_protocol", "execution"):
            _object(document.get(field), f"campaign_lock.{field}")
        if study_kind == "matched_search" and comparison is None and paired_protocol(document['evaluation_protocol']) is None:
            raise ValueError("single-environment Campaign requires a fixed-baseline paired assay")
        if paired_protocol(document['evaluation_protocol']) is not None:
            execution = document['execution']
            if set(execution) != {'target', 'executor_revision', 'broker_execution_sha256',
                                  'gpu', 'sandbox', 'fixed_baseline', 'runtime_config'}:
                raise ValueError('paired Campaign execution fields differ')
            if study_kind != 'matched_search' or (comparison is not None and native_backend(comparison) is None):
                raise ValueError('paired Campaign requires a same-backend native comparison')
            _digest(execution['broker_execution_sha256'], 'execution.broker_execution_sha256')
            executor = _object(execution['executor_revision'], 'execution.executor_revision')
            if set(executor) != {'executor_id', 'path', 'canonical_sha256'}:
                raise ValueError('paired Campaign Executor must be resolved')
            _digest(executor['canonical_sha256'], 'execution.executor_revision.canonical_sha256')
            runtime = _object(execution['runtime_config'], 'execution.runtime_config')
            if (set(runtime) != {'path', 'sha256'} or not isinstance(runtime['path'], str)
                or not Path(runtime['path']).is_absolute() or '..' in Path(runtime['path']).parts):
                raise ValueError('paired Campaign runtime binding differs')
            _digest(runtime['sha256'], 'runtime_config.sha256')
            for arm in arms.values():
                provider = arm['provider']
                _digest(arm['toolchain_sha256'], 'arm.toolchain_sha256')
                _digest(provider['executable_sha256'], 'provider.executable_sha256')
                _name(provider['revision'], 'provider.revision')
                for field in ('qualification', 'qualification_anchor'):
                    reference = _object(provider[field], f'provider.{field}')
                    if set(reference) != {'path', 'canonical_sha256'} or not Path(str(reference['path'])).is_absolute():
                        raise ValueError('paired Campaign qualification binding differs')
                    _digest(reference['canonical_sha256'], f'provider.{field}.canonical_sha256')
            fixed = _object(document['execution'].get('fixed_baseline'), 'execution.fixed_baseline')
            if set(fixed) != {'bundle_path', 'candidate'} or not Path(str(fixed['bundle_path'])).is_absolute():
                raise ValueError('paired Campaign fixed baseline binding differs')
            bound_baseline = candidate_from_identity(fixed['candidate'])
            if bound_baseline.target != execution['target']:
                raise ValueError('paired Campaign baseline target differs')
        analysis = _object(document.get("analysis_plan"), "campaign_lock.analysis_plan")
        analysis_sha = sha256(_canonical_json_bytes(analysis)).hexdigest()
        if document.get("analysis_plan_sha256") != analysis_sha:
            raise ValueError("Campaign Lock Analysis Plan bytes differ")
        experimental_unit = _name(
            analysis.get("experimental_unit"), "campaign_lock.analysis_plan.experimental_unit"
        )
        expected_unit = "run" if study_kind == "matched_search" else "case_route"
        if experimental_unit != expected_unit:
            raise ValueError("Campaign Lock experimental unit differs from Study kind")
        raw_estimand = analysis.get("estimand")
        if claim_scope == "system_qualification_only":
            if analysis != _SYSTEM_QUALIFICATION_ANALYSIS_PLAN:
                raise ValueError("system qualification Campaign Lock Analysis Plan differs")
            estimand = None
        elif claim_scope == "artifact_optimization_only":
            if analysis != _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN:
                raise ValueError("artifact optimization Campaign Lock Analysis Plan differs")
            estimand = None
        elif study_kind == "matched_search":
            version = _scientific_analysis_plan_version(analysis, "campaign_lock.analysis_plan")
            policy = native_backend(comparison)
            if version != (policy.analysis_version if policy is not None else "two_part_v2"):
                raise ValueError("Campaign Lock treatment and analysis arms differ")
            estimand = _name(raw_estimand, "campaign_lock.analysis_plan.estimand")
        else:
            estimand = _name(raw_estimand, "campaign_lock.analysis_plan.estimand")
        detached = cast(Mapping[str, object], json.loads(_canonical_json_bytes(document)))
        return cls(
            document=detached,
            canonical_sha256=sha256(_canonical_json_bytes(document)).hexdigest(),
            study_id=_name(study.get("study_id"), "campaign_lock.study.study_id"),
            study_kind=study_kind,
            claim_scope=claim_scope,
            agent_interface=agent_interface,
            workload_id=_name(
                workload.get("workload_id"), "campaign_lock.workload.workload_id"
            ),
            compiler_revision_id=_name(
                compiler.get("revision_id"),
                "campaign_lock.compiler_revision.revision_id",
            ),
            run_order=run_order,
            experimental_unit=experimental_unit,
            estimand=estimand,
            analysis_plan=MappingProxyType(dict(analysis)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CampaignLock":
        """Load a Campaign Lock from canonical JSON."""

        source = Path(path).resolve(strict=True)
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CampaignRef:
    """Read-only reference to one Campaign's lock and evidence root."""

    lock: CampaignLock
    evidence_root: Path


@dataclass(frozen=True)
class StudyReport:
    """Policy-bound report over structurally and semantically audited Runs."""

    study_id: str
    claim_scope: str
    system_qualification_passed: bool | None
    estimand: str | None
    campaign_complete: bool
    archive_integrity_passed: bool
    filesystem_custody_verified: bool
    semantic_replay_passed: bool
    estimand_available: bool
    missing_run_count: int
    estimate: Mapping[str, object] | None
    uncertainty: Mapping[str, object] | None
    descriptive: Mapping[str, object]
    run_inclusion: tuple["AnalysisInclusion", ...]
    run_audits: tuple[RunAudit, ...]


@dataclass(frozen=True)
class AnalysisInclusion:
    """Preregistered inclusion decision kept separate from archive/adherence facts."""

    run_id: str
    qualification_endpoint_included: bool
    conditional_performance_included: bool
    reason: str


@dataclass(frozen=True)
class TurnRequest:
    """Lab-owned request for exactly one provider Turn."""

    run_id: str
    arm: str
    turn: int
    cumulative_provider_tokens: int
    thread_id: str | None
    feedback: Mapping[str, object]
    maximum_candidates_per_turn: int
    state_card: Mapping[str, object] | None = None



class RunProvider(Protocol):
    """Provider seam returning raw-normalized Turn evidence, never a Run terminal."""

    provider_revision: str
    qualification_sha256: str
    executable_sha256: str
    configuration: Mapping[str, object]

    def turn(self, request: TurnRequest) -> ProviderTurn:
        """Execute exactly one initial or resumed Turn."""


class RunEvaluator(Protocol):
    """Common Evaluation seam used identically by both treatments."""

    protocol_sha256: str
    protocol: Mapping[str, object]

    def evaluate(
        self,
        candidate: LaunchableCandidate,
        *,
        case_id: str,
        purpose: str,
    ) -> LogicalEvaluationAttempt:
        """Return one logical attempt retaining every bounded broker job."""
