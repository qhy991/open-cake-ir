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

from .endpoints import analysis_without_endpoint_policy
from .efficiency_policy import analysis_without_performance_policy, performance_reporting_policy
from ._documents import _canonical_json_bytes, _digest, _name, _object, differs
from ._policies import (
    untimed,
    _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN,
    _ATTRIBUTION_EVALUATION,
    _LEGACY_ATTRIBUTION_EVALUATION,
    _MATCHED_CLAIM_SCOPES,
    _MATCHED_RALPH_EVENT_VOCABULARY_V1,
    _ONE_RUN_PER_ARM_SCOPES,
    _RALPH_STUDY_FIELDS,
    _SYSTEM_QUALIFICATION_ANALYSIS_PLAN,
    _matched_evidence_policy_version,
    _scientific_analysis_plan_version,
)
from .pairing import comparison_arm, native_backend, matched_run_arms
from .providers import ProviderTurn
from .reference_access import validate_declarations
from .run_controls import validate_run_controls
from .ralph import RalphBudget
from .selection import _EMPIRICAL_SELECTION
from .task_package import TASK_AGENTS_RALPH_V1
from .toolchains import single_environment_backends


def _live_study_kind(kind: object, context: str) -> None:
    if kind != "matched_search":
        state = "retired" if kind == "portfolio" else "unsupported"
        raise ValueError(f"{context} kind {kind!r} is {state}; supported kind: 'matched_search'")


def _analysis_estimand(
    analysis: Mapping[str, object], *, claim_scope: str,
    comparison: str | None, context: str, lock: bool,
) -> str | None:
    """The one rule for which Analysis Plan a claim scope admits, and its estimand.

    The Study carries the plan and the Lock copies it, so both documents are checked
    here rather than once each; `lock` only selects the document's name in the message.
    """
    document = "Campaign Lock " if lock else ""
    if claim_scope == "system_qualification_only":
        if analysis_without_endpoint_policy(analysis) != _SYSTEM_QUALIFICATION_ANALYSIS_PLAN:
            raise differs(
                f"system qualification {document}Analysis Plan",
                expected=_SYSTEM_QUALIFICATION_ANALYSIS_PLAN,
                observed=analysis_without_endpoint_policy(analysis),
            )
        return None
    if claim_scope == "artifact_optimization_only":
        observed = analysis_without_endpoint_policy(
            analysis_without_performance_policy(analysis, claim_scope)
        )
        if observed != _ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN:
            raise differs(
                f"artifact optimization {document}Analysis Plan",
                expected=_ARTIFACT_OPTIMIZATION_ANALYSIS_PLAN, observed=observed,
            )
        return None
    version = _scientific_analysis_plan_version(analysis, f"{context}.analysis_plan")
    policy = native_backend(comparison)
    expected = policy.analysis_version if policy is not None else "two_part_v2"
    if version != expected:
        raise differs(
            "Campaign Lock treatment and analysis arms" if lock
            else "scientific treatment and analysis arm assignment",
            expected=expected, observed=version,
        )
    return _name(analysis.get("estimand"), f"{context}.analysis_plan.estimand")


def _matched_study_shape(document: Mapping[str, object]) -> tuple[str, ...]:
    """Every rule a matched-search Study document satisfies by its own bytes.

    A binding marker (`{"binding": "campaign_lock"}`) and a resolved leaf both pass
    here; byte identity of referenced files, the frozen Compiler and Executor, and
    anything that needs the Workload are preflight's, because they need the checkout.
    Returns the predeclared run order.
    """
    claim_scope = cast(str, document["claim_scope"])
    arms = _object(document.get("arms"), "study.arms")
    comparison = comparison_arm(arms)
    policy = native_backend(comparison)
    single_environment = comparison is None
    matched_run_arms(arms, claim_scope)
    open_cake = _object(arms.get("open_cake"), "study.arms.open_cake")
    comparison_arm_document = (
        _object(arms[comparison], f"study.arms.{comparison}") if comparison is not None else {}
    )
    validate_declarations(arms)
    has_empirical_policy = "candidate_selection" in open_cake
    budget = _object(document.get("budget"), "study.budget")
    if has_empirical_policy and (
        claim_scope != "artifact_optimization_only"
        or open_cake.get("candidate_selection") != {"kind": _EMPIRICAL_SELECTION}
        or "maximum_candidates_per_turn" not in budget
    ):
        raise ValueError("empirical selection requires artifact_optimization_only candidate-set policy and an explicit model")
    open_cake_fields = {
        "environment_kind", "reference_access", "provider", "scaffold", "compiler_revision",
        "lowering_route", "schedule_skeleton", "tool_surface", "feedback",
    }
    if has_empirical_policy:
        open_cake_fields.add("candidate_selection")
    comparison_fields = {
        "environment_kind", "reference_access", "provider", "scaffold", "launch_contract",
        "candidate_skeleton", "toolchain_sha256", "tool_surface", "feedback",
    }
    if single_environment:
        comparison_fields = set()
        open_cake_fields.update({"input_format", "toolchain_sha256"})
        if open_cake.get("input_format") != "schedule_or_python_v1":
            raise ValueError("single-environment optimization requires the Python-enabled authoring contract")
    if policy is not None:
        open_cake_fields.update({"input_format", "toolchain_sha256"})
        comparison_fields -= {"launch_contract", "candidate_skeleton"}
        comparison_fields.add("baseline")
        if (open_cake.get("input_format") != "schedule_or_python_v1"
            or comparison_arm_document.get("baseline") != {"binding": "open_cake_lowering"}
            or open_cake.get("toolchain_sha256") != comparison_arm_document.get("toolchain_sha256")):
            raise differs(
                "same-backend native input, baseline or common toolchain binding",
                expected={"input_format": "schedule_or_python_v1",
                          "baseline": {"binding": "open_cake_lowering"},
                          "toolchain_sha256": open_cake.get("toolchain_sha256")},
                observed={"input_format": open_cake.get("input_format"),
                          "baseline": comparison_arm_document.get("baseline"),
                          "toolchain_sha256": comparison_arm_document.get("toolchain_sha256")},
            )
    if set(open_cake) != open_cake_fields or set(comparison_arm_document) != comparison_fields:
        raise differs(
            "Study Contract Authoring Environment fields differ",
            expected={"open_cake": sorted(open_cake_fields), comparison: sorted(comparison_fields)},
            observed={"open_cake": sorted(open_cake), comparison: sorted(comparison_arm_document)},
        )
    if (open_cake.get("environment_kind") != "open_cake"
            or comparison_arm_document.get("environment_kind") != comparison):
        raise differs(
            "Study Contract Authoring Environment kinds differ",
            expected={"open_cake": "open_cake", comparison: comparison},
            observed={"open_cake": open_cake.get("environment_kind"),
                      comparison: comparison_arm_document.get("environment_kind")},
        )
    route = open_cake.get("lowering_route")
    # The one arm of an artifact-optimization Study lowers through a backend whose
    # toolchain row admits it alone; a two-arm Study lowers through the comparison arm's.
    admitted_routes = (single_environment_backends() if single_environment
                       else {policy.backend if policy is not None else "triton"})
    if (not isinstance(route, Mapping) or set(route) != {"backend", "entry_point"}
        or route.get("backend") not in admitted_routes or not isinstance(route.get("entry_point"), str)
        or not route["entry_point"].isidentifier()):
        raise differs(
            "Study Contract Open Cake lowering route",
            expected={"backend": sorted(admitted_routes), "entry_point": "<identifier>"},
            observed=route,
        )
    for owner, field, keys in (
        ("open_cake", "schedule_skeleton", {"path", "canonical_sha256"}),
        ("open_cake", "scaffold", {"path", "sha256"}),
        *(
            (("direct_cuda", "launch_contract", {"path", "sha256"}),
             ("direct_cuda", "candidate_skeleton", {"path", "sha256"}))
            if comparison == "direct_cuda" else ()
        ),
    ):
        reference = _object(arms[owner].get(field), f"study.arms.{owner}.{field}")
        if set(reference) != keys:
            noun = {"schedule_skeleton": "Schedule skeleton reference", "scaffold": "scaffold reference",
                    "launch_contract": "direct launch contract reference",
                    "candidate_skeleton": "direct candidate skeleton reference"}[field]
            raise differs(f"Study Contract {noun}", expected=sorted(keys), observed=sorted(reference))
    if comparison is not None and (
            open_cake.get("provider") != comparison_arm_document.get("provider")
            or open_cake.get("scaffold") != comparison_arm_document.get("scaffold")):
        raise differs(
            "matched Authoring Environments differ in provider or scaffold",
            expected={"provider": comparison_arm_document.get("provider"),
                      "scaffold": comparison_arm_document.get("scaffold")},
            observed={"provider": open_cake.get("provider"), "scaffold": open_cake.get("scaffold")},
        )
    expected_open_cake_tools = (["submit_schedule_or_python"] if policy is not None or single_environment
                                else ["submit_schedule"])
    expected_comparison_tools = (None if comparison is None
                                 else [policy.submit_tool] if policy is not None else ["submit_cuda"])
    if (open_cake.get("tool_surface") != expected_open_cake_tools
            or (comparison is not None
                and comparison_arm_document.get("tool_surface") != expected_comparison_tools)):
        raise differs(
            "Study Contract Authoring Environment tool surfaces differ",
            expected={"open_cake": expected_open_cake_tools, comparison: expected_comparison_tools},
            observed={"open_cake": open_cake.get("tool_surface"),
                      comparison: comparison_arm_document.get("tool_surface")},
        )
    evaluation = _object(document.get("evaluation_protocol"), "study.evaluation_protocol")
    attribution_evaluation = evaluation.get("attribution_evaluation")
    # A Study for a target whose backend names no timing source carries a
    # measurement-coverage limitation instead of a paired assay. `evaluation_policy` has
    # written that shape since the DCU was admitted, and this gate never accepted it, so
    # no such Study reached a Campaign: its attribution is `correctness_only`, which was
    # not in the set below, and its arms can be given neither a qualified latency nor a
    # profile. Both halves are admitted here, from the one predicate the policy uses.
    # Whether the Study's coverage claim is true of the device is not checked here: the
    # registry that owns which backend admits a target and what source it may name lives
    # in the task layer, and `lab` does not import it (tests/contracts/test_task_boundaries.py).
    # `TaskLab.preflight` checks the claim against that owner before delegating here.
    no_timed_assay = untimed(evaluation)
    validate_run_controls(document)
    if no_timed_assay:
        timed_feedback, profile_feedback = [], []
    else:
        timed_feedback = ["qualified_timing"]
        profile_feedback = ["profile"] if attribution_evaluation is not None else []
    expected_open_cake_feedback = ["findings", "correctness", *timed_feedback, *profile_feedback]
    expected_comparison_feedback = (
        None if comparison is None else ["compile", "correctness", *timed_feedback, *profile_feedback]
    )
    if open_cake.get("feedback") != expected_open_cake_feedback or (
            comparison is not None
            and comparison_arm_document.get("feedback") != expected_comparison_feedback):
        raise differs(
            "Study Contract Authoring Environment feedback",
            expected={"open_cake": expected_open_cake_feedback, comparison: expected_comparison_feedback},
            observed={"open_cake": open_cake.get("feedback"),
                      comparison: comparison_arm_document.get("feedback")},
        )
    # Allocation order, finite budgets and run protocol.
    allocation = _object(document.get("allocation"), "study.allocation")
    order = allocation.get("order")
    if allocation.get("method") != "predeclared_balanced_blocks" or not isinstance(order, list):
        raise differs(
            "Study Contract allocation",
            expected={"method": "predeclared_balanced_blocks", "order": "<list>"},
            observed={"method": allocation.get("method"), "order": order},
        )
    run_order = tuple(_name(value, "study.allocation.order[]") for value in order)
    expected_arms = matched_run_arms(arms, claim_scope)
    if len(run_order) != len(set(run_order)) or sorted(
        name.rsplit("-", 1)[0] for name in run_order
    ) != expected_arms:
        required = "one" if claim_scope in _ONE_RUN_PER_ARM_SCOPES else "three"
        raise ValueError(
            f"Study Contract must predeclare {required} independent Run(s) per arm: "
            f"expected arms {expected_arms!r}, observed order {list(run_order)!r}"
        )
    _analysis_estimand(
        _object(document.get("analysis_plan"), "study.analysis_plan"),
        claim_scope=claim_scope, comparison=comparison,
        context="study", lock=False,
    )
    evidence_version = _matched_evidence_policy_version(
        _object(document.get("evidence"), "study.evidence"), "study.evidence"
    )
    if evidence_version != _MATCHED_RALPH_EVENT_VOCABULARY_V1:
        raise differs(
            "Study agent interface and Evidence policy",
            expected=_MATCHED_RALPH_EVENT_VOCABULARY_V1, observed=evidence_version,
        )
    return run_order


@dataclass(frozen=True)
class StudyContract:
    """Matched-search execution and data-use authority.

    `load` is the one place a Study document's shape is decided. The typed projections
    below read the document, so `dataclasses.replace(study, document=...)` with resolved
    binding leaves keeps them current.
    """

    document: Mapping[str, object]
    source_path: Path
    study_id: str
    schema_version: int
    state: str
    canonical_sha256: str
    # Predeclared Run order.
    run_order: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return cast(str, self.document["kind"])

    @property
    def claim_scope(self) -> str:
        return cast(str, self.document["claim_scope"])

    @property
    def arms(self) -> Mapping[str, Mapping[str, object]]:
        return cast(Mapping[str, Mapping[str, object]], self.document["arms"])

    @property
    def comparison(self) -> str | None:
        """The declared comparison arm, or None for a single-environment Study."""
        return comparison_arm(self.arms)

    @property
    def budget(self) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.document["budget"])

    @property
    def run_protocol(self) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.document["run_protocol"])

    @property
    def evaluation_protocol(self) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.document["evaluation_protocol"])

    @property
    def execution(self) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.document["execution"])

    @property
    def analysis_plan(self) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.document["analysis_plan"])

    @property
    def evidence_policy(self) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.document["evidence"])

    @classmethod
    def load(cls, path: str | Path) -> "StudyContract":
        """Load the currently supported closed Study Contract variant."""

        source = Path(path).resolve(strict=True)
        document = _object(json.loads(source.read_text(encoding="utf-8")), "study")
        kind = document.get("kind")
        schema_version = document.get("schema_version")
        _live_study_kind(kind, "Study")
        fields = _RALPH_STUDY_FIELDS if schema_version == 2 else set()
        if set(document) != fields:
            raise differs(
                "study root fields or schema_version",
                expected={"matched_search": (2, sorted(_RALPH_STUDY_FIELDS))},
                observed={"kind": kind, "schema_version": schema_version, "fields": sorted(document)},
            )
        state = document.get("state")
        if state not in {"template", "frozen"}:
            raise differs(
                "Study state or kind",
                expected={"state": ["frozen", "template"], "kind": ["matched_search"]},
                observed={"state": state, "kind": kind},
            )
        study_id = _name(document.get("study_id"), "study.study_id")
        if schema_version == 2:
            interface = _object(document.get("agent_interface"), "study.agent_interface")
            if interface != {"schema_version": 1, "kind": TASK_AGENTS_RALPH_V1}:
                raise differs(
                    "Study Ralph agent interface",
                    expected={"schema_version": 1, "kind": TASK_AGENTS_RALPH_V1}, observed=interface,
                )
        claim_scope = _name(document.get("claim_scope"), "study.claim_scope")
        if claim_scope not in _MATCHED_CLAIM_SCOPES:
            raise differs(
                "matched Study Contract claim scope",
                expected=sorted(_MATCHED_CLAIM_SCOPES), observed=claim_scope,
            )
        object_fields = (
            "workload", "arms", "allocation", "budget", "run_protocol",
            "evaluation_protocol", "execution", "analysis_plan", "evidence",
        )
        for field in object_fields:
            _object(document.get(field), f"study.{field}")
        performance_reporting_policy(document["analysis_plan"], claim_scope)
        run_order: tuple[str, ...] = ()
        run_order = _matched_study_shape(document)
        detached = cast(Mapping[str, object], json.loads(_canonical_json_bytes(document)))
        return cls(
            document=detached,
            source_path=source,
            study_id=study_id,
            schema_version=cast(int, schema_version),
            state=cast(str, state),
            canonical_sha256=sha256(_canonical_json_bytes(document)).hexdigest(),
            run_order=run_order,
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
            raise differs(
                "Campaign Lock fields or schema_version differ",
                expected={"schema_version": 1, "fields": sorted(fields)},
                observed={"schema_version": document.get("schema_version"), "fields": sorted(document)},
            )
        study = _object(document.get("study"), "campaign_lock.study")
        _live_study_kind(study.get("kind"), "Campaign Lock Study")
        workload = _object(document.get("workload"), "campaign_lock.workload")
        compiler = _object(
            document.get("compiler_revision"), "campaign_lock.compiler_revision"
        )
        resolved = _object(document.get("resolved_inputs"), "campaign_lock.resolved_inputs")
        if set(study) != {"study_id", "kind", "claim_scope", "canonical_sha256"}:
            raise differs(
                "Campaign Lock study fields differ",
                expected=["canonical_sha256", "claim_scope", "kind", "study_id"], observed=sorted(study),
            )
        if set(workload) != {"workload_id", "path", "canonical_sha256"}:
            raise differs(
                "Campaign Lock workload fields differ",
                expected=["canonical_sha256", "path", "workload_id"], observed=sorted(workload),
            )
        if set(compiler) != {"revision_id", "path"}:
            raise differs(
                "Campaign Lock Compiler Revision fields differ",
                expected=["path", "revision_id"], observed=sorted(compiler),
            )
        for context, digest in (
            ("study", study.get("canonical_sha256")),
            ("workload", workload.get("canonical_sha256")),
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
        analysis = _object(document.get("analysis_plan"), "campaign_lock.analysis_plan")
        performance_reporting_policy(analysis, claim_scope)
        if claim_scope not in _MATCHED_CLAIM_SCOPES:
            raise differs(
                "matched Campaign Lock claim scope",
                expected=sorted(_MATCHED_CLAIM_SCOPES), observed=claim_scope,
            )
        if set(resolved) != {
            "arm_environments", "arm_environment_sha256", "budget",
            "run_protocol", "evidence_policy", "agent_interface",
        }:
            raise ValueError("matched Campaign Lock requires the Ralph interface")
        interface = _object(resolved["agent_interface"], "campaign_lock.agent_interface")
        if interface != {"schema_version": 1, "kind": TASK_AGENTS_RALPH_V1}:
            raise differs(
                "Campaign Lock Ralph agent interface",
                expected={"schema_version": 1, "kind": TASK_AGENTS_RALPH_V1}, observed=interface,
            )
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
        validate_declarations(arms)
        comparison = comparison_arm(arms)
        if set(arm_hashes) != set(arms):
            raise differs(
                "Campaign Lock Authoring Environment set",
                expected=sorted(arms), observed=sorted(arm_hashes),
            )
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
            observed_digest = sha256(_canonical_json_bytes(environment)).hexdigest()
            if digest != observed_digest:
                raise differs(
                    f"Campaign Lock {arm_name} environment bytes differ",
                    expected=digest, observed=observed_digest,
                )
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
                raise differs(
                    "Campaign Lock empirical selection policy",
                    expected={"claim_scope": "artifact_optimization_only",
                              "budget.maximum_candidates_per_turn": "declared",
                              "candidate_selection": {"kind": _EMPIRICAL_SELECTION, "model": "<object>"}},
                    observed={"claim_scope": claim_scope,
                              "budget.maximum_candidates_per_turn": budget.get("maximum_candidates_per_turn"),
                              "candidate_selection": (
                                  {key: (selection[key] if key == "kind" else "<object>")
                                   for key in selection}
                                  if isinstance(selection, Mapping) else selection)},
                )
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
            raise differs(
                "matched Campaign Lock Run allocation",
                expected=expected_arms, observed=list(run_order),
            )
        for field in ("evaluation_protocol", "execution"):
            _object(document.get(field), f"campaign_lock.{field}")
        if comparison is None and paired_protocol(document['evaluation_protocol']) is None:
            raise ValueError("single-environment Campaign requires a fixed-baseline paired assay")
        if paired_protocol(document['evaluation_protocol']) is not None:
            execution = document['execution']
            paired_fields = {'target', 'executor_revision', 'broker_execution_sha256',
                             'gpu', 'sandbox', 'fixed_baseline', 'runtime_config'}
            if set(execution) != paired_fields:
                raise differs(
                    'paired Campaign execution fields differ',
                    expected=sorted(paired_fields), observed=sorted(execution),
                )
            if comparison is not None and native_backend(comparison) is None:
                raise ValueError('paired Campaign requires a same-backend native comparison')
            _digest(execution['broker_execution_sha256'], 'execution.broker_execution_sha256')
            executor = _object(execution['executor_revision'], 'execution.executor_revision')
            if set(executor) != {'executor_id', 'path'}:
                raise differs(
                    'paired Campaign Executor must be resolved: reference fields differ',
                    expected=['executor_id', 'path'], observed=sorted(executor),
                )
            _name(executor['executor_id'], 'execution.executor_revision.executor_id')
            runtime = _object(execution['runtime_config'], 'execution.runtime_config')
            if (set(runtime) != {'path', 'sha256'} or not isinstance(runtime['path'], str)
                or not Path(runtime['path']).is_absolute() or '..' in Path(runtime['path']).parts):
                raise differs(
                    'paired Campaign runtime binding',
                    expected={'fields': ['path', 'sha256'], 'path': '<absolute, no ..>'},
                    observed={'fields': sorted(runtime), 'path': runtime.get('path')},
                )
            _digest(runtime['sha256'], 'runtime_config.sha256')
            for arm in arms.values():
                provider = arm['provider']
                _digest(arm['toolchain_sha256'], 'arm.toolchain_sha256')
                _digest(provider['executable_sha256'], 'provider.executable_sha256')
                _name(provider['revision'], 'provider.revision')
                for field in ('qualification', 'qualification_anchor'):
                    reference = _object(provider[field], f'provider.{field}')
                    if set(reference) != {'path', 'canonical_sha256'} or not Path(str(reference['path'])).is_absolute():
                        raise differs(
                            f'paired Campaign qualification binding provider.{field}',
                            expected={'fields': ['canonical_sha256', 'path'], 'path': '<absolute>'},
                            observed={'fields': sorted(reference), 'path': reference.get('path')},
                        )
                    _digest(reference['canonical_sha256'], f'provider.{field}.canonical_sha256')
            fixed = _object(document['execution'].get('fixed_baseline'), 'execution.fixed_baseline')
            if set(fixed) not in (
                {'bundle_path', 'candidate'},
                {'bundle_path', 'candidate', 'selection'},
            ) or not Path(str(fixed['bundle_path'])).is_absolute():
                raise differs(
                    'paired Campaign fixed baseline binding',
                    expected={'fields': [['bundle_path', 'candidate'], ['bundle_path', 'candidate', 'selection']],
                              'bundle_path': '<absolute>'},
                    observed={'fields': sorted(fixed), 'bundle_path': fixed.get('bundle_path')},
                )
            if 'selection' in fixed:
                from .incumbents import validate_baseline_selection
                validate_baseline_selection(fixed['selection'])
            bound_baseline = candidate_from_identity(fixed['candidate'])
            if bound_baseline.target != execution['target']:
                raise differs(
                    'paired Campaign baseline target',
                    expected=execution['target'], observed=bound_baseline.target,
                )
        analysis_sha = sha256(_canonical_json_bytes(analysis)).hexdigest()
        if document.get("analysis_plan_sha256") != analysis_sha:
            raise differs(
                "Campaign Lock Analysis Plan bytes differ",
                expected=analysis_sha, observed=document.get("analysis_plan_sha256"),
            )
        experimental_unit = _name(
            analysis.get("experimental_unit"), "campaign_lock.analysis_plan.experimental_unit"
        )
        expected_unit = "run"
        if experimental_unit != expected_unit:
            raise differs(
                "Campaign Lock experimental unit differs from Study kind",
                expected=expected_unit, observed=experimental_unit,
            )
        estimand = _analysis_estimand(
            analysis, claim_scope=claim_scope,
            comparison=comparison,
            context="campaign_lock", lock=True,
        )
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

    def run_specification(self, run_id: str):
        """Project the external matched-Study format into the common Run authority."""
        from .run_spec import RunSpecification
        if run_id not in self.run_order:
            raise ValueError('Run was not allocated by this Campaign')
        resolved = self.document['resolved_inputs']
        # The legacy input format encodes assignment in the id; the execution
        # model below carries it explicitly and never decodes a run id.
        condition = run_id.rsplit('-', 1)[0]
        return RunSpecification.from_dict({
            'schema_version': 1, 'run_id': run_id,
            'sequence': self.run_order.index(run_id) + 1,
            'assignment': {'study_id': self.study_id,
                           'study_sha256': self.document['study']['canonical_sha256'],
                           'condition_id': condition},
            'workload': self.document['workload'],
            'compiler_revision': self.document['compiler_revision'],
            'authoring': resolved['arm_environments'][condition],
            'reference_inputs': ({'baseline_schedule': resolved['arm_environments']['open_cake']['schedule_skeleton']}
                                 if resolved['arm_environments'][condition]['environment_kind'] in {'native_triton', 'native_cute_dsl'} else {}),
            'budget': resolved['budget'], 'run_protocol': resolved['run_protocol'],
            'agent_interface': resolved['agent_interface'],
            'evidence_policy': resolved['evidence_policy'],
            'evaluation_protocol': self.document['evaluation_protocol'],
            'execution': self.document['execution'],
            'endpoint_policy': self.analysis_plan.get('endpoint_policy'),
        })

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
    environment_kind: str = "open_cake"



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
