from __future__ import annotations
import json,math,time,statistics,re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping,Protocol,cast
from open_cake_ir.evidence import EvidenceStore,RunAudit
from open_cake_ir.lab.custody import admit_new_campaign_path
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.tasks.workloads import load_workload
from .seed import KernelSeed
from .portfolio import PortfolioArtifact,PortfolioEvaluationReceipt,replay_portfolio_receipt
from open_cake_ir.lab.core import _ARM_ARTIFACT_ROLES, CampaignLock, CampaignRef, StudyContract, _candidate_artifact_media_type, _canonical_json_bytes, _digest, _object, _project_path, _resolve_compiler_reference, _resolve_executor_reference, _validate_executor_revision

def _portfolio_endpoint(receipt: PortfolioEvaluationReceipt) -> dict[str, object]:
    return {
        "correctness_by_case": dict(receipt.correctness_by_case),
        "kernel_measurement_quality_by_case": dict(
            receipt.kernel_measurement_quality_by_case
        ),
        "dispatcher_measurement_quality_by_case": dict(
            receipt.dispatcher_measurement_quality_by_case
        ),
        "pooled_medians_ms": {
            case_id: dict(values) for case_id, values in receipt.pooled_medians_ms.items()
        },
        "unsupported_rejected_before_launch": receipt.unsupported_rejected_before_launch,
        "route_counts": dict(receipt.route_counts),
        "unsupported_kernel_call_delta": receipt.unsupported_kernel_call_delta,
        "receipt_sha256": receipt.canonical_sha256,
    }


class PortfolioAssay(Protocol):
    """Common Evaluation implementation for the frozen portfolio sequence."""

    protocol_sha256: str

    def prepare(self, campaign_lock: CampaignLock) -> PortfolioArtifact:
        """Build/load the Portfolio after the Terminal Archive already exists."""

    def evaluate(self, campaign_lock: CampaignLock) -> PortfolioEvaluationReceipt:
        """Return typed observations; Lab still owns the Run terminal."""


@dataclass(frozen=True)
class ClaimView:
    """Scope-limited claims derived from one audited Portfolio terminal."""

    heldout_correctness_supported: bool
    dispatcher_correctness_supported: bool
    stable_kernel_performance_supported: bool
    stable_heldout_dispatcher_performance_supported: bool
    arbitrary_shape_generalization_supported: bool
    serving_supported: bool
    paper_result_reproduced: bool


@dataclass(frozen=True)
class PortfolioStudyReport:
    """Portfolio audit with correctness and measurement quality kept orthogonal."""

    study_id: str
    campaign_complete: bool
    archive_integrity_passed: bool
    filesystem_custody_verified: bool
    semantic_replay_passed: bool
    semantic_replay_error: str | None
    protocol_adherence: str
    endpoint: Mapping[str, object] | None
    claim_view: ClaimView
    run_audit: RunAudit | None


class PortfolioStudyMixin:
    def _preflight_portfolio(self, study: StudyContract) -> CampaignLock:
        """Resolve the real r43-r45 second use case without adding a runtime mode."""

        workload_ref = _object(study.document["workload"], "study.workload")
        if set(workload_ref) != {"path", "canonical_sha256"}:
            raise ValueError("portfolio Workload reference differs")
        workload_relative, workload_path = _project_path(
            self._root, workload_ref["path"], "study.workload.path"
        )
        workload = load_workload(workload_path)
        if workload.canonical_sha256 != _digest(
            workload_ref["canonical_sha256"], "study.workload.canonical_sha256"
        ):
            raise ValueError("portfolio Workload bytes differ")

        gate, compiler_relative, _compiler_reference = (
            _resolve_compiler_reference(
                self._root,
                study.document["compiler_revision"],
                "study.compiler_revision",
                template=study.state == "template",
            )
        )

        seed_ref = _object(study.document["kernel_seed"], "study.kernel_seed")
        if set(seed_ref) != {"path", "canonical_sha256"}:
            raise ValueError("portfolio Kernel Seed reference differs")
        seed_relative, seed_path = _project_path(
            self._root, seed_ref["path"], "study.kernel_seed.path"
        )
        seed = KernelSeed.load(self._root, seed_path)
        if seed.canonical_sha256 != _digest(
            seed_ref["canonical_sha256"], "study.kernel_seed.canonical_sha256"
        ) or seed.workload_sha256 != workload.canonical_sha256:
            raise ValueError("portfolio Kernel Seed bytes or Workload binding differ")

        case_roles = _object(study.document["case_roles"], "study.case_roles")
        if set(case_roles) != {"anchor", "held_out"}:
            raise ValueError("portfolio case roles differ")
        anchor = case_roles.get("anchor")
        held_out = case_roles.get("held_out")
        if (
            not isinstance(anchor, list)
            or len(anchor) != 1
            or not isinstance(held_out, list)
            or len(held_out) != 2
            or any(not isinstance(case, str) or not case for case in anchor + held_out)
            or len(set(anchor + held_out)) != 3
        ):
            raise ValueError("portfolio requires one anchor and two held-out cases")
        for case_id in anchor + held_out:
            workload.case(case_id)
        if anchor != ["headline_b32"] or held_out != ["b32_smoke", "public_b1"]:
            raise ValueError("portfolio frozen case roles differ")

        specialization = _object(
            study.document["specialization_policy"], "study.specialization_policy"
        )
        if specialization != {
            "kind": "exact_shape",
            "seed": "frozen",
            "retuning": "forbidden",
            "tail_policy": "reject",
        }:
            raise ValueError("portfolio specialization policy differs")
        dispatch = _object(study.document["dispatch_policy"], "study.dispatch_policy")
        if dispatch != {
            "key": ["B", "N", "K", "D"],
            "mapping_owner": "portfolio_manifest",
            "unsupported": "reject_before_launch",
            "fallback": "forbidden",
        }:
            raise ValueError("portfolio dispatch policy differs")
        evaluation = _object(
            study.document["evaluation_protocol"], "study.evaluation_protocol"
        )
        if evaluation != {
            "sequence": [
                "direct_preflight",
                "dispatcher_preflight",
                "kernel_timing",
                "dispatcher_timing",
                "postflight",
                "unsupported_probe",
            ],
            "cohorts_per_boundary": 5,
            "samples_per_cohort": 25,
            "maximum_cv": 0.05,
            "l2_flush_bytes": 268435456,
            "kernel_boundary": "cupti_target_kernel",
            "dispatcher_boundary": "host_validation_selection_launch_sync",
            "resampling": "forbidden",
        }:
            raise ValueError("portfolio Evaluation Protocol differs")
        execution = _object(study.document["execution"], "study.execution")
        if set(execution) != {
            "target",
            "executor_revision",
            "gpu",
            "sandbox",
        }:
            raise ValueError("portfolio execution fields differ")
        if execution.get("target") != "sm_100a" or execution.get("sandbox") != "workspace-write":
            raise ValueError("portfolio execution target or sandbox differs")
        executor_reference = _resolve_executor_reference(
            self._root,
            execution.get("executor_revision"),
            "study.execution",
            template=study.state == "template",
        )
        if _object(execution.get("gpu"), "study.execution.gpu") != {
            "name": "NVIDIA B200",
            "count": 1,
            "mode": "exclusive",
        }:
            raise ValueError("portfolio GPU admission differs")
        analysis = _object(study.document["analysis_plan"], "study.analysis_plan")
        if analysis != {
            "experimental_unit": "case_route",
            "estimand": "descriptive_exact_shape_correctness_and_measurement_quality",
            "correctness_scope": "three_frozen_cases_only",
            "measurement_quality": "per_case_per_boundary_cv",
            "cross_shape_contrast": "forbidden",
            "causal_claim": "forbidden",
            "missingness": "any_missing_declared_observation_makes_portfolio_endpoint_unavailable",
            "resampling": "forbidden",
        }:
            raise ValueError("portfolio Analysis Plan differs")
        evidence_policy = _object(study.document["evidence"], "study.evidence")
        if evidence_policy != {
            "schema_version": 2,
            "terminal_archive_required": True,
            "legacy_lineage": [
                "r43_harness_fault",
                "r44_host_contract_fault",
                "r45_unstable_timing",
            ],
            "legacy_claim_map_sha256": "7aae7a64a0638c0bf3ebfbffb88515ae6dce6329e6bb611d21c6052982ae2288",
        }:
            raise ValueError("portfolio Evidence policy differs")

        resolved_execution = cast(
            dict[str, object], json.loads(_canonical_json_bytes(execution))
        )
        resolved_execution["executor_revision"] = executor_reference
        lock_document: dict[str, object] = {
            "schema_version": 1,
            "study": {
                "study_id": study.study_id,
                "kind": "portfolio",
                "claim_scope": study.document["claim_scope"],
                "canonical_sha256": study.canonical_sha256,
            },
            "workload": {
                "workload_id": workload.workload_id,
                "path": workload_relative,
                "canonical_sha256": workload.canonical_sha256,
            },
            "compiler_revision": {
                "revision_id": gate.compiler_revision_id,
                "path": compiler_relative,
                "canonical_sha256": gate.compiler_revision_sha256,
            },
            "resolved_inputs": {
                "kernel_seed": {
                    "seed_id": seed.seed_id,
                    "path": seed_relative,
                    "canonical_sha256": seed.canonical_sha256,
                },
                "case_roles": case_roles,
                "specialization_policy": specialization,
                "dispatch_policy": dispatch,
                "evidence_policy": evidence_policy,
            },
            "run_order": ["portfolio-1"],
            "evaluation_protocol": evaluation,
            "execution": resolved_execution,
            "analysis_plan": analysis,
            "analysis_plan_sha256": sha256(_canonical_json_bytes(analysis)).hexdigest(),
        }
        return CampaignLock.from_dict(lock_document)


    def execute_portfolio(
        self,
        lock: CampaignLock,
        evidence_root: str | Path,
        *,
        assay: PortfolioAssay,
    ) -> CampaignRef:
        """Execute the sole frozen portfolio Run through Evaluation and Evidence."""

        if lock.study_kind != "portfolio" or lock.run_order != ("portfolio-1",):
            raise ValueError("portfolio execution requires a portfolio Campaign Lock")
        _validate_executor_revision(
            self._root,
            _object(lock.document["execution"], "campaign_lock.execution"),
            "campaign_lock.execution",
        )
        root = admit_new_campaign_path(
            self._root,
            evidence_root,
            role="Campaign Evidence root",
        )
        evidence = EvidenceStore.create(root)
        ledger = evidence.start_run(
            "portfolio-1",
            authority_sha256=lock.canonical_sha256,
            authority=lock.document,
        )
        ledger.append("run_started", {"sequence": 1, "replacement_run": False})
        try:
            if hasattr(assay, "set_observer"):

                def observe(kind: str, payload: Mapping[str, object]) -> None:
                    observation = evidence.put(
                        _canonical_json_bytes(payload), media_type="application/json"
                    )
                    ledger.append(
                        "portfolio_observation",
                        {
                            "observation_kind": kind,
                            "objects": [observation.reference("portfolio_observation")],
                        },
                    )

                assay.set_observer(observe)
            resolved = _object(lock.document["resolved_inputs"], "resolved_inputs")
            seed = _object(resolved["kernel_seed"], "resolved_inputs.kernel_seed")
            workload = _object(lock.document["workload"], "campaign_lock.workload")
            protocol_sha256 = sha256(
                _canonical_json_bytes(lock.document["evaluation_protocol"])
            ).hexdigest()
            artifact = assay.prepare(lock)
            if (
                assay.protocol_sha256 != protocol_sha256
                or artifact.workload_sha256 != workload["canonical_sha256"]
                or artifact.seed_sha256 != seed["canonical_sha256"]
            ):
                raise ValueError("Portfolio Assay does not match the Campaign Lock")
            expected_cases = set(
                cast(list[str], _object(resolved["case_roles"], "case_roles")["anchor"])
                + cast(list[str], _object(resolved["case_roles"], "case_roles")["held_out"])
            )
            if {entry.case_id for entry in artifact.entries} != expected_cases:
                raise ValueError("PortfolioArtifact cases differ from the Campaign Lock")
            artifact_references: list[dict[str, object]] = []
            for entry in artifact.entries:
                candidate = entry.candidate
                if (
                    not _ARM_ARTIFACT_ROLES["open_cake"]
                    <= set(candidate.artifact_payloads)
                    or set(candidate.artifact_payloads) != set(candidate.artifact_roles)
                ):
                    raise ValueError("Portfolio candidate artifact custody is incomplete")
                for role, payload in sorted(candidate.artifact_payloads.items()):
                    item = evidence.put(
                        payload, media_type=_candidate_artifact_media_type(role)
                    )
                    artifact_references.append(
                        item.reference(f"{entry.case_id}_{role}")
                    )
            artifact_object = evidence.put(
                _canonical_json_bytes(artifact.document), media_type="application/json"
            )
            artifact_references.append(
                artifact_object.reference("portfolio_artifact_manifest")
            )
            ledger.append(
                "portfolio_sealed",
                {
                    "portfolio_sha256": artifact.canonical_sha256,
                    "objects": artifact_references,
                },
            )
            receipt = assay.evaluate(lock)
            if (
                receipt.portfolio_sha256 != artifact.canonical_sha256
                or receipt.evaluation_protocol_sha256 != protocol_sha256
            ):
                raise ValueError("Portfolio receipt does not match the Campaign Lock")
            receipt_object = evidence.put(
                _canonical_json_bytes(receipt.document),
                media_type="application/json",
            )
            endpoint = _portfolio_endpoint(receipt)
            ledger.append(
                "portfolio_evaluated",
                {"objects": [receipt_object.reference("portfolio_evaluation_receipt")]},
            )
            ledger.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint=endpoint,
            )
        except Exception as error:
            fault = (
                error.protocol_adherence
                if isinstance(error, RunProtocolFault)
                else "harness_fault"
            )
            fault_payload: dict[str, object] = {
                "fault": fault,
                "exception_type": type(error).__name__,
            }
            if isinstance(error, RunProtocolFault) and error.artifact_payloads:
                references = []
                rejected = []
                for role, payload in sorted(error.artifact_payloads.items()):
                    try:
                        references.append(
                            evidence.put(payload, media_type="application/octet-stream").reference(
                                role
                            )
                        )
                    except (OSError, ValueError):
                        rejected.append(role)
                if references:
                    fault_payload["objects"] = references
                if rejected:
                    fault_payload["artifact_rejections"] = rejected
            ledger.append("run_fault", fault_payload)
            ledger.seal(
                protocol_adherence=fault,
                endpoint_observation="missing",
            )
        return CampaignRef(lock=lock, evidence_root=evidence.root)


    def audit_portfolio(self, campaign: CampaignRef) -> PortfolioStudyReport:
        """Derive the bounded r45-style Claim View from one terminal archive."""

        if campaign.lock.study_kind != "portfolio":
            raise ValueError("portfolio audit requires a portfolio Campaign Lock")
        store = EvidenceStore.open(campaign.evidence_root)
        try:
            audit = store.audit_run("portfolio-1")
        except (OSError, ValueError, json.JSONDecodeError):
            audit = None
        complete = audit is not None and audit.authority_sha256 == campaign.lock.canonical_sha256
        archive_integrity = bool(
            complete and audit is not None and audit.archive_integrity
        )
        filesystem_custody_verified = bool(
            complete and audit is not None and audit.filesystem_custody_verified
        )
        semantic_replay = False
        semantic_replay_error: str | None = None
        endpoint: Mapping[str, object] | None = None
        if archive_integrity and audit is not None:
            try:
                events = store.replay_events("portfolio-1")

                def object_for(kind: str, role: str) -> Mapping[str, object]:
                    matching = [event for event in events if event.get("kind") == kind]
                    if len(matching) != 1:
                        raise ValueError(f"portfolio event {kind!r} coverage differs")
                    payload = _object(matching[0].get("payload"), f"event.{kind}.payload")
                    objects = payload.get("objects")
                    if not isinstance(objects, list):
                        raise ValueError(f"portfolio event {kind!r} objects differ")
                    selected = [item for item in objects if isinstance(item, Mapping) and item.get("role") == role]
                    if len(selected) != 1:
                        raise ValueError(f"portfolio object role {role!r} coverage differs")
                    return cast(Mapping[str, object], selected[0])

                artifact_document = json.loads(
                    store.read_object(
                        object_for("portfolio_sealed", "portfolio_artifact_manifest")
                    )
                )
                artifact = PortfolioArtifact.from_document(artifact_document)
                for entry in artifact.entries:
                    for role, expected_sha256 in entry.candidate.artifact_roles.items():
                        reference = object_for(
                            "portfolio_sealed", f"{entry.case_id}_{role}"
                        )
                        payload = store.read_object(reference)
                        if (
                            reference.get("sha256") != expected_sha256
                            or sha256(payload).hexdigest() != expected_sha256
                        ):
                            raise ValueError(
                                "replayed PortfolioArtifact object lineage differs"
                            )
                resolved = _object(
                    campaign.lock.document["resolved_inputs"], "resolved_inputs"
                )
                seed = _object(resolved["kernel_seed"], "resolved_inputs.kernel_seed")
                workload = _object(campaign.lock.document["workload"], "workload")
                if (
                    artifact.seed_sha256 != seed["canonical_sha256"]
                    or artifact.workload_sha256 != workload["canonical_sha256"]
                ):
                    raise ValueError("replayed PortfolioArtifact authority differs")
                protocol_sha256 = sha256(
                    _canonical_json_bytes(campaign.lock.document["evaluation_protocol"])
                ).hexdigest()
                receipt_document = json.loads(
                    store.read_object(
                        object_for(
                            "portfolio_evaluated", "portfolio_evaluation_receipt"
                        )
                    )
                )
                receipt = replay_portfolio_receipt(
                    receipt_document,
                    artifact,
                    expected_protocol_sha256=protocol_sha256,
                )
                endpoint = _portfolio_endpoint(receipt)
                if audit.endpoint != endpoint:
                    raise ValueError("portfolio terminal endpoint differs from replay")
                semantic_replay = True
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                semantic_replay = False
                semantic_replay_error = f"{type(error).__name__}: {error}"
        correctness = endpoint.get("correctness_by_case") if endpoint is not None else None
        kernel_quality = (
            endpoint.get("kernel_measurement_quality_by_case") if endpoint is not None else None
        )
        dispatcher_quality = (
            endpoint.get("dispatcher_measurement_quality_by_case") if endpoint is not None else None
        )
        all_correct = (
            isinstance(correctness, Mapping)
            and len(correctness) == 3
            and all(value is True for value in correctness.values())
        )
        all_kernel_stable = (
            isinstance(kernel_quality, Mapping)
            and len(kernel_quality) == 3
            and all(value == "stable" for value in kernel_quality.values())
        )
        resolved = _object(campaign.lock.document["resolved_inputs"], "resolved_inputs")
        case_roles = _object(resolved["case_roles"], "resolved_inputs.case_roles")
        held_out = cast(list[str], case_roles["held_out"])
        heldout_dispatch_stable = isinstance(dispatcher_quality, Mapping) and all(
            dispatcher_quality.get(case_id) == "stable" for case_id in held_out
        )
        adhered = audit is not None and audit.protocol_adherence == "adhered"
        supported = (
            archive_integrity
            and filesystem_custody_verified
            and adhered
            and semantic_replay
        )
        claim_view = ClaimView(
            heldout_correctness_supported=supported and all_correct,
            dispatcher_correctness_supported=supported and all_correct,
            stable_kernel_performance_supported=supported and all_correct and all_kernel_stable,
            stable_heldout_dispatcher_performance_supported=(
                supported and all_correct and heldout_dispatch_stable
            ),
            arbitrary_shape_generalization_supported=False,
            serving_supported=False,
            paper_result_reproduced=False,
        )
        return PortfolioStudyReport(
            study_id=campaign.lock.study_id,
            campaign_complete=complete,
            archive_integrity_passed=archive_integrity,
            filesystem_custody_verified=filesystem_custody_verified,
            semantic_replay_passed=semantic_replay,
            semantic_replay_error=semantic_replay_error,
            protocol_adherence=audit.protocol_adherence if audit is not None else "missing",
            endpoint=endpoint,
            claim_view=claim_view,
            run_audit=audit,
        )
