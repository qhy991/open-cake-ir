from __future__ import annotations

import contextlib
import dataclasses
import grp
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation import (  # noqa: E402
    BrokerAttempt,
    EvaluationReceipt,
    LaunchableCandidate,
    LogicalEvaluationAttempt,
    NCU_ATTRIBUTION_METRICS,
    build_ncu_attribution_profile,
)
from open_cake_ir.evidence import EvidenceStore  # noqa: E402
from open_cake_ir.lab import (  # noqa: E402
    CANDIDATE_SET_ENVELOPE_V1,
    CODEX_DISABLED_FEATURES,
    BoundedBrokerEvaluator,
    CampaignLock,
    CommandBrokerSubmitter,
    EnvironmentResult,
    ExecutorRevision,
    Lab,
    ProviderAuxiliaryActivity,
    ProviderQualificationReceipt,
    ProviderTurn,
    RunProtocolFault,
    RalphBudget,
    RalphController,
    TASK_AGENTS_RALPH_V1,
    TurnObservation,
    materialize_task_package,
    project_checkpoints,
    render_task_package,
    verify_task_package,
)


def _profile_fixture(candidate_sha256: str, case_id: str, kernel_name: str) -> bytes:
    values = (95, 2, 2, 16, 8, 61.0, 24.0, 41.0, 37.0, 18.0, 3.0)
    lines = ['"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"']
    for index, (metric, value) in enumerate(zip(NCU_ATTRIBUTION_METRICS, values)):
        unit = "%" if "pct" in metric else "count"
        lines.append(f'"{index}","{kernel_name}","{metric}","{unit}","{value}"')
    return build_ncu_attribution_profile(
        candidate_sha256=candidate_sha256,
        case_id=case_id,
        kernel_name=kernel_name,
        ncu_version="2026.1.1.0",
        ncu_executable_sha256="e" * 64,
        stdout=("\n".join(lines) + "\n").encode(),
        stderr=b"==PROF== fixture\n",
    )


def _execute(lab, lock, evidence_root, *, provider, **kwargs):
    provider.packages = {
        run_id: render_task_package(lab._root, lock, run_id)
        for run_id in lock.run_order
    }
    return lab.execute(lock, evidence_root, provider=provider, **kwargs)


class FakeEnvironment:
    def __init__(self, arm: str, authority_document) -> None:
        self.arm = arm
        self.media_type = (
            "application/vnd.open-cake.schedule+json"
            if arm == "open_cake"
            else "text/x-cuda"
        )
        self.authority_document = authority_document
        self.canonical_sha256 = sha256(
            json.dumps(
                authority_document, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def build(self, submission):
        source_role = "lowered_source" if self.arm == "open_cake" else "authored_source"
        turn = json.loads(submission.payload)["turn"]
        entry_point = f"{self.arm}_turn_{turn}"
        launch_manifest = json.dumps(
            {
                "schema_version": 1,
                "abi": "flash_kmeans_assign_v1",
                "target": "sm_100a",
                "kernel_name": entry_point,
                "grid": [1, 1, 1],
                "block": [32, 1, 1],
                "dynamic_shared_memory_bytes": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        payloads = {
            source_role: submission.payload,
            "ptx": b"ptx:" + submission.payload,
            "cubin": b"cubin:" + submission.payload,
            "launch_manifest": launch_manifest,
        }
        if self.arm == "open_cake":
            payloads["compiler_expanded_source"] = b"expanded:" + submission.payload
        else:
            payloads["sass"] = b"sass:" + submission.payload
        artifacts = {role: sha256(payload).hexdigest() for role, payload in payloads.items()}
        return EnvironmentResult(
            "launchable",
            submission.sha256,
            LaunchableCandidate(
                candidate_sha256=submission.sha256,
                target="sm_100a",
                entry_point=entry_point,
                artifact_roles=artifacts,
                launch_spec_sha256=artifacts["launch_manifest"],
                artifact_payloads=payloads,
            ),
            {},
        )


class FakeProvider:
    provider_revision = "fixture-provider-candidate-set-ralph-v2"
    qualification_sha256 = "d693060b7bb0af69dacf9205f51aeb07b89955800015cb6b4b7c53c91b987a0d"
    executable_sha256 = "d" * 64
    configuration = {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "max",
        "service_tier": "default",
        "output_schema_sha256": "5b3b813d98ddae93fe9ff5cf0f1568d64721fa31029cb77a0ad5f2dfc800ed18",
        "removed_environment": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "sandbox": "workspace-write",
        "cwd_policy": "independent_task_workspace",
        "reference_visibility": "workspace_task_files",
        "disabled_features": list(CODEX_DISABLED_FEATURES),
        "submission_contract": CANDIDATE_SET_ENVELOPE_V1,
    }

    def __init__(self) -> None:
        self.requests = []
        self.threads: dict[str, str] = {}

    def turn(self, request):
        self.requests.append(request)
        thread_id = self.threads.setdefault(
            request.run_id, f"00000000-0000-0000-0000-{len(self.threads) + 1:012d}"
        )
        if request.thread_id is not None and request.thread_id != thread_id:
            raise ValueError("resume thread differs")
        payload = json.dumps(
            {"run_id": request.run_id, "turn": request.turn},
            sort_keys=True,
        ).encode()
        candidate_name = (
            "candidate-set.json"
            if "submission_contract" in self.configuration
            else ("candidate.json" if request.arm == "open_cake" else "candidate.cu")
        )
        change = "add" if request.turn == 1 else "update"
        file_item = {
            "id": f"file-{request.turn}",
            "type": "file_change",
            "changes": [{"path": f"/fixture/{candidate_name}", "kind": change}],
        }
        terminal = json.dumps(
            {
                "arm": request.arm,
                "candidate_written": True,
                "kind": "open_cake_ir_turn",
                "tool_calls": 1,
                "turn": request.turn,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        raw_events = b"\n".join(
            json.dumps(event, separators=(",", ":")).encode()
            for event in (
                {"type": "thread.started", "thread_id": thread_id},
                {"type": "turn.started"},
                {
                    "type": "item.started",
                    "item": {**file_item, "status": "in_progress"},
                },
                {
                    "type": "item.completed",
                    "item": {**file_item, "status": "completed"},
                },
                {
                    "type": "item.completed",
                    "item": {
                        "id": "message",
                        "type": "agent_message",
                        "text": terminal,
                    },
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 70000, "output_tokens": 10000},
                },
            )
        ) + b"\n"
        return ProviderTurn(
            thread_id=thread_id,
            provider_tokens=80000,
            candidates=(payload,),
            candidate_sha256s=(sha256(payload).hexdigest(),),
            raw_events=raw_events,
            raw_events_sha256=sha256(raw_events).hexdigest(),
            terminal_message=terminal,
            terminal_message_count=1,
            normalization="single_exact",
            reference_bundle=(
                self.packages[request.run_id].evidence_bundle(request.state_card)
            ),
        )


class CandidateSetFakeProvider(FakeProvider):
    pass


class RalphFakeProvider(FakeProvider):
    provider_revision = "fixture-provider-candidate-set-ralph-v2"
    qualification_sha256 = "d693060b7bb0af69dacf9205f51aeb07b89955800015cb6b4b7c53c91b987a0d"
    configuration = {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "max",
        "service_tier": "default",
        "output_schema_sha256": "5b3b813d98ddae93fe9ff5cf0f1568d64721fa31029cb77a0ad5f2dfc800ed18",
        "removed_environment": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "sandbox": "workspace-write",
        "cwd_policy": "independent_task_workspace",
        "reference_visibility": "workspace_task_files",
        "disabled_features": list(CODEX_DISABLED_FEATURES),
        "submission_contract": CANDIDATE_SET_ENVELOPE_V1,
    }

    def __init__(self, packages) -> None:
        super().__init__()
        self.packages = packages

    def turn(self, request):
        result = super().turn(request)
        if request.state_card is None:
            raise ValueError("Ralph fixture requires a StateCard")
        return dataclasses.replace(
            result,
            reference_bundle=self.packages[request.run_id].evidence_bundle(
                request.state_card
            ),
        )


def _enable_candidate_set(document: dict[str, object], maximum: int) -> None:
    """Make a copied fixture Study use the one candidate-set authority."""

    budget = document["budget"]
    assert isinstance(budget, dict)
    budget["maximum_candidates_per_turn"] = maximum
    receipt_path = "contracts/providers/fixture-provider-candidate-set-ralph-v2.json"
    receipt = ProviderQualificationReceipt.load(ROOT / receipt_path)
    arms = document["arms"]
    assert isinstance(arms, dict)
    for arm, raw_environment in arms.items():
        assert isinstance(raw_environment, dict)
        provider = raw_environment["provider"]
        assert isinstance(provider, dict)
        provider["revision"] = receipt.provider_revision
        provider["qualification"] = {
            "path": receipt_path,
            "canonical_sha256": receipt.canonical_sha256,
        }



class FakeEvaluator:
    def __init__(
        self,
        protocol,
        protocol_sha256: str,
        workload_sha256: str,
        *,
        raw_kernel_calls: int = 1,
    ) -> None:
        self.protocol = protocol
        self.protocol_sha256 = protocol_sha256
        self.workload_sha256 = workload_sha256
        self.raw_kernel_calls = raw_kernel_calls
        self.calls = 0

    def evaluate(self, candidate, *, case_id, purpose):
        arm, raw_turn = candidate.entry_point.rsplit("_turn_", 1)
        turn = int(raw_turn)
        latency = (1.0 if arm == "open_cake" else 2.0) - (turn - 1) * 0.1
        launch_receipt = json.dumps(
            {"candidate_sha256": candidate.candidate_sha256, "purpose": purpose},
            sort_keys=True,
        ).encode()
        attribution = purpose == "attribution"
        timing = (
            None
            if attribution
            else {"measurement_quality_passed": True, "pooled_median_ms": latency}
        )
        result_artifacts = (
            {
                "correctness_output": "correctness-output.json",
                "launch_receipt": "launch-receipt.json",
                "profile": "profile.json",
            }
            if attribution
            else {
                "correctness_output": "correctness-output.json",
                "launch_receipt": "launch-receipt.json",
                "timing_samples": "timing-samples.json",
            }
        )
        artifact_payloads = {
            "correctness_output": json.dumps(
                {"tie_aware_distance_match": True}, sort_keys=True
            ).encode(),
            "launch_receipt": launch_receipt,
            **(
                {
                    "profile": _profile_fixture(
                        candidate.candidate_sha256, case_id, candidate.entry_point
                    )
                }
                if attribution
                else {"timing_samples": json.dumps([latency] * 125).encode()}
            ),
        }
        receipt = EvaluationReceipt(
            candidate_sha256=candidate.candidate_sha256,
            workload_sha256=self.workload_sha256,
            evaluation_protocol_sha256=self.protocol_sha256,
            purpose=purpose,
            case_id=case_id,
            correctness_passed=True,
            correctness={"tie_aware_distance_match": True},
            kernel_calls=1,
            fallback_calls=0,
            launch_receipt_sha256=sha256(launch_receipt).hexdigest(),
            timing=timing,
            artifact_payloads=artifact_payloads,
        )
        self.calls += 1
        raw_result = json.dumps(
            {
                "schema_version": 1,
                "job_id": f"gpuq-{self.calls:012x}",
                "mode": "exclusive",
                "admitted": True,
                "error": None,
                "failure_class": None,
                "counters": {
                    "compiler_invocations": 0,
                    "module_loads": 1,
                    "preflight_calls": 1,
                    "kernel_calls": self.raw_kernel_calls,
                    "timing_samples": 0 if attribution else 125,
                    "fallback_calls": 0,
                },
                "receipt": {
                    "correctness_passed": True,
                    "correctness": {"tie_aware_distance_match": True},
                    "kernel_calls": 1,
                    "fallback_calls": 0,
                    "timing": timing,
                    "artifacts": result_artifacts,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        attempt = BrokerAttempt(
            job_id=f"gpuq-{self.calls:012x}",
            mode="exclusive",
            candidate_sha256=candidate.candidate_sha256,
            manifest_sha256=candidate.launch_spec_sha256,
            policy_sha256=self.protocol_sha256,
            evaluator_arguments_sha256=sha256(
                f"{case_id}:{purpose}".encode()
            ).hexdigest(),
            admitted=True,
            error=None,
            compiler_invocations=0,
            module_loads=1,
            preflight_calls=1,
            kernel_calls=1,
            timing_samples=0 if attribution else 125,
            fallback_calls=0,
            receipt=receipt,
            artifact_payloads={
                "broker_record": raw_result,
                "stdout": b"completed\n",
                "stderr": b"",
                "evaluator_result": raw_result,
            },
        )
        return LogicalEvaluationAttempt(candidate.candidate_sha256, (attempt,), receipt)


class FindingRoutingContractTests(unittest.TestCase):
    def test_environment_retains_hints_in_the_existing_agent_channel(self) -> None:
        from open_cake_ir.compiler import Compiler
        from open_cake_ir.lab.environments import OpenCakeEnvironment

        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        assessment = compiler.assess_file(ROOT / "corpus/schedules/tinygemm2-stage4-split-k.json")
        rows = OpenCakeEnvironment._finding_rows(assessment)
        self.assertEqual(rows, [item.to_dict() for item in assessment.findings + assessment.guidance])
        self.assertTrue(any(item["severity"] == "report" for item in rows))
        self.assertTrue(any(item["severity"] == "hint" for item in rows))
        self.assertTrue(all(not item["blocks_acceptance"] and not item["blocks_lowering"]
                            for item in rows))

    def test_routing_ignores_advisory_codes_and_keeps_blocking_order_deterministic(self) -> None:
        from open_cake_ir.compiler import Finding, FindingCategory, FindingSeverity
        from open_cake_ir.lab.routing import route_rejection

        hint = Finding("BACKEND_OPERATION_UNEMITTABLE", "operations[2]", "advisory only",
                       FindingCategory.HARDWARE_CONFORMANCE, FindingSeverity.HINT)
        blocking = [
            Finding("OP_DEPENDENCY_CYCLE", "operations[1]", "dependency cycle",
                    FindingCategory.PROGRAM_SAFETY),
            Finding("NAME_DUPLICATE", "buffers", "duplicate buffer name",
                    FindingCategory.SCHEDULE_SEMANTICS),
        ]
        rows = [item.to_dict() for item in blocking + [hint]]
        forward = route_rejection({"stage": "assessment", "findings": rows})
        backward = route_rejection({"stage": "assessment", "findings": list(reversed(rows))})
        self.assertEqual(forward, backward)
        self.assertEqual(forward.destination, "candidate")
        self.assertEqual(forward.reason, "a gate refused it: NAME_DUPLICATE, OP_DEPENDENCY_CYCLE")
        backend = dataclasses.replace(hint, severity=FindingSeverity.BLOCKING)
        self.assertEqual(
            route_rejection({"stage": "assessment", "findings": [backend.to_dict()]}).destination,
            "ir_vocabulary",
        )


class LabContractTests(unittest.TestCase):
    def test_execute_refuses_evidence_inside_the_checkout_before_side_effects(self) -> None:
        lock = Lab(ROOT).preflight(ROOT / "contracts/studies/matched-search-system-qualification-ralph-template.json")
        with tempfile.TemporaryDirectory(prefix=".campaign-custody-", dir=ROOT) as directory:
            evidence_root = Path(directory) / "evidence"

            with self.assertRaisesRegex(ValueError, "outside the project checkout"):
                Lab(ROOT).execute(
                    lock,
                    evidence_root,
                    provider=object(),
                    environments={},
                    evaluator=object(),
                )

            self.assertFalse(evidence_root.exists())

    def test_current_study_templates_resolve_exact_current_revisions(self) -> None:
        for name in (
            "matched-search-infrastructure-template.json",
            "matched-search-system-qualification-ralph-template.json",
            "artifact-optimization-ralph-template.json",
            "flash-kmeans-r45-portfolio-reconstruction-template.json",
            "matched-search-clean-start-reference-template.json",
            "matched-search-system-qualification-ralph-template.json",
            "artifact-optimization-ralph-template.json",
        ):
            path = ROOT / "contracts/studies" / name
            study = json.loads(path.read_text(encoding="utf-8"))
            compiler_binding = (
                study["compiler_revision"]
                if study["kind"] == "portfolio"
                else study["arms"]["open_cake"]["compiler_revision"]
            )
            self.assertEqual(study["state"], "template", name)
            self.assertEqual(compiler_binding, {"binding": "current_release"}, name)
            self.assertEqual(
                study["execution"]["executor_revision"],
                {"binding": "current_release"},
                name,
            )

            lock = Lab(ROOT).preflight(path)
            executor = lock.document["execution"]["executor_revision"]
            current = json.loads(
                (ROOT / "inventory/EXECUTOR_REVISIONS.json").read_text(encoding="utf-8")
            )["current"]
            self.assertEqual(executor["executor_id"], current["executor_id"], name)
            self.assertEqual(
                executor["canonical_sha256"],
                current["canonical_sha256"],
                name,
            )
            compiler = lock.document["compiler_revision"]
            current_compiler = json.loads(
                (ROOT / "compiler/revision.lock.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                compiler["revision_id"], current_compiler["revision_id"], name
            )
            if lock.study_kind == "matched_search":
                self.assertEqual(
                    lock.document["evaluation_protocol"]["attribution_evaluation"],
                    "correctness_then_profile_each_search_survivor",
                    name,
                )

    def test_study_revision_binding_forms_are_closed_by_state(self) -> None:
        source = ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        template = json.loads(source.read_text(encoding="utf-8"))
        resolved = Lab(ROOT).preflight(source).document
        exact_compiler = dict(resolved["compiler_revision"])
        exact_executor = dict(resolved["execution"]["executor_revision"])
        cases = (
            (
                "template-inline-compiler",
                "template",
                exact_compiler,
                {"binding": "current_release"},
                "template Compiler binding",
            ),
            (
                "template-inline-executor",
                "template",
                {"binding": "current_release"},
                exact_executor,
                "template Executor binding",
            ),
            (
                "frozen-moving-compiler",
                "frozen",
                {"binding": "current_release"},
                exact_executor,
                "frozen Study cannot follow the current Compiler",
            ),
            (
                "frozen-moving-executor",
                "frozen",
                exact_compiler,
                {"binding": "current_release"},
                "frozen Study cannot follow the current Executor",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, state, compiler, executor, error in cases:
                with self.subTest(name=name):
                    study = json.loads(json.dumps(template))
                    study["state"] = state
                    study["arms"]["open_cake"]["compiler_revision"] = compiler
                    study["execution"]["executor_revision"] = executor
                    path = Path(directory) / f"{name}.json"
                    path.write_text(json.dumps(study), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        Lab(ROOT).preflight(path)

    def test_v25_retains_the_reference_bundle_or_records_a_missing_endpoint(self) -> None:
        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-system-qualification-ralph-template.json"
        )
        resolved = lock.document["resolved_inputs"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        def execute(provider, evidence_root):
            return _execute(lab,
                lock,
                evidence_root,
                provider=provider,
                environments={
                    name: FakeEnvironment(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            campaign = execute(FakeProvider(), Path(directory) / "retained")
            report = lab.audit(campaign)
            evidence = EvidenceStore.open(campaign.evidence_root)
            for run_id in lock.run_order:
                expected = render_task_package(ROOT, lock, run_id)
                for event in evidence.replay_events(run_id):
                    if event["kind"] != "provider_turn_completed":
                        continue
                    references = [
                        item
                        for item in event["payload"]["objects"]
                        if item["role"] == "provider_reference_bundle"
                    ]
                    self.assertEqual(len(references), 1)
                    self.assertEqual(json.loads(evidence.read_object(references[0]))["task_markdown"], expected.task_markdown)

            self.assertTrue(report.archive_integrity_passed)
            self.assertTrue(report.semantic_replay_passed)
            self.assertTrue(report.system_qualification_passed)

            run_id = lock.run_order[0]
            audit = evidence.audit_run(run_id)
            without_bundle = json.loads(
                json.dumps(evidence.replay_events(run_id))
            )
            for event in without_bundle:
                if event["kind"] == "provider_turn_completed":
                    event["payload"]["objects"] = [
                        item
                        for item in event["payload"]["objects"]
                        if item["role"] != "provider_reference_bundle"
                    ]
            with mock.patch.object(
                evidence,
                "replay_events",
                return_value=tuple(without_bundle),
            ):
                self.assertFalse(
                    lab._replay_matched_run(evidence, audit, lock)
                )

        class MissingBundleProvider(FakeProvider):
            def turn(self, request):
                return dataclasses.replace(
                    super().turn(request),
                    reference_bundle=None,
                )

        with tempfile.TemporaryDirectory() as directory:
            campaign = execute(
                MissingBundleProvider(),
                Path(directory) / "missing",
            )
            report = lab.audit(campaign)

        self.assertTrue(report.archive_integrity_passed)
        self.assertTrue(report.semantic_replay_passed)
        self.assertFalse(report.system_qualification_passed)
        self.assertEqual(report.missing_run_count, len(lock.run_order))
        self.assertEqual(
            {audit.protocol_adherence for audit in report.run_audits},
            {"harness_fault"},
        )



    def test_system_qualification_preflight_binds_non_scientific_one_run_per_arm(self) -> None:
        lock = Lab(ROOT).preflight(
            ROOT / "contracts/studies/matched-search-system-qualification-ralph-template.json"
        )

        self.assertEqual(lock.run_order, ("open_cake-1", "direct_cuda-1"))
        self.assertEqual(lock.claim_scope, "system_qualification_only")
        self.assertIsNone(lock.estimand)

    def test_artifact_optimization_preflight_binds_full_features_without_an_estimand(self) -> None:
        lock = Lab(ROOT).preflight(
            ROOT / "contracts/studies/artifact-optimization-ralph-template.json"
        )

        self.assertEqual(lock.run_order, ("open_cake-1", "direct_cuda-1"))
        self.assertEqual(lock.claim_scope, "artifact_optimization_only")
        self.assertIsNone(lock.estimand)
        providers = lock.document["resolved_inputs"]["arm_environments"]
        for environment in providers.values():
            provider = environment["provider"]
            self.assertEqual(provider["disabled_features"], [])
            self.assertEqual(provider["event_contract"], "tool_rich_candidate_v1")


    def test_closed_provider_receipt_cannot_authorize_artifact_optimization(self) -> None:
        study = json.loads(
            (ROOT / "contracts/studies/artifact-optimization-ralph-template.json").read_text()
        )
        for arm in study["arms"].values():
            provider = arm["provider"]
            provider["revision"] = "codex-cli-0.144.3-sha134063e133f0"
            provider["executable_sha256"] = (
                "134063e133f0b4244fa3b251acf973d4fe4b4aeeacbdc135211bf480f59f1477"
            )
            provider["qualification"] = {
                "path": "contracts/providers/codex-cli-0.144.3-live-r4.json",
                "canonical_sha256": (
                    "86fb53a1f51845dee11e105eefe8bb01e5563dc0217cd7c9c7573d8199a0e707"
                ),
            }
            provider["qualification_anchor"] = {
                "path": "evidence/qualifications/codex-cli-0.144.3-live-r4-anchor.json",
                "canonical_sha256": (
                    "aac57d142aa66ed8f94e1adcffa8c198bccfee7ec73d36991f42a587368fecbd"
                ),
            }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(
                ValueError,
                "provider qualification bytes or capability differs",
            ):
                Lab(ROOT).preflight(path)

    def test_fixture_provider_qualification_forbids_an_external_anchor(self) -> None:
        study = json.loads(
            (
                ROOT
                / "contracts/studies/matched-search-system-qualification-ralph-template.json"
            ).read_text()
        )
        anchor = {"path": "anchor.json", "canonical_sha256": "a" * 64}
        study["arms"]["open_cake"]["provider"]["qualification_anchor"] = anchor
        study["arms"]["direct_cuda"]["provider"]["qualification_anchor"] = anchor
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(
                ValueError, "fixture provider qualification anchor must be null"
            ):
                Lab(ROOT).preflight(path)

    def test_system_qualification_rejects_a_scientific_estimand(self) -> None:
        study = json.loads(
            (
                ROOT
                / "contracts/studies/matched-search-system-qualification-ralph-template.json"
            ).read_text()
        )
        study["analysis_plan"]["estimand"] = "forbidden pilot contrast"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(
                ValueError, "system qualification Analysis Plan differs"
            ):
                Lab(ROOT).preflight(path)

    def test_preflight_rejects_a_changed_direct_candidate_skeleton(self) -> None:
        study = json.loads(
            (ROOT / "contracts/studies/matched-search-infrastructure-template.json").read_text()
        )
        study["arms"]["direct_cuda"]["candidate_skeleton"]["sha256"] = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(
                ValueError, "direct candidate skeleton bytes differ"
            ):
                Lab(ROOT).preflight(path)

    def test_reasoning_effort_requires_its_matching_qualification(self) -> None:
        study = json.loads(
            (ROOT / "contracts/studies/matched-search-infrastructure-template.json").read_text()
        )
        for arm in study["arms"].values():
            arm["provider"]["reasoning_effort"] = "xhigh"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(
                ValueError, "provider qualification bytes or capability differs"
            ):
                Lab(ROOT).preflight(path)

    def test_system_qualification_passes_with_replayed_evaluations_below_checkpoint(self) -> None:
        class UnderCheckpointProvider(FakeProvider):
            def turn(self, request):
                observed = super().turn(request)
                events = [json.loads(line) for line in observed.raw_events.splitlines()]
                events[-1]["usage"] = {
                    "input_tokens": 60000,
                    "output_tokens": 10000,
                }
                raw_events = b"".join(
                    json.dumps(event, separators=(",", ":")).encode() + b"\n"
                    for event in events
                )
                return ProviderTurn(
                    thread_id=observed.thread_id,
                    provider_tokens=70000,
                    candidates=observed.candidates,
                    candidate_sha256s=observed.candidate_sha256s,
                    raw_events=raw_events,
                    raw_events_sha256=sha256(raw_events).hexdigest(),
                    terminal_message=observed.terminal_message,
                    terminal_message_count=observed.terminal_message_count,
                    normalization=observed.normalization,
                    reference_bundle=observed.reference_bundle,
                )

        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-system-qualification-ralph-template.json"
        )
        provider = UnderCheckpointProvider()
        resolved = lock.document["resolved_inputs"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            campaign = _execute(lab,
                lock,
                Path(directory).resolve() / "evidence",
                provider=provider,
                environments={
                    name: FakeEnvironment(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            report = lab.audit(campaign)

        self.assertEqual(len(provider.requests), 4)
        self.assertTrue(report.system_qualification_passed)
        self.assertEqual(report.missing_run_count, 2)
        self.assertTrue(
            all(
                run["endpoint_observation"] == "missing"
                and run["evaluation_receipt_count"] >= 1
                for run in report.descriptive["runs"]
            )
        )
        self.assertIsNone(report.estimand)
        self.assertFalse(report.estimand_available)
        self.assertIsNone(report.estimate)
        self.assertIsNone(report.uncertainty)
        self.assertNotIn("qualification_rate", report.descriptive)
        self.assertNotIn("median_confirmed_latency_ms", report.descriptive)
        self.assertNotIn("paired_runs", report.descriptive)
        self.assertEqual(
            {item.reason for item in report.run_inclusion},
            {"system_qualification_not_scientific_data"},
        )
        self.assertTrue(
            all(
                not item.qualification_endpoint_included
                and not item.conditional_performance_included
                for item in report.run_inclusion
            )
        )

    def test_artifact_optimization_promotes_per_run_without_scientific_analysis(self) -> None:
        class OptimizationProvider(FakeProvider):
            provider_revision = "fixture-provider-optimization-candidate-set-ralph-v1"
            qualification_sha256 = (
                "fb39b90b6dec56953562879ff011fe9deab839928bfcdb1d32e6d8c6b79180d8"
            )
            configuration = {
                "model": "gpt-5.6-sol",
                "reasoning_effort": "max",
                "service_tier": "default",
                "output_schema_sha256": (
                    "c1166659951f1919b539a2a8a35b154ccff4c8a0121d233f7658e8dbc175a70a"
                ),
                "removed_environment": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
                "sandbox": "workspace-write",
                "cwd_policy": "independent_task_workspace",
                "reference_visibility": "workspace_task_files",
                "disabled_features": [],
                "event_contract": "tool_rich_candidate_v1",
                "submission_contract": CANDIDATE_SET_ENVELOPE_V1,
            }

            def turn(self, request):
                observed = super().turn(request)
                events = [json.loads(line) for line in observed.raw_events.splitlines()]
                events[2:2] = [
                    {
                        "type": "item.started",
                        "item": {
                            "id": f"command-{request.turn}",
                            "type": "command_execution",
                            "command": "pwd",
                            "status": "in_progress",
                        },
                    },
                    {
                        "type": "item.completed",
                        "item": {
                            "id": f"command-{request.turn}",
                            "type": "command_execution",
                            "command": "pwd",
                            "status": "completed",
                        },
                    },
                ]
                terminal = json.loads(events[-2]["item"]["text"])
                terminal.pop("tool_calls")
                events[-2]["item"]["text"] = json.dumps(
                    terminal,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                raw_events = b"".join(
                    json.dumps(event, separators=(",", ":")).encode() + b"\n"
                    for event in events
                )
                return ProviderTurn(
                    thread_id=observed.thread_id,
                    provider_tokens=observed.provider_tokens,
                    candidates=observed.candidates,
                    candidate_sha256s=observed.candidate_sha256s,
                    raw_events=raw_events,
                    raw_events_sha256=sha256(raw_events).hexdigest(),
                    terminal_message=events[-2]["item"]["text"],
                    terminal_message_count=1,
                    normalization="single_exact",
                    reference_bundle=observed.reference_bundle,
                    tool_activity=(
                        ProviderAuxiliaryActivity(
                            item_id=f"command-{request.turn}",
                            item_type="command_execution",
                            status="completed",
                        ),
                        ProviderAuxiliaryActivity(
                            item_id=f"file-{request.turn}",
                            item_type="file_change",
                            status="completed",
                        ),
                    ),
                )

        lab = Lab(ROOT)
        lock = lab.preflight(ROOT / "contracts/studies/artifact-optimization-ralph-template.json")
        resolved = lock.document["resolved_inputs"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            campaign = _execute(lab,
                lock,
                Path(directory).resolve() / "evidence",
                provider=OptimizationProvider(),
                environments={
                    name: FakeEnvironment(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            report = lab.audit(campaign)

        self.assertTrue(report.campaign_complete)
        self.assertTrue(report.semantic_replay_passed)
        self.assertIsNone(report.estimand)
        self.assertFalse(report.estimand_available)
        self.assertIsNone(report.estimate)
        self.assertIsNone(report.uncertainty)
        self.assertTrue(report.descriptive["artifact_optimization_complete"])
        self.assertEqual(
            set(report.descriptive["promoted_artifacts"]),
            {"open_cake-1", "direct_cuda-1"},
        )
        self.assertTrue(
            all(
                artifact["turn"] == 2
                for artifact in report.descriptive["promoted_artifacts"].values()
            )
        )
        self.assertIsNone(report.estimand)
        self.assertFalse(report.estimand_available)
        self.assertIsNone(report.estimate)
        self.assertIsNone(report.uncertainty)
        self.assertNotIn("qualification_rate", report.descriptive)
        self.assertNotIn("median_confirmed_latency_ms", report.descriptive)
        self.assertNotIn("paired_runs", report.descriptive)
        self.assertEqual(
            {item.reason for item in report.run_inclusion},
            {"artifact_optimization_not_scientific_data"},
        )
        self.assertTrue(
            all(
                not item.qualification_endpoint_included
                and not item.conditional_performance_included
                for item in report.run_inclusion
            )
        )

    def test_system_qualification_requires_a_replayed_evaluation_in_every_run(self) -> None:
        class RejectingEnvironment(FakeEnvironment):
            def build(self, submission):
                return EnvironmentResult(
                    "rejected",
                    submission.sha256,
                    None,
                    {"stage": "compile", "passed": False},
                )

        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-system-qualification-ralph-template.json"
        )
        resolved = lock.document["resolved_inputs"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            campaign = _execute(lab,
                lock,
                Path(directory).resolve() / "evidence",
                provider=FakeProvider(),
                environments={
                    name: RejectingEnvironment(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            report = lab.audit(campaign)

        self.assertTrue(report.campaign_complete)
        self.assertTrue(report.archive_integrity_passed)
        self.assertTrue(report.semantic_replay_passed)
        self.assertFalse(report.system_qualification_passed)
        self.assertFalse(report.estimand_available)
        self.assertTrue(
            all(
                run["evaluation_receipt_count"] == 0
                for run in report.descriptive["runs"]
            )
        )

    def test_command_broker_evaluator_retains_raw_job_and_receipt_artifacts(self) -> None:
        workload_path = ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        workload = json.loads(workload_path.read_text())
        workload_sha = sha256(
            json.dumps(
                workload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        protocol = {"case_id": "headline_b32", "kind": "fixture"}
        protocol_sha = sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payloads = {
            "cubin": b"cubin",
            "launch_manifest": b"manifest",
        }
        candidate = LaunchableCandidate(
            candidate_sha256="a" * 64,
            target="sm_100a",
            entry_point="kernel",
            artifact_roles={
                role: sha256(payload).hexdigest() for role, payload in payloads.items()
            },
            launch_spec_sha256=sha256(payloads["launch_manifest"]).hexdigest(),
            artifact_payloads=payloads,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text(
                "import argparse,json,pathlib,sys\n"
                "p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');a=p.parse_args()\n"
                "r=pathlib.Path(a.request).parent\n"
                "for n,v in [('correctness.json',{'passed':True}),('launch.json',{'calls':1}),('timing.json',{'cohorts_ms':[[1.0]]})]: (r/n).write_text(json.dumps(v))\n"
                "o={'schema_version':1,'job_id':'gpuq-000000000000','mode':'exclusive','admitted':True,'error':None,'failure_class':None,'counters':{'compiler_invocations':0,'module_loads':1,'preflight_calls':1,'kernel_calls':1,'timing_samples':125,'fallback_calls':0},'receipt':{'correctness_passed':True,'correctness':{'passed':True},'kernel_calls':1,'fallback_calls':0,'timing':{'measurement_quality_passed':True,'pooled_median_ms':1.0},'artifacts':{'correctness_output':'correctness.json','launch_receipt':'launch.json','timing_samples':'timing.json'}}}\n"
                "print('[gpu-run] accepted job gpuq-000000000001 label=fixture mode=exclusive gpus=1',file=sys.stderr)\n"
                "pathlib.Path(a.output).write_text(json.dumps(o,sort_keys=True,separators=(',',':')))\n"
            )
            submitter = CommandBrokerSubmitter(
                command=(sys.executable, str(worker)),
                workload_path=workload_path,
                workload_sha256=workload_sha,
                protocol_sha256=protocol_sha,
                cwd=root,
                executor=ExecutorRevision.load(
                    ROOT,
                    ROOT
                    / json.loads(
                        (ROOT / "inventory/EXECUTOR_REVISIONS.json").read_text(
                            encoding="utf-8"
                        )
                    )["current"]["path"],
                ),
                service_user=pwd.getpwuid(os.geteuid()).pw_name,
                service_group=grp.getgrgid(os.getegid()).gr_name,
            )
            logical = BoundedBrokerEvaluator(protocol, submitter).evaluate(
                candidate,
                case_id="headline_b32",
                purpose="search",
            )

        self.assertIsNotNone(logical.final_receipt)
        self.assertEqual(len(logical.attempts), 1)
        self.assertEqual(logical.attempts[0].job_id, "gpuq-000000000001")
        self.assertEqual(
            set(logical.attempts[0].artifact_payloads),
            {"broker_record", "stdout", "stderr", "evaluator_result"},
        )
        self.assertEqual(
            set(logical.final_receipt.artifact_payloads),
            {"correctness_output", "launch_receipt", "timing_samples"},
        )

    def test_missing_broker_result_retains_process_output(self) -> None:
        workload_path = ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        workload = json.loads(workload_path.read_text())
        workload_sha = sha256(
            json.dumps(
                workload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        protocol = {"case_id": "headline_b32", "kind": "fixture"}
        protocol_sha = sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payloads = {"cubin": b"cubin", "launch_manifest": b"manifest"}
        candidate = LaunchableCandidate(
            candidate_sha256="a" * 64,
            target="sm_100a",
            entry_point="kernel",
            artifact_roles={
                role: sha256(payload).hexdigest()
                for role, payload in payloads.items()
            },
            launch_spec_sha256=sha256(payloads["launch_manifest"]).hexdigest(),
            artifact_payloads=payloads,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text(
                "import sys\n"
                "print('worker-out')\n"
                "print('worker-err', file=sys.stderr)\n"
                "raise SystemExit(3)\n"
            )
            submitter = CommandBrokerSubmitter(
                command=(sys.executable, str(worker)),
                workload_path=workload_path,
                workload_sha256=workload_sha,
                protocol_sha256=protocol_sha,
                cwd=root,
                executor=ExecutorRevision.load(
                    ROOT,
                    ROOT
                    / json.loads(
                        (ROOT / "inventory/EXECUTOR_REVISIONS.json").read_text(
                            encoding="utf-8"
                        )
                    )["current"]["path"],
                ),
                service_user=pwd.getpwuid(os.geteuid()).pw_name,
                service_group=grp.getgrgid(os.getegid()).gr_name,
            )
            with self.assertRaises(RunProtocolFault) as raised:
                submitter.submit(
                    candidate,
                    case_id="headline_b32",
                    purpose="search",
                    attempt=1,
                )

        self.assertEqual(raised.exception.protocol_adherence, "broker_fault")
        self.assertEqual(
            raised.exception.artifact_payloads,
            {"broker_stdout": b"worker-out\n", "broker_stderr": b"worker-err\n"},
        )

    def test_preflight_rejects_a_draft_compiler_revision(self) -> None:
        study = json.loads((ROOT / "contracts/studies/matched-search-infrastructure-template.json").read_text())
        draft = json.loads((ROOT / "compiler/revision.json").read_text())
        draft_sha256 = sha256(
            json.dumps(
                draft,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        study["arms"]["open_cake"]["compiler_revision"] = {
            "path": "compiler/revision.json",
            "canonical_sha256": draft_sha256,
        }
        study["state"] = "frozen"
        study["execution"]["executor_revision"] = {
            field: value
            for field, value in json.loads(
                (ROOT / "inventory/EXECUTOR_REVISIONS.json").read_text(
                    encoding="utf-8"
                )
            )["current"].items()
            if field in {"executor_id", "path", "canonical_sha256"}
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(ValueError, "released"):
                Lab(ROOT).preflight(path)

    def test_preflight_rejects_an_unsupported_analysis_plan(self) -> None:
        study = json.loads((ROOT / "contracts/studies/matched-search-infrastructure-template.json").read_text())
        study["analysis_plan"]["contrast"] = "unsupported_nonsense"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(ValueError, "analysis_plan"):
                Lab(ROOT).preflight(path)

    def test_preflight_rejects_an_unknown_event_vocabulary(self) -> None:
        study = json.loads(
            (
                ROOT
                / "contracts/studies/matched-search-infrastructure-template.json"
            ).read_text()
        )
        study["evidence"]["event_vocabulary"] = "unpublished_matched_events"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(ValueError, "study.evidence"):
                Lab(ROOT).preflight(path)

    def test_lab_owns_two_turn_resume_budget_evaluation_and_terminal(self) -> None:
        lab = Lab(ROOT)
        lock = lab.preflight(ROOT / "contracts/studies/matched-search-infrastructure-template.json")
        provider = FakeProvider()
        resolved = lock.document["resolved_inputs"]
        arm_environments = resolved["arm_environments"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as parent:
            campaign = _execute(lab,
                lock,
                Path(parent).resolve() / "campaign-evidence",
                provider=provider,
                environments={
                    "open_cake": FakeEnvironment(
                        "open_cake", arm_environments["open_cake"]
                    ),
                    "direct_cuda": FakeEnvironment(
                        "direct_cuda", arm_environments["direct_cuda"]
                    ),
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            report = lab.audit(campaign)

        self.assertEqual([request.run_id for request in provider.requests[::2]], list(lock.run_order))
        self.assertEqual(len(provider.requests), 12)
        self.assertTrue(all(request.thread_id is None for request in provider.requests[::2]))
        self.assertTrue(all(request.thread_id is not None for request in provider.requests[1::2]))
        self.assertEqual(report.claim_scope, "scientific_matched_search")
        self.assertIsNone(report.system_qualification_passed)
        self.assertTrue(report.estimand_available)
        self.assertTrue(report.semantic_replay_passed)
        self.assertEqual(
            report.estimate["median_confirmed_latency_ms"],
            {"open_cake": 1.0, "direct_cuda": 2.0},
        )
        self.assertAlmostEqual(report.estimate["ratio_of_arm_medians"], 2.0)
        self.assertEqual(report.estimate["qualification_rate_difference"], 0.0)
        self.assertEqual({item.reason for item in report.run_inclusion}, {"included"})

    def test_two_part_estimand_allows_an_observed_candidate_failure(self) -> None:
        """A negative authoring result is data, not an external missing Run."""

        class RejectOpenCakeOne(FakeEnvironment):
            def build(self, submission):
                if json.loads(submission.payload)["run_id"] == "open_cake-1":
                    return EnvironmentResult(
                        "rejected",
                        submission.sha256,
                        None,
                        {"stage": "correctness", "passed": False},
                    )
                return super().build(submission)

        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        )
        resolved = lock.document["resolved_inputs"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            campaign = _execute(lab,
                lock,
                Path(directory).resolve() / "evidence",
                provider=FakeProvider(),
                environments={
                    name: RejectOpenCakeOne(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            report = lab.audit(campaign)

        self.assertTrue(report.semantic_replay_passed)
        self.assertEqual(report.missing_run_count, 0)
        self.assertTrue(report.estimand_available)
        self.assertAlmostEqual(
            report.estimate["qualification_rate_difference"], -1 / 3
        )
        self.assertEqual(
            report.descriptive["endpoint_counts"]["open_cake"],
            {"prescheduled": 3, "observed": 3, "qualified": 2, "missing": 0},
        )
        inclusion = {item.run_id: item for item in report.run_inclusion}
        self.assertTrue(inclusion["open_cake-1"].qualification_endpoint_included)
        self.assertFalse(inclusion["open_cake-1"].conditional_performance_included)

    def test_two_part_estimand_does_not_count_an_external_fault_as_failure(self) -> None:
        class FaultDirectCudaTwo(FakeProvider):
            def turn(self, request):
                if request.run_id == "direct_cuda-2":
                    raise RunProtocolFault("provider_fault", "fixture provider fault")
                return super().turn(request)

        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        )
        resolved = lock.document["resolved_inputs"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            campaign = _execute(lab,
                lock,
                Path(directory).resolve() / "evidence",
                provider=FaultDirectCudaTwo(),
                environments={
                    name: FakeEnvironment(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            report = lab.audit(campaign)

        self.assertTrue(report.semantic_replay_passed)
        self.assertEqual(report.missing_run_count, 1)
        self.assertFalse(report.estimand_available)
        self.assertIsNone(report.estimate)
        self.assertEqual(
            report.descriptive["endpoint_counts"]["direct_cuda"],
            {"prescheduled": 3, "observed": 2, "qualified": 2, "missing": 1},
        )
        self.assertEqual(
            report.descriptive["qualification_rate_among_observed"]["direct_cuda"],
            1.0,
        )

    def test_missing_archive_is_counted_as_missing_data(self) -> None:
        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
            for run_id in lock.run_order[:-1]:
                run = evidence.start_run(
                    run_id,
                    authority_sha256=lock.canonical_sha256,
                    authority=lock.document,
                )
                run.seal(
                    protocol_adherence="provider_fault",
                    endpoint_observation="missing",
                )
            report = lab.audit(lab.reference_campaign(lock, evidence.root))

        self.assertFalse(report.campaign_complete)
        self.assertEqual(report.missing_run_count, 6)

    def test_accepted_candidate_carries_environment_findings_into_the_next_turn(self) -> None:
        # A report is by construction attached to a candidate that builds, so the accepted
        # path is the only way one can reach the agent. The Lab used to overwrite the
        # Environment's feedback with the measurement alone, which left every report the
        # verifier produced unobservable to the author that could act on it.
        from open_cake_ir.compiler import Compiler
        from open_cake_ir.lab.environments import OpenCakeEnvironment

        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        assessment = compiler.assess_file(ROOT / "corpus/schedules/tinygemm2-stage4-split-k.json")
        diagnostics = OpenCakeEnvironment._finding_rows(assessment)
        self.assertTrue(any(item["severity"] == "hint" for item in diagnostics))

        class ReportingEnvironment(FakeEnvironment):
            def build(self, submission):
                result = super().build(submission)
                return EnvironmentResult(
                    result.disposition,
                    result.submission_sha256,
                    result.launchable,
                    {
                        "stage": "built",
                        "findings": diagnostics,
                    },
                )

        lab = Lab(ROOT)
        lock = lab.preflight(ROOT / "contracts/studies/matched-search-infrastructure-template.json")
        provider = FakeProvider()
        arm_environments = lock.document["resolved_inputs"]["arm_environments"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as parent:
            campaign = _execute(lab,
                lock,
                Path(parent).resolve() / "campaign-evidence",
                provider=provider,
                environments={
                    arm: ReportingEnvironment(arm, arm_environments[arm])
                    for arm in ("open_cake", "direct_cuda")
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )

        resumed = [request for request in provider.requests if request.thread_id is not None]
        self.assertTrue(resumed)
        for request in resumed:
            self.assertEqual(request.feedback["kind"], "evaluation")
            self.assertEqual(request.feedback["findings"], diagnostics)

    def test_semantic_replay_rejects_raw_broker_counter_that_differs_from_ledger(self) -> None:
        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        )
        resolved = lock.document["resolved_inputs"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            campaign = _execute(lab,
                lock,
                Path(directory).resolve() / "evidence",
                provider=FakeProvider(),
                environments={
                    name: FakeEnvironment(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                    raw_kernel_calls=2,
                ),
            )
            report = lab.audit(campaign)

        self.assertTrue(report.archive_integrity_passed)
        self.assertFalse(report.semantic_replay_passed)
        self.assertFalse(report.estimand_available)

    def test_semantic_replay_rejects_an_extra_archived_provider_item(self) -> None:
        class ExtraItemProvider(FakeProvider):
            def turn(self, request):
                observed = super().turn(request)
                events = [
                    json.loads(line) for line in observed.raw_events.splitlines()
                ]
                events.insert(
                    -1,
                    {
                        "type": "item.updated",
                        "item": {"id": "item-extra", "type": "agent_message"},
                    },
                )
                raw_events = b"".join(
                    json.dumps(event, separators=(",", ":")).encode() + b"\n"
                    for event in events
                )
                return ProviderTurn(
                    thread_id=observed.thread_id,
                    provider_tokens=observed.provider_tokens,
                    candidates=observed.candidates,
                    candidate_sha256s=observed.candidate_sha256s,
                    raw_events=raw_events,
                    raw_events_sha256=sha256(raw_events).hexdigest(),
                    terminal_message=observed.terminal_message,
                    terminal_message_count=observed.terminal_message_count,
                    normalization=observed.normalization,
                    reference_bundle=observed.reference_bundle,
                )

        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        )
        resolved = lock.document["resolved_inputs"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            campaign = _execute(lab,
                lock,
                Path(directory).resolve() / "evidence",
                provider=ExtraItemProvider(),
                environments={
                    name: FakeEnvironment(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            report = lab.audit(campaign)

        self.assertTrue(report.archive_integrity_passed)
        self.assertFalse(report.semantic_replay_passed)
        self.assertFalse(report.estimand_available)

    def test_invalid_provider_set_identity_stops_before_build_or_gpu(self) -> None:
        """The Lab owns the authoring bound and the raw provider-event seal."""

        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        )
        resolved = lock.document["resolved_inputs"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        for failure in ("raw_events_sha256", "candidate_count"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                builds: list[str] = []

                class RecordingEnvironment(FakeEnvironment):
                    def build(self, submission):
                        builds.append(submission.sha256)
                        return super().build(submission)

                class InvalidProvider(FakeProvider):
                    def turn(self, request):
                        observed = super().turn(request)
                        if failure == "raw_events_sha256":
                            return dataclasses.replace(
                                observed, raw_events_sha256="f" * 64
                            )
                        payloads = tuple(
                            json.dumps(
                                {
                                    "run_id": request.run_id,
                                    "turn": request.turn,
                                    "variant": index,
                                },
                                sort_keys=True,
                            ).encode()
                            for index in range(
                                request.maximum_candidates_per_turn + 1
                            )
                        )
                        return dataclasses.replace(
                            observed,
                            candidates=payloads,
                            candidate_sha256s=tuple(
                                sha256(payload).hexdigest() for payload in payloads
                            ),
                        )

                evaluator = FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                )
                campaign = _execute(lab,
                    lock,
                    Path(directory).resolve() / "evidence",
                    provider=InvalidProvider(),
                    environments={
                        name: RecordingEnvironment(name, document)
                        for name, document in resolved["arm_environments"].items()
                    },
                    evaluator=evaluator,
                )
                report = lab.audit(campaign)

                self.assertEqual(builds, [])
                self.assertEqual(evaluator.calls, 0)
                self.assertTrue(report.archive_integrity_passed)
                self.assertTrue(report.semantic_replay_passed)
                self.assertEqual(
                    {audit.protocol_adherence for audit in report.run_audits},
                    {"provider_fault"},
                )

    def test_r42_turn_discrete_missing_cell_keeps_estimand_unavailable(self) -> None:
        lab = Lab(ROOT)
        lock = lab.preflight(ROOT / "contracts/studies/matched-search-infrastructure-template.json")
        legacy = json.loads(
            (ROOT / "evidence/historical/legacy/r42-index.json").read_text()
        )
        latencies = {}
        for pair in legacy["descriptive_results"]["paired_available_runs"]:
            repetition = pair["repetition"]
            latencies[f"open_cake-{repetition}"] = pair["cake_best_latency_ms"]
            if pair["cuda_best_latency_ms"] is not None:
                latencies[f"direct_cuda-{repetition}"] = pair["cuda_best_latency_ms"]
        self.assertFalse(legacy["checkpoint"]["representation_effect_available"])
        self.assertEqual(
            legacy["checkpoint"]["cuda_repetition_3"]["turn_1_provider_tokens"],
            103579,
        )
        self.assertEqual(
            legacy["checkpoint"]["cuda_repetition_3"]["turn_2_provider_tokens"],
            329934,
        )
        with tempfile.TemporaryDirectory() as directory:
            evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
            for run_id in lock.run_order:
                run = evidence.start_run(
                    run_id,
                    authority_sha256=lock.canonical_sha256,
                    authority=lock.document,
                )
                if run_id == "direct_cuda-3":
                    run.seal(
                        protocol_adherence="adhered",
                        endpoint_observation="no_qualified_candidate",
                        endpoint={"qualified_by_budget": False, "budget": 150000},
                    )
                else:
                    run.seal(
                        protocol_adherence="adhered",
                        endpoint_observation="qualified",
                        endpoint={
                            "qualified_by_budget": True,
                            "budget": 150000,
                            "best_candidate_sha256": "a" * 64,
                            "best_confirmed_latency_ms": latencies[run_id],
                        },
                    )
            report = lab.audit(lab.reference_campaign(lock, evidence.root))

        self.assertFalse(report.estimand_available)
        self.assertEqual(report.missing_run_count, 0)
        self.assertIsNone(report.estimate)
        self.assertEqual(report.descriptive["paired_runs"][2]["direct_cuda_latency_ms"], None)
        inclusion = {item.run_id: item.reason for item in report.run_inclusion}
        self.assertEqual(
            inclusion["direct_cuda-3"], "observed_no_qualified_candidate_at_checkpoint"
        )
        direct_three = next(
            item for item in report.run_inclusion if item.run_id == "direct_cuda-3"
        )
        self.assertTrue(direct_three.qualification_endpoint_included)
        self.assertFalse(direct_three.conditional_performance_included)

    def test_candidate_rejection_is_observed_and_later_turn_cannot_backfill(self) -> None:
        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        )
        resolved = lock.document["resolved_inputs"]

        class RejectFirst(FakeEnvironment):
            def build(self, submission):
                if json.loads(submission.payload)["turn"] == 1:
                    return EnvironmentResult(
                        "rejected",
                        submission.sha256,
                        None,
                        {"stage": "correctness", "passed": False},
                    )
                return super().build(submission)

        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            campaign = _execute(lab,
                lock,
                Path(directory).resolve() / "evidence",
                provider=FakeProvider(),
                environments={
                    name: RejectFirst(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            report = lab.audit(campaign)

        self.assertTrue(report.semantic_replay_passed)
        self.assertFalse(report.estimand_available)
        self.assertEqual(report.missing_run_count, 0)
        self.assertEqual(
            {audit.endpoint_observation for audit in report.run_audits},
            {"no_qualified_candidate"},
        )
        self.assertTrue(
            all(item.qualification_endpoint_included for item in report.run_inclusion)
        )

    def test_protocol_failures_are_intact_but_not_included(self) -> None:
        lab = Lab(ROOT)
        lock = lab.preflight(ROOT / "contracts/studies/matched-search-infrastructure-template.json")
        with tempfile.TemporaryDirectory() as directory:
            evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
            for run_id in lock.run_order:
                run = evidence.start_run(
                    run_id,
                    authority_sha256=lock.canonical_sha256,
                    authority=lock.document,
                )
                run.seal(protocol_adherence="provider_fault", endpoint_observation="missing")
            report = lab.audit(lab.reference_campaign(lock, evidence.root))

        self.assertTrue(report.campaign_complete)
        self.assertTrue(report.archive_integrity_passed)
        self.assertFalse(report.estimand_available)
        self.assertEqual(report.missing_run_count, 6)
        self.assertEqual({item.reason for item in report.run_inclusion}, {"protocol_deviation"})

    def test_contamination_uses_the_same_terminal_schema_without_replacement(self) -> None:
        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        )
        resolved = lock.document["resolved_inputs"]

        class ContaminatedProvider(FakeProvider):
            def turn(self, request):
                raise RunProtocolFault(
                    "contamination",
                    "workspace changed before Turn",
                    artifact_payloads={
                        "provider_stderr": b"OPENAI_API_KEY=must-not-enter-evidence"
                    },
                )

        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            campaign = _execute(lab,
                lock,
                Path(directory).resolve() / "evidence",
                provider=ContaminatedProvider(),
                environments={
                    name: FakeEnvironment(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            report = lab.audit(campaign)

        self.assertTrue(report.archive_integrity_passed)
        self.assertEqual(
            {audit.protocol_adherence for audit in report.run_audits},
            {"contamination"},
        )
        self.assertEqual(report.missing_run_count, 6)

    def test_checkpoints_do_not_backfill_a_crossing_turn(self) -> None:
        r40_failure = project_checkpoints(
            turns=(TurnObservation(1, 40969, "a" * 64, True, 1.1),),
            checkpoints=(50000, 100000, 150000),
            terminal_provider_tokens=40969,
        )
        self.assertEqual([checkpoint.state for checkpoint in r40_failure], ["unreached"] * 3)

        r42_cuda_three = project_checkpoints(
            turns=(
                TurnObservation(1, 103579, "b" * 64, False, None),
                TurnObservation(2, 329934, "c" * 64, True, 5.054870),
            ),
            checkpoints=(150000,),
            terminal_provider_tokens=329934,
        )
        self.assertEqual(r42_cuda_three[0].state, "reached_no_qualified_candidate")
        self.assertIsNone(r42_cuda_three[0].best_candidate_sha256)

    def test_portfolio_semantic_replay_keeps_correctness_separate_from_timing(self) -> None:
        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/flash-kmeans-r45-portfolio-reconstruction-template.json"
        )
        from open_cake_ir.compiler import Compiler
        from open_cake_ir.evaluation import (
            PortfolioArtifact,
            PortfolioCaseObservation,
            WorkloadContract,
            evaluate_portfolio_observations,
        )
        from open_cake_ir.lab import KernelSeed, lower_specialists

        workload = WorkloadContract.load(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        seed = KernelSeed.load(
            ROOT, ROOT / "contracts/kernel-seeds/r42-cake-r1-turn1-v3.json"
        )
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
        cases = {
            case_id: workload.case(case_id)["shape"]
            for case_id in ("headline_b32", "b32_smoke", "public_b1")
        }
        lowerings = lower_specialists(compiler, seed, cases)
        candidates = {}
        for item in lowerings:
            source = item.lowering.source.encode()
            manifest = json.dumps(
                {"case_id": item.case_id, "grid": item.assessment.analysis["grid"]},
                sort_keys=True,
            ).encode()
            payloads = {
                "lowered_source": source,
                "compiler_expanded_source": b"expanded:" + source,
                "ptx": b"ptx:" + item.case_id.encode(),
                "cubin": b"cubin:" + item.case_id.encode(),
                "launch_manifest": manifest,
            }
            candidates[item.case_id] = LaunchableCandidate(
                candidate_sha256=item.lowering.schedule_sha256,
                target="sm_100a",
                entry_point=item.lowering.route.entry_point,
                artifact_roles={
                    role: sha256(payload).hexdigest()
                    for role, payload in payloads.items()
                },
                launch_spec_sha256=sha256(manifest).hexdigest(),
                artifact_payloads=payloads,
            )
        artifact = PortfolioArtifact.build(workload, seed.canonical_sha256, candidates)
        stable = tuple(tuple([1.0] * 25) for _ in range(5))
        unstable = tuple(
            tuple([1.0, 2.0] * 12 + [1.0]) if index == 0 else tuple([1.0] * 25)
            for index in range(5)
        )
        observations = {}
        for case_id in candidates:
            observations[case_id] = PortfolioCaseObservation(
                direct_preflight_correct=True,
                dispatcher_preflight_correct=True,
                postflight_correct=True,
                kernel_cohorts_ms=stable,
                dispatcher_cohorts_ms=(stable if case_id == "headline_b32" else unstable),
                correctness_receipts={
                    "direct_preflight": {"passed": True},
                    "dispatcher_preflight": {"passed": True},
                    "postflight": {"passed": True},
                },
                candidate_record_sha256=candidates[case_id].canonical_sha256,
                cubin_sha256=candidates[case_id].artifact_roles["cubin"],
                launch_spec_sha256=candidates[case_id].launch_spec_sha256,
                module_admission={
                    "candidate_record_sha256": candidates[case_id].canonical_sha256,
                    "cubin_sha256": candidates[case_id].artifact_roles["cubin"],
                    "launch_spec_sha256": candidates[case_id].launch_spec_sha256,
                    "module_loaded": True,
                    "gpu_uuid": "GPU-fixture",
                    "broker_job_id": "gpuq-000000000001",
                },
            )
        route_counts = {
            "automatic_retries": 0,
            "compiler_invocations": 3,
            "module_loads": 3,
            "module_unloads": 3,
            "direct_preflight_calls": 3,
            "dispatcher_preflight_calls": 3,
            "cupti_candidate_calls": 540,
            "host_dispatch_calls": 450,
            "l2_flush_calls": 450,
            "dispatcher_postflight_calls": 3,
            "unsupported_probes": 1,
            "candidate_kernel_calls": 999,
            "dispatcher_kernel_calls": 456,
            "fallback_calls": 0,
            "selections": {case_id: 152 for case_id in observations},
        }
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        receipt = evaluate_portfolio_observations(
            artifact,
            observations,
            evaluation_protocol_sha256=protocol_sha256,
            route_counts=route_counts,
            unsupported_kernel_call_delta=0,
        )

        class Assay:
            def __init__(self, artifact_value, protocol_value, receipt_value):
                self.artifact = artifact_value
                self.protocol_sha256 = protocol_value
                self.receipt = receipt_value

            def prepare(self, campaign_lock):
                return self.artifact

            def evaluate(self, campaign_lock):
                return self.receipt

        with tempfile.TemporaryDirectory() as directory:
            campaign = lab.execute_portfolio(
                lock,
                Path(directory).resolve() / "portfolio-evidence",
                assay=Assay(artifact, protocol_sha256, receipt),
            )
            report = lab.audit_portfolio(campaign)

        self.assertTrue(report.claim_view.heldout_correctness_supported, report)
        self.assertTrue(report.semantic_replay_passed)
        self.assertTrue(report.claim_view.dispatcher_correctness_supported)
        self.assertTrue(report.claim_view.stable_kernel_performance_supported)
        self.assertFalse(
            report.claim_view.stable_heldout_dispatcher_performance_supported
        )
        self.assertFalse(report.claim_view.arbitrary_shape_generalization_supported)
        self.assertFalse(report.claim_view.serving_supported)
        self.assertFalse(report.claim_view.paper_result_reproduced)

    def test_preflight_resolves_variant_specific_inputs_into_one_lock(self) -> None:
        matched = Lab(ROOT).preflight(
            ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        )
        portfolio = Lab(ROOT).preflight(
            ROOT / "contracts/studies/flash-kmeans-r45-portfolio-reconstruction-template.json"
        )

        self.assertEqual(matched.study_kind, "matched_search")
        self.assertEqual(matched.experimental_unit, "run")
        self.assertEqual(len(matched.run_order), 6)
        self.assertEqual(portfolio.study_kind, "portfolio")
        self.assertEqual(portfolio.experimental_unit, "case_route")
        self.assertEqual(portfolio.run_order, ("portfolio-1",))


if __name__ == "__main__":
    unittest.main()


class CandidateSetFilterTest(unittest.TestCase):
    """Every candidate is built and sealed; only the bounded subset is evaluated.

    This is the paper's pre-GPU filter. Compile time is spent on the whole set precisely
    so that device time is not, so a Turn that writes three candidates must show three
    builds and three sealed objects, while the evaluated subset must follow the filter
    order rather than provider arrival order.
    """

    def test_the_set_is_built_and_ordered_before_one_is_evaluated(self) -> None:
        built: list[str] = []

        class CostedEnvironment(FakeEnvironment):
            unscored_variant: int | None = None

            def build(self, submission):
                built.append(submission.sha256)
                result = super().build(submission)
                # The last candidate written is the cheapest, so ordering by cost and
                # ordering by arrival disagree -- which is what makes the assertion mean
                # something.
                index = json.loads(submission.payload)["variant"]
                from open_cake_ir.compiler.ranking import Cost

                if index == 1:
                    return EnvironmentResult(
                        "rejected",
                        result.submission_sha256,
                        None,
                        {
                            "stage": "verification",
                            "findings": [
                                {
                                    "code": "FIXTURE_REJECTED",
                                    "blocks_acceptance": True,
                                }
                            ],
                        },
                    )

                return EnvironmentResult(
                    result.disposition,
                    result.submission_sha256,
                    result.launchable,
                    result.feedback,
                    result.artifact_payloads,
                    cost=(
                        None
                        if index == self.unscored_variant
                        else Cost(
                            schedule_id=f"v{index}",
                            ctas=1,
                            ctas_per_multiprocessor=1,
                            binding_resource="registers",
                            device_fill=0.25 * (index + 1),
                        )
                    ),
                )

        class SetProvider(CandidateSetFakeProvider):
            def turn(self, request):
                observed = super().turn(request)
                payloads = tuple(
                    json.dumps(
                        {"run_id": request.run_id, "turn": request.turn, "variant": i},
                        sort_keys=True,
                    ).encode()
                    for i in range(3)
                )
                return ProviderTurn(
                    thread_id=observed.thread_id,
                    provider_tokens=observed.provider_tokens,
                    candidates=payloads,
                    candidate_sha256s=tuple(sha256(p).hexdigest() for p in payloads),
                    raw_events=observed.raw_events,
                    raw_events_sha256=observed.raw_events_sha256,
                    terminal_message=observed.terminal_message,
                    terminal_message_count=observed.terminal_message_count,
                    normalization=observed.normalization,
                    reference_bundle=observed.reference_bundle,
                )

        lab = Lab(ROOT)
        source = ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        _enable_candidate_set(document, 3)
        with tempfile.TemporaryDirectory() as parent:
            study_path = Path(parent) / "candidate-set-study.json"
            study_path.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            lock = lab.preflight(study_path)
            arms = lock.document["resolved_inputs"]["arm_environments"]
            protocol_sha256 = sha256(
                json.dumps(
                    lock.document["evaluation_protocol"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            campaign = _execute(lab,
                lock,
                Path(parent).resolve() / "campaign-evidence",
                provider=SetProvider(),
                environments={
                    arm: CostedEnvironment(arm, arms[arm]) for arm in arms
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            store = EvidenceStore.open(campaign.evidence_root)
            run_id = next(iter(lock.run_order))
            run_events = store.replay_events(run_id)
            events = [
                event
                for event in run_events
                if event["kind"] == "candidate_set_filtered"
            ]
            rejections = [
                event
                for event in run_events
                if event["kind"] == "candidate_rejected"
            ]
            evaluated_by_turn: dict[str, dict[int, list[str]]] = {
                "search": {},
                "attribution": {},
            }
            selected_by_turn: dict[int, str] = {}
            for event in run_events:
                payload = event["payload"]
                if event["kind"] == "candidate_evaluated" and payload[
                    "purpose"
                ] in evaluated_by_turn:
                    evaluated_by_turn[payload["purpose"]].setdefault(
                        payload["turn"], []
                    ).append(payload["candidate_sha256"])
                elif event["kind"] == "candidate_selected":
                    selected_by_turn[payload["turn"]] = payload[
                        "candidate_sha256"
                    ]
            campaign_wide = sum(
                1
                for identifier in lock.run_order
                for event in store.replay_events(identifier)
                if event["kind"] == "candidate_set_filtered"
            )
            report = lab.audit(campaign)
            lock_path = Path(parent) / "campaign.lock.json"
            lock_path.write_text(
                json.dumps(lock.document, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            fresh_process = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; sys.path.insert(0, sys.argv[3] + '/src'); "
                        "from open_cake_ir.lab import CampaignLock, Lab; "
                        "lock=CampaignLock.load(sys.argv[1]); lab=Lab(sys.argv[3]); "
                        "report=lab.audit(lab.reference_campaign(lock, sys.argv[2])); "
                        "raise SystemExit(0 if report.semantic_replay_passed else 1)"
                    ),
                    str(lock_path),
                    str(campaign.evidence_root),
                    str(ROOT),
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            class TamperedSelectionReplay:
                def replay_events(self, identifier):
                    replayed = json.loads(json.dumps(store.replay_events(identifier)))
                    alternatives = next(
                        event["payload"]["order"]
                        for event in replayed
                        if event["kind"] == "candidate_set_filtered"
                    )
                    for event in replayed:
                        if event["kind"] == "candidate_selected":
                            event["payload"]["candidate_sha256"] = alternatives[1][
                                "candidate_sha256"
                            ]
                            break
                    return replayed

                def read_object(self, reference):
                    return store.read_object(reference)

            audit = store.audit_run(run_id)
            tampered_selection_rejected = not lab._replay_matched_run(
                TamperedSelectionReplay(), audit, lock
            )

            class MissingAttributionReplay:
                def replay_events(self, identifier):
                    replayed = json.loads(json.dumps(store.replay_events(identifier)))
                    for index, event in enumerate(replayed):
                        if (
                            event["kind"] == "candidate_evaluated"
                            and event["payload"]["purpose"] == "attribution"
                        ):
                            del replayed[index]
                            break
                    return replayed

                def read_object(self, reference):
                    return store.read_object(reference)

            missing_attribution_rejected = not lab._replay_matched_run(
                MissingAttributionReplay(), audit, lock
            )

            class ExtraEventReplay:
                def replay_events(self, identifier):
                    replayed = json.loads(json.dumps(store.replay_events(identifier)))
                    replayed.insert(
                        -2,
                        {
                            "kind": "unpublished_metric",
                            "payload": {"value": 1},
                        },
                    )
                    return replayed

                def read_object(self, reference):
                    return store.read_object(reference)

            extra_event_rejected = not lab._replay_matched_run(
                ExtraEventReplay(), audit, lock
            )

            class ChangedRunStartReplay:
                def replay_events(self, identifier):
                    replayed = json.loads(json.dumps(store.replay_events(identifier)))
                    started = next(
                        event for event in replayed if event["kind"] == "run_started"
                    )
                    started["payload"]["assigned_arm"] = "tampered_arm"
                    return replayed

                def read_object(self, reference):
                    return store.read_object(reference)

            changed_run_start_rejected = not lab._replay_matched_run(
                ChangedRunStartReplay(), audit, lock
            )

        self.assertTrue(events)
        first = events[0]["payload"]
        self.assertEqual(first["submitted"], 3)
        self.assertEqual(first["launchable"], 2)
        # Ordered by cost, so the fullest device -- written last -- leads, and the one
        # the rejected member sorts last and retains its full rejection evidence.
        self.assertEqual(
            [row["cost"] and row["cost"]["device_fill"] for row in first["order"]],
            [0.75, 0.25, None],
        )
        self.assertEqual(len(rejections), len(events))
        self.assertTrue(
            all(event["payload"]["routed_to"] == "candidate" for event in rejections)
        )
        # And every candidate was built, not only the survivor -- counted across the whole
        # campaign, because the recording environment is shared by every run and arm.
        self.assertEqual(len(built), 3 * campaign_wide)
        self.assertTrue(report.archive_integrity_passed)
        self.assertTrue(report.semantic_replay_passed)
        self.assertEqual(fresh_process.returncode, 0, fresh_process.stderr.decode())
        self.assertTrue(tampered_selection_rejected)
        self.assertEqual(
            evaluated_by_turn["attribution"], evaluated_by_turn["search"]
        )
        self.assertTrue(
            any(
                candidate != selected_by_turn[turn]
                for turn, candidates in evaluated_by_turn["attribution"].items()
                for candidate in candidates
            )
        )
        self.assertTrue(missing_attribution_rejected)
        self.assertTrue(extra_event_rejected)
        self.assertTrue(changed_run_start_rejected)

        # The same set with its first launchable member unscored is not partially
        # reordered. Variant 2 has a cost and would lead under the former behavior;
        # variant 0 is unknown, so both launchable members retain provider order.
        CostedEnvironment.unscored_variant = 0
        try:
            with tempfile.TemporaryDirectory() as parent:
                study_path = Path(parent) / "partial-order-study.json"
                study_path.write_text(
                    json.dumps(document, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8",
                )
                lock = lab.preflight(study_path)
                arms = lock.document["resolved_inputs"]["arm_environments"]
                protocol_sha256 = sha256(
                    json.dumps(
                        lock.document["evaluation_protocol"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                campaign = _execute(lab,
                    lock,
                    Path(parent).resolve() / "partial-order-evidence",
                    provider=SetProvider(),
                    environments={
                        arm: CostedEnvironment(arm, arms[arm]) for arm in arms
                    },
                    evaluator=FakeEvaluator(
                        lock.document["evaluation_protocol"],
                        protocol_sha256,
                        lock.document["workload"]["canonical_sha256"],
                    ),
                )
                partial_store = EvidenceStore.open(campaign.evidence_root)
                partial = next(
                    event["payload"]
                    for event in partial_store.replay_events(lock.run_order[0])
                    if event["kind"] == "candidate_set_filtered"
                )
                partial_report = lab.audit(campaign)
        finally:
            CostedEnvironment.unscored_variant = None

        self.assertEqual(
            [row["cost"] and row["cost"]["device_fill"] for row in partial["order"]],
            [None, 0.75, None],
        )
        self.assertEqual(
            [row["disposition"] for row in partial["order"]],
            ["launchable", "launchable", "rejected"],
        )
        self.assertTrue(partial_report.semantic_replay_passed)

    def test_every_member_of_an_all_rejected_set_replays(self) -> None:
        class RejectAllEnvironment(FakeEnvironment):
            def build(self, submission):
                return EnvironmentResult(
                    "rejected",
                    submission.sha256,
                    None,
                    {
                        "stage": "verification",
                        "findings": [
                            {"code": "FIXTURE_REJECTED", "blocks_acceptance": True}
                        ],
                    },
                )

        class ThreeCandidateProvider(CandidateSetFakeProvider):
            def turn(self, request):
                observed = super().turn(request)
                payloads = tuple(
                    json.dumps(
                        {
                            "run_id": request.run_id,
                            "turn": request.turn,
                            "variant": index,
                        },
                        sort_keys=True,
                    ).encode()
                    for index in range(3)
                )
                return dataclasses.replace(
                    observed,
                    candidates=payloads,
                    candidate_sha256s=tuple(
                        sha256(payload).hexdigest() for payload in payloads
                    ),
                )

        lab = Lab(ROOT)
        lock = lab.preflight(
            ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        )
        resolved = lock.document["resolved_inputs"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            campaign = _execute(lab,
                lock,
                Path(directory).resolve() / "evidence",
                provider=ThreeCandidateProvider(),
                environments={
                    name: RejectAllEnvironment(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            store = EvidenceStore.open(campaign.evidence_root)
            report = lab.audit(campaign)
            run_events = store.replay_events(lock.run_order[0])

        filters = [
            event for event in run_events if event["kind"] == "candidate_set_filtered"
        ]
        rejections = [
            event for event in run_events if event["kind"] == "candidate_rejected"
        ]
        selections = [
            event for event in run_events if event["kind"] == "candidate_selected"
        ]
        self.assertTrue(report.semantic_replay_passed)
        self.assertEqual(len(rejections), 3 * len(filters))
        self.assertTrue(
            all(
                event["payload"]["reason"] == "all_candidates_rejected"
                for event in selections
            )
        )


class AttributionAssayIntegrationTest(unittest.TestCase):
    """A Study that declares attribution gets it; one that does not is unchanged.

    Profiling costs device time, so it is declared rather than assumed. What must not
    happen is the opposite -- a Campaign that declared it and silently did not run it
    would leave the loop's third stage looking complete in the contract and absent in the
    evidence.
    """

    def _run(self, *, declare_attribution: bool):
        purposes: list[str] = []

        class RecordingEvaluator(FakeEvaluator):
            def evaluate(self, candidate, *, case_id, purpose):
                purposes.append(purpose)
                if purpose != "attribution":
                    return super().evaluate(candidate, case_id=case_id, purpose=purpose)
                launch = json.dumps(
                    {"candidate_sha256": candidate.candidate_sha256, "purpose": purpose},
                    sort_keys=True,
                ).encode()
                correctness = json.dumps(
                    {"passed": True, "metrics": {"tie_aware_distance_match": True}},
                    sort_keys=True,
                ).encode()
                profile = _profile_fixture(
                    candidate.candidate_sha256, case_id, candidate.entry_point
                )
                receipt = EvaluationReceipt(
                    candidate_sha256=candidate.candidate_sha256,
                    workload_sha256=self.workload_sha256,
                    evaluation_protocol_sha256=self.protocol_sha256,
                    purpose="attribution",
                    case_id=case_id,
                    correctness_passed=True,
                    correctness={"tie_aware_distance_match": True},
                    kernel_calls=1,
                    fallback_calls=0,
                    launch_receipt_sha256=sha256(launch).hexdigest(),
                    timing=None,
                    artifact_payloads={
                        "correctness_output": correctness,
                        "launch_receipt": launch,
                        "profile": profile,
                    },
                )
                self.calls += 1
                job_id = f"gpuq-{self.calls:012x}"
                raw_result = json.dumps(
                    {
                        "schema_version": 1,
                        "job_id": job_id,
                        "mode": "exclusive",
                        "admitted": True,
                        "error": None,
                        "failure_class": None,
                        "counters": {
                            "compiler_invocations": 0,
                            "module_loads": 1,
                            "preflight_calls": 1,
                            "kernel_calls": 1,
                            "timing_samples": 0,
                            "fallback_calls": 0,
                        },
                        "receipt": {
                            "correctness_passed": True,
                            "correctness": {"tie_aware_distance_match": True},
                            "kernel_calls": 1,
                            "fallback_calls": 0,
                            "timing": None,
                            "artifacts": {
                                "correctness_output": "correctness-output.json",
                                "launch_receipt": "launch-receipt.json",
                                "profile": "profile.json",
                            },
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                attempt = BrokerAttempt(
                    job_id=job_id,
                    mode="exclusive",
                    candidate_sha256=candidate.candidate_sha256,
                    manifest_sha256=candidate.launch_spec_sha256,
                    policy_sha256=self.protocol_sha256,
                    evaluator_arguments_sha256=sha256(
                        f"{case_id}:{purpose}".encode()
                    ).hexdigest(),
                    admitted=True,
                    error=None,
                    compiler_invocations=0,
                    module_loads=1,
                    preflight_calls=1,
                    kernel_calls=1,
                    timing_samples=0,
                    fallback_calls=0,
                    receipt=receipt,
                    artifact_payloads={
                        "broker_record": raw_result,
                        "evaluator_result": raw_result,
                        "stdout": b"completed\n",
                        "stderr": b"",
                    },
                )
                return LogicalEvaluationAttempt(
                    candidate_sha256=candidate.candidate_sha256,
                    attempts=(attempt,),
                    final_receipt=receipt,
                )

        lab = Lab(ROOT)
        source = ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        current_lock = lab.preflight(source)
        document["state"] = "frozen"
        document["arms"]["open_cake"]["compiler_revision"] = dict(
            current_lock.document["compiler_revision"]
        )
        document["execution"]["executor_revision"] = dict(
            current_lock.document["execution"]["executor_revision"]
        )
        document["study_id"] += (
            "-attribution-fixture" if declare_attribution else "-legacy-feedback-fixture"
        )
        if not declare_attribution:
            # Model the bounded legacy spelling before the current successor opted in.
            # The temporary Study is frozen before preflight; no Campaign Lock is edited.
            document["evaluation_protocol"].pop("attribution_evaluation")
            for arm in document["arms"].values():
                arm["feedback"].remove("profile")
        stack = contextlib.ExitStack()
        with stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            study_path = Path(directory) / "successor.json"
            study_path.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            lock = lab.preflight(study_path)
        arms = lock.document["resolved_inputs"]["arm_environments"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        with tempfile.TemporaryDirectory() as parent:
            provider = FakeProvider()
            campaign = _execute(lab,
                lock,
                Path(parent).resolve() / "campaign-evidence",
                provider=provider,
                environments={arm: FakeEnvironment(arm, arms[arm]) for arm in arms},
                evaluator=RecordingEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            semantic_replay_passed = lab.audit(campaign).semantic_replay_passed
        resumed_profiles = [
            request.feedback.get("profile")
            for request in provider.requests
            if request.thread_id is not None
        ]
        return purposes, semantic_replay_passed, resumed_profiles

    def test_attribution_runs_only_when_the_study_declares_it(self) -> None:
        without, without_replay, without_profiles = self._run(
            declare_attribution=False
        )
        with_attribution, with_replay, with_profiles = self._run(
            declare_attribution=True
        )
        self.assertNotIn("attribution", without)
        self.assertIn("attribution", with_attribution)
        self.assertTrue(without_replay)
        self.assertTrue(with_replay)
        self.assertEqual(without_profiles, [None] * len(without_profiles))
        self.assertTrue(with_profiles)
        self.assertTrue(
            all(profile["kind"] == "ncu_kernel_attribution" for profile in with_profiles)
        )


class SearchBudgetTest(unittest.TestCase):
    """A malformed search budget is refused at the boundary, not partway through a run.

    `searches_per_turn` reaches a slice of the ranked candidates, so zero would index an
    empty list and a string would raise while the run was already under way. A run that
    faults has spent the GPU time this Lab exists to gate before spending it.
    """

    def _preflight(self, value, materiality=None, maximum_candidates=None):
        lab = Lab(ROOT)
        source = ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        if value is None:
            document["evaluation_protocol"].pop("searches_per_turn", None)
        else:
            document["evaluation_protocol"]["searches_per_turn"] = value
        document["evaluation_protocol"].pop("search_materiality_ratio", None)
        if maximum_candidates is not None:
            _enable_candidate_set(document, maximum_candidates)
        elif isinstance(value, int) and not isinstance(value, bool) and value > 1:
            _enable_candidate_set(document, value)
        if materiality is not None:
            document["evaluation_protocol"]["search_materiality_ratio"] = materiality
        with tempfile.TemporaryDirectory() as directory:
            study_path = Path(directory) / "successor.json"
            study_path.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            return lab.preflight(study_path)

    def test_only_a_positive_count_is_admitted(self) -> None:
        for value in (0, -1, "two", 1.5, True, [2]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self._preflight(value)

        # Searching more than one candidate without saying what counts as material is
        # asking the loop to call every inversion a cost-model defect.
        with self.assertRaisesRegex(ValueError, "search_materiality_ratio"):
            self._preflight(2)

        # Absent means one, which is the behaviour every Study had before the field.
        lock = self._preflight(None)
        self.assertNotIn("searches_per_turn", lock.document["evaluation_protocol"])
        # And a real budget survives into the Lock, where the run reads it without
        # re-parsing: preflight is the only place this value is judged.
        lock = self._preflight(3, materiality=1.05)
        self.assertEqual(lock.document["evaluation_protocol"]["searches_per_turn"], 3)
        self.assertEqual(
            lock.document["evaluation_protocol"]["search_materiality_ratio"], 1.05
        )
        with self.assertRaisesRegex(ValueError, "exceeds maximum_candidates_per_turn"):
            self._preflight(3, materiality=1.05, maximum_candidates=2)


class StructurallyDistinctCandidatesTest(unittest.TestCase):
    """Two spellings of one program consume one search slot, and the author is told.

    The paper's first stage asks for structurally distinct candidates. A provider that
    renames a Schedule and submits it again produces different bytes, a different seal and
    the same kernel; measuring it twice spends GPU time to learn what the first
    measurement already said.
    """

    def _run(self, *, same_program: bool):
        class TwinEnvironment(FakeEnvironment):
            def build(self, submission):
                variant = json.loads(submission.payload)["variant"]
                result = super().build(submission)
                return EnvironmentResult(
                    result.disposition,
                    result.submission_sha256,
                    result.launchable,
                    result.feedback,
                    result.artifact_payloads,
                    cost=result.cost,
                    # The first two collapse in the duplicate case; the third is still a
                    # distinct program and must receive the search slot that the second
                    # spelling did not consume.
                    semantic_sha256=(
                        ("a" if variant < 2 else "b") * 64
                        if same_program
                        else f"{variant:064d}"
                    ),
                )

        class CandidateSetProvider(CandidateSetFakeProvider):
            def turn(self, request):
                observed = super().turn(request)
                payloads = tuple(
                    json.dumps(
                        {"run_id": request.run_id, "turn": request.turn, "variant": i},
                        sort_keys=True,
                    ).encode()
                    for i in range(3)
                )
                return ProviderTurn(
                    thread_id=observed.thread_id,
                    provider_tokens=observed.provider_tokens,
                    candidates=payloads,
                    candidate_sha256s=tuple(sha256(p).hexdigest() for p in payloads),
                    raw_events=observed.raw_events,
                    raw_events_sha256=observed.raw_events_sha256,
                    terminal_message=observed.terminal_message,
                    terminal_message_count=observed.terminal_message_count,
                    normalization=observed.normalization,
                    reference_bundle=observed.reference_bundle,
                )

        lab = Lab(ROOT)
        source = ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        document["evaluation_protocol"]["searches_per_turn"] = 2
        document["evaluation_protocol"]["search_materiality_ratio"] = 1.05
        _enable_candidate_set(document, 3)
        with tempfile.TemporaryDirectory() as directory:
            study_path = Path(directory) / "successor.json"
            study_path.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            lock = lab.preflight(study_path)
            arms = lock.document["resolved_inputs"]["arm_environments"]
            protocol_sha256 = sha256(
                json.dumps(
                    lock.document["evaluation_protocol"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            campaign = _execute(lab,
                lock,
                Path(directory) / "campaign-evidence",
                provider=CandidateSetProvider(),
                environments={arm: TwinEnvironment(arm, arms[arm]) for arm in arms},
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            store = EvidenceStore.open(campaign.evidence_root)
            events = [
                event
                for identifier in lock.run_order
                for event in store.replay_events(identifier)
            ]
            searched = sum(
                1
                for event in events
                if event["kind"] == "candidate_evaluated"
                and event["payload"]["purpose"] == "search"
            )
            collapses = [
                event["payload"]
                for event in events
                if event["kind"] == "diagnosis_routed"
                and "collapsed" in event["payload"]
            ]
            audit = store.audit_run(lock.run_order[0])

            class ChangedDiagnosisReplay:
                def replay_events(self, identifier):
                    replayed = json.loads(json.dumps(store.replay_events(identifier)))
                    for event in replayed:
                        if (
                            event["kind"] == "diagnosis_routed"
                            and "collapsed" in event["payload"]
                        ):
                            event["payload"]["collapsed"][0]["semantic_sha256"] = (
                                "f" * 64
                            )
                            break
                    return replayed

                def read_object(self, reference):
                    return store.read_object(reference)

            if same_program:
                with self.assertRaisesRegex(
                    ValueError, "diagnosis_routed is not derived"
                ):
                    lab._replay_matched_run(ChangedDiagnosisReplay(), audit, lock)
                changed_diagnosis_rejected = True
            else:
                changed_diagnosis_rejected = not lab._replay_matched_run(
                    ChangedDiagnosisReplay(), audit, lock
                )
            return searched, collapses, changed_diagnosis_rejected

    def test_one_program_is_measured_once_and_reported(self) -> None:
        distinct_searched, distinct_collapses, distinct_tamper_rejected = self._run(
            same_program=False
        )
        self.assertEqual(distinct_collapses, [])
        self.assertFalse(distinct_tamper_rejected)

        same_searched, same_collapses, same_tamper_rejected = self._run(
            same_program=True
        )
        self.assertTrue(same_collapses)
        self.assertTrue(same_tamper_rejected)
        for payload in same_collapses:
            self.assertEqual(payload["routed_to"], "candidate")
            for entry in payload["collapsed"]:
                # The report has to name which candidate it stood in for, or the author
                # is told they repeated themselves without being told what they repeated.
                self.assertNotEqual(
                    entry["candidate_sha256"], entry["same_program_as"]
                )
        # The duplicate saves one measurement and the third, distinct candidate fills
        # the declared second slot. A slice taken before deduplication would search only
        # one program here.
        self.assertEqual(same_searched, distinct_searched)


class QualifiedCandidateSelectionTest(unittest.TestCase):
    """Qualification chooses the winner, and its findings stay paired with it."""

    def test_an_unstable_fastest_candidate_cannot_win_or_lend_findings(self) -> None:
        class SelectionEnvironment(FakeEnvironment):
            def build(self, submission):
                variant = json.loads(submission.payload)["variant"]
                result = super().build(submission)
                candidate = result.launchable
                assert candidate is not None
                entry_point = f"{self.arm}_turn_{variant + 1}"
                artifacts = dict(candidate.artifact_payloads)
                manifest = json.loads(artifacts["launch_manifest"])
                manifest["kernel_name"] = entry_point
                artifacts["launch_manifest"] = json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")
                ).encode()
                roles = {
                    role: sha256(payload).hexdigest()
                    for role, payload in artifacts.items()
                }
                return EnvironmentResult(
                    result.disposition,
                    result.submission_sha256,
                    LaunchableCandidate(
                        candidate_sha256=candidate.candidate_sha256,
                        target=candidate.target,
                        entry_point=entry_point,
                        artifact_roles=roles,
                        launch_spec_sha256=roles["launch_manifest"],
                        artifact_payloads=artifacts,
                    ),
                    {"findings": [{"variant": variant}]},
                    result.artifact_payloads,
                )

        class CandidateSetProvider(CandidateSetFakeProvider):
            def turn(self, request):
                observed = super().turn(request)
                payloads = tuple(
                    json.dumps(
                        {
                            "run_id": request.run_id,
                            "turn": request.turn,
                            "variant": variant,
                        },
                        sort_keys=True,
                    ).encode()
                    for variant in range(2)
                )
                return ProviderTurn(
                    thread_id=observed.thread_id,
                    provider_tokens=observed.provider_tokens,
                    candidates=payloads,
                    candidate_sha256s=tuple(
                        sha256(payload).hexdigest() for payload in payloads
                    ),
                    raw_events=observed.raw_events,
                    raw_events_sha256=observed.raw_events_sha256,
                    terminal_message=observed.terminal_message,
                    terminal_message_count=observed.terminal_message_count,
                    normalization=observed.normalization,
                    reference_bundle=observed.reference_bundle,
                )

        class QualityEvaluator(FakeEvaluator):
            def __init__(self, *arguments):
                super().__init__(*arguments)
                self.selections: list[tuple[str, int]] = []

            def evaluate(self, candidate, *, case_id, purpose):
                variant = int(candidate.entry_point.rsplit("_", 1)[1]) - 1
                self.selections.append((purpose, variant))
                result = super().evaluate(
                    candidate,
                    case_id=case_id,
                    purpose=purpose,
                )
                receipt = result.final_receipt
                assert receipt is not None and len(result.attempts) == 1
                if purpose == "attribution":
                    return result
                # Variant zero is faster but unstable. Variant one is the sole result
                # whose measurement is eligible to choose and confirm.
                latency = 0.1 if variant == 0 else 1.0
                quality = variant == 1
                timing = {
                    "measurement_quality_passed": quality,
                    "pooled_median_ms": latency,
                }
                samples = (
                    [0.08] * 62 + [0.1] + [0.12] * 62
                    if variant == 0
                    else [latency] * 125
                )
                receipt = dataclasses.replace(
                    receipt,
                    timing=timing,
                    artifact_payloads={
                        **receipt.artifact_payloads,
                        "timing_samples": json.dumps(samples).encode(),
                    },
                )
                attempt = result.attempts[0]
                raw = json.loads(attempt.artifact_payloads["broker_record"])
                raw["receipt"]["timing"] = timing
                raw_bytes = json.dumps(
                    raw, sort_keys=True, separators=(",", ":")
                ).encode()
                attempt = dataclasses.replace(
                    attempt,
                    receipt=receipt,
                    artifact_payloads={
                        **attempt.artifact_payloads,
                        "broker_record": raw_bytes,
                        "evaluator_result": raw_bytes,
                    },
                )
                return LogicalEvaluationAttempt(
                    candidate.candidate_sha256,
                    (attempt,),
                    receipt,
                )

        lab = Lab(ROOT)
        source = ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        document["evaluation_protocol"]["searches_per_turn"] = 2
        document["evaluation_protocol"]["search_materiality_ratio"] = 1.05
        _enable_candidate_set(document, 2)
        with tempfile.TemporaryDirectory() as directory:
            study_path = Path(directory) / "successor.json"
            study_path.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            lock = lab.preflight(study_path)
            arms = lock.document["resolved_inputs"]["arm_environments"]
            protocol_sha256 = sha256(
                json.dumps(
                    lock.document["evaluation_protocol"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            provider = CandidateSetProvider()
            evaluator = QualityEvaluator(
                lock.document["evaluation_protocol"],
                protocol_sha256,
                lock.document["workload"]["canonical_sha256"],
            )
            campaign = _execute(lab,
                lock,
                Path(directory) / "campaign-evidence",
                provider=provider,
                environments={
                    arm: SelectionEnvironment(arm, arms[arm]) for arm in arms
                },
                evaluator=evaluator,
            )
            store = EvidenceStore.open(campaign.evidence_root)
            selections = [
                event["payload"]
                for run_id in lock.run_order
                for event in store.replay_events(run_id)
                if event["kind"] == "candidate_selected"
            ]
            report = lab.audit(campaign)

        self.assertTrue(report.archive_integrity_passed)
        self.assertTrue(report.semantic_replay_passed)
        self.assertTrue(selections)
        self.assertTrue(
            all(
                payload["reason"] == "lowest_qualified_search_latency"
                and payload["qualified_search_candidates"]
                == [payload["candidate_sha256"]]
                for payload in selections
            )
        )
        self.assertNotIn(("confirmatory", 0), evaluator.selections)
        self.assertIn(("confirmatory", 1), evaluator.selections)
        next_turns = [request for request in provider.requests if request.turn == 2]
        self.assertTrue(next_turns)
        self.assertTrue(
            all(request.feedback["findings"] == [{"variant": 1}] for request in next_turns)
        )


class CostModelRouteTest(unittest.TestCase):
    """The fourth destination, unlocked by evaluating more than one candidate.

    Deciding the order was wrong needs two measurements to compare, so this route stayed
    named-but-uninferred while a Turn evaluated one candidate. With `searches_per_turn`
    above one it becomes derivable, and the loop can finally record that the candidate was
    fine and the ranking was not.
    """

    def _run(
        self,
        *,
        searches_per_turn: int,
        materiality: float = 1.05,
        rank_with_cost: bool = True,
    ):
        class MisrankingEnvironment(FakeEnvironment):
            def build(self, submission):
                from open_cake_ir.compiler.ranking import Cost

                variant = json.loads(submission.payload)["variant"]
                result = super().build(submission)
                candidate = result.launchable
                assert candidate is not None
                # The evaluator derives latency from the entry point, so folding the
                # variant into it makes the later variant measurably faster while the
                # cost below ranks it second. The filter is wrong on purpose.
                renamed = LaunchableCandidate(
                    candidate_sha256=candidate.candidate_sha256,
                    target=candidate.target,
                    entry_point=f"{self.arm}_turn_{variant + 1}",
                    artifact_roles=candidate.artifact_roles,
                    launch_spec_sha256=candidate.launch_spec_sha256,
                    artifact_payloads=candidate.artifact_payloads,
                )
                return EnvironmentResult(
                    result.disposition,
                    result.submission_sha256,
                    renamed,
                    result.feedback,
                    result.artifact_payloads,
                    cost=(
                        Cost(
                            schedule_id=f"v{variant}",
                            ctas=1,
                            ctas_per_multiprocessor=1,
                            binding_resource="registers",
                            device_fill=0.9 - 0.4 * variant,
                        )
                        if rank_with_cost
                        else None
                    ),
                )

        class TwoCandidateProvider(CandidateSetFakeProvider):
            def turn(self, request):
                observed = super().turn(request)
                payloads = tuple(
                    json.dumps(
                        {"run_id": request.run_id, "turn": request.turn, "variant": i},
                        sort_keys=True,
                    ).encode()
                    for i in range(2)
                )
                return ProviderTurn(
                    thread_id=observed.thread_id,
                    provider_tokens=observed.provider_tokens,
                    candidates=payloads,
                    candidate_sha256s=tuple(sha256(p).hexdigest() for p in payloads),
                    raw_events=observed.raw_events,
                    raw_events_sha256=observed.raw_events_sha256,
                    terminal_message=observed.terminal_message,
                    terminal_message_count=observed.terminal_message_count,
                    normalization=observed.normalization,
                    reference_bundle=observed.reference_bundle,
                )

        lab = Lab(ROOT)
        source = ROOT / "contracts/studies/matched-search-infrastructure-template.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        document["evaluation_protocol"]["searches_per_turn"] = searches_per_turn
        document["evaluation_protocol"].pop("search_materiality_ratio", None)
        if searches_per_turn > 1:
            document["evaluation_protocol"]["search_materiality_ratio"] = materiality
        _enable_candidate_set(document, 2)
        with tempfile.TemporaryDirectory() as directory:
            study_path = Path(directory) / "successor.json"
            study_path.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            lock = lab.preflight(study_path)
            arms = lock.document["resolved_inputs"]["arm_environments"]
            protocol_sha256 = sha256(
                json.dumps(
                    lock.document["evaluation_protocol"],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            campaign = _execute(lab,
                lock,
                Path(directory) / "campaign-evidence",
                provider=TwoCandidateProvider(),
                environments={arm: MisrankingEnvironment(arm, arms[arm]) for arm in arms},
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            store = EvidenceStore.open(campaign.evidence_root)
            return [
                event["payload"]
                for identifier in lock.run_order
                for event in store.replay_events(identifier)
                if event["kind"] == "diagnosis_routed"
            ]

    def test_a_wrong_order_is_the_cost_models_not_the_candidates(self) -> None:
        # One search per Turn: nothing to compare, so nothing is claimed.
        self.assertEqual(self._run(searches_per_turn=1), [])

        # The evaluator puts the second variant about 1.11x ahead of the first.
        routed = self._run(searches_per_turn=2, materiality=1.05)
        self.assertTrue(routed)
        for payload in routed:
            self.assertEqual(payload["routed_to"], "cost_model")
            # The claim has to name both candidates, or it is an accusation with no
            # evidence attached to it.
            self.assertNotEqual(payload["ranked_first"], payload["measured_first"])
            self.assertGreaterEqual(
                payload["observed_ratio"], payload["materiality_ratio"]
            )

        # The same inversion, against a Study that counts nothing under 1.5x. The order
        # is still wrong in the sense that the second candidate measured faster, and the
        # loop still carries that one forward -- but "the cost model got it wrong" is not
        # a claim this measurement supports, so it is not made. Most inversions live here:
        # 24 of 37 candidates at one shape sit within 6% of the best.
        self.assertEqual(self._run(searches_per_turn=2, materiality=1.5), [])

    def test_provider_order_without_cost_is_not_a_cost_model_claim(self) -> None:
        self.assertEqual(
            self._run(searches_per_turn=2, rank_with_cost=False),
            [],
        )


class BrokerExecutionDigestTest(unittest.TestCase):
    """The digest a live Study pins its broker to.

    `execution.broker_execution_sha256` is frozen into a Study Contract when a live
    qualification is taken, and `execute_matched_from_config` recomputes it and refuses a
    Campaign whose broker no longer matches. Preflight only checks the field's shape, so
    this function is the whole of that binding -- and it had no test, which meant the one
    property that makes it worth having could stop holding silently.
    """

    def _digest(self, argv, root, **overrides):
        from open_cake_ir.lab import broker_execution_sha256

        arguments = {
            "cwd": root,
            "project_root": root,
            "timeout_seconds": 60,
            "service_user": "gpuq",
            "service_group": "gpuq-users",
        }
        arguments.update(overrides)
        return broker_execution_sha256(tuple(argv), **arguments)

    def test_the_digest_follows_the_content_of_what_it_will_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binary = root / "broker"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o700)
            script = root / "worker.py"
            script.write_text("print('one')\n")
            argv = [str(binary), "--script", str(script)]

            first = self._digest(argv, root)
            self.assertEqual(first, self._digest(argv, root))

            # An absolute file argument is part of what runs, so its content is part of
            # the identity. This is what makes editing the evaluator invalidate a Study
            # that was frozen against the old one, instead of running a different program
            # under the same seal.
            script.write_text("print('two')\n")
            self.assertNotEqual(first, self._digest(argv, root))

            # And so is the executable itself.
            script.write_text("print('one')\n")
            self.assertEqual(first, self._digest(argv, root))
            binary.write_text("#!/bin/sh\nexit 1\n")
            binary.chmod(0o700)
            self.assertNotEqual(first, self._digest(argv, root))

    def test_the_policy_it_refuses_is_the_policy_it_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binary = root / "broker"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o700)
            argv = [str(binary)]

            for label, overrides in (
                ("cwd outside the project root", {"cwd": root.parent}),
                ("no timeout", {"timeout_seconds": 0}),
                ("no service user", {"service_user": ""}),
                ("no service group", {"service_group": ""}),
            ):
                with self.subTest(refusal=label):
                    with self.assertRaises(ValueError):
                        self._digest(argv, root, **overrides)

            with self.assertRaisesRegex(ValueError, "empty"):
                self._digest([], root)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                self._digest([str(root / "absent")], root)


class RalphTaskInterfaceTests(unittest.TestCase):
    def test_retired_study_and_lock_interfaces_are_refused(self) -> None:
        from open_cake_ir.lab import StudyContract
        template = ROOT / "contracts/studies/matched-search-system-qualification-ralph-template.json"
        study = json.loads(template.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            legacy = json.loads(json.dumps(study))
            legacy["schema_version"] = 1
            del legacy["agent_interface"]
            path.write_text(json.dumps(legacy))
            with self.assertRaisesRegex(ValueError, "schema_version"):
                StudyContract.load(path)
            study["arms"]["open_cake"]["prompt_template"] = {"path": "missing.md"}
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(ValueError, "Authoring Environment fields"):
                Lab(ROOT).preflight(path)
        lock = Lab(ROOT).preflight(template)
        legacy_lock = json.loads(json.dumps(lock.document))
        del legacy_lock["resolved_inputs"]["agent_interface"]
        with self.assertRaisesRegex(ValueError, "Ralph interface"):
            CampaignLock.from_dict(legacy_lock)
        legacy_lock = json.loads(json.dumps(lock.document))
        arm = legacy_lock["resolved_inputs"]["arm_environments"]["open_cake"]
        arm["prompt_template"] = {"path": "missing.md"}
        with self.assertRaisesRegex(ValueError, "prompt_template"):
            CampaignLock.from_dict(legacy_lock)

    def test_ralph_budget_controls_tokens_time_turns_and_evaluations(self) -> None:
        now = [10.0]
        budget = RalphBudget.from_mapping(
            {
                "limit": 100,
                "maximum_turns": 3,
                "maximum_candidates_per_turn": 2,
                "wall_time_seconds": 20,
                "active_authoring_time_seconds": 10,
                "evaluation_limits": {
                    "search": 4,
                    "confirmatory": 2,
                    "attribution": 4,
                },
            }
        )
        controller = RalphController(
            budget,
            searches_per_turn=2,
            profile_each_search_survivor=True,
            clock=lambda: now[0],
        )
        self.assertIsNone(
            controller.stop_reason(turn=1, cumulative_provider_tokens=0)
        )
        started = controller.begin_authoring()
        now[0] += 4
        controller.end_authoring(started)
        for purpose in ("search", "search", "confirmatory", "attribution", "attribution"):
            controller.record_evaluation(purpose)
        card = controller.state_card(
            turn=2,
            cumulative_provider_tokens=60,
            feedback={"kind": "evaluation"},
        )
        self.assertEqual(card["remaining"]["provider_tokens"], 40)
        self.assertEqual(card["evaluation_counts"]["search"], 2)
        self.assertIsNone(
            controller.stop_reason(turn=2, cumulative_provider_tokens=60)
        )
        self.assertEqual(
            controller.stop_reason(turn=2, cumulative_provider_tokens=100),
            "provider_token_limit",
        )
        now[0] = 31
        self.assertEqual(
            controller.stop_reason(turn=2, cumulative_provider_tokens=60),
            "wall_time_limit",
        )
        now[0] = 10
        active_controller = RalphController(
            budget,
            searches_per_turn=2,
            profile_each_search_survivor=True,
            clock=lambda: now[0],
        )
        started = active_controller.begin_authoring()
        now[0] += 11
        active_controller.end_authoring(started)
        self.assertEqual(
            active_controller.stop_reason(
                turn=2, cumulative_provider_tokens=1
            ),
            "active_authoring_time_limit",
        )
        self.assertEqual(
            controller.stop_reason(turn=4, cumulative_provider_tokens=1),
            "maximum_turns",
        )
        evaluation_controller = RalphController(
            budget,
            searches_per_turn=2,
            profile_each_search_survivor=True,
            clock=lambda: 10.0,
        )
        for purpose in (
            "search",
            "search",
            "search",
            "confirmatory",
            "attribution",
            "attribution",
            "attribution",
        ):
            evaluation_controller.record_evaluation(purpose)
        self.assertEqual(
            evaluation_controller.stop_reason(
                turn=2, cumulative_provider_tokens=1
            ),
            "evaluation_budget",
        )

    def test_task_package_is_complete_read_only_and_arm_specific(self) -> None:
        lock = Lab(ROOT).preflight(
            ROOT
            / "contracts/studies/matched-search-system-qualification-ralph-template.json"
        )
        self.assertEqual(lock.agent_interface, TASK_AGENTS_RALPH_V1)
        open_package = render_task_package(ROOT, lock, "open_cake-1")
        cuda_package = render_task_package(ROOT, lock, "direct_cuda-1")
        self.assertIn("schedule-skeleton.json", open_package.task_markdown)
        self.assertIn("schedule-authoring.md", open_package.task_markdown)
        self.assertNotIn("candidate-skeleton.cu", open_package.task_markdown)
        self.assertIn("candidate-skeleton.cu", cuda_package.task_markdown)
        self.assertIn("cuda-launch-abi.json", cuda_package.task_markdown)
        self.assertNotIn("schedule-skeleton.json", cuda_package.task_markdown)
        for package in (open_package, cuda_package):
            self.assertIn("Write only `candidate-set.json`", package.agents_markdown)
            self.assertNotIn("confirmed_latency_ms", package.agents_markdown)
            with tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                materialize_task_package(workspace, package)
                verify_task_package(workspace, package)
                self.assertEqual(
                    {path.name for path in workspace.iterdir()},
                    {"TASK.md", "AGENTS.md"},
                )
                self.assertEqual((workspace / "TASK.md").stat().st_mode & 0o222, 0)
                (workspace / "TASK.md").chmod(0o644)
                (workspace / "TASK.md").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "TASK.md custody differs"):
                    verify_task_package(workspace, package)

    def test_ralph_system_qualification_executes_two_turns_and_replays(self) -> None:
        lab = Lab(ROOT, clock=lambda: 0.0)
        lock = lab.preflight(
            ROOT
            / "contracts/studies/matched-search-system-qualification-ralph-template.json"
        )
        packages = {
            run_id: render_task_package(ROOT, lock, run_id)
            for run_id in lock.run_order
        }
        resolved = lock.document["resolved_inputs"]
        protocol_sha256 = sha256(
            json.dumps(
                lock.document["evaluation_protocol"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        provider = RalphFakeProvider(packages)
        with tempfile.TemporaryDirectory() as directory:
            campaign = _execute(lab,
                lock,
                Path(directory) / "evidence",
                provider=provider,
                environments={
                    name: FakeEnvironment(name, document)
                    for name, document in resolved["arm_environments"].items()
                },
                evaluator=FakeEvaluator(
                    lock.document["evaluation_protocol"],
                    protocol_sha256,
                    lock.document["workload"]["canonical_sha256"],
                ),
            )
            report = lab.audit(campaign)
            evidence = EvidenceStore.open(campaign.evidence_root)
            for run_id in lock.run_order:
                events = evidence.replay_events(run_id)
                provider_events = [
                    event
                    for event in events
                    if event["kind"] == "provider_turn_completed"
                ]
                self.assertEqual(len(provider_events), 2)
                checkpoint = next(
                    event for event in events if event["kind"] == "checkpoints_projected"
                )["payload"]
                self.assertEqual(
                    checkpoint["ralph"]["terminal_reason"],
                    "provider_token_limit",
                )
                bundles = []
                for event in provider_events:
                    reference = next(
                        item
                        for item in event["payload"]["objects"]
                        if item["role"] == "provider_reference_bundle"
                    )
                    bundles.append(json.loads(evidence.read_object(reference)))
                self.assertEqual(
                    [bundle["state_card"]["iteration"] for bundle in bundles],
                    [1, 2],
                )
                self.assertTrue(
                    all(
                        bundle["task_markdown"] == packages[run_id].task_markdown
                        and bundle["agents_markdown"] == packages[run_id].agents_markdown
                        for bundle in bundles
                    )
                )
            tampered_run = lock.run_order[0]
            tampered_events = json.loads(
                json.dumps(evidence.replay_events(tampered_run))
            )
            first_provider_event = next(
                event
                for event in tampered_events
                if event["kind"] == "provider_turn_completed"
            )
            original_reference = next(
                item
                for item in first_provider_event["payload"]["objects"]
                if item["role"] == "provider_reference_bundle"
            )
            tampered_bundle = json.loads(evidence.read_object(original_reference))
            tampered_bundle["task_markdown"] += "\nunauthorized change\n"
            writer = EvidenceStore.writer(campaign.evidence_root)
            tampered_object = writer.put(
                json.dumps(
                    tampered_bundle,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
                media_type="application/json",
            )
            first_provider_event["payload"]["objects"] = [
                (
                    tampered_object.reference("provider_reference_bundle")
                    if item["role"] == "provider_reference_bundle"
                    else item
                )
                for item in first_provider_event["payload"]["objects"]
            ]
            with mock.patch.object(
                evidence,
                "replay_events",
                return_value=tuple(tampered_events),
            ):
                self.assertFalse(
                    lab._replay_matched_run(
                        evidence,
                        evidence.audit_run(tampered_run),
                        lock,
                    )
                )
            wrong_stop_events = json.loads(
                json.dumps(evidence.replay_events(tampered_run))
            )
            wrong_checkpoint = next(
                event
                for event in wrong_stop_events
                if event["kind"] == "checkpoints_projected"
            )
            wrong_checkpoint["payload"]["ralph"]["terminal_reason"] = (
                "wall_time_limit"
            )
            with mock.patch.object(
                evidence,
                "replay_events",
                return_value=tuple(wrong_stop_events),
            ):
                self.assertFalse(
                    lab._replay_matched_run(
                        evidence,
                        evidence.audit_run(tampered_run),
                        lock,
                    )
                )
        self.assertEqual(len(provider.requests), 4)
        self.assertTrue(report.archive_integrity_passed)
        self.assertTrue(report.semantic_replay_passed)
        self.assertTrue(report.system_qualification_passed)
        self.assertIsNone(report.estimand)
        self.assertIsNone(report.estimate)

    def test_ralph_artifact_template_keeps_promotion_non_scientific(self) -> None:
        lock = Lab(ROOT).preflight(
            ROOT / "contracts/studies/artifact-optimization-ralph-template.json"
        )
        self.assertEqual(lock.agent_interface, TASK_AGENTS_RALPH_V1)
        self.assertEqual(lock.claim_scope, "artifact_optimization_only")
        self.assertEqual(lock.run_order, ("open_cake-1", "direct_cuda-1"))
        self.assertIsNone(lock.estimand)
        self.assertEqual(
            lock.document["analysis_plan"]["comparative_statistics"],
            "forbidden",
        )
        for environment in lock.document["resolved_inputs"][
            "arm_environments"
        ].values():
            provider = environment["provider"]
            self.assertEqual(provider["disabled_features"], [])
            self.assertEqual(provider["event_contract"], "tool_rich_candidate_v1")
            self.assertEqual(
                provider["reference_visibility"], "workspace_task_files"
            )
            self.assertNotIn("prompt_template", environment)


class RuntimeReferenceCustodyTest(unittest.TestCase):
    """The gate between a runtime config and the files a live Campaign will read.

    `execute_matched_from_config` resolves every raw reference through this, and it is the
    whole of the property that a config cannot point a Campaign at a file outside the
    project. It had no test: a line trace found lab/compose.py at 197 of 197 statements
    never executed, and most of that module needs a live Campaign. This part does not.

    Each refusal below is a different way in, not a variation on one.
    """

    def _resolve(self, root, path_value, digest=None, name="reference"):
        from open_cake_ir.lab.compose import _raw_reference_path

        if digest is None:
            candidate = root / path_value if isinstance(path_value, str) else None
            digest = (
                sha256(candidate.read_bytes()).hexdigest()
                if candidate is not None and candidate.is_file()
                else "0" * 64
            )
        return _raw_reference_path(root, {"path": path_value, "sha256": digest}, name)

    def test_a_reference_inside_the_project_resolves_to_its_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "inner").mkdir()
            target = root / "inner" / "runtime.json"
            target.write_text("{}\n")
            self.assertEqual(self._resolve(root, "inner/runtime.json"), target)

    def test_every_way_out_of_the_project_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory).resolve()
            root = outer / "project"
            (root / "inner").mkdir(parents=True)
            target = root / "inner" / "runtime.json"
            target.write_text("{}\n")
            secret = outer / "secret.json"
            secret.write_text("{}\n")

            (root / "link.json").symlink_to(secret)
            # A symlinked *directory* passes the final-component check, so the resolved
            # path has to be tested against the root as well. If that ordering were the
            # other way round this would be the way through.
            (root / "elsewhere").symlink_to(outer)

            for label, value in (
                ("an absolute path", str(secret)),
                ("a parent traversal", "../secret.json"),
                ("a backslash separator", "inner\\runtime.json"),
                ("a symlinked file", "link.json"),
                ("a symlinked directory component", "elsewhere/secret.json"),
                ("a directory rather than a file", "inner"),
                ("a path that is not a string", 7),
                ("an empty path", ""),
            ):
                with self.subTest(refusal=label):
                    with self.assertRaises(ValueError):
                        self._resolve(root, value)

            with self.subTest(refusal="substituted bytes"):
                with self.assertRaisesRegex(ValueError, "bytes differ"):
                    self._resolve(root, "inner/runtime.json", digest="0" * 64)

            with self.subTest(refusal="fields differ"):
                from open_cake_ir.lab.compose import _raw_reference_path

                with self.assertRaisesRegex(ValueError, "fields differ"):
                    _raw_reference_path(root, {"path": "inner/runtime.json"}, "reference")


class EmpiricalSelectionContractTests(unittest.TestCase):
    """Actual CPU Lab path with a prospective fixture closure, never a host admission."""

    @classmethod
    def setUpClass(cls):
        from open_cake_ir.compiler import Compiler
        from open_cake_ir.evaluation import WorkloadContract
        from open_cake_ir.lab.environments import _empirical_context
        from tests.contracts.test_authoring_environment import _headline_schedule, _synthetic_flash_model

        cls.temporary = tempfile.TemporaryDirectory(prefix="empirical-selection-contract-")
        cls.parent = Path(cls.temporary.name).resolve()
        cls.root = cls.parent / "prospective-project"
        shutil.copytree(ROOT, cls.root, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        # Exercise the current event branch with a cycle-derived prospective Revision
        # in this disposable project. The copied host document is a synthetic contract
        # fixture: no host admission or provider/GPU qualification is performed.
        inventory_path = cls.root / "inventory/EXECUTOR_REVISIONS.json"
        inventory = json.loads(inventory_path.read_text())
        descriptor = json.loads((cls.root / inventory["current"]["path"]).read_text())
        host_fixture = cls.parent / "synthetic-host-environment.json"
        host_fixture.write_text(json.dumps(descriptor["host_environment"]))
        prepared = subprocess.run(
            ["bash", str(cls.root / "tools/release_executor_cycle.sh"), "--host-environment", str(host_fixture)],
            cwd=cls.root, env={**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if prepared.returncode:
            raise RuntimeError(prepared.stdout.decode() + prepared.stderr.decode())
        inventory = json.loads(inventory_path.read_text())
        cls.executor = ExecutorRevision.load(cls.root, cls.root / inventory["current"]["path"])
        cls.lab = Lab(cls.root)
        cls.workload = WorkloadContract.load(cls.root / "contracts/workloads/flash-kmeans-assign-v2.json")
        cls.compiler = Compiler.load(cls.root, cls.root / "compiler/revision.lock.json")
        cls.study = json.loads((cls.root / "contracts/studies/artifact-optimization-ralph-template.json").read_text())
        cls.study["arms"]["open_cake"]["candidate_selection"] = {"kind": "external_empirical_advisory_v1"}
        cls.study["evaluation_protocol"].pop("attribution_evaluation")
        cls.study["evaluation_protocol"].pop("search_materiality_ratio")
        cls.study["evaluation_protocol"]["searches_per_turn"] = 1
        for arm in cls.study["arms"].values():
            arm["feedback"].remove("profile")
        assessment = cls.compiler.assess(_headline_schedule(cls.workload))
        cls.compiler_ref = {"revision_id": assessment.compiler_revision_id, "canonical_sha256": assessment.compiler_revision_sha256}
        cls.context = _empirical_context(cls.executor, workload_sha256=cls.workload.canonical_sha256, case_id="headline_b32")
        cls.model = _synthetic_flash_model(cls.workload, cls.compiler_ref, cls.context)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def preflight(self, *, model=None, study=None):
        directory = Path(tempfile.mkdtemp(dir=self.parent))
        study_path = directory / "study.json"
        model_path = directory / "model.json"
        study_path.write_text(json.dumps(self.study if study is None else study))
        model_path.write_text(json.dumps(self.model if model is None else model))
        lock = self.lab.preflight(study_path, empirical_cost_model_path=model_path)
        return lock, directory, model_path

    def test_policy_admission_and_model_freeze(self):
        lock, directory, model_path = self.preflight()
        self.assertEqual(lock.document["resolved_inputs"]["arm_environments"]["open_cake"]["candidate_selection"]["model"], self.model)
        self.assertEqual(self.study["arms"]["open_cake"]["candidate_selection"], {"kind": "external_empirical_advisory_v1"})
        model_path.write_text("not the model anymore")
        lock_path = directory / "lock.json"
        lock_path.write_text(json.dumps(lock.document))
        self.assertEqual(CampaignLock.load(lock_path).document, lock.document)
        for scope in ("scientific_matched_search", "system_qualification_only"):
            study = json.loads(json.dumps(self.study))
            study["claim_scope"] = scope
            path = directory / (scope + ".json")
            path.write_text(json.dumps(study))
            with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "artifact_optimization_only"):
                self.lab.preflight(path, empirical_cost_model_path=model_path)
        with self.assertRaisesRegex(ValueError, "explicit model"):
            self.lab.preflight(directory / "study.json")
        no_policy = json.loads(json.dumps(self.study))
        del no_policy["arms"]["open_cake"]["candidate_selection"]
        path = directory / "no-policy.json"
        path.write_text(json.dumps(no_policy))
        with self.assertRaisesRegex(ValueError, "explicit model"):
            self.lab.preflight(path, empirical_cost_model_path=model_path)
        legacy = json.loads(json.dumps(self.study))
        del legacy["evidence"]["event_vocabulary"]
        with self.assertRaisesRegex(ValueError, "study.evidence is unsupported"):
            self.preflight(study=legacy)
        path = self.root / "contracts/studies/flash-kmeans-r45-portfolio-reconstruction-template.json"
        with self.assertRaisesRegex(ValueError, "artifact_optimization_only"):
            self.lab.preflight(path, empirical_cost_model_path=model_path)

    def test_context_and_revision_matching_is_exact(self):
        from open_cake_ir.lab.environments import _EmpiricalSelection
        mutations = [
            ("timer", "different timer"), ("cache_protocol", "different cache"),
            ("input_scope", "different workload or case"),
            ("runtime", {"compiler_version": "different", "executor_revision": self.executor.canonical_sha256}),
            ("runtime", {"compiler_version": self.context["runtime"]["compiler_version"]}),
            ("runtime", {**self.context["runtime"], "unmapped": "runtime"}),
            ("runtime", {**self.context["runtime"], "executor_revision": "b" * 64}),
        ]
        for field, value in mutations:
            model = json.loads(json.dumps(self.model))
            model["context"][field] = value
            with self.subTest(context=field, value=value):
                selection = _EmpiricalSelection({"kind": "external_empirical_advisory_v1", "model": model}, context=self.context, compiler_revision_id=self.compiler_ref["revision_id"], compiler_revision_sha256=self.compiler_ref["canonical_sha256"], target="sm_100a")
                result = selection.estimate(model["curves"][0]["template"])
                self.assertFalse(result["covered"])
                self.assertIsNone(result["predicted_kernel_us"])
                self.assertIn("context differs", result["reason"])
        for field, value in (("compiler_revision_id", "different"), ("compiler_revision_sha256", "0" * 64), ("target", "different")):
            arguments = {"compiler_revision_id": self.compiler_ref["revision_id"], "compiler_revision_sha256": self.compiler_ref["canonical_sha256"], "target": "sm_100a", field: value}
            selection = _EmpiricalSelection({"kind": "external_empirical_advisory_v1", "model": self.model}, context=self.context, **arguments)
            self.assertFalse(selection.estimate(self.model["curves"][0]["template"])["covered"])

    def test_policy_refuses_unproven_native_and_tile_assays(self):
        from open_cake_ir.evaluation import WorkloadContract
        from open_cake_ir.lab import OpenCakeEnvironment

        paired = json.loads((self.root / "contracts/studies/matched-search-triton-optimization-template.json").read_text())
        paired["claim_scope"] = "artifact_optimization_only"
        paired["arms"]["open_cake"]["candidate_selection"] = {"kind": "external_empirical_advisory_v1"}
        with self.assertRaisesRegex(ValueError, "Flash/direct-CUDA assay"):
            self.preflight(study=paired)

        workload_path = "contracts/workloads/rmsnorm-fp32-v1.json"
        workload = WorkloadContract.load(self.root / workload_path)
        direct_tile = json.loads(json.dumps(self.study))
        direct_tile["workload"] = {"path": workload_path, "canonical_sha256": workload.canonical_sha256}
        with self.assertRaisesRegex(ValueError, "Flash/direct-CUDA assay"):
            self.preflight(study=direct_tile)

        authority = {**paired["arms"]["open_cake"], "candidate_selection": {
            "kind": "external_empirical_advisory_v1", "model": self.model,
        }}
        for python_input in (True, False):
            environment_authority = dict(authority)
            if not python_input:
                environment_authority.pop("input_format")
            with self.subTest(python_input=python_input), self.assertRaisesRegex(ValueError, "Flash/direct-CUDA assay"):
                OpenCakeEnvironment(self.compiler, object(), authority_document=environment_authority,
                    workload=workload, case_id="primary", executor=self.executor)

        lock, _, _ = self.preflight()
        changed = json.loads(json.dumps(lock.document))
        arms = changed["resolved_inputs"]["arm_environments"]
        arms["native_triton"] = arms.pop("direct_cuda")
        arms["native_triton"]["environment_kind"] = "native_triton"
        changed["resolved_inputs"]["arm_environment_sha256"] = {
            name: sha256(json.dumps(arm, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for name, arm in arms.items()
        }
        with self.assertRaisesRegex(ValueError, "Flash/direct-CUDA assay"):
            CampaignLock.from_dict(changed)

    def test_both_author_interfaces_keep_full_model_only_in_the_lock(self):
        from open_cake_ir.lab.task_package import build_run_reference_documents
        model = json.loads(json.dumps(self.model))
        model["reported_evidence"]["raw_observations"] = list(range(10000))
        model["reported_evidence"]["unbounded_report"] = "private supplier report" * 10000
        for interface in ("ralph",):
            study = json.loads(json.dumps(self.study))
            if interface == "ralph":
                study = json.loads((self.root / "contracts/studies/artifact-optimization-ralph-template.json").read_text())
                study["arms"]["open_cake"]["candidate_selection"] = {"kind": "external_empirical_advisory_v1"}
            with self.subTest(interface=interface):
                lock, _, _ = self.preflight(model=model, study=study)
                arm = lock.document["resolved_inputs"]["arm_environments"]["open_cake"]
                documents = build_run_reference_documents(self.root, lock, arm)
                authority = json.loads(documents["run-authority.json"])
                projected = authority["authoring_environment"]["candidate_selection"]
                self.assertEqual(projected, {
                    "kind": "external_empirical_advisory_v1",
                    "model": {key: model[key] for key in (
                        "model_id", "compiler_revision_id", "compiler_revision_sha256", "target",
                    )},
                })
                self.assertLess(len(documents["run-authority.json"]), 10000)
                visible = b"\n".join(documents.values()).decode()
                if interface == "ralph":
                    package = render_task_package(self.root, lock, "open_cake-1")
                    visible += package.task_markdown + package.agents_markdown
                for forbidden in ("raw_observations", "unbounded_report", "private supplier report", "varying_dimensions"):
                    self.assertNotIn(forbidden, visible)
                self.assertEqual(arm["candidate_selection"]["model"], model)

    def run_campaign(self, *, model=None, include_rejected=False, via_cli=False):
        from open_cake_ir.lab import OpenCakeEnvironment
        lock, directory, model_path = self.preflight(model=model)
        if via_cli:
            cli_lock = directory / "cli-campaign.lock.json"
            completed = subprocess.run(
                [sys.executable, "-m", "open_cake_ir.cli", "--project-root", str(self.root),
                 "lab", "preflight", str(directory / "study.json"),
                 "--empirical-cost-model", str(model_path), "--output", str(cli_lock)],
                cwd=self.root, env={**os.environ, "PYTHONPATH": str(self.root / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(CampaignLock.load(cli_lock).document, lock.document)
            lock = CampaignLock.load(cli_lock)
        arms = lock.document["resolved_inputs"]["arm_environments"]
        source_model = self.model

        class Provider(FakeProvider):
            provider_revision = arms["open_cake"]["provider"]["revision"]
            qualification_sha256 = arms["open_cake"]["provider"]["qualification"]["canonical_sha256"]
            configuration = {**FakeProvider.configuration, "disabled_features": [], "event_contract": "tool_rich_candidate_v1", "output_schema_sha256": arms["open_cake"]["provider"]["output_schema"]["sha256"]}

            def turn(self, request):
                observed = super().turn(request)
                events = [json.loads(line) for line in observed.raw_events.splitlines()]
                terminal = json.loads(events[-2]["item"]["text"])
                terminal.pop("tool_calls")
                events[-2]["item"]["text"] = json.dumps(terminal, sort_keys=True, separators=(",", ":"))
                raw = b"".join(json.dumps(event).encode() + b"\n" for event in events)
                payloads = observed.candidates
                if request.arm == "open_cake":
                    schedules = [json.loads(json.dumps(curve["template"])) for curve in source_model["curves"]]
                    for i, schedule in enumerate(schedules):
                        schedule["schedule_id"] = f"synthetic-{i}-turn-{request.turn}"
                    if include_rejected:
                        invalid = json.loads(json.dumps(schedules[0]))
                        invalid["buffers"][0]["dtype"] = "fp32"
                        schedules.append(invalid)
                    payloads = tuple(json.dumps(schedule).encode() for schedule in schedules)
                return dataclasses.replace(observed, candidates=payloads, candidate_sha256s=tuple(sha256(value).hexdigest() for value in payloads), raw_events=raw, raw_events_sha256=sha256(raw).hexdigest(), terminal_message=events[-2]["item"]["text"], tool_activity=(ProviderAuxiliaryActivity(item_id=f"file-{request.turn}", item_type="file_change", status="completed"),))

        class Toolchain:
            def __init__(self):
                self.requests = []

            def build(self, request):
                self.requests.append(request)
                manifest = json.dumps({"schema_version": 1, "abi": "flash_kmeans_assign_v1", "target": "sm_100a", "kernel_name": "open_cake_turn_1", "grid": [1, 1, 1], "block": [32, 1, 1], "dynamic_shared_memory_bytes": 0}, sort_keys=True, separators=(",", ":")).encode()
                payloads = {"lowered_source": request.source, "compiler_expanded_source": b"synthetic expanded", "ptx": b"synthetic ptx", "cubin": b"synthetic cubin", "launch_manifest": manifest}
                roles = {role: sha256(value).hexdigest() for role, value in payloads.items()}
                return LaunchableCandidate(candidate_sha256=request.candidate_sha256, target=request.target, entry_point="open_cake_turn_1", artifact_roles=roles, launch_spec_sha256=roles["launch_manifest"], artifact_payloads=payloads)

        provider = Provider()
        toolchain = Toolchain()
        environment = OpenCakeEnvironment(self.compiler, toolchain, authority_document=arms["open_cake"], workload=self.workload, case_id="headline_b32", executor=self.executor)
        model_path.unlink()  # Runtime and replay must depend on the lock, not this file.
        campaign = _execute(self.lab, lock, directory / "evidence", provider=provider, environments={"open_cake": environment, "direct_cuda": FakeEnvironment("direct_cuda", arms["direct_cuda"])}, evaluator=FakeEvaluator(lock.document["evaluation_protocol"], sha256(json.dumps(lock.document["evaluation_protocol"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(), lock.document["workload"]["canonical_sha256"]))
        return campaign, provider, toolchain

    def test_actual_search_feedback_and_fresh_process_replay(self):
        model = json.loads(json.dumps(self.model))
        model["reported_evidence"]["raw_observations"] = list(range(10000))
        campaign, provider, toolchain = self.run_campaign(model=model, include_rejected=True, via_cli=True)
        report = self.lab.audit(campaign)
        self.assertTrue(report.semantic_replay_passed)
        self.assertTrue(report.archive_integrity_passed)
        store = EvidenceStore.open(campaign.evidence_root)
        events = store.replay_events("open_cake-1")
        filters = [event["payload"] for event in events if event["kind"] == "candidate_set_filtered"]
        searches = [event["payload"] for event in events if event["kind"] == "candidate_evaluated" and event["payload"]["purpose"] == "search"]
        self.assertEqual(len(toolchain.requests), 2 * len(filters))
        for filtered, search in zip(filters, searches):
            self.assertEqual([row["empirical_cost"]["predicted_kernel_us"] if row["empirical_cost"] else None for row in filtered["order"]], [10, 20, None])
            self.assertTrue(filtered["candidate_selection"]["order_applied"])
            self.assertEqual(search["candidate_sha256"], filtered["order"][0]["candidate_sha256"])
            self.assertEqual(filtered["order"][-1]["disposition"], "rejected")
        request = next(item for item in provider.requests if item.arm == "open_cake" and item.turn == 2)
        self.assertEqual(request.feedback["candidate_selection"], {**filters[0]["candidate_selection"], "order": filters[0]["order"]})
        self.assertNotIn("raw_observations", json.dumps(request.feedback["candidate_selection"]))
        self.assertLess(len(json.dumps(request.feedback["candidate_selection"])), 4000)
        self.assertEqual(campaign.lock.document["resolved_inputs"]["arm_environments"]["open_cake"]["candidate_selection"]["model"]["reported_evidence"]["raw_observations"], list(range(10000)))
        self.assertEqual(request.thread_id, provider.threads["open_cake-1"])
        lock_path = campaign.evidence_root.parent / "lock.json"
        lock_path.write_text(json.dumps(campaign.lock.document))
        completed = subprocess.run([sys.executable, "-m", "open_cake_ir.cli", "--project-root", str(self.root), "lab", "audit", "--lock", str(lock_path), "--evidence-root", str(campaign.evidence_root)], cwd=self.root, env={**os.environ, "PYTHONPATH": str(self.root / "src"), "PYTHONDONTWRITEBYTECODE": "1"}, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertTrue(json.loads(completed.stdout)["semantic_replay_passed"])
        audit = store.audit_run("open_cake-1")
        for mutation in ("prediction", "order", "coverage", "selection", "omit"):
            class Tampered:
                def replay_events(self, run_id):
                    changed = json.loads(json.dumps(events))
                    item = next(event["payload"] for event in changed if event["kind"] == "candidate_set_filtered")
                    if mutation == "prediction": item["order"][0]["empirical_cost"]["predicted_kernel_us"] += 1
                    elif mutation == "order": item["order"][:2] = reversed(item["order"][:2])
                    elif mutation == "coverage": item["order"][0]["empirical_cost"]["covered"] = False
                    elif mutation == "selection": item["candidate_selection"]["order_applied"] = False
                    else: item["order"].pop()
                    return tuple(changed)
                def read_object(self, reference):
                    return store.read_object(reference)
            with self.subTest(tamper=mutation):
                self.assertFalse(self.lab._replay_matched_run(Tampered(), audit, campaign.lock))
        for mutation in ("coefficients", "context"):
            changed = json.loads(json.dumps(campaign.lock.document))
            arm = changed["resolved_inputs"]["arm_environments"]["open_cake"]
            model = arm["candidate_selection"]["model"]
            if mutation == "coefficients": model["curves"][0]["points"][0]["kernel_us"] += 1
            else: model["context"]["runtime"]["executor_revision"] = "0" * 64
            changed["resolved_inputs"]["arm_environment_sha256"]["open_cake"] = sha256(json.dumps(arm, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            with self.subTest(frozen_model_tamper=mutation):
                self.assertFalse(self.lab._replay_matched_run(store, audit, CampaignLock.from_dict(changed)))

    def test_partial_coverage_and_ties_preserve_provider_order(self):
        for condition in ("partial", "tie", "context"):
            model = json.loads(json.dumps(self.model))
            if condition == "partial": model["curves"] = model["curves"][1:]
            elif condition == "tie": model["curves"][0]["points"] = json.loads(json.dumps(model["curves"][1]["points"]))
            else: model["context"]["timer"] = "incompatible timer"
            with self.subTest(condition=condition):
                campaign, _, _ = self.run_campaign(model=model)
                store = EvidenceStore.open(campaign.evidence_root)
                events = store.replay_events("open_cake-1")
                first = next(event["payload"] for event in events if event["kind"] == "candidate_set_filtered")
                authored = next(event["payload"] for event in events if event["kind"] == "provider_turn_completed")
                original = [item["sha256"] for item in authored["objects"] if item["role"].startswith("candidate_submission_")]
                self.assertEqual([row["candidate_sha256"] for row in first["order"]], original)
                self.assertEqual(first["candidate_selection"]["order_applied"], condition == "tie")
                self.assertTrue(self.lab.audit(campaign).semantic_replay_passed)
                class Reordered:
                    def replay_events(self, run_id):
                        changed = json.loads(json.dumps(events))
                        row = next(event["payload"] for event in changed if event["kind"] == "candidate_set_filtered")
                        row["order"].reverse()
                        return tuple(changed)
                    def read_object(self, reference):
                        return store.read_object(reference)
                self.assertFalse(self.lab._replay_matched_run(Reordered(), store.audit_run("open_cake-1"), campaign.lock))
