"""Prospective vector endpoints through CPU Lab consumers; no release or host claims."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.checkpoints import TurnObservation, project_checkpoints
from open_cake_ir.lab.endpoints import NORMAL_BUDGET_TERMINAL, endpoint_policy, matched_endpoint
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.tasks.runtime import TaskLab
from tests.contracts import test_diagnosis_feedback as diagnosis_consumers
from tests.contracts.test_lab import FakeProvider, FakeEnvironment, FakeEvaluator, _execute


class TerminalProjectionTests(unittest.TestCase):
    def test_every_normal_axis_keeps_unreached_token_checkpoint_and_completed_best(self):
        turns = [TurnObservation(1, 20, "a" * 64, True, 1.0),
                 TurnObservation(2, 40, "b" * 64, True, 0.8),
                 TurnObservation(3, 60, "c" * 64, True, 0.8)]
        checkpoints = project_checkpoints(turns=turns, checkpoints=[30, 100], terminal_provider_tokens=60)
        self.assertEqual([row.state for row in checkpoints], ["reached_with_best", "unreached"])
        self.assertEqual(checkpoints[0].best_candidate_sha256, "a" * 64)
        for reason in ("provider_token_limit", "maximum_turns", "wall_time_limit", "active_authoring_time_limit", "evaluation_budget"):
            state, value = matched_endpoint(checkpoint=checkpoints[-1], observations=turns,
                terminal_provider_tokens=60, protocol_adherence="adhered", terminal_reason=reason,
                analysis={"endpoint_policy": NORMAL_BUDGET_TERMINAL})
            self.assertEqual(state, "qualified")
            self.assertEqual(value, {"qualified_by_budget": True, "budget": 60,
                "best_candidate_sha256": "b" * 64, "best_confirmed_latency_ms": 0.8,
                "terminal_reason": reason, "observation_basis": NORMAL_BUDGET_TERMINAL})
        self.assertEqual(matched_endpoint(checkpoint=checkpoints[-1], observations=turns,
            terminal_provider_tokens=60, protocol_adherence="adhered", terminal_reason="maximum_turns", analysis={}), ("missing", None))
        for fault in ("provider_fault", "broker_fault", "harness_fault"):
            self.assertEqual(matched_endpoint(checkpoint=checkpoints[-1], observations=turns,
                terminal_provider_tokens=60, protocol_adherence=fault, terminal_reason=fault,
                analysis={"endpoint_policy": NORMAL_BUDGET_TERMINAL}), ("missing", None))

    def test_policy_is_closed_and_requires_a_normal_stop(self):
        checkpoint = project_checkpoints(turns=[], checkpoints=[100], terminal_provider_tokens=0)[0]
        for value in (None, {}, True, "normal_budget_terminal_v2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                endpoint_policy({"endpoint_policy": value})
        with self.assertRaises(ValueError):
            matched_endpoint(checkpoint=checkpoint, observations=[], terminal_provider_tokens=0,
                protocol_adherence="adhered", terminal_reason=None, analysis={"endpoint_policy": NORMAL_BUDGET_TERMINAL})
        state, value = matched_endpoint(checkpoint=checkpoint, observations=[], terminal_provider_tokens=0,
            protocol_adherence="adhered", terminal_reason="wall_time_limit", analysis={"endpoint_policy": NORMAL_BUDGET_TERMINAL})
        self.assertEqual(state, "no_qualified_candidate")
        self.assertEqual(value["budget"], 0)


class TerminalRunTests(unittest.TestCase):
    def setUp(self):
        diagnosis_consumers.DiagnosisRunTests.setUp(self)  # Explicit Compiler/Executor consumer dependencies.
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.enterContext(patch.dict(os.environ, {"OPEN_CAKE_CUSTODY_DIRECTORY": str(self.directory / "registry")}))

    def run_campaign(self, *, policy=True, scope="scientific_matched_search", provider=None,
                     rejected=False, reject_first=False, clock=None, limits=None, evaluator_class=FakeEvaluator):
        from open_cake_ir.lab.environments import EnvironmentResult
        directory = Path(tempfile.mkdtemp(dir=self.directory))
        template = "artifact-optimization-ralph-template.json" if scope == "artifact_optimization_only" else "matched-search-infrastructure-template.json"
        document = json.loads((ROOT / "contracts/studies" / template).read_text())
        document["analysis_plan"].pop("endpoint_policy", None)
        if policy:
            document["analysis_plan"]["endpoint_policy"] = NORMAL_BUDGET_TERMINAL
        document["budget"].update(limit=500000, checkpoints=[100000, 500000], maximum_turns=2)
        if limits:
            document["budget"].update(limits)
        study = directory / "study.json"
        study.write_text(json.dumps(document))
        lab = TaskLab(ROOT, **({"clock": clock} if clock else {}))
        lock = lab.preflight(study)
        arms = lock.document["resolved_inputs"]["arm_environments"]
        if scope == "artifact_optimization_only" and provider is None:
            from open_cake_ir.lab.provider_policy import provider_configuration
            from open_cake_ir.lab.provider_documents import ProviderAuxiliaryActivity
            declaration = arms["open_cake"]["provider"]
            class ArtifactProvider(FakeProvider):
                provider_revision = declaration["revision"]
                qualification_sha256 = declaration["qualification"]["canonical_sha256"]
                configuration = provider_configuration(declaration, scope, arms=arms)
                def turn(self, request):
                    observed = super().turn(request)
                    events = [json.loads(line) for line in observed.raw_events.splitlines()]
                    terminal = json.loads(events[-2]["item"]["text"])
                    terminal.pop("tool_calls")
                    events[-2]["item"]["text"] = json.dumps(terminal, sort_keys=True, separators=(",", ":"))
                    raw = b"\n".join(json.dumps(event).encode() for event in events) + b"\n"
                    return dataclasses.replace(observed, raw_events=raw,
                        raw_events_sha256=sha256(raw).hexdigest(), terminal_message=events[-2]["item"]["text"],
                        tool_activity=(ProviderAuxiliaryActivity(item_id=f"file-{request.turn}", item_type="file_change", status="completed"),))
            provider = ArtifactProvider()
        provider = provider or FakeProvider()
        class Environment(FakeEnvironment):
            def build(self, submission):
                return (EnvironmentResult("rejected", submission.sha256, None,
                    {"stage": "assessment", "code": "CPU_FIXTURE_REFUSAL"})
                    if rejected or (reject_first and json.loads(submission.payload)["turn"] == 1)
                    else super().build(submission))
        protocol = lock.document["evaluation_protocol"]
        evaluator = evaluator_class(protocol, sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), lock.document["workload"]["canonical_sha256"])
        campaign = _execute(lab, lock, directory / "evidence", provider=provider,
            environments={arm: Environment(arm, authority) for arm, authority in arms.items()}, evaluator=evaluator)
        return lab, campaign, provider

    def test_actual_normal_turn_stop_observes_endpoint_without_checkpoint_backfill(self):
        for policy in (False, True):
            for rejected in (False, True):
                with self.subTest(policy=policy, rejected=rejected):
                    lab, campaign, provider = self.run_campaign(policy=policy, rejected=rejected)
                    report = lab.audit(campaign)
                    self.assertTrue(report.semantic_replay_passed)
                    self.assertEqual(report.missing_run_count, 0 if policy else len(campaign.lock.run_order))
                    for audit in report.run_audits:
                        if policy:
                            self.assertEqual(audit.endpoint_observation, "no_qualified_candidate" if rejected else "qualified")
                            self.assertEqual(audit.endpoint["budget"], 160000)
                            self.assertEqual(audit.endpoint["terminal_reason"], "maximum_turns")
                        else:
                            self.assertEqual(audit.endpoint_observation, "missing")
                            self.assertIsNone(audit.endpoint)
                        observed = report.descriptive["terminal_observations"][audit.run_id]
                        self.assertEqual(observed["token_limit_checkpoint_state"], "unreached")
                    self.assertEqual(report.descriptive["reference_access_by_arm"],
                        {arm: declaration.get("reference_access") for arm, declaration in campaign.lock.document["resolved_inputs"]["arm_environments"].items()})

    def test_later_confirmation_defines_terminal_without_backfilling_earlier_rejection(self):
        lab, campaign, _ = self.run_campaign(reject_first=True)
        report = lab.audit(campaign)
        self.assertTrue(report.semantic_replay_passed)
        self.assertTrue(report.estimand_available)
        store = EvidenceStore.open(campaign.evidence_root)
        for audit in report.run_audits:
            self.assertEqual(audit.endpoint_observation, "qualified")
            self.assertEqual(audit.endpoint["budget"], 160000)
            projection = next(event["payload"]["checkpoints"] for event in store.replay_events(audit.run_id)
                              if event["kind"] == "checkpoints_projected")
            self.assertEqual(projection[0]["state"], "reached_no_qualified_candidate")
            self.assertIsNone(projection[0]["best_candidate_sha256"])
            self.assertEqual(projection[-1]["state"], "unreached")

    def test_actual_other_budget_axes_and_simultaneous_time_limits(self):
        class Clock:
            value = 0.0
            def __call__(self): return self.value
        for reason in ("provider_token_limit", "wall_time_limit", "active_authoring_time_limit", "evaluation_budget"):
            clock = Clock()
            class TimedProvider(FakeProvider):
                def turn(self, request):
                    observed = super().turn(request)
                    if reason in {"wall_time_limit", "active_authoring_time_limit"}:
                        clock.value += 2
                    return observed
            limits = {"maximum_turns": 4}
            if reason == "provider_token_limit":
                limits.update(limit=80000, checkpoints=[40000, 80000])
            elif reason == "wall_time_limit":
                # Both time axes reach together; existing Ralph priority chooses wall.
                limits.update(wall_time_seconds=1, active_authoring_time_seconds=1)
            elif reason == "active_authoring_time_limit":
                limits.update(wall_time_seconds=100, active_authoring_time_seconds=1)
            else:
                limits["evaluation_limits"] = {"search": 2, "confirmatory": 1, "attribution": 2}
            with self.subTest(reason=reason):
                lab, campaign, provider = self.run_campaign(provider=TimedProvider(), clock=clock, limits=limits)
                report = lab.audit(campaign)
                self.assertTrue(report.semantic_replay_passed)
                for audit in report.run_audits:
                    self.assertEqual(audit.endpoint_observation, "qualified")
                    self.assertEqual(audit.endpoint["terminal_reason"], reason)
                    self.assertEqual(audit.endpoint["budget"], 80000)

    def test_failed_call_unknown_usage_remains_missing_despite_prior_confirmation(self):
        class FaultProvider(FakeProvider):
            def turn(self, request):
                if request.turn == 2:
                    raise RunProtocolFault("provider_fault", "CPU fixture unavailable usage")
                return super().turn(request)
        lab, campaign, provider = self.run_campaign(provider=FaultProvider())
        report = lab.audit(campaign)
        self.assertTrue(report.semantic_replay_passed)
        self.assertFalse(report.estimand_available)
        for audit in report.run_audits:
            self.assertEqual(audit.endpoint_observation, "missing")
            self.assertIsNone(audit.endpoint)
            events = EvidenceStore.open(campaign.evidence_root).replay_events(audit.run_id)
            fault = next(event["payload"] for event in events if event["kind"] == "run_fault")
            self.assertEqual(fault["terminal_provider_tokens_scope"], "known_subtotal")
            self.assertEqual(fault["terminal_provider_tokens"], 80000)
            self.assertEqual(report.descriptive["terminal_observations"][audit.run_id]["missing_reason"], "protocol_fault")

    def test_partial_second_turn_confirmation_fault_cannot_supply_terminal_candidate(self):
        class PartialEvaluator(FakeEvaluator):
            def evaluate(self, candidate, *, case_id, purpose):
                if candidate.entry_point.endswith("_turn_2") and purpose == "confirmatory":
                    raise RunProtocolFault("broker_fault", "CPU fixture partial confirmation")
                return super().evaluate(candidate, case_id=case_id, purpose=purpose)
        lab, campaign, provider = self.run_campaign(evaluator_class=PartialEvaluator)
        report = lab.audit(campaign)
        self.assertTrue(report.semantic_replay_passed)
        self.assertFalse(report.estimand_available)
        for audit in report.run_audits:
            self.assertEqual(audit.protocol_adherence, "broker_fault")
            self.assertEqual(audit.endpoint_observation, "missing")
            self.assertIsNone(audit.endpoint)
            events = EvidenceStore.open(campaign.evidence_root).replay_events(audit.run_id)
            confirmations = [event["payload"]["turn"] for event in events
                             if event["kind"] == "candidate_evaluated" and event["payload"]["purpose"] == "confirmatory"]
            self.assertEqual(confirmations, [1])
            starts = [event["payload"] for event in events if event["kind"] == "evaluation_attempt_started"]
            self.assertTrue(all(set(value) == {"turn", "purpose", "candidate_sha256"} for value in starts))
            self.assertEqual(sum(value["purpose"] == "confirmatory" for value in starts), 2)
            checkpoint = next(event["payload"] for event in events if event["kind"] == "checkpoints_projected")
            self.assertEqual(checkpoint["ralph"]["evaluation_counts"]["confirmatory"], 2)
            self.assertEqual(report.descriptive["terminal_observations"][audit.run_id]["logical_evaluation_invocation_counts"]["confirmatory"], 2)
            self.assertEqual(report.descriptive["evaluation_receipt_counts"][audit.run_id], 5)

    def test_evaluator_invocation_order_and_identity_are_independently_replayed(self):
        from copy import deepcopy
        from types import SimpleNamespace
        from open_cake_ir.lab.evaluation_lifecycle import replay_evaluation_invocations
        lab, campaign, _ = self.run_campaign()
        store = EvidenceStore.open(campaign.evidence_root)
        audit = store.audit_run(campaign.lock.run_order[0])
        original = list(store.replay_events(audit.run_id))
        receipts = {(event["payload"]["turn"], event["payload"]["purpose"], event["payload"]["candidate_sha256"]): SimpleNamespace(correctness_passed=True)
                    for event in original if event["kind"] == "candidate_evaluated"}
        budget = campaign.lock.document["resolved_inputs"]["budget"]
        protocol = campaign.lock.document["evaluation_protocol"]
        starts = [index for index, event in enumerate(original) if event["kind"] == "evaluation_attempt_started"]
        completed = next(index for index, event in enumerate(original) if event["kind"] == "evaluation_attempt_completed")
        for mutation in ("missing start", "duplicate start", "completion before start", "unsealed candidate", "wrong purpose", "extra field", "boolean turn", "unfinished normal", "wrong fault stage", "completion after fault"):
            events = deepcopy(original)
            start = starts[0]
            if mutation == "missing start":
                events.pop(start)
            elif mutation == "duplicate start":
                events.insert(start, deepcopy(events[start]))
            elif mutation == "completion before start":
                events[start], events[completed] = events[completed], events[start]
            elif mutation == "unsealed candidate":
                events[start]["payload"]["candidate_sha256"] = "f" * 64
            elif mutation == "wrong purpose":
                events[start]["payload"]["purpose"] = "confirmatory"
            elif mutation == "extra field":
                events[start]["payload"]["gpu_dispatch"] = True
            elif mutation == "boolean turn":
                events[start]["payload"]["turn"] = True
            else:
                events = events[:start + 1]
                if mutation != "unfinished normal":
                    events.append({"kind": "run_fault", "payload": {"turn": 1, "stage": "provider" if mutation == "wrong fault stage" else "evaluation"}})
                    if mutation == "completion after fault":
                        events.append(deepcopy(original[completed]))
                events.extend(deepcopy(original[-2:]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                # The concrete lifecycle validator refuses; no unrelated archive
                # checksum or provider assertion supplies this negative result.
                replay_evaluation_invocations(events, receipts=receipts, budget=budget, protocol=protocol)
        self.assertEqual(replay_evaluation_invocations(original, receipts=receipts, budget=budget, protocol=protocol),
                         {"search": 2, "confirmatory": 2, "attribution": 2})
        altered = deepcopy(original)
        altered.pop(starts[0])
        with patch.object(store, "replay_events", return_value=tuple(altered)):
            self.assertFalse(lab._replay_matched_run(store, audit, campaign.lock))

    def test_public_preflight_rejects_unknown_endpoint_policy(self):
        document = json.loads((ROOT / "contracts/studies/matched-search-infrastructure-template.json").read_text())
        for value in (None, {}, "normal_budget_terminal_v2"):
            document["analysis_plan"]["endpoint_policy"] = value
            study = self.directory / "invalid-plan.json"
            study.write_text(json.dumps(document))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "endpoint policy"):
                TaskLab(ROOT).preflight(study)

    def test_zero_work_normal_wall_stop_and_endpoint_tampering(self):
        class Clock:
            def __init__(self): self.value = 0
            def __call__(self):
                self.value += 10
                return self.value
        lab, campaign, provider = self.run_campaign(clock=Clock(), limits={"wall_time_seconds": 1, "active_authoring_time_seconds": 1})
        report = lab.audit(campaign)
        self.assertEqual(provider.requests, [])
        self.assertTrue(report.semantic_replay_passed)
        for audit in report.run_audits:
            self.assertEqual(audit.endpoint_observation, "no_qualified_candidate")
            self.assertEqual(audit.endpoint["budget"], 0)
            self.assertEqual(audit.endpoint["terminal_reason"], "wall_time_limit")
        store = EvidenceStore.open(campaign.evidence_root)
        audit = report.run_audits[0]
        for mutation in ({"terminal_reason": "maximum_turns"}, {"extra": True}, {"budget": 1}, {"budget": False}, {"budget": 0.0}, {"qualified_by_budget": 0}):
            altered = dataclasses.replace(audit, endpoint={**audit.endpoint, **mutation})
            # Terminal event must also agree before reaching the independent projection.
            events = [dict(event) for event in store.replay_events(audit.run_id)]
            events[-1]["payload"] = {**events[-1]["payload"], "endpoint": dict(altered.endpoint)}
            with patch.object(store, "replay_events", return_value=tuple(events)):
                self.assertFalse(lab._replay_matched_run(store, altered, campaign.lock))

    def test_insufficient_initial_evaluation_capacity_stays_a_preflight_refusal(self):
        with self.assertRaisesRegex(ValueError, "cannot admit one complete Turn"):
            self.run_campaign(limits={
                "evaluation_limits": {"search": 1, "confirmatory": 1, "attribution": 2}})
        self.assertEqual(list(self.directory.rglob("evidence")), [])

    def test_one_invalid_run_does_not_erase_other_runs_artifacts_or_receipts(self):
        from open_cake_ir.lab.reporting import audit_campaign
        lab, campaign, provider = self.run_campaign(scope="artifact_optimization_only")
        baseline = lab.audit(campaign)
        self.assertTrue(baseline.semantic_replay_passed)
        self.assertTrue(all(baseline.descriptive["promoted_artifacts"].values()))
        for bad_run in (campaign.lock.run_order[0], campaign.lock.run_order[-1]):
            examined = []
            def replay(evidence, audit, lock):
                examined.append(audit.run_id)
                return False if audit.run_id == bad_run else lab._replay_matched_run(evidence, audit, lock)
            report = audit_campaign(campaign, replay_run=replay)
            self.assertEqual(examined, list(campaign.lock.run_order))
            self.assertFalse(report.semantic_replay_passed)
            self.assertFalse(report.descriptive["artifact_optimization_complete"])
            self.assertFalse(report.estimand_available)
            self.assertIsNone(report.descriptive["promoted_artifacts"][bad_run])
            for good in set(campaign.lock.run_order) - {bad_run}:
                self.assertEqual(report.descriptive["promoted_artifacts"][good], baseline.descriptive["promoted_artifacts"][good])
                self.assertEqual(report.descriptive["evaluation_receipt_counts"][good], baseline.descriptive["evaluation_receipt_counts"][good])
