"""CPU execution/replay evidence for candidate-bound Ralph feedback.

These explicit doubles exercise the shared protocol, never a GPU, real provider,
Compiler release or host qualification.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from open_cake_ir.evaluation import LogicalEvaluationAttempt
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.environments import EnvironmentResult
from open_cake_ir.tasks.runtime import TaskLab
from tests.contracts.test_diagnosis_feedback import DiagnosisRunTests
from tests.contracts.test_lab import (
    FakeEnvironment,
    FakeEvaluator,
    FakeProvider,
    ROOT,
    _enable_candidate_set,
    _execute,
    _submission_envelope,
)


class _ThreeCandidates(FakeProvider):
    def turn(self, request):
        observed = super().turn(request)
        candidates = tuple(
            json.dumps(
                {"run_id": request.run_id, "turn": request.turn, "variant": index},
                sort_keys=True, separators=(",", ":"),
            ).encode()
            for index in range(3)
        )
        return dataclasses.replace(
            observed,
            candidates=candidates,
            candidate_sha256s=tuple(sha256(value).hexdigest() for value in candidates),
            raw_submission=_submission_envelope(request.arm, candidates),
        )


class _FeedbackEnvironment(FakeEnvironment):
    def __init__(self, arm, authority, peer):
        super().__init__(arm, authority)
        self.peer = peer

    def build(self, submission, *, compilation=None):
        variant = json.loads(submission.payload)["variant"]
        if self.peer == "all_rejected" or self.peer == "rejected" and variant == 2:
            return EnvironmentResult(
                "rejected", submission.sha256, None,
                {"stage": "compile", "diagnostic": f"syntax variant {variant}"},
            )
        result = super().build(submission, compilation=compilation)
        return dataclasses.replace(
            result,
            feedback={"findings": [{"code": f"FIXTURE_VARIANT_{variant}",
                                    "path": f"operations[{variant}]",
                                    "message": f"retained candidate {variant}",
                                    "severity": "report", "category": "hardware_conformance",
                                    "blocks_acceptance": False, "blocks_lowering": False}]},
            semantic_sha256=("a" if variant in (0, 2) else "b") * 64
            if self.peer == "duplicate" else None,
        )


class _QualityEvaluator(FakeEvaluator):
    """First candidate is faster but unstable; second is the qualified winner."""
    def candidate_position(self, candidate):
        arm, _ = super().candidate_position(candidate)
        role = "lowered_source" if arm == "open_cake" else "authored_source"
        variant = json.loads(candidate.artifact_payloads[role])["variant"]
        return arm, variant + 1

    def evaluate(self, candidate, *, case_id, purpose):
        result = super().evaluate(candidate, case_id=case_id, purpose=purpose)
        if purpose == "attribution":
            return result
        _, position = self.candidate_position(candidate)
        variant = position - 1
        latency = 0.1 if variant == 0 else 1.0
        quality = variant != 0
        timing = {"measurement_quality_passed": quality, "pooled_median_ms": latency}
        samples = [0.08] * 62 + [0.1] + [0.12] * 62 if variant == 0 else [latency] * 125
        receipt = dataclasses.replace(
            result.final_receipt,
            timing=timing,
            artifact_payloads={**result.final_receipt.artifact_payloads,
                               "timing_samples": json.dumps(samples).encode()},
        )
        attempt = result.attempts[0]
        raw = json.loads(attempt.artifact_payloads["broker_record"])
        raw["receipt"]["timing"] = timing
        raw_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        attempt = dataclasses.replace(
            attempt,
            receipt=receipt,
            artifact_payloads={**attempt.artifact_payloads,
                               "broker_record": raw_bytes, "evaluator_result": raw_bytes},
        )
        return LogicalEvaluationAttempt(candidate.candidate_sha256, (attempt,), receipt)


class CandidateFeedbackTests(unittest.TestCase):
    # Reuse only the semantic Compiler/Executor seams, not the other class's tests.
    setUp = DiagnosisRunTests.setUp

    @contextlib.contextmanager
    def campaign(self, peer="unsearched"):
        with tempfile.TemporaryDirectory() as directory:
            document = json.loads((ROOT / "contracts/studies/matched-search-infrastructure-template.json").read_text())
            _enable_candidate_set(document, 3)
            document["budget"].update(maximum_turns=2, limit=160000,
                                      checkpoints=[80000, 160000])
            document["evaluation_protocol"]["searches_per_turn"] = 3 if peer == "duplicate" else 2
            study = Path(directory) / "study.json"
            study.write_text(json.dumps(document))
            lab = TaskLab(ROOT, clock=lambda: 0.0)
            lock = lab.preflight(study)
            provider = _ThreeCandidates()
            protocol = lock.document["evaluation_protocol"]
            evaluator = _QualityEvaluator(
                protocol,
                sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                lock.document["workload"]["canonical_sha256"],
            )
            arms = lock.document["resolved_inputs"]["arm_environments"]
            campaign = _execute(
                lab, lock, Path(directory) / "evidence", provider=provider,
                environments={arm: _FeedbackEnvironment(arm, authority, peer)
                              for arm, authority in arms.items()},
                evaluator=evaluator,
            )
            yield lab, lock, provider, campaign, EvidenceStore.open(campaign.evidence_root)

    def assert_no_measurement(self, row):
        for key in ("correctness_passed", "candidate_disposition", "measurement_quality",
                    "search_qualified", "search_latency_ms", "profile", "baseline_comparison"):
            self.assertNotIn(key, row)

    def test_second_candidate_identity_and_each_peer_reach_real_next_turn(self):
        for peer, expected_status in (("unsearched", "not_evaluated"),
                                      ("rejected", "build_rejected"),
                                      ("duplicate", "duplicate")):
            with self.subTest(peer=peer), self.campaign(peer) as (lab, lock, provider, campaign, store):
                requests = [request for request in provider.requests if request.turn == 2]
                self.assertEqual(len(requests), len(lock.run_order))
                for request in requests:
                    events = store.replay_events(request.run_id)
                    actions = next(event["payload"]["actions"] for event in events
                                   if event["kind"] == "author_actions_resolved" and event["payload"]["turn"] == 1)
                    expected_ids = [row["candidate_sha256"] for row in actions]
                    feedback = request.feedback
                    self.assertEqual(feedback["source_turn"], 1)
                    self.assertEqual(feedback["selected_candidate_sha256"], expected_ids[1])
                    rows = feedback["candidate_results"]
                    self.assertEqual([row["candidate_index"] for row in rows], [0, 1, 2])
                    self.assertEqual([row["candidate_sha256"] for row in rows], expected_ids)
                    self.assertEqual([row["status"] for row in rows], ["evaluated", "evaluated", expected_status])
                    self.assertEqual([row["selected"] for row in rows], [False, True, False])
                    self.assertTrue(rows[0]["correctness_passed"])
                    self.assertFalse(rows[0]["search_qualified"])
                    self.assertEqual(rows[0]["measurement_quality"], "unstable")
                    self.assertEqual(rows[0]["search_latency_ms"], 0.1)
                    self.assertTrue(rows[1]["correctness_passed"])
                    self.assertTrue(rows[1]["search_qualified"])
                    self.assertEqual(rows[1]["search_latency_ms"], 1.0)
                    self.assertEqual(rows[0]["findings"][0]["code"], "FIXTURE_VARIANT_0")
                    self.assertEqual(rows[1]["findings"][0]["code"], "FIXTURE_VARIANT_1")
                    self.assertIsNotNone(rows[0]["profile"])
                    self.assertIsNotNone(rows[1]["profile"])
                    self.assert_no_measurement(rows[2])
                    if peer == "duplicate":
                        self.assertEqual(rows[2]["same_program_as"], expected_ids[0])
                    self.assertEqual(request.state_card["previous_feedback"], feedback)
                    terminal = next(event["payload"]["state"] for event in events
                                    if event["kind"] == "search_completed")
                    self.assertEqual(terminal["previous_feedback"]["source_turn"], 2)
                    final_selection = [event["payload"] for event in events
                                       if event["kind"] == "candidate_selected"][-1]
                    self.assertEqual(terminal["previous_feedback"]["selected_candidate_sha256"],
                                     final_selection["candidate_sha256"])
                self.assertTrue(lab.audit(campaign).semantic_replay_passed)

    def test_all_build_rejections_select_diagnostics_without_claiming_evaluation(self):
        with self.campaign("all_rejected") as (lab, lock, provider, campaign, store):
            requests = [request for request in provider.requests if request.turn == 2]
            self.assertEqual(len(requests), len(lock.run_order))
            for request in requests:
                feedback = request.feedback
                self.assertEqual(feedback["source_turn"], 1)
                self.assertEqual(len(feedback["candidate_results"]), 3)
                self.assertEqual(feedback["selected_candidate_sha256"],
                                 feedback["candidate_results"][0]["candidate_sha256"])
                for index, row in enumerate(feedback["candidate_results"]):
                    self.assertEqual(row["status"], "build_rejected")
                    self.assertEqual(row["selected"], index == 0)
                    self.assert_no_measurement(row)
                self.assertFalse(any(event["kind"] == "evaluation_attempt_started"
                                     for event in store.replay_events(request.run_id)))
            self.assertTrue(lab.audit(campaign).semantic_replay_passed)

    @staticmethod
    def mutate_feedback(feedback, mutation):
        rows = feedback["candidate_results"]
        if mutation == "wrong_selected":
            feedback["selected_candidate_sha256"] = rows[0]["candidate_sha256"]
            rows[0]["selected"], rows[1]["selected"] = True, False
        elif mutation == "swapped_identity":
            rows[0]["candidate_sha256"], rows[1]["candidate_sha256"] = (
                rows[1]["candidate_sha256"], rows[0]["candidate_sha256"])
        elif mutation == "deleted_peer":
            rows.pop()
        elif mutation == "forged_unsearched":
            rows[2].update(status="evaluated", correctness_passed=True,
                           candidate_disposition="qualified", measurement_quality="stable",
                           search_qualified=True, search_latency_ms=0.01, findings=[])
        elif mutation == "wrong_turn":
            feedback["source_turn"] += 1
        else:
            raise AssertionError(mutation)

    def test_replay_rejects_forged_next_turn_feedback_even_with_consistent_rubric(self):
        with self.campaign() as (lab, lock, provider, campaign, store):
            self.assertTrue(lab.audit(campaign).semantic_replay_passed)
            request = next(request for request in provider.requests
                           if request.turn == 2 and request.arm == "open_cake")
            events = store.replay_events(request.run_id)
            turn = next(event for event in events if event["kind"] == "provider_turn_completed"
                        and event["payload"]["turn"] == 2)
            reference = next(item for item in turn["payload"]["objects"]
                             if item["role"] == "provider_reference_bundle")
            read_object = store.read_object
            original = read_object(reference)
            bundle = json.loads(original)
            audit = store.audit_run(request.run_id)
            for mutation in ("wrong_selected", "swapped_identity", "deleted_peer",
                             "forged_unsearched", "wrong_turn"):
                with self.subTest(mutation=mutation):
                    state = copy.deepcopy(bundle["state_card"])
                    self.mutate_feedback(state["previous_feedback"], mutation)
                    # Re-derive the rubric as the author interface does. A rejection
                    # must bind feedback to measurements, not only to its own rubric.
                    forged = provider.packages[request.run_id].evidence_bundle(state)
                    def read_forged(item):
                        return forged if item.get("sha256") == reference["sha256"] else read_object(item)
                    with patch.object(store, "read_object", side_effect=read_forged):
                        self.assertFalse(lab._replay_matched_run(store, audit, lock))
            # Semantic probes modify only the mocked read; frozen evidence stays intact.
            self.assertEqual(read_object(reference), original)

    def test_replay_rejects_forged_terminal_feedback_without_a_future_turn(self):
        with self.campaign() as (lab, lock, provider, campaign, store):
            self.assertTrue(lab.audit(campaign).semantic_replay_passed)
            run_id = next(run_id for run_id in lock.run_order if run_id.startswith("open_cake"))
            original = [dict(event) for event in store.replay_events(run_id)]
            audit = store.audit_run(run_id)
            for mutation in ("swapped_identity", "deleted_peer", "forged_unsearched"):
                with self.subTest(mutation=mutation):
                    events = copy.deepcopy(original)
                    for event in events:
                        if event["kind"] == "search_completed":
                            self.mutate_feedback(event["payload"]["state"]["previous_feedback"], mutation)
                        elif event["kind"] == "checkpoints_projected":
                            self.mutate_feedback(event["payload"]["ralph"]["previous_feedback"], mutation)
                    with patch.object(store, "replay_events", return_value=tuple(events)):
                        self.assertFalse(lab._replay_matched_run(store, audit, lock))
            self.assertEqual([dict(event) for event in store.replay_events(run_id)], original)


if __name__ == "__main__":
    unittest.main()
