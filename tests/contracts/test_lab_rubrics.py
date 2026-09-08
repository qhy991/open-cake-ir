"""CPU probes of delivered guidance, retained evidence and provider replay."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.archive import _archive_provider_turn
from open_cake_ir.lab.providers import (
    CANDIDATE_SET_ENVELOPE_V1, CodexInvocationBuilder, CodexRunProvider,
    ProviderQualificationReceipt,
)
from open_cake_ir.lab.ralph import RalphBudget, RalphController
from open_cake_ir.lab.replay_provider import _replay_provider_turns
from open_cake_ir.lab.rubrics import derive_rubric
from open_cake_ir.lab.task_package import TaskPackage, materialize_task_package
from tests.contracts.test_lab import FakeProvider


def states(rubric):
    return {row["criterion"]: row["state"] for row in rubric["criteria"]}


def evaluation(**updates):
    return {
        "kind": "evaluation", "candidate_disposition": "qualified",
        "measurement_quality": "stable", "confirmed": True,
        "search_latency_ms": 1.0, "confirmed_latency_ms": 1.1,
        "findings": [], "profile": {"metrics": {"occupancy": "unknown"}},
        **updates,
    }


class RubricContractTests(unittest.TestCase):
    def test_initial_unknown_absent_and_malformed_stay_distinct(self):
        self.assertEqual(states(derive_rubric({"kind": "initial"}))["admission"], "initial")
        for value, expected in ((None, "unknown"), ([], "malformed"), (False, "malformed")):
            with self.subTest(value=value):
                self.assertEqual(set(states(derive_rubric(value)).values()), {expected})
        self.assertEqual(set(states(derive_rubric()).values()), {"absent"})
        missing = states(derive_rubric({}))
        self.assertEqual(missing["admission"], "unknown")
        self.assertEqual(missing["confirmation"], "absent")
        unknown = derive_rubric({"kind": "future_feedback", "candidate_disposition": "future_result",
                                "measurement_quality": "unknown", "confirmed": None,
                                "profile": "unknown", "findings": "unknown"})
        self.assertEqual(set(states(unknown).values()), {"unknown"})
        malformed = states(derive_rubric({"kind": [], "candidate_disposition": True,
                                        "measurement_quality": {}, "confirmed": "true",
                                        "profile": 3, "findings": [None]}))
        self.assertEqual(set(malformed.values()), {"malformed"})
        for key in ("blocks_lowering", "blocks_acceptance"):
            self.assertEqual(states(derive_rubric({"findings": [{key: "false"}]}))["findings"], "malformed")

    def test_evaluation_axes_preserve_independent_reported_states(self):
        expected = {"admission": "observed", "correctness": "qualified", "measurement": "stable",
                    "confirmation": "confirmed", "profile": "observed", "findings": "none"}
        self.assertEqual(states(derive_rubric(evaluation())), expected)
        for update, axis, status in (
            ({"confirmed": False}, "confirmation", "unconfirmed"),
            ({"measurement_quality": "unstable", "confirmed": False}, "measurement", "unstable"),
            ({"measurement_quality": "not_measured"}, "measurement", "not_measured"),
            ({"candidate_disposition": "correctness_rejected"}, "correctness", "correctness_rejected"),
            ({"profile": None}, "profile", "unavailable"),
            ({"profile": {}}, "profile", "unknown"),
        ):
            with self.subTest(update=update):
                self.assertEqual(states(derive_rubric(evaluation(**update)))[axis], status)
        no_profile = evaluation()
        del no_profile["profile"]
        self.assertEqual(states(derive_rubric(no_profile))["profile"], "absent")
        self.assertIn("not evidence of a bottleneck or causal mechanism", json.dumps(derive_rubric(evaluation())))

    def test_local_refusal_preserves_raw_findings_without_routing_or_mutation(self):
        finding = {"code": "UNSEEN_BACKEND_CAPABILITY", "path": "operations[3]",
                   "category": "hardware_conformance", "message": "cannot lower this operation",
                   "severity": "blocking", "blocks_acceptance": False, "blocks_lowering": True,
                   "source_location": {"line": 7}, "declared_resource": "registers"}
        feedback = {"stage": "assessment", "findings": [finding],
                    "calibration_available": False, "private_future_field": {"x": [1, 2]}}
        before = copy.deepcopy(feedback)
        # The pure projection has no resource access; references stay relative to this bundle.
        with patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected read")):
            rubric = derive_rubric(feedback)
            self.assertEqual(rubric, derive_rubric(feedback))
        self.assertEqual(feedback, before)
        self.assertEqual(states(rubric)["admission"], "refused")
        self.assertEqual(states(rubric)["correctness"], "absent")
        self.assertNotIn("destination", json.dumps(rubric))
        self.assertIn("need not mean the candidate is wrong", json.dumps(rubric))
        for criterion in rubric["criteria"]:
            for pointer in criterion["evidence"]:
                self.assertIn(pointer.removeprefix("/state_card/previous_feedback/"), feedback)
        rubric["criteria"][0]["evidence"].clear()
        self.assertEqual(feedback, before)
        compile_feedback = derive_rubric({"stage": "compile", "diagnostic": "toolchain refused"})
        self.assertEqual(states(compile_feedback)["admission"], "refused")
        self.assertEqual(states(compile_feedback)["correctness"], "absent")

    def test_real_provider_delivers_retains_and_replays_each_rubric(self):
        # Real Ralph, provider, archive and replay paths; only process execution is a CPU fixture.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            executable = root / "codex"
            executable.write_bytes(b"CPU CLI fixture")
            executable.chmod(0o700)
            helper = root / "codex-code-mode-host"
            helper.write_bytes(b"CPU Code Mode host fixture")
            helper.chmod(0o700)
            workspace = root / "workspace"
            workspace.mkdir()
            schema = root / "schema.json"
            schema.write_text("{}")
            package = TaskPackage("open_cake-1", "open_cake", "# Task\n中文\n", "# Rules\n")
            materialize_task_package(workspace, package)
            builder = CodexInvocationBuilder(
                executable=executable, provider_revision="rubric-cpu-fixture",
                model="gpt-5.6-sol", reasoning_effort="max", service_tier="default",
                workspace=workspace, output_schema=schema, removed_environment=("OPENAI_API_KEY",),
                submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                cwd_policy="independent_task_workspace", reference_visibility="workspace_task_files",
            )
            qualification = ProviderQualificationReceipt(
                provider_revision=builder.provider_revision,
                executable_sha256=sha256(executable.read_bytes()).hexdigest(),
                configuration_sha256=builder.configuration_sha256,
                initial_and_resume_equivalent=True, file_lifecycle_observed=True,
                usage_observed=True, qualified=True, scope="live_two_turn_current_provider",
            )
            fake = FakeProvider()
            fake.packages = {package.run_id: package}
            invocations = []

            class Adapter:
                def execute(self, invocation, **kwargs):
                    invocations.append(invocation)
                    # FakeProvider supplies existing closed-event-contract bytes, without an external call.
                    result = fake.turn(current_request)
                    kwargs["candidate_path"].write_bytes(result.raw_submission)
                    return replace(result, reference_bundle=None)

            provider = CodexRunProvider(qualification=qualification, builders={package.run_id: builder},
                                        task_packages={package.run_id: package}, adapter=Adapter())
            controller = RalphController(RalphBudget(10_000_000, 20, 1, 100, 100, 20, 20, 20),
                                         searches_per_turn=1, profile_each_search_survivor=True, clock=lambda: 0.0)
            feedbacks = [
                {"kind": "initial"}, {"stage": "assessment", "error": "unsupported lowering"},
                evaluation(), evaluation(confirmed=False), evaluation(measurement_quality="unstable", confirmed=False),
                evaluation(profile=None), {"kind": "future_feedback"}, {"kind": "evaluation", "confirmed": "true"},
            ]
            evidence = EvidenceStore.create(root / "evidence")
            events = []
            ledger = SimpleNamespace(append=lambda kind, payload: events.append({"kind": kind, "payload": payload}))
            cumulative = 0
            thread = None
            for turn, feedback in enumerate(feedbacks, 1):
                before = copy.deepcopy(feedback)
                state = controller.state_card(turn=turn, cumulative_provider_tokens=cumulative, feedback=feedback)
                current_request = SimpleNamespace(run_id=package.run_id, arm=package.arm, turn=turn,
                    cumulative_provider_tokens=cumulative, thread_id=thread, feedback=feedback,
                    maximum_candidates_per_turn=1, state_card=state)
                result = provider.turn(current_request)
                thread = result.thread_id
                cumulative += result.provider_tokens
                _archive_provider_turn(arm=package.arm, candidate_media_type="application/json",
                    cumulative_tokens=cumulative, evidence=evidence, ledger=ledger, maximum_candidates_per_turn=1,
                    provider_document={}, provider_turn=result, thread_id=thread, turn_number=turn)
                reference = next(item for item in events[-1]["payload"]["objects"] if item["role"] == "provider_reference_bundle")
                retained = evidence.read_object(reference)
                self.assertEqual(retained, result.reference_bundle)
                self.assertTrue(invocations[-1].argv[-1].endswith(retained.decode()))
                bundle = json.loads(retained)
                self.assertEqual(bundle["rubric"], derive_rubric(before))
                self.assertEqual(bundle["state_card"]["previous_feedback"], before)
                self.assertEqual(feedback, before)
                self.assertEqual(controller.stop_reason(turn=turn, cumulative_provider_tokens=cumulative), None)
                self.assertEqual(dict(state["evaluation_counts"]), {"search": 0, "confirmatory": 0, "attribution": 0})
            self.assertEqual({p.name for p in workspace.iterdir()}, {"TASK.md", "AGENTS.md", "candidate-set.json"})

            def replay(rows):
                return _replay_provider_turns(arm=package.arm, audit=SimpleNamespace(run_id=package.run_id),
                    event_contract="closed_file_change_v1", evidence=evidence, expected_task_package=package,
                    maximum_candidates_per_turn=1, provider_authority={}, provider_events=rows)

            self.assertIsNotNone(replay(events))
            for tamper in ("state", "guidance", "missing", "extra"):
                with self.subTest(tamper=tamper):
                    changed = copy.deepcopy(events)
                    references = changed[0]["payload"]["objects"]
                    original = next(item for item in references if item["role"] == "provider_reference_bundle")
                    bundle = json.loads(evidence.read_object(original))
                    if tamper == "missing":
                        del bundle["rubric"]
                    elif tamper == "extra":
                        bundle["rubric"]["score"] = 100
                    else:
                        bundle["rubric"]["criteria"][0][tamper] = "forged"
                    # Re-seal the replacement through CAS: rejection must come from semantic replay.
                    replacement = evidence.put(json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(), media_type="application/json")
                    references[references.index(original)] = replacement.reference("provider_reference_bundle")
                    self.assertIsNone(replay(changed))


if __name__ == "__main__":
    unittest.main()
