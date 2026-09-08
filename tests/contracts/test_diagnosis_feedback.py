"""CPU diagnosis seams; explicit drafts/doubles never qualify a host or candidate."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler, CompilerError, LoweringRefusedError
from open_cake_ir.lab import CandidateSubmission
from open_cake_ir.lab.routing import route_rejection
from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment
from open_cake_ir.tasks.workloads import load_workload
from tests.contracts.test_authoring_environment import RecordingToolchain, _headline_schedule


class DiagnosisSeamTests(unittest.TestCase):
    def environment(self, *, python=False):
        # This is the explicit prospective source domain, not the stale released lock.
        draft = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        # Explicit interface double permits exercising a prospective Environment seam;
        # no released descriptor or admission record is created or rewritten.
        compiler = Mock(wraps=draft)
        compiler.state = "released"
        workload = load_workload(ROOT / "contracts/workloads/flash-kmeans-assign-v2.json")
        study = json.loads((ROOT / "contracts/studies/matched-search-infrastructure-template.json").read_text())
        if python:
            study["arms"]["open_cake"]["input_format"] = "schedule_or_python_v1"
        toolchain = RecordingToolchain()
        environment = TaskOpenCakeEnvironment(compiler, toolchain,
            authority_document=study["arms"]["open_cake"], workload=workload, case_id="headline_b32")
        submission = CandidateSubmission.seal(environment.media_type, json.dumps(_headline_schedule(workload)).encode())
        return compiler, environment, submission, toolchain

    def test_controlled_lowering_refusal_reaches_router_without_build(self):
        compiler, environment, submission, toolchain = self.environment()
        with patch.object(compiler, "lower", side_effect=LoweringRefusedError("Schedule does not determine its source: missing body")):
            result = environment.build(submission)
        self.assertEqual(result.disposition, "rejected")
        self.assertEqual(result.feedback["stage"], "lowering")
        self.assertEqual(result.feedback["code"], "LOWERING_UNDETERMINED")
        self.assertEqual(route_rejection(result.feedback).destination, "ir_vocabulary")
        self.assertEqual(toolchain.requests, [])

    def test_actual_python_refusal_preserves_top_level_location_for_peer(self):
        from open_cake_ir.lab.diagnoses import rejected_peer_feedback
        _, environment, _, toolchain = self.environment(python=True)
        submission = CandidateSubmission.seal(environment.media_type, json.dumps({"python_source": "def broken(:\n"}).encode())
        result = environment.build(submission)
        self.assertEqual(result.disposition, "rejected")
        self.assertEqual(result.feedback["code"], "PYTHON_SYNTAX")
        peer = rejected_peer_feedback([(submission, result)], arm="open_cake")[0]
        self.assertEqual(peer["source_location"], {key: value for key, value in result.feedback["source_location"].items() if key != "filename"})
        self.assertEqual(peer["source_location"]["line"], 1)
        self.assertNotIn("filename", peer["source_location"])
        self.assertEqual(toolchain.requests, [])

    def test_unrelated_compiler_fault_is_not_candidate_rejection(self):
        compiler, environment, submission, toolchain = self.environment()
        with patch.object(compiler, "lower", side_effect=CompilerError("Revision source binding differs")):
            with self.assertRaisesRegex(CompilerError, "Revision source"):
                environment.build(submission)
        self.assertEqual(toolchain.requests, [])

    def test_compile_route_uses_frozen_arm_not_candidate_feedback(self):
        for arm in ("direct_cuda", "native_triton", "native_cute_dsl"):
            with self.subTest(arm=arm):
                self.assertEqual(route_rejection({"stage": "compile", "source_role": "lowered_source"}, arm=arm).destination, "candidate")
        self.assertEqual(route_rejection({"stage": "compile"}, arm="open_cake").destination, "verifier")
        with self.assertRaises(ValueError):
            route_rejection({"stage": "compile"}, arm="unknown")


class DiagnosisProjectionTests(unittest.TestCase):
    def test_all_peers_bounded_without_source_or_artifact_leakage(self):
        from open_cake_ir.lab.diagnoses import rejected_peer_feedback
        from open_cake_ir.lab.environments import EnvironmentResult
        built = []
        for index in range(3):
            submission = CandidateSubmission.seal("text/x-cuda", f"candidate {index}".encode())
            result = EnvironmentResult("rejected", submission.sha256, None,
                {"stage": "assessment", "error": "x" * 4000, "source": "hidden implementation",
                 "findings": [{"code": f"BAD_{n}", "path": f"operations[{n}]", "message": "m" * 900,
                               "blocks_acceptance": True, "unbounded": {"source": "hidden"}}
                              for n in range(12)]}, {"source": b"hidden implementation"})
            built.append((submission, result))
        rows = rejected_peer_feedback(built, arm="direct_cuda")
        self.assertEqual([row["candidate_sha256"] for row in rows], [submission.sha256 for submission, _ in built])
        for row in rows:
            self.assertEqual(len(row["findings"]), 8)
            self.assertEqual(row["omitted_findings"], 4)
            self.assertTrue(row["text_truncated"])
            self.assertEqual(len(row["error"]), 512)
            self.assertEqual(row["findings"][0]["path"], "operations[0]")
        self.assertNotIn("hidden", json.dumps(rows))
        self.assertNotIn("unbounded", json.dumps(rows))

    def test_qsa_authored_compile_refusal_uses_common_router(self):
        from open_cake_ir.tasks.qsa.feedback import qsa_evaluation_feedback
        from tests.contracts.test_qsa_feedback import _completed
        value = _completed()
        value.update(outcome="rejected", validity="invalid")
        value["stages"] = [{"id": "compile", "status": "failed", "summary": "syntax error"}]
        for arm, destination in (("direct_cuda", "candidate"), ("open_cake", "verifier")):
            feedback = qsa_evaluation_feedback(value, arm=arm)
            self.assertEqual(feedback["routed_to"], destination)
            self.assertEqual(feedback["routing_reason"], route_rejection({"stage": "compile"}, arm=arm).reason)


class DiagnosisRunTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        from tests.contracts._executor_fixture import SemanticExecutorFixture, compiler_reference
        self.enterContext(SemanticExecutorFixture())
        # Diagnosis consumer test only: Compiler admission has its own release tests.
        # Exact interface fixture binds the same declared input at preflight/replay;
        # it neither writes a release nor represents a successful Corpus Gate.
        reference = compiler_reference(ROOT)
        gate = SimpleNamespace(compiler_revision_id=reference["revision_id"],
                               compiler_revision_sha256=reference["canonical_sha256"])
        def resolve(root, value, context, *, template):
            if value != ({"binding": "current_release"} if template else reference):
                raise ValueError("diagnosis fixture Compiler reference differs")
            return gate, reference["path"], reference
        def load(root, value, context):
            if value != reference:
                raise ValueError("diagnosis fixture Compiler reference differs")
        self.enterContext(patch("open_cake_ir.lab.preflight._resolve_compiler_reference", resolve))
        self.enterContext(patch("open_cake_ir.lab.bindings.load_compiler_reference", load))

    def test_real_next_turn_and_replay_retain_all_rejected_peers(self):
        import dataclasses
        import tempfile
        from hashlib import sha256
        from open_cake_ir.evidence import EvidenceStore
        from open_cake_ir.lab.environments import EnvironmentResult
        from open_cake_ir.tasks.runtime import TaskLab
        from tests.contracts.test_lab import FakeProvider, FakeEnvironment, FakeEvaluator, _enable_candidate_set, _execute, _submission_envelope

        class ThreeCandidates(FakeProvider):
            def turn(self, request):
                turn = super().turn(request)
                candidates = tuple(json.dumps({"turn": request.turn, "variant": i}, sort_keys=True, separators=(",", ":")).encode() for i in range(3))
                return dataclasses.replace(turn, candidates=candidates,
                    candidate_sha256s=tuple(sha256(value).hexdigest() for value in candidates),
                    raw_submission=_submission_envelope(request.arm, candidates))

        for mixed in (False, True):
            class Rejections(FakeEnvironment):
                def build(self, submission):
                    variant = json.loads(submission.payload)["variant"]
                    if mixed and variant == 0:
                        return super().build(submission)
                    return EnvironmentResult("rejected", submission.sha256, None,
                        {"stage": "compile", "diagnostic": f"syntax variant {variant}"})
            with self.subTest(mixed=mixed), tempfile.TemporaryDirectory() as directory:
                document = json.loads((ROOT / "contracts/studies/matched-search-infrastructure-template.json").read_text())
                _enable_candidate_set(document, 3)
                document["budget"]["maximum_turns"] = 2
                study = Path(directory) / "study.json"
                study.write_text(json.dumps(document))
                lab = TaskLab(ROOT)
                lock = lab.preflight(study)
                arms = lock.document["resolved_inputs"]["arm_environments"]
                provider = ThreeCandidates()
                protocol = lock.document["evaluation_protocol"]
                evaluator = FakeEvaluator(protocol, sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), lock.document["workload"]["canonical_sha256"])
                campaign = _execute(lab, lock, Path(directory) / "evidence", provider=provider,
                    environments={arm: Rejections(arm, authority) for arm, authority in arms.items()}, evaluator=evaluator)
                next_turns = [request for request in provider.requests if request.turn == 2]
                self.assertEqual(len(next_turns), len(lock.run_order))
                for request in next_turns:
                    peers = request.feedback["rejected_candidates"]
                    self.assertEqual(len(peers), 2 if mixed else 3)
                    self.assertEqual([row["diagnostic"] for row in peers],
                                     [f"syntax variant {i}" for i in range(1 if mixed else 0, 3)])
                    self.assertTrue(all(row["routed_to"] == ("verifier" if request.arm == "open_cake" else "candidate") for row in peers))
                    if mixed:
                        self.assertEqual(request.feedback["kind"], "evaluation")
                report = lab.audit(campaign)
                self.assertTrue(report.semantic_replay_passed)
                store = EvidenceStore.open(campaign.evidence_root)
                run_id = next(run for run in lock.run_order if run.startswith("direct_cuda"))
                audit = store.audit_run(run_id)
                events = [dict(event) for event in store.replay_events(run_id)]
                rejected = next(event for event in events if event["kind"] == "candidate_rejected")
                rejected["payload"] = {**rejected["payload"], "routed_to": "verifier"}
                # In-memory tamper at the independent semantic boundary; archive untouched.
                with patch.object(store, "replay_events", return_value=tuple(events)):
                    self.assertFalse(lab._replay_matched_run(store, audit, lock))


class DiagnosisSummaryTests(unittest.TestCase):
    def test_cross_root_counts_are_read_only_deduplicated_and_policy_scoped(self):
        import os
        import shutil
        import subprocess
        import tempfile
        from hashlib import sha256
        from open_cake_ir.evidence import EvidenceStore
        from open_cake_ir.serialization import canonical_json_bytes
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict(os.environ, {"OPEN_CAKE_CUSTODY_DIRECTORY": str(root / "registry")}):
                paths = []
                for index in range(2):
                    store = EvidenceStore.create(root / f"evidence-{index}")
                    authority = {"campaign_id": f"fixture-{index}", "execution": {
                        "executor_revision": {"executor_id": f"fixture-executor-{index}", "canonical_sha256": str(index) * 64}},
                        "resolved_inputs": {"evidence_policy": {"event_vocabulary": f"fixture-{index}"}}}
                    run = store.start_run("direct_cuda-1", authority=authority,
                        authority_sha256=sha256(canonical_json_bytes(authority)).hexdigest())
                    run.append("candidate_rejected", {"turn": 1, "candidate_sha256": "a" * 64,
                        "routed_to": "candidate", "routing_reason": "retained prior routing", "feedback": {"diagnostic": "do not print source text"}})
                    run.append("diagnosis_routed", {"turn": 1, "routed_to": "cost_model", "routing_reason": "ranking inversion"})
                    run.seal(protocol_adherence="adhered", endpoint_observation="observed", endpoint={"kind": "fixture"})
                    paths.append(store.root)
                copied = root / "copied"
                shutil.copytree(paths[0], copied)
                paths.append(copied)
                before = {path: (path.read_bytes(), path.stat().st_mode) for evidence_root in paths for path in evidence_root.rglob("*") if path.is_file()}
                completed = subprocess.run([sys.executable, str(ROOT / "tools/summarize_diagnoses.py"), *map(str, paths)],
                    capture_output=True, text=True, check=True)
                result = json.loads(completed.stdout)
                self.assertEqual(len(result["groups"]), 2)
                self.assertEqual(len(result["runs"]), 2)
                for group in result["groups"]:
                    self.assertEqual(group["counts"], {"candidate_rejected:candidate": 1, "diagnosis_routed:cost_model": 1})
                self.assertTrue(any(not run["filesystem_custody_verified"] for run in result["runs"]))
                self.assertNotIn("do not print source text", completed.stdout)
                self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mode) for path in before})
                # An incomplete new root is refused, not silently omitted or repaired.
                broken = EvidenceStore.create(root / "incomplete")
                broken.start_run("direct_cuda-1", authority=authority,
                    authority_sha256=sha256(canonical_json_bytes(authority)).hexdigest())
                refused = subprocess.run([sys.executable, str(ROOT / "tools/summarize_diagnoses.py"), str(broken.root)], capture_output=True, text=True)
                self.assertEqual(refused.returncode, 1)
                self.assertIn("archive integrity failed", refused.stderr)
                self.assertEqual(refused.stdout, "")
