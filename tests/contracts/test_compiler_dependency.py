"""Compiler dependency admission at independent CPU worker/build boundaries."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.lab.bindings import load_compiler_reference
from open_cake_ir.tasks import evaluate as worker
from tests.contracts._executor_fixture import compiler_reference

ROOT = Path(__file__).resolve().parents[2]


class CompilerDependencyTests(unittest.TestCase):
    def test_exact_released_dependency_resolves_its_own_target(self):
        revision = load_compiler_reference(ROOT, compiler_reference(ROOT), "fixture")
        self.assertEqual(revision.revision_id, compiler_reference(ROOT)["revision_id"])
        self.assertEqual(revision.targets["apple_gpu_family7"].target_id, "apple_gpu_family7")

    def test_missing_moving_stale_and_extra_dependency_fields_refuse(self):
        reference = compiler_reference(ROOT)
        for value in (None, {"binding": "current_release"}, {**reference, "revision_id": "wrong"},
                      {**reference, "canonical_sha256": "0" * 64}, {**reference, "extra": True}):
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                load_compiler_reference(ROOT, value, "fixture")

    def test_changed_target_is_refused_by_compiler_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            shutil.copytree(ROOT / "compiler", root / "compiler")
            target = root / "compiler/targets/apple_gpu_family7.json"
            document = json.loads(target.read_bytes())
            document["device_names"] = ["substituted test target"]
            target.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, "[Tt]arget"):
                load_compiler_reference(root, compiler_reference(ROOT), "fixture")

    def test_worker_refuses_dependency_before_workload_or_artifact_interpretation(self):
        for reference, target in ((None, "sm_100a"),
                ({**compiler_reference(ROOT), "revision_id": "wrong"}, "sm_100a"),
                (compiler_reference(ROOT), "unbound_target")):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "request.json"
                path.write_text(json.dumps({"compiler_revision": reference, "target": target}))
                with patch.object(worker.ExecutorRevision, "load_reference", return_value=object()), \
                     patch.object(worker, "load_workload") as workload:
                    with self.assertRaisesRegex(ValueError, "Compiler|compiler"):
                        worker._load_authority(path)
                    workload.assert_not_called()

    def test_qsa_stage_command_binds_exact_dependency(self):
        from types import SimpleNamespace
        from open_cake_ir.tasks.qsa import evaluate as qsa
        reference = compiler_reference(ROOT)
        executor = SimpleNamespace(executor_id="CPU-fixture", canonical_sha256="e" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.json"
            for value in (reference, {**reference, "revision_id": "other"}):
                path.write_text(json.dumps({"stages": [{"id": "correctness", "judge": {
                    "identity": "CPU-fixture@" + "e" * 64,
                    "command": ["python", "--compiler-reference", json.dumps(value)]}}]}))
                with patch.object(qsa, "resolve_executor", return_value=executor), \
                     patch.dict("os.environ", {"KERNELINFRA_TASK": str(path), "KERNELINFRA_STAGE_ID": "correctness"}):
                    if value == reference:
                        self.assertIs(qsa._executor(ROOT, reference), executor)
                    else:
                        with self.assertRaisesRegex(ValueError, "Compiler command binding"):
                            qsa._executor(ROOT, reference)

    def test_qsa_profiler_child_cannot_bypass_compiler_admission(self):
        from open_cake_ir.tasks.qsa import evaluate as qsa
        with patch.object(qsa, "_profile_child") as child:
            with self.assertRaisesRegex(ValueError, "Compiler Revision"):
                qsa.main(["--project-root", str(ROOT), "--nvcc", "/not-invoked/nvcc",
                    "--cuobjdump", "/not-invoked/cuobjdump", "--profile-child", "--build-root", str(ROOT),
                    "--compiler-reference", json.dumps({**compiler_reference(ROOT), "revision_id": "wrong"})])
            child.assert_not_called()

    def test_replay_refuses_foreign_dependency_even_with_consistent_request_identity(self):
        from dataclasses import replace
        from hashlib import sha256
        from open_cake_ir.evaluation import LaunchableCandidate
        from open_cake_ir.evidence import EvidenceStore
        from open_cake_ir.lab.archive import _archive_logical_attempt, _logical_attempt_document
        from open_cake_ir.lab.replay_attempts import _replay_broker_attempt_ledger
        from tests.contracts.test_lab import FakeEvaluator
        payload = b"CPU fixture executable; never launched"
        digest = sha256(payload).hexdigest()
        candidate = LaunchableCandidate("a" * 64, "sm_100a", "open_cake_turn_1",
            {"cubin": digest}, digest, {"cubin": payload})
        logical = FakeEvaluator({}, "b" * 64, "c" * 64).evaluate(candidate, case_id="fixture", purpose="search")
        for foreign in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                evidence = EvidenceStore.create(Path(temporary).resolve() / "evidence")
                attempt = logical.attempts[0]
                request = json.loads(attempt.artifact_payloads["evaluator_request"])
                if foreign:
                    request["compiler_revision"]["revision_id"] = "another-compiler"
                authority = {k: v for k, v in request.items() if k != "attempt"}
                encoded = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                attempt = replace(attempt, evaluator_arguments_sha256=sha256(encoded(authority)).hexdigest(),
                    artifact_payloads={**attempt.artifact_payloads, "evaluator_request": encoded(request)})
                observed = replace(logical, attempts=(attempt,))
                refs = _archive_logical_attempt(evidence, observed)
                def replay():
                    _replay_broker_attempt_ledger(evidence, refs, _logical_attempt_document(observed),
                        candidate=candidate, protocol_sha256="b" * 64, final_receipt=logical.final_receipt,
                        compiler_reference=compiler_reference(ROOT))
                if foreign:
                    with self.assertRaisesRegex(ValueError, "Compiler dependency differs"):
                        replay()
                else:
                    replay()
