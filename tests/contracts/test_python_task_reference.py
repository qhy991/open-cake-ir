"""Python starter projection through the existing Lab task package; no execution."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from open_cake_ir.compiler import frontend
from open_cake_ir.lab.python_reference import bind_python_reference, read_skeleton
from open_cake_ir.lab.task_package import build_run_reference_documents, _document_sections

ROOT = Path(__file__).resolve().parents[2]


class PythonTaskReferenceTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / "examples/python/metal_rmsnorm.py").read_text()
        self.document = frontend.parse(self.source).document

    def test_only_workload_metadata_changes_in_bound_python(self):
        prepared = {**self.document, "metadata": {"workload_contract_sha256": "a" * 64}}
        bound = bind_python_reference(self.source, prepared, filename="starter.py")
        self.assertEqual(frontend.parse(bound.decode()).document, prepared)
        self.assertIn(b"@cake.schedule", bound)
        self.assertIn(b"lm.reduce", bound)
        self.assertIn("```python", _document_sections({"schedule-starter.py": bound}, access="known_kernel_reproduction"))

    def test_task_preparation_cannot_silently_retarget_or_reshape_python(self):
        for key, value in (("target", "apple_gpu_family7"), ("schedule_id", "another-program")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "only metadata"):
                bind_python_reference(self.source, {**self.document, key: value}, filename="starter.py")

    def test_reading_a_python_skeleton_never_executes_module_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "starter.py"
            marker = Path(temporary) / "executed"
            path.write_text(f"open({str(marker)!r}, 'w').write('bad')\n" + self.source)
            with self.assertRaises(frontend.FrontendError):
                read_skeleton(path)
            self.assertFalse(marker.exists())
            path.write_text(self.source)
            self.assertEqual(read_skeleton(path), self.document)

    def test_explicit_metal_binding_uses_common_task_preparation_without_cuda_fallback(self):
        from open_cake_ir.lab.pairing import bind_baseline
        from open_cake_ir.tasks.authoring import prepare_schedule
        tensors = [SimpleNamespace(name=b["name"], shape=tuple(b["shape"]), dtype=b["dtype"], mode=b["mode"])
                   for b in self.document["buffers"] if b["space"] == "global"]
        workload = SimpleNamespace(document={"semantics": {"target": "apple_gpu_family8"}},
            canonical_sha256="a" * 64, tensor_abi=lambda _: tensors)
        with self.assertRaisesRegex(ValueError, "lowering backend"):
            bind_baseline(self.document, workload, "primary")
        prepared = prepare_schedule(self.document, workload, "primary", {
            "input_format": "schedule_or_python_v1", "lowering_route": self.document["lowering"]})
        self.assertEqual(prepared["target"], "apple_gpu_family8")
        self.assertEqual(prepared["lowering"]["backend"], "metal")
        self.assertEqual(prepared["metadata"]["workload_contract_sha256"], "a" * 64)

    def test_common_retention_uses_actual_backend_products(self):
        from open_cake_ir.lab.archive import _arm_artifact_roles, _candidate_artifact_media_type
        self.assertEqual(_arm_artifact_roles("open_cake", "apple_gpu_family7"), {
            "lowered_source", "metal_binary_archive", "metal_build_report", "launch_manifest"})
        self.assertEqual(_arm_artifact_roles("open_cake", "sm_103a"), {
            "lowered_source", "compiler_expanded_source", "ptx", "cubin", "launch_manifest"})
        with self.assertRaisesRegex(ValueError, "compiled target"):
            _arm_artifact_roles("direct_cuda", "apple_gpu_family7")
        self.assertEqual(_candidate_artifact_media_type("metal_binary_archive"), "application/octet-stream")
        self.assertEqual(_candidate_artifact_media_type("metal_build_report"), "application/json")

    def test_existing_open_cake_environment_accepts_bound_python_metal_before_builder(self):
        from hashlib import sha256
        from open_cake_ir.compiler import Compiler
        from open_cake_ir.evaluation.core import LaunchableCandidate
        from open_cake_ir.lab.environments import CandidateSubmission, OpenCakeEnvironment
        from tests.contracts.test_metal_artifacts import abi_fixture
        from tools.metal import rmsnorm
        workload = abi_fixture()
        source = rmsnorm.source(3, 7, target="apple_gpu_family7")
        document = frontend.parse(source).document
        prepared = {**document, "metadata": {"workload_contract_sha256": workload.canonical_sha256}}
        bound = bind_python_reference(source, prepared, filename="candidate.py")
        observed = []
        class Builder:
            def build(self, request):
                observed.append(request)
                archive = b"explicit nonexecuted builder test double"
                payloads = {"lowered_source": request.source, "metal_binary_archive": archive}
                return LaunchableCandidate(request.candidate_sha256, request.target, request.entry_point,
                    {role: sha256(data).hexdigest() for role, data in payloads.items()}, "b" * 64, payloads)
        environment = OpenCakeEnvironment(Compiler.load(ROOT, "compiler/revision.lock.json"), Builder(),
            authority_document={"input_format": "schedule_or_python_v1", "lowering_route": prepared["lowering"]},
            workload=workload, case_id="odd")
        result = environment.build(CandidateSubmission.seal(environment.media_type,
            json.dumps({"python_source": bound.decode()}).encode()))
        self.assertEqual(result.disposition, "launchable")
        # A richer envelope is refused for what it actually is. Reporting it as a
        # lowering route the actor never changed cost a live Campaign three Turns.
        rejected = environment.build(CandidateSubmission.seal(environment.media_type,
            json.dumps({"python_source": bound.decode(), "candidate_id": "a",
                        "hypothesis": "shorter scalar tail"}).encode()))
        self.assertNotEqual(rejected.disposition, "launchable")
        message = json.dumps(dict(rejected.feedback))
        self.assertIn("python_source alone", message)
        self.assertIn("candidate_id", message)
        self.assertIn("hypothesis", message)
        self.assertNotIn("lowering route", message)
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].target, "apple_gpu_family7")
        self.assertEqual(observed[0].source_role, "lowered_source")
        self.assertIsNone(result.cost)

    def test_common_reference_builder_publishes_python_with_bound_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative, contents in {
                "starter.py": self.source, "workload.json": "{}", "target.json": "{}", "scaffold.md": "fixture",
                "docs/PYTHON_FRONTEND.md": "Python frontend fixture",
                "compiler/AUTHORING_CONTRACT.md": "Schedule authoring fixture",
                "examples/python/fma.py": (ROOT / "examples/python/fma.py").read_text(),
                "compiler.json": json.dumps({"target_definitions": {"apple_gpu_family8": {"path": "target.json"}}}),
            }.items():
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(contents)
            lock = SimpleNamespace(document={"workload": {"path": "workload.json", "canonical_sha256": "a" * 64},
                "compiler_revision": {"path": "compiler.json"}, "study": {},
                "execution": {"target": "apple_gpu_family8"}, "evaluation_protocol": {"case_id": "primary"},
                "resolved_inputs": {"budget": {}, "run_protocol": {}}})
            arm = {"environment_kind": "open_cake", "reference_access": "known_kernel_reproduction", "input_format": "schedule_or_python_v1",
                "schedule_skeleton": {"path": "starter.py"}, "scaffold": {"path": "scaffold.md"},
                "lowering_route": self.document["lowering"]}
            lock.document["resolved_inputs"]["arm_environments"] = {"open_cake": arm}
            def prepare(document, workload, case_id, authority):
                return {**document, "metadata": {"workload_contract_sha256": workload.canonical_sha256}}
            docs = build_run_reference_documents(root, lock, arm,
                workload_contract=SimpleNamespace(canonical_sha256="a" * 64), prepare_schedule=prepare)
            self.assertIn("schedule-starter.py", docs)
            self.assertNotIn("schedule-skeleton.json", docs)
            self.assertNotIn("paired-triton-authoring.md", docs)
            self.assertIn("python-frontend.md", docs)
            self.assertIn("python-example.py", docs)
            # Python-enabled arms retain their authoring surface even when the
            # workload provides its initial Schedule as JSON.
            (root / "starter.json").write_text(json.dumps(self.document))
            arm["schedule_skeleton"] = {"path": "starter.json"}
            json_starter_docs = build_run_reference_documents(root, lock, arm,
                workload_contract=SimpleNamespace(canonical_sha256="a" * 64), prepare_schedule=prepare)
            self.assertIn("schedule-skeleton.json", json_starter_docs)
            self.assertEqual(json_starter_docs["python-frontend.md"], docs["python-frontend.md"])
            self.assertEqual(json_starter_docs["python-example.py"], docs["python-example.py"])
            self.assertEqual(frontend.parse(docs["schedule-starter.py"].decode()).document["metadata"],
                             {"workload_contract_sha256": "a" * 64})


if __name__ == "__main__":
    unittest.main()
