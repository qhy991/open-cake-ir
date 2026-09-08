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
        self.assertIn("```python", _document_sections({"starter.py": bound}))

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

    def test_common_reference_builder_publishes_python_with_bound_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative, contents in {
                "starter.py": self.source, "workload.json": "{}", "target.json": "{}", "scaffold.md": "fixture",
                "docs/PYTHON_FRONTEND.md": "Python frontend fixture",
                "compiler.json": json.dumps({"target_definitions": {"apple_gpu_family8": {"path": "target.json"}}}),
            }.items():
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(contents)
            lock = SimpleNamespace(document={"workload": {"path": "workload.json", "canonical_sha256": "a" * 64},
                "compiler_revision": {"path": "compiler.json"}, "study": {},
                "execution": {"target": "apple_gpu_family8"}, "evaluation_protocol": {"case_id": "primary"},
                "resolved_inputs": {"budget": {}, "run_protocol": {}}})
            arm = {"environment_kind": "open_cake", "input_format": "schedule_or_python_v1",
                "schedule_skeleton": {"path": "starter.py"}, "scaffold": {"path": "scaffold.md"},
                "lowering_route": self.document["lowering"]}
            def prepare(document, workload, case_id, authority):
                return {**document, "metadata": {"workload_contract_sha256": workload.canonical_sha256}}
            docs = build_run_reference_documents(root, lock, arm,
                workload_contract=SimpleNamespace(canonical_sha256="a" * 64), prepare_schedule=prepare)
            self.assertIn("schedule-starter.py", docs)
            self.assertNotIn("schedule-skeleton.json", docs)
            self.assertNotIn("paired-triton-authoring.md", docs)
            self.assertEqual(frontend.parse(docs["schedule-starter.py"].decode()).document["metadata"],
                             {"workload_contract_sha256": "a" * 64})


if __name__ == "__main__":
    unittest.main()
