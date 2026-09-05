from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/gpu/state_store_b200_correctness"
sys.path.insert(0, str(ROOT / "src"))


def _generator():
    path = EXAMPLE / "prepare_candidate.py"
    spec = importlib.util.spec_from_file_location("state_store_b200_prepare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load state-store B200 generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StateStoreB200CorrectnessContractTests(unittest.TestCase):
    def test_compiler_release_drift_is_refused_before_creating_output(self) -> None:
        generator = _generator()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            # The historical v40 object and this v41 implementation must not mix.
            with mock.patch.object(
                generator, "FIXED_SOURCE_COMMIT",
                "7fdac036626a4eaf0a0e047dca1565c0a88f19aa",
            ):
                with self.assertRaisesRegex(RuntimeError, "Compiler lock differs"):
                    generator.prepare(output)
            self.assertFalse(output.exists())

    def test_source_archive_without_frozen_git_object_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "archive"
            shutil.copytree(ROOT / "src", archive / "src")
            script = archive / "examples/gpu/state_store_b200_correctness/prepare_candidate.py"
            script.parent.mkdir(parents=True)
            shutil.copyfile(EXAMPLE / "prepare_candidate.py", script)
            output = Path(directory) / "bundle"
            completed = subprocess.run(
                [sys.executable, "-I", str(script), "--output-root", str(output)],
                cwd=archive, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("requires a Git checkout containing Compiler commit", completed.stderr)
            self.assertFalse(output.exists())

    def test_compiler_imported_from_another_checkout_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            foreign_src = Path(directory) / "src"
            shutil.copytree(ROOT / "src", foreign_src)
            output = Path(directory) / "bundle"
            completed = subprocess.run(
                [sys.executable, "-I", "-c",
                 "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
                 "import open_cake_ir.compiler; "
                 "runpy.run_path(sys.argv.pop(1),run_name='__main__')",
                 str(foreign_src), str(EXAMPLE / "prepare_candidate.py"),
                 "--output-root", str(output)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Compiler from a different checkout", completed.stderr)
            self.assertFalse(output.exists())

    def test_frozen_task_is_one_broker_correctness_stage(self) -> None:
        task = json.loads((EXAMPLE / "task.json").read_text(encoding="utf-8"))
        self.assertEqual(
            task["task_id"],
            "open-cake-state-store-b8x128-b200-correctness-v5",
        )
        self.assertIn("Frozen Compiler v41 successor", task["description"])
        self.assertIn("GPU validation pending", task["description"])
        self.assertEqual(len(task["stages"]), 1)
        stage = task["stages"][0]
        self.assertEqual(stage["kind"], "correctness")
        self.assertEqual(stage.get("execution", "broker"), "broker")
        self.assertEqual(stage["resources"]["mode"], "shared")
        self.assertEqual(stage["resources"]["gpu_count"], 1)
        self.assertNotIn("benchmark", json.dumps(task))
        self.assertNotIn("profile", json.dumps(task))

    def test_catalog_is_fixed_to_the_declared_b200_endpoint(self) -> None:
        catalog = json.loads((EXAMPLE / "catalog.json").read_text(encoding="utf-8"))
        self.assertEqual(len(catalog["nodes"]), 1)
        node = catalog["nodes"][0]
        self.assertEqual(node["ssh"], "verda-b200x4")
        self.assertEqual(
            node["socket"],
            "/tmp/kernelinfra-open-cake-state-store-b200-v5.sock",
        )
        self.assertIn("state-store-b200-v5/gpu-infra", node["kernelctl"])
        self.assertEqual(set(node["capabilities"]), {"b200", "cuda", "sm100"})

    def test_judge_is_syntax_valid_and_declares_four_full_outputs(self) -> None:
        source = (EXAMPLE / "evaluate.py").read_text(encoding="utf-8")
        compile(source, str(EXAMPLE / "evaluate.py"), "exec")
        self.assertIn('"state_before"', source)
        self.assertIn('"update_before"', source)
        self.assertIn('"actual_state_after"', source)
        self.assertIn("state_mismatch_count", source)
        self.assertIn("update_mismatch_count", source)
        self.assertIn("state_data_ptr_unchanged", source)
        self.assertIn("wrapper_returned_empty_tuple", source)


if __name__ == "__main__":
    unittest.main()
