from __future__ import annotations
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

class TaskOwnershipTests(unittest.TestCase):
    def test_common_layers_do_not_import_task_implementations(self):
        for package in ("lab", "evaluation"):
            for path in (ROOT/"src/open_cake_ir"/package).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    names=[]
                    if isinstance(node, ast.Import):
                        names=[item.name for item in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names=[importlib.util.resolve_name("."*node.level+(node.module or ""), "open_cake_ir."+package)
                               if node.level else node.module or ""]
                    self.assertFalse(any(name.startswith("open_cake_ir.tasks") for name in names),path)

    def test_task_loader_keeps_exact_semantics_and_no_generic_default(self):
        from open_cake_ir.evaluation import WorkloadContract
        from open_cake_ir.tasks.workloads import load_workload
        import json
        self.assertFalse(hasattr(WorkloadContract,"load"))
        workload=load_workload(ROOT/"contracts/workloads/flash-kmeans-assign-v2.json")
        self.assertEqual(workload.target,"sm_100a")
        self.assertEqual([arg.name for arg in workload.tensor_abi("headline_b32")],
                         ["tokens","centroids","centroid_sq","assignments"])
        document=workload.document
        document["semantics"]["tie_break"]="highest_index"
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.json";path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):load_workload(path)
            for operator in (None, [], {}, True):
                document["operator"] = operator
                path.write_text(json.dumps(document))
                with self.subTest(operator=operator), self.assertRaises(ValueError):
                    load_workload(path)

    def test_qsa_cannot_create_an_alternate_turn(self):
        from open_cake_ir.tasks.qsa import feedback
        from open_cake_ir import lab, evaluation
        self.assertFalse(hasattr(feedback,"qsa_next_turn_request"))
        self.assertFalse(hasattr(lab,"qsa_evaluation_feedback"))
        self.assertFalse(hasattr(evaluation,"materialize_qsa_case"))
        result=subprocess.run([sys.executable,"-m","open_cake_ir.tasks.qsa.project_feedback","turn"],
                              cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn("invalid choice",result.stderr)

    def test_executor_closure_includes_nested_task_implementations(self):
        from tools.release_executor import _source_paths
        sources={path.relative_to(ROOT).as_posix() for path in _source_paths(ROOT)}
        expected={path.relative_to(ROOT).as_posix() for path in (ROOT/"src/open_cake_ir/tasks").rglob("*.py")}
        self.assertTrue(expected <= sources)
        self.assertIn("src/open_cake_ir/tasks/qsa/assets/qsa_direct_reference_v1.cu",sources)
