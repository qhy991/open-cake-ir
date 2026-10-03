"""One authoring selection authority; source checks do not claim runtime readiness."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.cli import main
from open_cake_ir.compiler import Compiler
from open_cake_ir.tasks import catalog
from open_cake_ir.tasks.catalog_cli import check_cell
from tools import launch_task, launch_task_matrix, rewrite_collection

ROOT = Path(__file__).resolve().parents[2]


class TaskCatalogTests(unittest.TestCase):
    def command(self, *arguments):
        with redirect_stdout(StringIO()) as output, redirect_stderr(StringIO()) as errors:
            code = main(["--project-root", str(ROOT), "tasks", *arguments])
        return code, output.getvalue(), errors.getvalue()

    def test_launchers_and_reference_selection_share_the_catalog(self):
        self.assertEqual(launch_task.TASKS, catalog.task_names())
        self.assertEqual(launch_task_matrix.ALL_TASKS, catalog.task_names(suite="portable"))
        self.assertEqual(len(launch_task.TASKS), len(set(launch_task.TASKS)))
        self.assertIs(launch_task._default_shape, catalog.default_shape)
        self.assertIs(launch_task_matrix._depth, catalog.matrix_depth)
        self.assertEqual(rewrite_collection.select_tasks(None), catalog.select_collection_tasks(ROOT, None))
        for row in rewrite_collection.select_tasks(None):
            self.assertIn(row["task"], launch_task.TASKS)
        with self.assertRaisesRegex(ValueError, "unknown authoring task"):
            catalog.default_shape("unregistered_task")

    def test_reference_only_tasks_remain_visible_and_cannot_be_selected(self):
        code, output, errors = self.command("list", "--suite", "flashinfer-rewrites", "--format", "json")
        self.assertEqual((code, errors), (0, ""))
        document = json.loads(output)
        self.assertEqual(document["task_count"], len(rewrite_collection.catalog()))
        blocked = next(row for row in document["tasks"] if row["id"] == "027_cake_kda_prefill")
        self.assertEqual(blocked["authoring"], "not_integrated")
        self.assertEqual(blocked["reason"], "full_workload_oracle_and_authoring_not_integrated")
        self.assertIsNone(blocked["backend"])
        with self.assertRaisesRegex(ValueError, blocked["reason"]):
            catalog.select_collection_tasks(ROOT, [blocked["id"]])
        code, output, errors = self.command("show", blocked["id"], "--format", "markdown")
        self.assertEqual(code, 0, errors)
        self.assertIn("待绑定完整任务", output)
        self.assertIn(blocked["reason"], output)

    def test_ready_reference_cannot_drift_away_from_the_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / catalog.COLLECTION_PATH
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"tasks": [{"id": "example", "task": "unregistered_task", "status": "ready"}]}))
            with self.assertRaisesRegex(ValueError, "not registered for authoring"):
                catalog.collection_rows(root)

    def test_listing_does_not_load_compiler_or_launch_a_provider(self):
        with patch.object(Compiler, "load", side_effect=AssertionError("registration view must not assess")), \
             patch.object(launch_task, "main", side_effect=AssertionError("must not launch")):
            code, output, errors = self.command("list", "--suite", "portable", "--format", "json")
        self.assertEqual((code, errors), (0, ""))
        document = json.loads(output)
        self.assertEqual(document["scope"], "task_registration_only")
        self.assertIsNone(document["source_commit"])
        self.assertEqual({row["task"] for row in document["tasks"]}, set(launch_task_matrix.ALL_TASKS))
        self.assertEqual(set(document["runtime_checks"].values()), {"not_examined"})

    def test_check_preserves_exact_target_and_shape_and_missing_runtime_evidence(self):
        code, output, errors = self.command("check", "--task", "silu", "--backend", "triton-metax", "--format", "json")
        self.assertEqual(code, 0, errors)
        document = json.loads(output)
        row = document["tasks"][0]
        self.assertEqual(row["source_status"], "source_generated")
        self.assertEqual(row["target"], "xcore1002")
        self.assertEqual(row["validation_cases"], 5)
        self.assertEqual(row["runtime_status"], "not_examined")
        self.assertEqual(document["corpus_gate"], "not_examined")
        self.assertTrue(document["source_commit"])
        code, output, errors = self.command("check", "--task", "silu", "--backend", "triton-metax", "--columns", "1000", "--format", "json")
        self.assertEqual(code, 1, errors)
        row = json.loads(output)["tasks"][0]
        self.assertEqual(row["shape"]["columns"], 1000)
        self.assertEqual(row["source_status"], "refused")
        self.assertEqual(row["refused_at"], "workload")
        self.assertIn("power-of-two", row["reason"])

    def test_unsupported_route_is_refused_without_substitution(self):
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        row = next(row for row in catalog.entries(ROOT) if row["task"] == "metax_fp8_gemm")
        result = check_cell(compiler, row, "triton-b300")
        self.assertEqual(result["target"], "sm_103a")
        self.assertEqual(result["source_status"], "refused")
        self.assertIn("requires triton-metax", result["reason"])

    def test_invalid_selection_fails_without_an_empty_success_or_traceback(self):
        for arguments in (("show", "not-a-task"),
                          ("check", "--task", "not-a-task", "--backend", "triton-b300"),
                          ("check", "--suite", "portable", "--family", "tinygemm", "--backend", "triton-b300"),
                          ("check", "--backend", "triton-b300", "--depth", "256")):
            code, output, errors = self.command(*arguments)
            self.assertEqual(code, 2)
            self.assertNotIn("Traceback", errors)
            self.assertFalse(output)

    def test_reference_block_is_kept_separate_from_source_lowering(self):
        code, output, errors = self.command("check", "--suite", "flashinfer-rewrites", "--task",
                                            "027_cake_kda_prefill", "--backend", "triton-b200", "--format", "json")
        self.assertEqual(code, 1, errors)
        row = json.loads(output)["tasks"][0]
        self.assertEqual(row["source_status"], "not_integrated")
        self.assertEqual(row["refused_at"], "authoring")
