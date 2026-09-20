"""AMD task entrypoints, frozen predecessor contracts and complete starter ABIs."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.aka_v3 import workload as aka
from open_cake_ir.tasks.workloads import create_task, load_workload, materialize_case, reference_outputs
from tools import launch_task, launch_task_matrix

ROOT = Path(__file__).resolve().parents[2]


class AkaTaskLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_frozen_b200_contracts_still_equal_their_original_documents(self):
        for path in (ROOT / "contracts/workloads").glob("aka-*-triton-b200-v1.json"):
            workload = load_workload(path)
            self.assertEqual(workload.document, json.loads(path.read_text()))
            self.assertEqual(workload.document["revision"], "1")

    def test_amd_successors_keep_all_input_cases_and_the_original_oracle(self):
        for name in aka.LAUNCHABLE_TASKS.values():
            old = WorkloadContract(aka.workload_document(name))
            new = WorkloadContract(aka.workload_document(name, backend="triton-gfx1151"))
            with self.subTest(task=name):
                aka.validate_aka_v3_contract(new.document)
                self.assertEqual(new.document["revision"], "2")
                self.assertEqual(new.target, "gfx1151")
                self.assertEqual(new.document["tensors"], old.document["tensors"])
                self.assertEqual(new.document["oracle"], old.document["oracle"])
                self.assertIn("historical_B200_evidence_does_not_transfer",
                              new.document["provenance"][0]["scope"])
                for case in new.case_ids:
                    inputs = materialize_case(new, case)
                    self.assertEqual(inputs, materialize_case(old, case))
                    self.assertEqual(reference_outputs(new, case, inputs),
                                     reference_outputs(old, case, inputs))

    def test_retargeting_a_frozen_predecessor_or_changing_its_oracle_is_refused(self):
        document = aka.workload_document("row_gather")
        document["semantics"]["target"] = "gfx1151"
        with self.assertRaisesRegex(ValueError, "identity, target"):
            aka.validate_aka_v3_contract(document)
        document = aka.workload_document("row_gather", backend="triton-gfx1151")
        for field in ("oracle", "validation"):
            altered = deepcopy(document)
            altered[field]["qualification"] = "GPU passed"
            with self.assertRaisesRegex(ValueError, "frozen semantic"):
                aka.validate_aka_v3_contract(altered)

    def test_unimplemented_starters_and_invalid_shapes_are_refused_by_their_owner(self):
        for task in ("histogram", "max_pool1d"):
            with self.assertRaisesRegex(ValueError, "no portable starter"):
                aka.workload_document(task, backend="triton-gfx1151")
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            aka.workload_document("residual_layernorm", backend="triton-gfx1151", columns=7)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            create_task("aka_momentum_sgd", backend="triton-gfx1151", rows=True, columns=8)
        with self.assertRaisesRegex(ValueError, "declares K"):
            create_task("aka_row_gather", backend="triton-gfx1151", depth=8)

    def test_momentum_extent_refuses_int32_overflow_before_native_compilation(self):
        for elements in (2**31, 2**31 + 1):
            with self.subTest(elements=elements):
                with self.assertRaisesRegex(ValueError, "signed int32 coordinates"):
                    create_task("aka_momentum_sgd", backend="triton-gfx1151", rows=1, columns=elements)
                with self.assertRaisesRegex(ValueError, "signed int32 coordinates"):
                    aka.workload_document("momentum_sgd", backend="triton-gfx1151", elements=elements)
        # Keep old B200 contract generation intact, while refusing its new starter.
        legacy = aka.workload_document("momentum_sgd", elements=2**31)
        aka.validate_aka_v3_contract(legacy)
        with self.assertRaisesRegex(ValueError, "signed int32 coordinates"):
            create_task("aka_momentum_sgd", backend="triton-b200", rows=1, columns=2**31)
        for elements in (1, 257, 2**31 - 1):
            document, source = create_task("aka_momentum_sgd", backend="triton-gfx1151", rows=1, columns=elements)
            assessment = self.compiler.assess(frontend.parse(source).document)
            with self.subTest(boundary=elements):
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                self.compiler.lower(assessment)

    def test_new_launcher_starters_lower_for_the_exact_amd_target_and_write_every_output(self):
        for task in (launch_task.ADD_RMSNORM_TASK, *aka.LAUNCHABLE_TASKS):
            rows, columns = launch_task._default_shape(task, None, None)
            document, source = create_task(task, backend="triton-gfx1151", rows=rows, columns=columns)
            schedule = frontend.parse(source).document
            assessment = self.compiler.assess(schedule)
            with self.subTest(task=task):
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                lowering = self.compiler.lower(assessment)
                self.assertEqual(lowering.target, "gfx1151")
                self.assertEqual(lowering.toolchain_requirements["code_object"], "hsaco")
                outputs = set(document["semantics"]["candidate_abi"]["outputs"])
                written = {name for op in schedule["operations"] if op["kind"] == "store"
                           for name in op["writes"]}
                self.assertEqual(written, outputs)

    def test_single_task_cli_reaches_stack_admission_for_each_new_task(self):
        with tempfile.TemporaryDirectory() as directory:
            for task in (launch_task.ADD_RMSNORM_TASK, *aka.LAUNCHABLE_TASKS):
                args = ["--task", task, "--backend", "triton-gfx1151", "--harness", "codex",
                        "--model", "not-invoked", "--effort", "high", "--baseline-only",
                        "--workspace", str(Path(directory) / task)]
                with self.subTest(task=task), patch.object(
                    launch_task, "_admit_stack", side_effect=RuntimeError("stack boundary")
                ), self.assertRaisesRegex(RuntimeError, "stack boundary"):
                    launch_task.main(args)

    def test_matrix_accepts_explicit_new_tasks_before_any_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            for task in (launch_task.ADD_RMSNORM_TASK, *aka.LAUNCHABLE_TASKS):
                args = ["--task", task, "--backend", "triton-gfx1151", "--harness", "codex",
                        "--model", "not-invoked", "--effort", "high",
                        "--workspace-root", str(Path(directory) / task)]
                with self.subTest(task=task), patch.object(
                    launch_task_matrix.subprocess, "run", side_effect=RuntimeError("baseline boundary")
                ), self.assertRaisesRegex(RuntimeError, "baseline boundary"):
                    launch_task_matrix.main(args)


if __name__ == "__main__":
    unittest.main()
