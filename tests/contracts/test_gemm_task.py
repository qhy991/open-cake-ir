"""The FP32 GEMM task: frozen contract, independent oracle and a lowering baseline.

No GPU runs here. Device correctness for this task is a separate evidence domain.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.apple import BACKENDS
from open_cake_ir.tasks.gemm import workload as gemm
from open_cake_ir.tasks.tiles.workload import _round
from open_cake_ir.tasks.workloads import create_task, load_workload, materialize_case, reference_outputs

ROOT = Path(__file__).resolve().parents[2]


class GemmTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")

    def task(self, rows=4, depth=8, columns=3, backend="metal-m2"):
        document, source = create_task("gemm_bias", backend=backend, rows=rows, depth=depth, columns=columns)
        return WorkloadContract(document), source

    def test_every_backend_freezes_one_device_shape_and_case_set(self):
        with tempfile.TemporaryDirectory() as directory:
            for backend, device in BACKENDS.items():
                with self.subTest(backend=backend):
                    document, _ = create_task("gemm_bias", backend=backend, rows=4, depth=8, columns=3)
                    path = Path(directory) / f"{backend}.json"
                    path.write_text(json.dumps(document))
                    workload = load_workload(path)
                    self.assertEqual(workload.target, device["target"])
                    self.assertEqual(workload.case_ids, tuple(gemm.CASES))
                    self.assertIn(backend, workload.workload_id)
                    self.assertTrue(workload.document["validation"]["all_cases_required"])
                    for case_id in workload.case_ids:
                        self.assertEqual(workload.tensor_abi(case_id), workload.tensor_abi("primary"))

    def test_frozen_contract_rejects_shape_domain_or_case_drift(self):
        document, _ = create_task("gemm_bias", backend="metal-m2", rows=4, depth=8, columns=3)
        changes = (
            lambda d: d["semantics"].update(target="apple_gpu_family7"),
            lambda d: d["semantics"].update(extra="ignored"),
            lambda d: d["semantics"]["candidate_abi"].update(inputs=["b", "a", "bias"]),
            lambda d: d["tensors"]["a"].update(max_abs=4.0),
            lambda d: d["tensors"]["out"].update(dtype="bf16"),
            lambda d: d["validation"].update(atol=1.0),
            lambda d: d["validation"].update(all_cases_required=False),
            lambda d: d["oracle"].update(callable="candidate.reference"),
            lambda d: d["cases"].pop(),
            lambda d: d["cases"][1]["shape"].update(K=9),
            lambda d: d.update(revision="2"),
            lambda d: d.update(state="draft"),
        )
        for change in changes:
            changed = deepcopy(document)
            change(changed)
            with self.subTest(document=changed), self.assertRaises(ValueError):
                gemm.validate_gemm_contract(changed)
        for options in ({"backend": "metal"}, {"rows": 0}, {"depth": True}, {"columns": -1},
                        {"rows": 2**20, "depth": 2**12}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_task("gemm_bias", **{"rows": 4, "depth": 8, "columns": 3, **options})
        # Only GEMM owns a K extent, and it cannot be omitted.
        with self.assertRaises(ValueError):
            create_task("gemm_bias", backend="metal-m2", rows=4, columns=3)
        with self.assertRaises(ValueError):
            create_task("rmsnorm", backend="metal-m2", rows=4, columns=3, depth=8)

    def test_oracle_is_independent_fsum_and_refuses_unusable_inputs(self):
        workload, _ = self.task(rows=3, depth=5, columns=2)
        for case_id in workload.case_ids:
            inputs = materialize_case(workload, case_id)
            self.assertEqual(inputs, materialize_case(workload, case_id))
            expected = reference_outputs(workload, case_id, inputs)["out"]
            self.assertEqual(len(expected), 3 * 2)
            for m in range(3):
                for n in range(2):
                    total = math.fsum(inputs["a"][m * 5 + k] * inputs["b"][k * 2 + n] for k in range(5))
                    # The oracle sums exactly and rounds once, so this is an equality.
                    self.assertEqual(expected[m * 2 + n], _round(total + inputs["bias"][n], "fp32"))
        inputs = materialize_case(workload, "primary")
        for replacement in ([], [0.0], [float("nan")] * 15, [float("inf")] * 15, [True] * 15, [1e30] * 15):
            with self.subTest(replacement=replacement[:1]), self.assertRaises(ValueError):
                reference_outputs(workload, "primary", {**inputs, "a": replacement})
        with self.assertRaises(ValueError):
            reference_outputs(workload, "primary", {"a": inputs["a"]})

    def test_starter_binds_the_workload_and_lowers_for_every_declared_case(self):
        for rows, depth, columns in ((1, 1, 1), (4, 8, 3), (3, 33, 5), (2, 64, 16)):
            with self.subTest(shape=(rows, depth, columns)):
                workload, source = self.task(rows, depth, columns)
                schedule = frontend.parse(source).document
                self.assertEqual(schedule["metadata"]["workload_contract_sha256"], workload.canonical_sha256)
                buffers = [b for b in schedule["buffers"] if b["space"] == "global"]
                for case_id in workload.case_ids:
                    self.assertEqual([(b["name"], tuple(b["shape"]), b["dtype"], b["mode"]) for b in buffers],
                                     [(a.name, a.shape, a.dtype, a.mode) for a in workload.tensor_abi(case_id)])
                assessment = self.compiler.assess(schedule)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                lowering = self.compiler.lower(assessment)
                self.assertEqual(lowering.toolchain_requirements["target"], workload.target)
                self.assertEqual(lowering.toolchain_requirements["threadgroups_per_grid"], [rows, 1, 1])
                self.assertFalse(assessment.calibration_available)


if __name__ == "__main__":
    unittest.main()
