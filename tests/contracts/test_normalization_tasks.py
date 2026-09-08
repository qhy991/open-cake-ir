"""Portable task contracts and CPU semantics; no native Metal or GPU execution."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.core import compare_tile_outputs
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.normalization import workload as normalization
from open_cake_ir.tasks.workloads import create_task, load_workload, materialize_case, reference_outputs
from open_cake_ir.tasks.tiles.workload import _round

ROOT = Path(__file__).resolve().parents[2]


class NormalizationTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")

    def task(self, name="rmsnorm", rows=2, columns=7):
        document, source = create_task(name, rows=rows, columns=columns)
        return WorkloadContract(document), source

    def test_registered_workloads_preserve_legacy_cuda_contracts(self):
        for revision, target in (("1", "sm_100a"), ("2", "sm_103a")):
            workload = load_workload(ROOT / f"contracts/workloads/rmsnorm-fp32-v{revision}.json")
            self.assertEqual(workload.target, target)
            self.assertEqual(len(workload.tensor_abi("primary")[0].shape), 3)
            inputs = materialize_case(workload, "tiny")
            self.assertEqual(len(reference_outputs(workload, "tiny", inputs)["y"]), 8)
        with tempfile.TemporaryDirectory() as directory:
            for name in normalization.TASKS:
                document, _ = create_task(name, rows=2, columns=7)
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps(document))
                workload = load_workload(path)
                self.assertEqual(workload.target, "apple_gpu_family7")
                self.assertEqual(workload.case_ids, tuple(normalization.CASES))
                self.assertTrue(workload.document["validation"]["all_cases_required"])
                for case_id in workload.case_ids:
                    self.assertEqual(workload.tensor_abi(case_id), workload.tensor_abi("primary"))

    def test_frozen_contract_rejects_target_domain_or_case_drift(self):
        document, _ = create_task("rmsnorm", rows=2, columns=7)
        changes = (
            lambda d: d["semantics"].update(target="apple_gpu_family8"),
            lambda d: d["semantics"].update(epsilon=1e-6),
            lambda d: d["semantics"].update(extra="ignored"),
            lambda d: d["semantics"]["candidate_abi"].update(inputs=["weight", "x"]),
            lambda d: d["tensors"]["x"].update(max_abs=512.0),
            lambda d: d["tensors"]["out"].update(dtype="bf16"),
            lambda d: d["tensors"]["x"].update(shape=["R", "R", "C"]),
            lambda d: d["validation"].update(atol=1.0),
            lambda d: d["validation"].update(all_cases_required=False),
            lambda d: d["oracle"].update(callable="candidate.reference"),
            lambda d: d["cases"].pop(),
            lambda d: d["cases"][1]["shape"].update(C=65),
            lambda d: d["cases"][0].update(seed=True),
            lambda d: d["cases"][0].update(mode="candidate_selected"),
            lambda d: d.update(revision="2"),
            lambda d: d.update(schema_version=True),
            lambda d: d.update(state="draft"),
        )
        for change in changes:
            changed = deepcopy(document)
            change(changed)
            with self.subTest(document=changed), self.assertRaises(ValueError):
                normalization.validate_normalization_contract(changed)
        for options in ({"backend": "metal"}, {"backend": "cuda"}, {"rows": True},
                        {"columns": 0}, {"rows": 2**30, "columns": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_task("rmsnorm", **options)

    def test_materialization_preserves_abi_fp32_bounds_and_seed_reproducibility(self):
        for name in normalization.TASKS:
            workload, _ = self.task(name, rows=2, columns=65)
            for case_id in workload.case_ids:
                with self.subTest(task=name, case=case_id):
                    inputs = materialize_case(workload, case_id)
                    self.assertEqual(inputs, materialize_case(workload, case_id))
                    args = [arg for arg in workload.tensor_abi(case_id) if arg.mode == "input"]
                    self.assertEqual(list(inputs), [arg.name for arg in args])
                    for arg in args:
                        self.assertEqual(len(inputs[arg.name]), math.prod(arg.shape))
                        bound = workload.document["tensors"][arg.name]["max_abs"]
                        self.assertTrue(all(math.isfinite(value) and abs(value) <= bound
                                            and _round(value, "fp32") == value for value in inputs[arg.name]))
                    self.assertEqual(inputs["weight"][16], 0.0)
                    self.assertTrue(any(value < 0 for value in inputs["weight"]))
                    if case_id == "zeros":
                        self.assertTrue(all(value == 0 for value in inputs["x"]))
                    elif case_id == "near_zero":
                        self.assertLessEqual(max(map(abs, inputs["x"])), 1e-4)
                    elif case_id == "mixed_magnitude":
                        self.assertGreater(max(map(abs, inputs["x"])), 16)
            self.assertNotEqual(materialize_case(workload, "primary")["x"],
                                materialize_case(workload, "alternating")["x"])

    def test_oracles_match_independent_small_formulas(self):
        rms, _ = self.task(columns=2, rows=1)
        inverse = 1 / math.sqrt(12.5 + 1e-5)
        actual = reference_outputs(rms, "primary", {"x": [3.0, 4.0], "weight": [1.0, -0.5]})
        self.assertEqual(actual["out"], [_round(3 * inverse, "fp32"), _round(-2 * inverse, "fp32")])
        layer, _ = self.task("layernorm", rows=1, columns=2)
        inverse = 1 / math.sqrt(1 + 1e-5)
        actual = reference_outputs(layer, "primary", {"x": [1.0, 3.0], "weight": [0.5, -1.0], "bias": [0.25, 0.75]})
        self.assertEqual(actual["out"], [_round(-0.5 * inverse + 0.25, "fp32"),
                                         _round(-inverse + 0.75, "fp32")])
        residual, _ = self.task("residual_rmsnorm", rows=1, columns=2)
        inputs = {"x": [1.0, 1.0], "weight": [1.0, 1.0], "residual": [2**-24, 3 * 2**-24]}
        combined = [1.0, 1.0 + 2**-22]  # Both halfway additions round to an even FP32 significand.
        inverse = 1 / math.sqrt(sum(value * value for value in combined) / 2 + 1e-5)
        self.assertEqual(reference_outputs(residual, "primary", inputs)["out"],
                         [_round(value * inverse, "fp32") for value in combined])
        single, _ = self.task("layernorm", rows=2, columns=1)
        self.assertEqual(reference_outputs(single, "primary", {"x": [7.0, -3.0], "weight": [1.5], "bias": [-0.25]})["out"], [-0.25, -0.25])

    def test_oracle_refuses_nonfinite_unrounded_and_incomplete_inputs(self):
        for name in normalization.TASKS:
            workload, _ = self.task(name)
            inputs = materialize_case(workload, "primary")
            for replacement in ([], [0.0], [float("nan")] * 14, [float("inf")] * 14,
                                [True] * 14, [0.1] * 14, [512.0] * 14):
                with self.subTest(task=name, replacement=replacement), self.assertRaises(ValueError):
                    reference_outputs(workload, "primary", {**inputs, "x": replacement})
            with self.assertRaises(ValueError):
                reference_outputs(workload, "primary", {"x": inputs["x"]})

    def test_starter_binds_workload_and_all_case_abis_for_m1_lowering(self):
        for name in normalization.TASKS:
            for width in (1, 7, 32, 65, 257, 1024, 4096):
                with self.subTest(task=name, width=width):
                    workload, source = self.task(name, columns=width)
                    schedule = frontend.parse(source).document
                    self.assertEqual(schedule["metadata"]["workload_contract_sha256"], workload.canonical_sha256)
                    buffers = [b for b in schedule["buffers"] if b["space"] == "global"]
                    for case_id in workload.case_ids:
                        self.assertEqual([(b["name"], tuple(b["shape"]), b["dtype"], b["mode"]) for b in buffers],
                                         [(a.name, a.shape, a.dtype, a.mode) for a in workload.tensor_abi(case_id)])
                    assessment = self.compiler.assess(schedule)
                    self.assertTrue(assessment.lowering_eligible, assessment.findings)
                    self.assertEqual(self.compiler.lower(assessment).toolchain_requirements["target"], workload.target)
                    self.assertFalse(assessment.calibration_available)

    def test_generated_starters_match_all_input_case_oracles_on_cpu(self):
        from tests.contracts import test_metal as cpu_contracts
        runner = cpu_contracts.MetalTests("runTest")
        runner.compiler = self.compiler
        for name in normalization.TASKS:
            for width, cases in ((1, ("primary",)), (7, tuple(normalization.CASES)), (65, ("mixed_magnitude",))):
                workload, source = self.task(name, columns=width)
                schedule = frontend.parse(source).document
                for case_id in cases:
                    with self.subTest(task=name, width=width, case=case_id):
                        inputs = materialize_case(workload, case_id)
                        expected = reference_outputs(workload, case_id, inputs)
                        outputs = runner.execute_body(schedule, inputs)
                        observed = {"out": outputs["out"]}
                        after = {key: outputs[key] for key in inputs}
                        passed, metrics = compare_tile_outputs(workload, inputs, expected, observed, after)
                        self.assertTrue(passed, metrics)


if __name__ == "__main__":
    unittest.main()
