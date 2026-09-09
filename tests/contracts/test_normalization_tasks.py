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

    def task(self, name="rmsnorm", rows=2, columns=7, backend="metal-m1-pro"):
        document, source = create_task(name, backend=backend, rows=rows, columns=columns)
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
                for backend, device in normalization.BACKENDS.items():
                    document, _ = create_task(name, backend=backend, rows=2, columns=7)
                    path = Path(directory) / f"{name}-{backend}.json"
                    path.write_text(json.dumps(document))
                    workload = load_workload(path)
                    self.assertEqual(workload.target, device["target"])
                    self.assertEqual(workload.case_ids, tuple(normalization.CASES))
                    self.assertTrue(workload.document["validation"]["all_cases_required"])
                    for case_id in workload.case_ids:
                        self.assertEqual(workload.tensor_abi(case_id), workload.tensor_abi("primary"))

    def test_softmax_owns_no_affine_parameter_epsilon_or_shared_tolerance(self):
        document, source = create_task("softmax", backend="metal-m2", rows=4, columns=7)
        workload = WorkloadContract(document)
        self.assertEqual([arg.name for arg in workload.tensor_abi("primary")], ["x", "out"])
        self.assertNotIn("epsilon", document["semantics"])
        self.assertEqual(document["semantics"]["arithmetic"]["shift"],
                         "subtract_row_maximum_before_exponentiation")
        # A distribution needs a tighter absolute bound than the scale-normalizing tasks.
        self.assertEqual((document["validation"]["atol"], document["validation"]["rtol"]), (2e-6, 2e-5))
        for other in ("rmsnorm", "layernorm", "residual_rmsnorm"):
            with self.subTest(task=other):
                sibling, _ = create_task(other, backend="metal-m2", rows=4, columns=7)
                self.assertIn("weight", sibling["tensors"])
                self.assertIn("epsilon", sibling["semantics"])
                self.assertEqual(sibling["validation"]["atol"], 2e-5)
        schedule = frontend.parse(source).document
        assessment = self.compiler.assess(schedule)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        self.assertIn("precise::exp", self.compiler.lower(assessment).source)

    def test_softmax_oracle_is_an_independent_shifted_fsum_distribution(self):
        workload, _ = self.task("softmax", rows=3, columns=17)
        for case_id in workload.case_ids:
            inputs = materialize_case(workload, case_id)
            out = reference_outputs(workload, case_id, inputs)["out"]
            self.assertEqual(len(out), 3 * 17)
            for start in range(0, len(out), 17):
                row = out[start:start + 17]
                self.assertTrue(all(0.0 <= value <= 1.0 for value in row))
                self.assertAlmostEqual(math.fsum(row), 1.0, places=5)
            for m in range(3):
                source_row = inputs["x"][m * 17:(m + 1) * 17]
                shift = max(source_row)
                weights = [math.exp(v - shift) for v in source_row]
                total = math.fsum(weights)
                for n, weight in enumerate(weights):
                    self.assertEqual(out[m * 17 + n], _round(weight / total, "fp32"))
        with self.assertRaises(ValueError):
            reference_outputs(workload, "primary", {"x": [float("nan")] * 51})

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
        for options in ({"backend": "metal"}, {"backend": "cuda"}, {"backend": "metal-m3"},
                        {"rows": True}, {"columns": 0}, {"rows": 2**30, "columns": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_task("rmsnorm", **options)

    def test_each_backend_freezes_one_device_and_never_mixes_with_another(self):
        documents = {backend: create_task("rmsnorm", backend=backend, rows=2, columns=7)[0]
                     for backend in normalization.BACKENDS}
        self.assertEqual(len({document["workload_id"] for document in documents.values()}), len(documents))
        for backend, document in documents.items():
            with self.subTest(backend=backend):
                device = normalization.BACKENDS[backend]
                self.assertEqual(document["semantics"]["target"], device["target"])
                self.assertEqual(normalization.device_name(device["target"]), device["device_name"])
                self.assertIn(backend, document["workload_id"])
                normalization.validate_normalization_contract(document)
        # Only the whole frozen document admits a backend; no field may be swapped alone.
        for backend, other in (("metal-m1-pro", "metal-m2"), ("metal-m2", "metal-m1-pro")):
            for field in ("workload_id", "provenance"):
                changed = deepcopy(documents[backend])
                changed[field] = deepcopy(documents[other][field])
                with self.subTest(backend=backend, field=field), self.assertRaises(ValueError):
                    normalization.validate_normalization_contract(changed)
        for target in ("sm_100a", "", None):
            with self.subTest(target=target), self.assertRaises(ValueError):
                normalization.device_name(target)

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
                    if "weight" in inputs:  # Softmax owns no affine parameter.
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
            if len(inputs) > 1:  # Softmax's only input is already the complete set.
                with self.subTest(task=name, missing=sorted(set(inputs) - {"x"})), self.assertRaises(ValueError):
                    reference_outputs(workload, "primary", {"x": inputs["x"]})

    def test_starter_binds_workload_and_all_case_abis_for_apple_lowering(self):
        for name in normalization.TASKS:
            for width in (1, 7, 32, 65, 257, 1024, 4096):
                lowered = {}
                for backend in normalization.BACKENDS:
                    with self.subTest(task=name, width=width, backend=backend):
                        workload, source = self.task(name, columns=width, backend=backend)
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
                        self.assertFalse(assessment.calibration_available)
                        lowered[backend] = (dict(lowering.toolchain_requirements), lowering.source)
                # Only the target commitment separates the Apple backends here, so the
                # CPU semantic contracts below stay valid for one emitted body.
                (first, *rest) = lowered.values()
                for requirements, source in rest:
                    self.assertEqual(source, first[1])
                    self.assertEqual({name: value for name, value in requirements.items() if name != "target"},
                                     {name: value for name, value in first[0].items() if name != "target"})

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
