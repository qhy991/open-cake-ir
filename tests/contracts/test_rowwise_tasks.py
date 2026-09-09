"""Portable backward-task contracts and CPU semantics; no native Metal or GPU execution.

The family's structure is a signed row reduction whose error the epilogue amplifies by
multiplication instead of normalizing away by division. These tests check what that
structure creates: that each task folds the statistics its own definition names, and that
the shape-derived allowance is a valid bound and not the thing doing the passing.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.core import compare_tile_outputs
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, admit_width
from open_cake_ir.tasks.rowwise import workload as rowwise
from open_cake_ir.tasks.workloads import create_task, materialize_case, reference_outputs
from open_cake_ir.tasks.tiles.workload import _round

ROOT = Path(__file__).resolve().parents[2]
# How many row reductions each task's definition requires, and whether any of them folds
# a product of two tensors rather than one loaded tile. Both differ between the two tasks,
# which is exactly why the contract states the reduction per task rather than per family.
REDUCTIONS = {
    "softmax_backward": (1, False),
    "layernorm_backward_input": (2, True),
    "rmsnorm_input_gradient": (1, True),
    # The only maximum in the set, and the only task whose fold selects rather than
    # accumulates, so no rounding survives its reduction at all.
    "absmax_rescale": (1, False),
    # Three sums collapsed into one value per row: the family's only rank-1 output.
    "cosine_similarity": (3, True),
}


class RowwiseTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")

    def task(self, name, rows=2, columns=8, backend="metal-m1-pro"):
        document, source = create_task(name, backend=backend, rows=rows, columns=columns)
        return WorkloadContract(document), source

    def test_each_task_folds_exactly_the_statistics_its_definition_names(self):
        for name, (count, folds_product) in REDUCTIONS.items():
            with self.subTest(task=name):
                workload, source = self.task(name)
                schedule = frontend.parse(source).document
                reduces = [op for op in schedule["operations"] if op["kind"] == "reduce"]
                self.assertEqual(len(reduces), count)
                # The softmax backward folds the upstream gradient it loaded and forms its
                # product afterwards; the LayerNorm input gradient folds a product.
                products = {op["writes"][0] for op in schedule["operations"]
                            if op["kind"] == "elementwise" and op["parameters"]["op"] == "mul"}
                self.assertEqual(any(set(op["reads"]) & products for op in reduces),
                                 folds_product)
                self.assertEqual(workload.document["semantics"]["arithmetic"]["reduction"],
                                 rowwise._REDUCTION[name])
                # Whatever it folds, the epilogue multiplies the result rather than
                # dividing by it, which is what the derived allowance exists for.
                self.assertNotIn("/ total", source)
                self.assertEqual(any(op["parameters"].get("op") == "max" for op in reduces),
                                 name == "absmax_rescale")

    def test_three_tensor_ranks_coexist_in_one_frozen_abi(self):
        """Per-row statistics and a per-feature scale are new shapes for a Lab task."""
        workload, _ = self.task("layernorm_backward_input", rows=3, columns=8)
        shapes = {arg.name: arg.shape for arg in workload.tensor_abi("primary")}
        self.assertEqual(shapes, {"dy": (3, 8), "x": (3, 8), "mean": (3,), "rstd": (3,),
                                  "gamma": (8,), "out": (3, 8)})
        # And the family's one collapsing output is rank-1 over the row extent.
        collapsed, _ = self.task("cosine_similarity", rows=3, columns=8)
        self.assertEqual({arg.name: arg.shape for arg in collapsed.tensor_abi("primary")},
                         {"a": (3, 8), "b": (3, 8), "out": (3,)})
        inputs = materialize_case(workload, "primary")
        self.assertEqual([len(inputs[name]) for name in ("dy", "x", "mean", "rstd", "gamma")],
                         [24, 24, 3, 3, 8])

    def test_oracles_match_independent_small_formulas(self):
        softmax, _ = self.task("softmax_backward", rows=1, columns=2)
        probabilities, gradients_in = [0.25, 0.75], [1.0, -0.5]
        total = math.fsum(gradients_in)
        self.assertEqual(
            reference_outputs(softmax, "primary", {"p": probabilities, "dp": gradients_in})["out"],
            [_round(g - p * total, "fp32") for p, g in zip(probabilities, gradients_in)])
        layer, _ = self.task("layernorm_backward_input", rows=1, columns=2)
        supplied = {"dy": [1.0, -2.0], "x": [1.5, -0.5], "mean": [0.5],
                    "rstd": [2.0], "gamma": [1.0, 0.5]}
        centered = [(1.5 - 0.5) * 2.0, (-0.5 - 0.5) * 2.0]
        weighted = [1.0 * 1.0, -2.0 * 0.5]
        first = math.fsum(g * h for g, h in zip(weighted, centered)) / 2
        second = math.fsum(weighted) / 2
        self.assertEqual(reference_outputs(layer, "primary", supplied)["out"],
                         [_round(2.0 * (g - second - h * first), "fp32")
                          for g, h in zip(weighted, centered)])
        # An all-zero row leaves both statistics at zero and the output at zero.
        for name in rowwise.TASKS:
            workload, _ = self.task(name, rows=1, columns=4)
            zeros = materialize_case(workload, "zeros")
            with self.subTest(task=name):
                self.assertTrue(all(value == 0.0
                                    for value in reference_outputs(workload, "zeros", zeros)["out"]))

    def test_the_allowance_is_derived_from_the_frozen_width_and_is_a_bound(self):
        """It must grow with the accumulation depth and must not be what passes the test."""
        previous = 0.0
        for columns in (1, 8, 64, 1024):
            document, _ = create_task("softmax_backward", rows=2, columns=columns)
            allowance = document["validation"]["atol"]
            with self.subTest(columns=columns):
                self.assertGreaterEqual(allowance, previous)
                self.assertEqual(allowance, rowwise._allowance("softmax_backward", columns)["atol"])
            previous = allowance
        self.assertGreater(previous, rowwise._allowance("softmax_backward", 1)["atol"])
        # The depth model is the emitted body's, not a guess: 32 lanes then a five-deep fold.
        self.assertEqual(rowwise._reduction_depth(1), 6)
        self.assertEqual(rowwise._reduction_depth(32), 6)
        self.assertEqual(rowwise._reduction_depth(1024), 37)

    def test_frozen_contract_rejects_target_domain_or_case_drift(self):
        document, _ = create_task("layernorm_backward_input", rows=2, columns=8)
        changes = (
            lambda d: d["semantics"].update(target="apple_gpu_family9"),
            lambda d: d["semantics"]["arithmetic"].update(reduction="none"),
            lambda d: d["semantics"]["candidate_abi"].update(inputs=["x", "dy", "mean", "rstd", "gamma"]),
            lambda d: d["tensors"]["mean"].update(shape=["R", "C"]),
            lambda d: d["tensors"]["gamma"].update(max_abs=16.0),
            lambda d: d["tensors"].pop("rstd"),
            lambda d: d["validation"].update(atol=1.0),
            lambda d: d["validation"].update(rtol=2e-4),
            lambda d: d["oracle"].update(callable="candidate.reference"),
            lambda d: d["provenance"].pop(),
            lambda d: d["provenance"][1].update(derived_parent_id="softmax_backward_v1"),
            lambda d: d["cases"].pop(),
            lambda d: d["cases"][0].update(seed=True),
            lambda d: d.update(operator="softmax_backward_fp32"),
            lambda d: d.update(state="draft"),
        )
        for change in changes:
            changed = deepcopy(document)
            change(changed)
            with self.subTest(document=changed), self.assertRaises(ValueError):
                rowwise.validate_rowwise_contract(changed)
        for options in ({"backend": "cuda"}, {"rows": 0}, {"columns": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_task("layernorm_backward_input", **options)

    def test_oracle_refuses_nonfinite_unrounded_and_incomplete_inputs(self):
        for name in rowwise.TASKS:
            workload, _ = self.task(name)
            inputs = materialize_case(workload, "primary")
            primary = workload.tensor_abi("primary")[0].name
            for replacement in ([], [0.0], [float("nan")] * 16, [float("inf")] * 16,
                                [True] * 16, [0.1] * 16, [8.0] * 16):
                with self.subTest(task=name, replacement=replacement), self.assertRaises(ValueError):
                    reference_outputs(workload, "primary", {**inputs, primary: replacement})
            if len(inputs) > 1:
                with self.subTest(task=name, missing=True), self.assertRaises(ValueError):
                    reference_outputs(workload, "primary", {primary: inputs[primary]})

    def test_one_task_freezes_for_an_apple_or_an_nvidia_device_unchanged(self):
        def device_bearing(document):
            stripped = deepcopy(document)
            stripped.pop("workload_id")
            stripped["semantics"].pop("target")
            stripped["provenance"][0].pop("scope")
            return stripped

        for name in rowwise.TASKS:
            frozen = {}
            for backend, device in BACKENDS.items():
                document, source = self.task(name, rows=2, columns=8, backend=backend)
                document = document.document
                with self.subTest(task=name, backend=backend):
                    self.assertEqual(document["semantics"]["target"], device["target"])
                    assessment = self.compiler.assess(frontend.parse(source).document)
                    self.assertTrue(assessment.lowering_eligible, assessment.findings)
                    self.assertIn(f'backend="{device["route"]}"', source)
                    # Neither task names an instruction contract, so nothing device
                    # specific reaches the operation body at all.
                    self.assertNotIn("instruction=", source)
                frozen[backend] = document
            (first, *rest) = frozen.values()
            for document in rest:
                with self.subTest(task=name, workload=document["workload_id"]):
                    self.assertEqual(device_bearing(document), device_bearing(first))

    def test_generated_starters_match_all_input_case_oracles_on_cpu(self):
        """And by a margin, so the derived allowance is not what makes this pass."""
        from tests.contracts import test_metal as cpu_contracts
        runner = cpu_contracts.MetalTests("runTest")
        runner.compiler = self.compiler
        for name in rowwise.TASKS:
            for width in (1, 8, 65, 256):
                workload, source = self.task(name, columns=width)
                schedule = frontend.parse(source).document
                allowance = workload.document["validation"]["atol"]
                for case_id in rowwise.CASES:
                    with self.subTest(task=name, width=width, case=case_id):
                        inputs = materialize_case(workload, case_id)
                        expected = reference_outputs(workload, case_id, inputs)
                        outputs = runner.execute_body(schedule, inputs)
                        passed, metrics = compare_tile_outputs(
                            workload, inputs, expected, {"out": outputs["out"]},
                            {key: outputs[key] for key in inputs})
                        self.assertTrue(passed, metrics)
                        observed = max(abs(a - b) for a, b
                                       in zip(expected["out"], outputs["out"]))
                        # The contract carries a worst case that assumes every rounding
                        # aligns; the portable semantics stay far inside it. If this ever
                        # tightens to the allowance, the body changed, not the bound.
                        self.assertLess(observed, allowance / 10 + 1e-6)


if __name__ == "__main__":
    unittest.main()
