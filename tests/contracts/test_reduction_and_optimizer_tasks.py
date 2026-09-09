"""Portable contracts for the column-reduction and optimizer-step families.

These two families exist because of what they ask a Schedule that no other family does.
The reductions family inverts which extent is parallel: one program per feature, folding
down the rows, writing rank-1 outputs. The optimizers family writes several buffers per
element and declares some of its inputs non-negative because a reciprocal square root
reads them. Both properties are asserted here rather than assumed.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.core import compare_tile_outputs
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS
from open_cake_ir.tasks.optimizers import workload as optimizers
from open_cake_ir.tasks.reductions import workload as reductions
from open_cake_ir.tasks.workloads import create_task, materialize_case, reference_outputs
from open_cake_ir.tasks.tiles.workload import _round

ROOT = Path(__file__).resolve().parents[2]


class ColumnReductionTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")

    def task(self, name, rows=8, columns=4, backend="metal-m1-pro"):
        document, source = create_task(name, backend=backend, rows=rows, columns=columns)
        return WorkloadContract(document), source

    def test_the_program_maps_over_features_and_folds_down_the_rows(self):
        """The inversion is the family, so it is asserted on the emitted Schedule."""
        for name in reductions.TASKS:
            with self.subTest(task=name):
                workload, source = self.task(name)
                schedule = frontend.parse(source).document
                axis, = schedule["program_map"]["axes"]
                # dimension 1 is the feature extent; every other family maps dimension 0.
                self.assertEqual(axis["dimension"], 1)
                self.assertTrue([op for op in schedule["operations"] if op["kind"] == "reduce"])
                for name_, shape in reductions.OUTPUTS[name]:
                    self.assertEqual(dict(
                        (arg.name, arg.shape) for arg in workload.tensor_abi("primary")
                    )[name_], (4,))
                self.assertEqual(
                    workload.document["semantics"]["arithmetic"]["program_axis"],
                    "one_program_per_feature_the_row_extent_is_the_sequential_one")

    def test_only_the_absmax_task_selects_instead_of_accumulating(self):
        for name in reductions.TASKS:
            _, source = self.task(name)
            reduces = [op for op in frontend.parse(source).document["operations"]
                       if op["kind"] == "reduce"]
            with self.subTest(task=name):
                self.assertEqual(any(op["parameters"]["op"] == "max" for op in reduces),
                                 name == "channel_absmax_scale")
        # Its allowance is therefore flat, while the accumulating ones grow with rows.
        flat = {create_task("channel_absmax_scale", rows=rows, columns=4)[0]["validation"]["atol"]
                for rows in (8, 64, 1024)}
        self.assertEqual(len(flat), 1)
        growing = [create_task("bias_gradient_reduction", rows=rows, columns=4)[0]
                   ["validation"]["atol"] for rows in (8, 64, 1024)]
        self.assertEqual(growing, sorted(growing))
        self.assertLess(growing[0], growing[-1])

    def test_oracles_match_independent_small_formulas(self):
        # Two rows of two features, inside the family's declared bound of two.
        values = [1.0, -2.0, 1.5, -0.5]
        moments, _ = self.task("per_channel_moments", rows=2, columns=2)
        self.assertEqual(reference_outputs(moments, "primary", {"x": values}),
                         {"mean": [_round(1.25, "fp32"), _round(-1.25, "fp32")],
                          "mean_square": [_round((1.0 + 2.25) / 2, "fp32"),
                                          _round((4.0 + 0.25) / 2, "fp32")]})
        absmax, _ = self.task("channel_absmax_scale", rows=2, columns=2)
        peaks = [1.5 + reductions.EPSILON, 2.0 + reductions.EPSILON]
        self.assertEqual(reference_outputs(absmax, "primary", {"x": values}),
                         {"amax": [_round(p, "fp32") for p in peaks],
                          "scale": [_round(p / reductions.FP8_E4M3_MAX, "fp32") for p in peaks]})
        bias, _ = self.task("bias_gradient_reduction", rows=2, columns=2)
        self.assertEqual(reference_outputs(bias, "primary",
                                           {"dout": values, "bias": [0.5, -0.5]})["dbias"],
                         [_round(0.5 + 2.5, "fp32"), _round(-0.5 - 2.5, "fp32")])

    def test_every_task_lowers_on_every_route_and_matches_the_cpu_semantics(self):
        from tests.contracts import test_metal as cpu_contracts
        runner = cpu_contracts.MetalTests("runTest")
        runner.compiler = self.compiler
        for name in reductions.TASKS:
            for backend, device in BACKENDS.items():
                with self.subTest(task=name, backend=backend):
                    _, source = self.task(name, rows=8, columns=4, backend=backend)
                    assessment = self.compiler.assess(frontend.parse(source).document)
                    self.assertTrue(assessment.lowering_eligible, assessment.findings)
                    self.assertIn(f'backend="{device["route"]}"', source)
            for rows in (1, 8, 32):
                workload, source = self.task(name, rows=rows, columns=5)
                schedule = frontend.parse(source).document
                for case_id in reductions.CASES:
                    with self.subTest(task=name, rows=rows, case=case_id):
                        inputs = materialize_case(workload, case_id)
                        expected = reference_outputs(workload, case_id, inputs)
                        outputs = runner.execute_body(schedule, inputs)
                        passed, metrics = compare_tile_outputs(
                            workload, inputs, expected, {k: outputs[k] for k in expected},
                            {k: outputs[k] for k in inputs})
                        self.assertTrue(passed, metrics)

    def test_frozen_contract_rejects_drift(self):
        document, _ = create_task("per_channel_moments", rows=8, columns=4)
        for change in (
            lambda d: d["semantics"]["arithmetic"].update(program_axis="row"),
            lambda d: d["semantics"]["candidate_abi"].update(outputs=["mean_square", "mean"]),
            lambda d: d["tensors"]["mean"].update(shape=["R"]),
            lambda d: d["validation"].update(atol=1.0),
            lambda d: d["provenance"].pop(),
            lambda d: d.update(operator="bias_gradient_reduction_fp32"),
        ):
            changed = deepcopy(document)
            change(changed)
            with self.subTest(document=changed), self.assertRaises(ValueError):
                reductions.validate_reductions_contract(changed)


class OptimizerTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")

    def task(self, name, rows=2, columns=8, backend="metal-m1-pro"):
        document, source = create_task(name, backend=backend, rows=rows, columns=columns)
        return WorkloadContract(document), source

    def test_every_accumulator_is_read_and_rewritten_as_its_own_buffer(self):
        for name in optimizers.TASKS:
            with self.subTest(task=name):
                workload, source = self.task(name)
                abi = workload.tensor_abi("primary")
                self.assertEqual([arg.name for arg in abi if arg.mode == "input"],
                                 list(optimizers.INPUTS[name]))
                self.assertEqual([arg.name for arg in abi if arg.mode == "output"],
                                 list(optimizers.OUTPUTS[name]))
                # One store per declared output; nothing is updated in place.
                stores = [op for op in frontend.parse(source).document["operations"]
                          if op["kind"] == "store"]
                self.assertEqual(len(stores), len(optimizers.OUTPUTS[name]))
                self.assertEqual(
                    workload.document["semantics"]["arithmetic"]["state"],
                    "every_accumulator_is_read_and_rewritten_as_a_separate_output_buffer")
                # No reduction: an optimizer step reads one coordinate per output.
                self.assertEqual([op for op in frontend.parse(source).document["operations"]
                                  if op["kind"] == "reduce"], [])

    def test_declared_nonnegative_accumulators_are_generated_and_enforced(self):
        for name, declared in optimizers.NONNEGATIVE.items():
            workload, _ = self.task(name)
            tensors = workload.document["tensors"]
            with self.subTest(task=name):
                self.assertEqual(sorted(n for n in tensors if tensors[n].get("nonnegative")),
                                 sorted(declared))
                for case_id in optimizers.CASES:
                    inputs = materialize_case(workload, case_id)
                    for tensor in declared:
                        self.assertTrue(all(value >= 0.0 for value in inputs[tensor]))
                if declared:
                    # The oracle refuses a negative accumulator rather than returning a NaN.
                    inputs = materialize_case(workload, "primary")
                    inputs[declared[0]] = [-1.0] * len(inputs[declared[0]])
                    with self.assertRaisesRegex(ValueError, "non-negative"):
                        reference_outputs(workload, "primary", inputs)

    def test_oracles_match_independent_small_formulas(self):
        sgd, _ = self.task("momentum_sgd", rows=1, columns=2)
        # Exact binary fractions: the ABI admits only values that survive FP32 rounding.
        supplied = {"param": [1.0, -1.0], "grad": [0.5, 0.25], "moment": [0.125, -0.25]}
        moments = [optimizers.MOMENTUM * m + optimizers.LEARNING_RATE * g
                   for m, g in zip(supplied["moment"], supplied["grad"])]
        self.assertEqual(reference_outputs(sgd, "primary", supplied),
                         {"moment_out": [_round(v, "fp32") for v in moments],
                          "param_out": [_round(p - v, "fp32")
                                        for p, v in zip(supplied["param"], moments)]})
        delta, _ = self.task("adadelta", rows=1, columns=1)
        supplied = {"param": [1.0], "grad": [0.5], "square_accumulator": [0.25],
                    "update_accumulator": [0.75]}
        h_next = (optimizers.ADADELTA_DECAY * 0.25
                  + (1 - optimizers.ADADELTA_DECAY) * 0.25)
        step = (math.sqrt(0.75 + optimizers.ADADELTA_EPSILON)
                / math.sqrt(h_next + optimizers.ADADELTA_EPSILON) * 0.5)
        actual = reference_outputs(delta, "primary", supplied)
        self.assertEqual(actual["square_out"], [_round(h_next, "fp32")])
        self.assertEqual(actual["param_out"],
                         [_round(1.0 + optimizers.ADADELTA_LEARNING_RATE * step, "fp32")])

    def test_every_task_lowers_on_every_route_and_matches_the_cpu_semantics(self):
        from tests.contracts import test_metal as cpu_contracts
        runner = cpu_contracts.MetalTests("runTest")
        runner.compiler = self.compiler
        for name in optimizers.TASKS:
            for backend, device in BACKENDS.items():
                with self.subTest(task=name, backend=backend):
                    _, source = self.task(name, backend=backend)
                    assessment = self.compiler.assess(frontend.parse(source).document)
                    self.assertTrue(assessment.lowering_eligible, assessment.findings)
                    self.assertIn(f'backend="{device["route"]}"', source)
            for width in (1, 8, 65):
                workload, source = self.task(name, columns=width)
                schedule = frontend.parse(source).document
                for case_id in optimizers.CASES:
                    with self.subTest(task=name, width=width, case=case_id):
                        inputs = materialize_case(workload, case_id)
                        expected = reference_outputs(workload, case_id, inputs)
                        outputs = runner.execute_body(schedule, inputs)
                        passed, metrics = compare_tile_outputs(
                            workload, inputs, expected, {k: outputs[k] for k in expected},
                            {k: outputs[k] for k in inputs})
                        self.assertTrue(passed, metrics)

    def test_hyperparameters_are_frozen_by_the_contract(self):
        for name in optimizers.TASKS:
            document, _ = create_task(name, rows=2, columns=8)
            with self.subTest(task=name):
                self.assertEqual(document["semantics"]["arithmetic"]["hyperparameters"],
                                 "frozen_by_this_contract_not_a_search_dimension")
                # Every constant the definition names is written into it, so a Schedule
                # cannot quietly substitute a different one.
                for constant in (optimizers.MOMENTUM if name == "momentum_sgd"
                                 else optimizers.ADAM_BETA1 if name == "adamw"
                                 else optimizers.ADADELTA_DECAY,):
                    self.assertIn(repr(constant), document["semantics"]["definition"])
        for change in (lambda d: d["semantics"]["arithmetic"].update(hyperparameters="free"),
                       lambda d: d["tensors"]["second"].pop("nonnegative")):
            changed = deepcopy(create_task("adamw", rows=2, columns=8)[0])
            change(changed)
            with self.subTest(document=changed), self.assertRaises(ValueError):
                optimizers.validate_optimizers_contract(changed)


if __name__ == "__main__":
    unittest.main()
