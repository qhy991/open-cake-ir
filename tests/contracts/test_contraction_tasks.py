"""Portable contracts for the arithmetic-bound family; no native Metal or GPU execution.

This family exists because everything else in the Lab is bandwidth-bound. The first test
measures that claim with the Compiler's own work model rather than asserting it, and the
rest check the structures a contraction creates: an operand that feeds many outputs, a
fold whose error the shape has to pay for, and -- for attention -- two chained
contractions with a data-dependent normalization between them.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.evaluation.core import compare_tile_outputs
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.contraction import workload as contraction
from open_cake_ir.tasks.devices import BACKENDS
from open_cake_ir.tasks.workloads import create_task, materialize_case, reference_outputs
from open_cake_ir.tasks.tiles.workload import _round

ROOT = Path(__file__).resolve().parents[2]
# The shape this family is frozen at for the measurements below.
SHAPE = {"rows": 1024, "depth": 128, "columns": 64}
# Every bandwidth-bound family, for the comparison that justifies this one existing.
ELEMENTWISE_FAMILIES = ("activation", "rowwise", "reductions", "optimizers")


class ContractionTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")

    def task(self, name, backend="metal-m1-pro", **shape):
        document, source = create_task(name, backend=backend, **{**SHAPE, **shape})
        return WorkloadContract(document), source

    def intensity(self, source):
        return work_bound(Schedule.from_dict(frontend.parse(source).document)).arithmetic_intensity

    def test_this_family_is_arithmetic_bound_and_the_others_are_not(self):
        """Measured with the Compiler's work model, not asserted.

        The separation is the reason the family exists: no Schedule for a task on the
        wrong side of it can trade arithmetic for traffic, because there is no arithmetic
        to trade.
        """
        bandwidth_bound = []
        for family in ELEMENTWISE_FAMILIES:
            module = importlib.import_module(f"open_cake_ir.tasks.{family}.workload")
            for name in module.TASKS:
                _, source = create_task(name, backend="metal-m1-pro", rows=128, columns=1024)
                bandwidth_bound.append((self.intensity(source), name))
        highest, name = max(bandwidth_bound)
        # Every elementwise and reduction task does a few operations per element it moves.
        self.assertLess(highest, 2.0, name)
        self.assertEqual(len(bandwidth_bound), 20)
        for task in contraction.TASKS:
            with self.subTest(task=task):
                _, source = self.task(task)
                # An order of magnitude past the whole rest of the set.
                self.assertGreater(self.intensity(source), 10.0 * highest)

    def test_intensity_grows_with_the_contracted_extent(self):
        """It is the contraction that buys the arithmetic, so it must show up as one."""
        previous = 0.0
        for depth in (32, 64, 128):
            _, source = self.task("gemm", depth=depth)
            with self.subTest(depth=depth):
                self.assertGreater(self.intensity(source), previous)
            previous = self.intensity(source)

    def test_each_task_contracts_something_structurally_different(self):
        for name in contraction.TASKS:
            with self.subTest(task=name):
                workload, source = self.task(name)
                schedule = frontend.parse(source).document
                reduces = [op for op in schedule["operations"] if op["kind"] == "reduce"]
                self.assertEqual(workload.document["semantics"]["arithmetic"]["reduction"],
                                 contraction._REDUCTION[name])
                # Every task here broadcasts a narrower operand across a loaded tile:
                # that is what makes one loaded byte feed many outputs.
                self.assertTrue(any("broadcast_axis" in op.get("parameters", {})
                                    for op in schedule["operations"]), name)
                if name == "attention_decode":
                    # Two contractions, a maximum and a normalizing sum between them.
                    self.assertEqual(len(reduces), 4)
                    self.assertEqual([op["parameters"]["op"] for op in reduces],
                                     ["sum", "max", "sum", "sum"])
                else:
                    self.assertEqual(len(reduces), 1)
        # Only the distance contracts a difference, which is what forbids a fused
        # multiply-accumulate instruction from serving it.
        squares = {name for name in contraction.TASKS
                   if any(op.get("parameters", {}).get("op") == "square"
                          for op in frontend.parse(self.task(name)[1]).document["operations"])}
        self.assertEqual(squares, {"pairwise_sqdist"})

    def test_the_naive_baseline_is_what_the_search_is_against(self):
        """Every baseline keeps the whole contracted operand resident, and says so.

        The Metal route's live-storage limit is the budget a tiled contraction would free,
        so a shape whose operand does not fit is refused with that exact finding rather
        than silently admitted.
        """
        for name in contraction.TASKS:
            with self.subTest(task=name):
                _, source = self.task(name, backend="metal-m2")
                self.assertIn("[:, :]", source)
                self.assertEqual(
                    self.task(name)[0].document["semantics"]["arithmetic"]["cost_regime"],
                    "arithmetic_bound_the_contracted_axis_feeds_many_outputs_per_loaded_byte")
        # Doubling both tiled extents exceeds the per-lane budget on Metal and is refused.
        _, oversized = self.task("gemm", backend="metal-m2", depth=256, columns=256)
        refused = self.compiler.assess(frontend.parse(oversized).document)
        self.assertFalse(refused.lowering_eligible)
        self.assertIn("METAL_PRIVATE_STORAGE_LIMIT",
                      [finding.code for finding in refused.findings])
        # The same shape is admitted on a route with no such per-lane budget.
        _, cuda = self.task("gemm", backend="triton-b200", depth=256, columns=256)
        self.assertTrue(self.compiler.assess(frontend.parse(cuda).document).lowering_eligible)

    def test_oracles_match_independent_small_formulas(self):
        gemm, _ = self.task("gemm", rows=1, depth=2, columns=2)
        supplied = {"a": [1.0, 2.0], "b": [0.5, -0.5, 1.0, 0.25], "bias": [0.125, -0.25]}
        self.assertEqual(reference_outputs(gemm, "primary", supplied)["out"],
                         [_round(1.0 * 0.5 + 2.0 * 1.0 + 0.125, "fp32"),
                          _round(1.0 * -0.5 + 2.0 * 0.25 - 0.25, "fp32")])
        # The gated variant is the same contraction with one epilogue applied.
        gated, _ = self.task("gemm_silu", rows=1, depth=2, columns=2)
        plain = reference_outputs(gemm, "primary", supplied)["out"]
        self.assertEqual(reference_outputs(gated, "primary", supplied)["out"],
                         [_round(v / (1.0 + math.exp(-v)), "fp32") for v in plain])
        distance, _ = self.task("pairwise_sqdist", rows=1, depth=2, columns=2)
        supplied = {"x": [1.0, 2.0], "c": [0.5, 1.0, -1.0, 0.5]}
        self.assertEqual(reference_outputs(distance, "primary", supplied)["out"],
                         [_round(0.25 + 1.0, "fp32"), _round(4.0 + 2.25, "fp32")])
        attention, _ = self.task("attention_decode", rows=1, depth=2, columns=2)
        supplied = {"q": [1.0, 0.0], "k": [1.0, 0.0, 0.0, 1.0], "v": [2.0, 0.0, 0.0, 2.0]}
        scale = 1.0 / math.sqrt(2)
        logits = [1.0 * scale, 0.0]
        weights = [math.exp(v - max(logits)) for v in logits]
        total = math.fsum(weights)
        self.assertEqual(reference_outputs(attention, "primary", supplied)["out"],
                         [_round(weights[0] * 2.0 / total, "fp32"),
                          _round(weights[1] * 2.0 / total, "fp32")])

    def test_attention_weights_form_a_distribution_over_the_keys(self):
        """The normalization between the two contractions is real, so check it is one."""
        workload, _ = self.task("attention_decode", rows=2, depth=8, columns=8)
        for case_id in contraction.CASES:
            inputs = materialize_case(workload, case_id)
            out = reference_outputs(workload, case_id, inputs)["out"]
            with self.subTest(case=case_id):
                # A convex combination of values cannot leave their range.
                self.assertLessEqual(max(map(abs, out)), max(map(abs, inputs["v"])) + 1e-6)

    def test_the_allowance_is_derived_from_the_contracted_extent(self):
        previous = 0.0
        for depth in (8, 32, 128):
            document, _ = create_task("gemm", **{**SHAPE, "depth": depth})
            with self.subTest(depth=depth):
                self.assertEqual(document["validation"]["atol"],
                                 contraction._allowance("gemm", depth, SHAPE["columns"])["atol"])
                self.assertGreater(document["validation"]["atol"], previous)
            previous = document["validation"]["atol"]

    def test_each_task_names_a_real_aka_qualified_parent(self):
        """Lineage is checkable against the dataset, not just plausible-looking text."""
        records = Path.home() / contraction.AKA_V7_RECORDS
        if not records.exists():
            self.skipTest("the AKA v7 records are not present on this machine")
        qualified = {}
        for line in records.read_text().splitlines():
            row = json.loads(line)
            qualified[row["derived_parent_id"]] = row["outcome"]
        for name, parent in contraction.AKA_PARENTS.items():
            with self.subTest(task=name):
                self.assertEqual(qualified.get(parent), "qualified", parent)
                document, _ = create_task(name, **SHAPE)
                lineage = [entry for entry in document["provenance"]
                           if entry["kind"] == "aka_qualified_parent_lineage"]
                self.assertEqual([entry["derived_parent_id"] for entry in lineage], [parent])
                # Semantics are inherited; a measured B200 result is not.
                self.assertIn("no_B200", lineage[0]["scope"])

    def test_frozen_contract_rejects_target_domain_or_case_drift(self):
        document, _ = create_task("attention_decode", **SHAPE)
        changes = (
            lambda d: d["semantics"].update(target="apple_gpu_family9"),
            lambda d: d["semantics"]["arithmetic"].update(cost_regime="memory_bound"),
            lambda d: d["semantics"]["arithmetic"].pop("scale"),
            lambda d: d["semantics"]["candidate_abi"].update(inputs=["k", "q", "v"]),
            lambda d: d["tensors"]["k"].update(shape=["K", "N"]),
            lambda d: d["tensors"]["q"].update(max_abs=16.0),
            lambda d: d["validation"].update(atol=1.0),
            lambda d: d["provenance"].pop(),
            lambda d: d["cases"][0]["shape"].update(K=64),
            lambda d: d.update(operator="contraction_gemm_fp32"),
            lambda d: d.update(state="draft"),
        )
        for change in changes:
            changed = deepcopy(document)
            change(changed)
            with self.subTest(document=changed), self.assertRaises(ValueError):
                contraction.validate_contraction_contract(changed)
        for options in ({"backend": "cuda"}, {"rows": 0}, {"depth": True}, {"columns": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_task("gemm", **{**SHAPE, **options})
        # A contraction declares a K extent; omitting it is an error, not a default.
        with self.assertRaisesRegex(ValueError, "K extent"):
            create_task("gemm", rows=8, columns=8)

    def test_oracle_refuses_nonfinite_unrounded_and_incomplete_inputs(self):
        for name in contraction.TASKS:
            workload, _ = self.task(name, rows=2, depth=8, columns=8)
            inputs = materialize_case(workload, "primary")
            primary = workload.tensor_abi("primary")[0].name
            for replacement in ([], [0.0], [float("nan")] * 16, [float("inf")] * 16,
                                [True] * 16, [0.1] * 16, [8.0] * 16):
                with self.subTest(task=name, replacement=replacement), self.assertRaises(ValueError):
                    reference_outputs(workload, "primary", {**inputs, primary: replacement})
            with self.subTest(task=name, missing=True), self.assertRaises(ValueError):
                reference_outputs(workload, "primary", {primary: inputs[primary]})

    def test_one_task_freezes_for_an_apple_or_an_nvidia_device_unchanged(self):
        def device_bearing(document):
            stripped = deepcopy(document)
            stripped.pop("workload_id")
            stripped["semantics"].pop("target")
            stripped["provenance"][0].pop("scope")
            return stripped

        for name in contraction.TASKS:
            frozen = {}
            for backend, device in BACKENDS.items():
                workload, source = self.task(name, backend=backend)
                with self.subTest(task=name, backend=backend):
                    assessment = self.compiler.assess(frontend.parse(source).document)
                    self.assertTrue(assessment.lowering_eligible, assessment.findings)
                    self.assertIn(f'backend="{device["route"]}"', source)
                    # Nothing device specific reaches a contraction's operation body.
                    self.assertNotIn("instruction=", source)
                frozen[backend] = workload.document
            (first, *rest) = frozen.values()
            for document in rest:
                with self.subTest(task=name, workload=document["workload_id"]):
                    self.assertEqual(device_bearing(document), device_bearing(first))

    def test_generated_starters_match_all_input_case_oracles_on_cpu(self):
        from tests.contracts import test_metal as cpu_contracts
        runner = cpu_contracts.MetalTests("runTest")
        runner.compiler = self.compiler
        for name in contraction.TASKS:
            for rows, depth, columns in ((1, 8, 8), (4, 8, 8), (2, 32, 8)):
                workload, source = self.task(name, rows=rows, depth=depth, columns=columns)
                schedule = frontend.parse(source).document
                allowance = workload.document["validation"]["atol"]
                for case_id in contraction.CASES:
                    with self.subTest(task=name, shape=(rows, depth, columns), case=case_id):
                        inputs = materialize_case(workload, case_id)
                        expected = reference_outputs(workload, case_id, inputs)
                        outputs = runner.execute_body(schedule, inputs)
                        observed = {key: outputs[key] for key in expected}
                        passed, metrics = compare_tile_outputs(
                            workload, inputs, expected, observed,
                            {key: outputs[key] for key in inputs})
                        self.assertTrue(passed, metrics)
                        worst = max(abs(a - b) for key in expected
                                    for a, b in zip(expected[key], observed[key]))
                        # The derived allowance is a worst case; the portable semantics
                        # stay well inside it, so it is not what makes this pass.
                        self.assertLess(worst, allowance / 5 + 1e-6)


if __name__ == "__main__":
    unittest.main()
