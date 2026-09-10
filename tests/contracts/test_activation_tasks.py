"""Portable element-local activation contracts and CPU semantics; no native Metal or GPU.

The family exists to isolate one transcendental with no reduction behind it, so these
tests assert the absence of a cross-lane exchange as carefully as they assert the math.
Most tasks are migrations of AKA qualified parents; one test binds each of those to the
exact review row it came from.
"""
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
from open_cake_ir.tasks.activation import workload as activation
from open_cake_ir.tasks.devices import BACKENDS, TRITON_MAXIMUM_TILE, admit_width
from open_cake_ir.tasks.normalization import workload as normalization
from open_cake_ir.tasks.workloads import create_task, load_workload, materialize_case, reference_outputs
from open_cake_ir.tasks.tiles.workload import _round

ROOT = Path(__file__).resolve().parents[2]

# The transcendental each lowered body must contain, which is what distinguishes these
# tasks from one another once the reduction is gone. Softsign and the rsqrt gradient are
# the family's only transcendental-free members: softsign composes an absolute value out
# of two ReLUs, and the rsqrt gradient is a cube and two scales. `None` asserts that no
# precise-namespace call appears at all, which is the stronger claim for those two.
TRANSCENDENTAL = {
    "silu": "precise::exp(", "swiglu": "precise::exp(",
    "selu": "precise::exp(", "softplus_gradient": "precise::exp(",
    "gelu_tanh": "precise::tanh(", "gelu_tanh_backward": "precise::tanh(",
    # The family's two transcendental-free members: both compose their piecewise shape
    # out of ReLU arms and never call into the precise namespace at all.
    "softsign": None, "prelu": None,
}


class ActivationTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")

    def task(self, name="silu", rows=2, columns=7, backend="metal-m1-pro"):
        document, source = create_task(name, backend=backend, rows=rows, columns=columns)
        return WorkloadContract(document), source

    def primary_inputs(self, name, values, second=None):
        """Build one oracle input mapping in the task's own ABI names and shapes."""
        declared = activation.INPUTS[name]
        supplied = {declared[0][0]: list(values)}
        for extra, shape in declared[1:]:
            fill = list(values if second is None else second)
            # A per-feature operand is one value per column, not one per element.
            supplied[extra] = fill if len(shape) == 2 else fill[:1] * 1
        return supplied

    def test_every_task_registers_one_backend_bound_frozen_document(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in activation.TASKS:
                for backend, device in activation.BACKENDS.items():
                    with self.subTest(task=name, backend=backend):
                        document, _ = create_task(name, backend=backend, rows=2, columns=8)
                        path = Path(directory) / f"{name}-{backend}.json"
                        path.write_text(json.dumps(document))
                        workload = load_workload(path)
                        self.assertEqual(workload.target, device["target"])
                        self.assertEqual(workload.case_ids, tuple(activation.CASES))
                        self.assertTrue(workload.document["validation"]["all_cases_required"])
                        for case_id in workload.case_ids:
                            self.assertEqual(workload.tensor_abi(case_id), workload.tensor_abi("primary"))
        # Distinct operators and backends never collide on one frozen identity.
        identities = {create_task(name, backend=backend, rows=2, columns=8)[0]["workload_id"]
                      for name in activation.TASKS for backend in activation.BACKENDS}
        self.assertEqual(len(identities), len(activation.TASKS) * len(activation.BACKENDS))
        self.assertEqual(set(activation.INPUTS), set(activation.TASKS))

    def test_each_migrated_task_names_the_aka_row_it_came_from(self):
        """Lineage is inspectable, and the named parent really is a qualified survivor."""
        rows = {json.loads(line)["derived_parent_id"]: json.loads(line)
                for line in (ROOT / activation.AKA_REVIEW).read_text().splitlines()}
        for name, parent in activation.AKA_V7_PARENTS.items():
            with self.subTest(task=name, dataset="v7"):
                document, _ = create_task(name, rows=2, columns=8)
                lineage = [entry for entry in document["provenance"]
                           if entry["kind"] == "aka_qualified_parent_lineage"]
                self.assertEqual([entry["derived_parent_id"] for entry in lineage], [parent])
                self.assertEqual(lineage[0]["path"], activation.AKA_V7_RECORDS)
        for name, parent in activation.AKA_PARENTS.items():
            with self.subTest(task=name):
                self.assertIn(parent, rows)
                row = rows[parent]
                self.assertEqual(row["final_status"], "dynamic_valid")
                self.assertEqual(row["current_ir_expressibility"], "expressible")
                document, _ = create_task(name, rows=2, columns=7)
                lineage = [entry for entry in document["provenance"]
                           if entry["kind"] == "aka_qualified_parent_lineage"]
                self.assertEqual([entry["derived_parent_id"] for entry in lineage], [parent])
                # The Apple Workload inherits semantics, never a measured B200 result.
                self.assertIn("no_B200", lineage[0]["scope"])
                self.assertFalse(row["performance_measured"])
        # A task with no recorded parent declares only its own specification.
        for name in (set(activation.TASKS) - set(activation.AKA_PARENTS)
                     - set(activation.AKA_V7_PARENTS)):
            with self.subTest(task=name):
                document, _ = create_task(name, rows=2, columns=7)
                self.assertEqual([entry["kind"] for entry in document["provenance"]],
                                 ["task_mathematical_specification"])

    def test_a_saved_forward_output_cannot_be_declared_with_an_impossible_sign(self):
        """A B300 run surfaced this: revision 1 admitted inputs the operator never sees.

        `y` here is a saved softplus output, and softplus(x) = log(1 + exp(x)) is strictly
        positive. Revision 1 declared it as a freely signed value bounded by 16, so
        mixed_magnitude generated y = -16, where 1 - exp(-y) reaches -8.9e6 and the output
        reaches 1.4e8 -- seven orders of magnitude past the operator's real range. The
        relative tolerance then admitted an absolute error of 2844, so that case tested
        nothing. Same class as the GEMM SiLU oracle overflow.
        """
        for name in activation.TASKS:
            declared = activation.NONNEGATIVE[name]
            workload, _ = self.task(name, rows=4, columns=8)
            tensors = workload.document["tensors"]
            with self.subTest(task=name):
                self.assertEqual(sorted(n for n in tensors if tensors[n].get("nonnegative")),
                                 sorted(declared))
                for case_id in activation.CASES:
                    supplied = materialize_case(workload, case_id)
                    for tensor in declared:
                        self.assertTrue(all(value >= 0.0 for value in supplied[tensor]))
                if declared:
                    supplied = materialize_case(workload, "primary")
                    supplied[declared[0]] = [-1.0] * len(supplied[declared[0]])
                    with self.assertRaisesRegex(ValueError, "non-negative"):
                        reference_outputs(workload, "primary", supplied)
        # The output now stays inside the range the operator can actually produce.
        workload, _ = self.task("softplus_gradient", rows=4, columns=8)
        for case_id in activation.CASES:
            supplied = materialize_case(workload, case_id)
            out = reference_outputs(workload, case_id, supplied)["out"]
            with self.subTest(case=case_id):
                self.assertLessEqual(max(map(abs, out)), activation.MAX_ABS + 1e-6)
        # The narrowed contract is a distinct revision, so a revision-1 document is refused.
        stale = deepcopy(workload.document)
        stale["revision"] = "1"
        with self.assertRaises(ValueError):
            activation.validate_activation_contract(stale)

    def test_the_family_owns_no_reduction_affine_parameter_or_epsilon(self):
        for name in activation.TASKS:
            with self.subTest(task=name):
                document, source = create_task(name, backend="metal-m2", rows=4, columns=7)
                self.assertEqual(document["semantics"]["arithmetic"]["reduction"],
                                 "none_every_output_reads_one_input_coordinate")
                self.assertNotIn("epsilon", document["semantics"])
                for absent in ("weight", "bias", "residual"):
                    self.assertNotIn(absent, document["tensors"])
                self.assertEqual([arg.name for arg in WorkloadContract(document).tensor_abi("primary")],
                                 [n for n, _ in activation.INPUTS[name]] + ["out"])
                schedule = frontend.parse(source).document
                self.assertEqual([op for op in schedule["operations"] if op["kind"] == "reduce"], [])
        # The bound is two exponents below the normalization families' so that the
        # sigmoid's exponential of the most negative admitted input stays finite in FP32.
        self.assertEqual(activation.MAX_ABS, 16.0)
        self.assertLess(activation.MAX_ABS, normalization.workload_document(
            "softmax", rows=2, columns=7)["tensors"]["x"]["max_abs"])

    def test_one_task_freezes_for_an_apple_or_an_nvidia_device_unchanged(self):
        """The mathematics is device-independent; only the device binding moves.

        Two contract fields and one Schedule decorator are the whole portability surface
        for this family, so a Workload frozen for a Metal device and the same Workload
        frozen for a CUDA device differ in exactly the target, the identity that names it
        and the route's own tanh spelling -- never in the definition, ABI, input cases,
        oracle or tolerance.
        """
        def device_bearing(document):
            """Strip the three fields that name the device, keep everything else."""
            stripped = deepcopy(document)
            stripped.pop("workload_id")
            stripped["semantics"].pop("target")
            stripped["provenance"][0].pop("scope")
            return stripped

        for name in activation.TASKS:
            frozen = {}
            for backend, device in BACKENDS.items():
                document, source = create_task(name, backend=backend, rows=2, columns=8)
                with self.subTest(task=name, backend=backend):
                    self.assertEqual(document["semantics"]["target"], device["target"])
                    assessment = self.compiler.assess(frontend.parse(source).document)
                    self.assertTrue(assessment.lowering_eligible, assessment.findings)
                    self.assertIn(f'backend="{device["route"]}"', source)
                    self.assertIn(device["tanh_contract"] if name == "gelu_tanh" else "lm.load",
                                  source)
                frozen[backend] = document
            (first, *rest) = frozen.values()
            for document in rest:
                with self.subTest(task=name, workload=document["workload_id"]):
                    # Definition, ABI, input cases, oracle and tolerance are the task;
                    # none of them may move when the device does.
                    self.assertEqual(device_bearing(document), device_bearing(first))
                    self.assertNotEqual(document["workload_id"], first["workload_id"])
                    self.assertNotEqual(document["semantics"]["target"],
                                        first["semantics"]["target"])

    def test_a_route_that_cannot_tile_a_width_refuses_before_freezing_it(self):
        """A Workload with no Schedule is worse than a refusal, so refuse at creation."""
        for columns in (1, 8, 1024):
            for backend in BACKENDS:
                with self.subTest(columns=columns, backend=backend):
                    create_task("silu", backend=backend, rows=2, columns=columns)
        for columns in (7, 65, 257, TRITON_MAXIMUM_TILE * 2):
            with self.subTest(columns=columns):
                # Metal stripes any width over 32 lanes; tl.arange needs a power of two.
                create_task("silu", backend="metal-m2", rows=2, columns=columns)
                with self.assertRaisesRegex(ValueError, "power-of-two"):
                    create_task("silu", backend="triton-b200", rows=2, columns=columns)

    def test_gelu_names_the_metal_tanh_contract_and_no_other(self):
        """The named contract is what makes precise::tanh reachable from an Apple Target.

        Authoring this task is what found the gap: tanh was mapped by the Metal emitter
        and required to name a contract that no Apple Target admitted, so no Apple
        Schedule could reach the mapping at all.
        """
        _, source = self.task("gelu_tanh", backend="metal-m2")
        schedule = frontend.parse(source).document
        assessment = self.compiler.assess(schedule)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        self.assertIn("precise::tanh(", self.compiler.lower(assessment).source)
        # The CUDA route reaches the same task through its own admitted spelling.
        _, cuda = self.task("gelu_tanh", backend="triton-b200", columns=8)
        cuda_assessment = self.compiler.assess(frontend.parse(cuda).document)
        self.assertTrue(cuda_assessment.lowering_eligible, cuda_assessment.findings)
        self.assertIn("libdevice.tanh(", self.compiler.lower(cuda_assessment).source)
        # Neither Target admits the other's spelling, which is why the contract is named.
        for contract in ("libdevice.tanh.f32", "ptx.fma.rn.f32", "metal.fast.tanh.f32"):
            borrowed = deepcopy(schedule)
            operation = next(op for op in borrowed["operations"] if op["id"] == "tanh")
            operation["parameters"]["instruction"]["contract"] = contract
            with self.subTest(contract=contract):
                refused = self.compiler.assess(borrowed)
                self.assertFalse(refused.lowering_eligible)
                self.assertIn("TARGET_INSTRUCTION_UNSUPPORTED",
                              [finding.code for finding in refused.findings])

    def test_no_task_asks_for_a_conditional_the_ir_does_not_have(self):
        """SELU and softsign are branch-shaped in the textbook and branch-free here.

        This is the AKA v6 review's own finding: absolute value is `relu(x) + relu(-x)`
        and SELU's positive arm cancels because `alpha * exp(0) - alpha` is zero. If the
        IR ever grows a compare/select, these two stop being the reason to add it.
        """
        for name in ("selu", "softsign"):
            with self.subTest(task=name):
                _, source = self.task(name)
                schedule = frontend.parse(source).document
                operations = [op["parameters"].get("op") for op in schedule["operations"]
                              if op["kind"] == "elementwise"]
                self.assertIn("relu", operations)
                self.assertTrue(set(operations) <= {"relu", "exp", "add", "sub", "mul", "div"},
                                operations)
        # The reference SELU is the piecewise function; the Schedule reaches it through
        # ReLU arms, and the two agree on the sign boundary exactly.
        selu, _ = self.task("selu", rows=1, columns=3)
        out = reference_outputs(selu, "primary", {"x": [0.0, -0.0, 1.0]})["out"]
        self.assertEqual([abs(out[0]), abs(out[1])], [0.0, 0.0])
        self.assertEqual(out[2], _round(activation.SELU_LAMBDA, "fp32"))

    def test_oracles_match_independent_small_formulas(self):
        values = [0.0, 1.0, -2.0]
        expected = {
            "silu": [v / (1.0 + math.exp(-v)) for v in values],
            "softsign": [v / (1.0 + abs(v)) for v in values],
            "selu": [activation.SELU_LAMBDA * (v if v > 0 else activation.SELU_ALPHA * (math.exp(v) - 1))
                     for v in values],
            "gelu_tanh": [0.5 * v * (1.0 + math.tanh(
                activation.GELU_INNER_SCALE * (v + activation.GELU_CUBIC_SCALE * v ** 3)))
                for v in values],
        }
        for name, reference in expected.items():
            with self.subTest(task=name):
                workload, _ = self.task(name, rows=1, columns=3)
                actual = reference_outputs(workload, "primary", {"x": values})["out"]
                for observed, wanted in zip(actual, reference):
                    self.assertAlmostEqual(observed, _round(wanted, "fp32"), places=6)
        # The two-input tasks read a second tensor rather than a second coordinate.
        second = [2.0, -0.5, 4.0]
        swiglu, _ = self.task("swiglu", rows=1, columns=3)
        self.assertEqual(reference_outputs(swiglu, "primary", {"x": values, "up": second})["out"],
                         [_round(v / (1.0 + math.exp(-v)) * p, "fp32")
                          for v, p in zip(values, second)])
        # y is a saved softplus output, so the contract admits only non-negative values.
        softplus, _ = self.task("softplus_gradient", rows=1, columns=3)
        saved = [0.0, 1.0, 2.0]
        self.assertEqual(reference_outputs(softplus, "primary", {"y": saved, "dy": second})["out"],
                         [_round(g * (1.0 - math.exp(-v)), "fp32") for v, g in zip(saved, second)])
        # PReLU's slope is per feature, so its second operand has the row's width.
        prelu, _ = self.task("prelu", rows=1, columns=3)
        self.assertEqual(reference_outputs(prelu, "primary", {"x": values, "slope": second})["out"],
                         [_round(v if v > 0 else v * s, "fp32") for v, s in zip(values, second)])

    def test_every_task_maps_an_all_zero_row_to_zero(self):
        """Every member of this family is zero at the origin once its gate is applied."""
        for name in activation.TASKS:
            with self.subTest(task=name):
                workload, _ = self.task(name, rows=1, columns=2)
                supplied = {arg.name: [0.0] * math.prod(arg.shape)
                            for arg in workload.tensor_abi("primary") if arg.mode == "input"}
                out = reference_outputs(workload, "primary", supplied)["out"]
                self.assertEqual([abs(value) for value in out], [0.0, 0.0])

    def test_materialization_preserves_abi_fp32_bounds_and_seed_reproducibility(self):
        for name in activation.TASKS:
            workload, _ = self.task(name, rows=2, columns=65)
            primary = activation.INPUTS[name][0][0]
            for case_id in workload.case_ids:
                with self.subTest(task=name, case=case_id):
                    inputs = materialize_case(workload, case_id)
                    self.assertEqual(inputs, materialize_case(workload, case_id))
                    args = [arg for arg in workload.tensor_abi(case_id) if arg.mode == "input"]
                    self.assertEqual(list(inputs), [arg.name for arg in args])
                    for arg in args:
                        self.assertEqual(len(inputs[arg.name]), math.prod(arg.shape))
                        self.assertTrue(all(math.isfinite(value) and abs(value) <= activation.MAX_ABS
                                            and _round(value, "fp32") == value
                                            for value in inputs[arg.name]))
                        # An admitted input never makes the sigmoid's exponential overflow.
                        self.assertTrue(all(math.isfinite(math.exp(-value)) for value in inputs[arg.name]))
                    if case_id == "zeros":
                        self.assertTrue(all(value == 0 for value in inputs[primary]))
                    elif case_id == "near_zero":
                        self.assertLessEqual(max(map(abs, inputs[primary])), 1e-4)
                    elif case_id == "mixed_magnitude":
                        self.assertEqual(max(map(abs, inputs[primary])), 16.0)
            self.assertNotEqual(materialize_case(workload, "primary")[primary],
                                materialize_case(workload, "alternating")[primary])

    def test_frozen_contract_rejects_target_domain_or_case_drift(self):
        document, _ = create_task("swiglu", rows=2, columns=7)
        changes = (
            lambda d: d["semantics"].update(target="apple_gpu_family9"),
            lambda d: d["semantics"].update(epsilon=1e-5),
            lambda d: d["semantics"]["arithmetic"].update(reduction="cta"),
            lambda d: d["semantics"]["candidate_abi"].update(inputs=["up", "x"]),
            lambda d: d["tensors"]["x"].update(max_abs=256.0),
            lambda d: d["tensors"].pop("up"),
            lambda d: d["tensors"]["out"].update(dtype="bf16"),
            lambda d: d["validation"].update(atol=1.0),
            lambda d: d["validation"].update(all_cases_required=False),
            lambda d: d["oracle"].update(callable="candidate.reference"),
            lambda d: d["cases"].pop(),
            lambda d: d["cases"][1]["shape"].update(C=65),
            lambda d: d["cases"][0].update(seed=True),
            lambda d: d.update(operator="silu_fp32"),
            lambda d: d.update(revision="2"),
            lambda d: d.update(state="draft"),
        )
        for change in changes:
            changed = deepcopy(document)
            change(changed)
            with self.subTest(document=changed), self.assertRaises(ValueError):
                activation.validate_activation_contract(changed)
        # A migrated task cannot silently drop, borrow or invent its recorded lineage.
        migrated, _ = create_task("softsign", rows=2, columns=7)
        for change in (lambda d: d["provenance"].pop(),
                       lambda d: d["provenance"][1].update(derived_parent_id="selu_contiguous_v9"),
                       lambda d: d["provenance"][1].update(kind="task_mathematical_specification")):
            changed = deepcopy(migrated)
            change(changed)
            with self.subTest(provenance=changed["provenance"]), self.assertRaises(ValueError):
                activation.validate_activation_contract(changed)
        for options in ({"backend": "metal"}, {"backend": "cuda"}, {"rows": True},
                        {"columns": 0}, {"rows": 2**30, "columns": 1}, {"depth": 8}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_task("swiglu", **options)

    def test_oracle_refuses_nonfinite_unrounded_and_incomplete_inputs(self):
        for name in activation.TASKS:
            workload, _ = self.task(name)
            primary = activation.INPUTS[name][0][0]
            inputs = materialize_case(workload, "primary")
            for replacement in ([], [0.0], [float("nan")] * 14, [float("inf")] * 14,
                                [True] * 14, [0.1] * 14, [32.0] * 14):
                with self.subTest(task=name, replacement=replacement), self.assertRaises(ValueError):
                    reference_outputs(workload, "primary", {**inputs, primary: replacement})
            if len(activation.INPUTS[name]) > 1:
                missing = activation.INPUTS[name][1][0]
                with self.subTest(task=name, missing=missing), self.assertRaises(ValueError):
                    reference_outputs(workload, "primary", {primary: inputs[primary]})

    def test_starter_binds_workload_and_all_case_abis_on_every_route(self):
        for name in activation.TASKS:
            for width in (1, 7, 32, 65, 257, 1024):
                lowered = {}
                for backend in activation.BACKENDS:
                    try:
                        admit_width(backend, width)
                    except ValueError:
                        continue  # This route cannot tile this width; covered separately.
                    with self.subTest(task=name, width=width, backend=backend):
                        workload, source = self.task(name, columns=width, backend=backend)
                        schedule = frontend.parse(source).document
                        self.assertEqual(schedule["metadata"]["workload_contract_sha256"],
                                         workload.canonical_sha256)
                        buffers = [b for b in schedule["buffers"] if b["space"] == "global"]
                        for case_id in workload.case_ids:
                            self.assertEqual([(b["name"], tuple(b["shape"]), b["dtype"], b["mode"])
                                              for b in buffers],
                                             [(a.name, a.shape, a.dtype, a.mode)
                                              for a in workload.tensor_abi(case_id)])
                        assessment = self.compiler.assess(schedule)
                        self.assertTrue(assessment.lowering_eligible, assessment.findings)
                        lowering = self.compiler.lower(assessment)
                        self.assertEqual(lowering.toolchain_requirements["target"], workload.target)
                        self.assertFalse(assessment.calibration_available)
                        route = activation.BACKENDS[backend]["route"]
                        if route == "metal":
                            # Whatever the task's own kernel is, no SIMD exchange belongs in it.
                            if TRANSCENDENTAL[name] is None:
                                self.assertNotIn("precise::", lowering.source)
                            else:
                                self.assertIn(TRANSCENDENTAL[name], lowering.source)
                            self.assertNotIn("simd_", lowering.source)
                        else:
                            self.assertIn("@triton.jit", lowering.source)
                            self.assertNotIn("tl.sum", lowering.source)
                        lowered.setdefault(route, []).append(
                            (dict(lowering.toolchain_requirements), lowering.source))
                # Two devices on one route differ only in the Target they commit to and
                # in the frozen identity the emitted header records.
                def anonymize(text):
                    for backend in activation.BACKENDS:
                        text = text.replace(backend, "<backend>")
                    return "\n".join(line for line in text.splitlines()
                                      if not line.startswith("# schedule_sha256="))
                for route, emitted in lowered.items():
                    (first, *rest) = emitted
                    for requirements, source in rest:
                        self.assertEqual(anonymize(source), anonymize(first[1]), route)
                        self.assertEqual({k: v for k, v in requirements.items() if k != "target"},
                                         {k: v for k, v in first[0].items() if k != "target"})

    def test_generated_starters_match_all_input_case_oracles_on_cpu(self):
        from tests.contracts import test_metal as cpu_contracts
        runner = cpu_contracts.MetalTests("runTest")
        runner.compiler = self.compiler
        for name in activation.TASKS:
            for width, cases in ((1, ("primary",)), (7, tuple(activation.CASES)),
                                 (65, ("mixed_magnitude", "near_zero"))):
                workload, source = self.task(name, columns=width)
                schedule = frontend.parse(source).document
                for case_id in cases:
                    with self.subTest(task=name, width=width, case=case_id):
                        inputs = materialize_case(workload, case_id)
                        expected = reference_outputs(workload, case_id, inputs)
                        outputs = runner.execute_body(schedule, inputs)
                        passed, metrics = compare_tile_outputs(
                            workload, inputs, expected, {"out": outputs["out"]},
                            {key: outputs[key] for key in inputs})
                        self.assertTrue(passed, metrics)


if __name__ == "__main__":
    unittest.main()
