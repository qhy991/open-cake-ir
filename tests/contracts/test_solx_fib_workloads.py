"""SoL-ExecBench (FlashInfer-Bench definitions) Workload admission, oracle and refusals.

Each refusal test names the guard it expects, so a hazard that happens to be blocked by
an unrelated rule is visible as such rather than counted as coverage.
"""
import json
import math
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib.authoring import starter_source
from open_cake_ir.tasks.solx_fib.workload import (
    ATOL, CASES, RTOL, SEED_ELEMENT_CAP, SPECS, TASKS,
    admitting_backends, default_rows, launchable_tasks, materialize_case,
    reference_outputs, row_spans, validate_solx_fib_contract, workload_document,
)
from open_cake_ir.tasks.tiles.workload import _round
from open_cake_ir.tasks.workloads import create_task, load_workload

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "contracts/workloads"
# The three captures whose hidden size is not a power of two.
PARTITIONED_TASKS = ("fib_fused_add_rmsnorm_h7168", "fib_rmsnorm_h1536", "fib_rmsnorm_h7168")


def _contract_path(task):
    document = workload_document(task, rows=default_rows(task),
                                 columns=SPECS[task]["hidden"], backend="triton-b300")
    return CONTRACTS / (document["workload_id"] + ".json"), document


def _tiny(task, rows=2, **kwargs):
    return WorkloadContract(workload_document(task, rows=rows,
                                              columns=SPECS[task]["hidden"], **kwargs))


def _upstream_reference(x, residual, weight, width, epsilon):
    """A direct transcription of the pack's torch reference in sequential FP32.

    It is written from the upstream definition, not from the module under test, so an
    agreement between them is evidence about the semantics rather than about one
    implementation reused twice.
    """
    result = []
    for start in range(0, len(x), width):
        if residual is None:
            row = [_round(x[start + i], "fp32") for i in range(width)]
        else:
            row = [_round(_round(x[start + i], "fp32") + _round(residual[start + i], "fp32"), "fp32")
                   for i in range(width)]
        mean = _round(sum(_round(value * value, "fp32") for value in row) / width, "fp32")
        inverse = 1.0 / math.sqrt(mean + epsilon)
        result.extend(_round(value * inverse * weight[i], "bf16") for i, value in enumerate(row))
    return result


class FamilyRegistrationTests(unittest.TestCase):
    def test_every_upstream_rmsnorm_capture_is_registered(self):
        self.assertEqual(len(TASKS), 9)
        self.assertEqual(set(TASKS), set(SPECS))
        self.assertEqual({spec["upstream"].split("/")[-1][:3] for spec in SPECS.values()},
                         {"001", "002", "003", "021", "022", "023", "024", "025", "026"})

    def test_epsilon_is_the_upstream_constant_and_is_not_one_family_value(self):
        # The two Llama-3.1-8B captures use 1e-5; every other capture uses 1e-6. A single
        # family constant would silently be wrong for seven of the nine.
        self.assertEqual({spec["epsilon"] for spec in SPECS.values()}, {1e-5, 1e-6})
        self.assertEqual(SPECS["fib_rmsnorm_h4096"]["epsilon"], 1e-5)
        self.assertEqual(SPECS["fib_fused_add_rmsnorm_h4096"]["epsilon"], 1e-5)
        self.assertEqual(SPECS["fib_rmsnorm_h2048"]["epsilon"], 1e-6)
        for task in launchable_tasks():
            _, document = _contract_path(task)
            self.assertEqual(document["semantics"]["epsilon"], SPECS[task]["epsilon"], task)

    def test_all_nine_tasks_have_an_admitted_route(self):
        self.assertEqual(set(launchable_tasks()), set(TASKS))
        for task in launchable_tasks():
            self.assertEqual(admitting_backends(task),
                             ("triton-b200", "triton-b300", "triton-gfx1151"), task)

    def test_partitioned_rows_cover_each_element_exactly_once(self):
        for task in PARTITIONED_TASKS:
            width = SPECS[task]["hidden"]
            spans = row_spans(width)
            self.assertGreater(len(spans), 1)
            self.assertEqual([i for start, stop in spans for i in range(start, stop)],
                             list(range(width)))
            self.assertTrue(all((stop - start) & (stop - start - 1) == 0
                                for start, stop in spans))

    def test_the_seed_extent_is_the_largest_upstream_batch_the_oracle_can_serve(self):
        for task, spec in SPECS.items():
            rows = default_rows(task)
            self.assertIn(rows, spec["batches"], task)
            self.assertLessEqual(rows * spec["hidden"], SEED_ELEMENT_CAP, task)
            larger = [batch for batch in spec["batches"] if batch > rows]
            self.assertTrue(all(batch * spec["hidden"] > SEED_ELEMENT_CAP for batch in larger), task)


class FrozenContractTests(unittest.TestCase):
    def test_one_committed_contract_per_launchable_task_and_no_others(self):
        committed = sorted(path.name for path in CONTRACTS.glob("solx-fib-*rmsnorm-*.json"))
        expected = sorted(_contract_path(task)[0].name for task in launchable_tasks())
        self.assertEqual(committed, expected)

    def test_committed_bytes_are_the_generator_output(self):
        for task in launchable_tasks():
            path, document = _contract_path(task)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), document, task)

    def test_each_contract_loads_through_the_common_registry(self):
        for task in launchable_tasks():
            path, _ = _contract_path(task)
            workload = load_workload(path)
            self.assertEqual(workload.document["operator"], TASKS[task][0], task)
            self.assertEqual(workload.target, "sm_103a", task)
            self.assertEqual(workload.case_ids, tuple(CASES), task)
            abi = workload.tensor_abi("primary")
            self.assertTrue(all(arg.dtype == "bf16" for arg in abi), task)
            names = [arg.name for arg in abi]
            self.assertEqual(names[-1], "out", task)
            self.assertEqual("residual" in names, SPECS[task]["residual"], task)

    def test_the_upstream_matched_ratio_gate_is_not_inherited(self):
        for task in launchable_tasks():
            path, _ = _contract_path(task)
            validation = json.loads(path.read_text(encoding="utf-8"))["validation"]
            self.assertEqual(validation["comparison"], "elementwise_atol_rtol", task)
            self.assertTrue(validation["all_cases_required"], task)
            self.assertNotIn("required_matched_ratio", validation, task)
            self.assertEqual((validation["atol"], validation["rtol"]), (ATOL, RTOL), task)
            self.assertLess(validation["rtol"], 1e-2, task)

    def test_restricted_upstream_implementations_are_declared_not_assumed(self):
        for task in launchable_tasks():
            path, _ = _contract_path(task)
            provenance = json.loads(path.read_text(encoding="utf-8"))["provenance"]
            restricted = [entry for entry in provenance if entry["kind"] == "restricted_artifact"]
            self.assertEqual(len(restricted), 2, task)
            for entry in restricted:
                self.assertEqual(entry["role"], "complete_low_level_target_implementation", task)
                self.assertIn("not_readable_by_clean_start", entry["scope"], task)
            upstream, = [entry for entry in provenance if entry["kind"] == "solx_pack_task"]
            self.assertEqual(upstream["upstream"], "flashinfer-bench", task)
            self.assertIn("semantic_source_only", upstream["scope"], task)
            # The whole upstream axis is recorded; only one of its values is examined.
            self.assertEqual(upstream["upstream_batch_axis"], list(SPECS[task]["batches"]), task)
            self.assertIn("not for memory-bandwidth saturation",
                          json.loads(path.read_text(encoding="utf-8"))["semantics"]["exclusions"], task)


class OracleTests(unittest.TestCase):
    def test_the_oracle_agrees_with_a_transcription_of_the_upstream_reference(self):
        for task in launchable_tasks():
            workload = _tiny(task, rows=2)
            width, epsilon = SPECS[task]["hidden"], SPECS[task]["epsilon"]
            for case_id in workload.case_ids:
                inputs = materialize_case(workload, case_id)
                expected = _upstream_reference(inputs["x"], inputs.get("residual"),
                                               inputs["weight"], width, epsilon)
                self.assertEqual(reference_outputs(workload, case_id, inputs)["out"],
                                 expected, (task, case_id))

    def test_every_declared_case_is_finite_and_exactly_representable_in_bf16(self):
        for task in launchable_tasks():
            workload = _tiny(task, rows=2)
            for case_id in workload.case_ids:
                inputs = materialize_case(workload, case_id)
                for name, vector in inputs.items():
                    self.assertTrue(all(_round(v, "bf16") == v for v in vector), (task, case_id, name))
                outputs = reference_outputs(workload, case_id, inputs)["out"]
                self.assertEqual(len(outputs), 2 * SPECS[task]["hidden"], (task, case_id))
                self.assertTrue(all(math.isfinite(v) and _round(v, "bf16") == v for v in outputs),
                                (task, case_id))

    def test_the_residual_operand_changes_the_result(self):
        # The fused capture is not the plain one with an ignored third input.
        fused = _tiny("fib_rmsnorm_h2048", rows=2), _tiny("fib_fused_add_rmsnorm_h2048", rows=2)
        plain_inputs = materialize_case(fused[0], "primary")
        fused_inputs = materialize_case(fused[1], "primary")
        self.assertIn("residual", fused_inputs)
        self.assertNotIn("residual", plain_inputs)
        self.assertNotEqual(reference_outputs(fused[1], "primary", fused_inputs)["out"],
                            reference_outputs(fused[0], "primary", plain_inputs)["out"])

    def test_epsilon_keeps_an_all_zero_row_finite_and_zero(self):
        for task in launchable_tasks():
            workload = _tiny(task, rows=1)
            inputs = materialize_case(workload, "zeros")
            self.assertTrue(all(v == 0.0 for v in inputs["x"]), task)
            self.assertEqual(set(reference_outputs(workload, "zeros", inputs)["out"]), {0.0}, task)

    def test_materialization_is_deterministic(self):
        workload = _tiny("fib_rmsnorm_h512", rows=2)
        self.assertEqual(materialize_case(workload, "primary"),
                         materialize_case(workload, "primary"))

    def test_two_tasks_at_the_same_shape_do_not_share_inputs(self):
        # Distinct per-task case seeds, so a task cannot pass by reusing a sibling's data.
        left = materialize_case(_tiny("fib_rmsnorm_h2048", rows=2), "primary")
        right = materialize_case(_tiny("fib_fused_add_rmsnorm_h2048", rows=2), "primary")
        self.assertNotEqual(left["x"], right["x"])

    def test_oracle_refuses_an_input_outside_the_declared_domain(self):
        workload = _tiny("fib_rmsnorm_h512", rows=1)
        inputs = materialize_case(workload, "primary")
        inputs["x"][0] = 512.0
        with self.assertRaisesRegex(ValueError, "bounded, finite bf16"):
            reference_outputs(workload, "primary", inputs)

    def test_oracle_refuses_a_value_that_is_not_representable_in_bf16(self):
        workload = _tiny("fib_rmsnorm_h512", rows=1)
        inputs = materialize_case(workload, "primary")
        inputs["x"][0] = 1.0000001
        with self.assertRaisesRegex(ValueError, "bounded, finite bf16"):
            reference_outputs(workload, "primary", inputs)

    def test_oracle_refuses_an_incomplete_input_abi(self):
        workload = _tiny("fib_fused_add_rmsnorm_h2048", rows=1)
        inputs = materialize_case(workload, "primary")
        del inputs["residual"]
        with self.assertRaisesRegex(ValueError, "complete input ABI"):
            reference_outputs(workload, "primary", inputs)


class RefusalTests(unittest.TestCase):
    def test_another_hidden_size_is_a_different_upstream_task(self):
        with self.assertRaisesRegex(ValueError, "different upstream task"):
            workload_document("fib_rmsnorm_h4096", rows=8, columns=2048, backend="triton-b300")

    def test_a_route_that_cannot_name_bf16_is_refused_by_dtype(self):
        with self.assertRaisesRegex(ValueError, r"cannot name dtype 'bf16'"):
            workload_document("fib_rmsnorm_h512", rows=8, columns=512, backend="metal-m1-pro")

    def test_a_target_that_does_not_declare_cast_is_refused_by_operation_kind(self):
        # gfx938 declares load/elementwise/reduce/store only, so this BF16 ABI has no
        # widening or narrowing conversion there. The refusal names that, not the dtype.
        with self.assertRaisesRegex(ValueError, r"does not admit operation kind 'cast'"):
            workload_document("fib_rmsnorm_h512", rows=8, columns=512, backend="triton-dcu")

    def test_a_valid_but_different_backend_is_admitted(self):
        document = workload_document("fib_rmsnorm_h512", rows=8, columns=512, backend="triton-b200")
        self.assertEqual(document["semantics"]["target"], "sm_100a")
        validate_solx_fib_contract(document)

    def test_a_changed_epsilon_is_refused_by_the_frozen_contract_guard(self):
        document = workload_document("fib_rmsnorm_h512", rows=8, columns=512, backend="triton-b300")
        document["semantics"]["epsilon"] = 1e-5
        with self.assertRaisesRegex(ValueError, "frozen semantic/ABI/input/validation contract differs"):
            validate_solx_fib_contract(document)

    def test_a_loosened_tolerance_is_refused_by_the_frozen_contract_guard(self):
        document = workload_document("fib_rmsnorm_h512", rows=8, columns=512, backend="triton-b300")
        document["validation"]["rtol"] = 1e-2
        with self.assertRaisesRegex(ValueError, "frozen semantic/ABI/input/validation contract differs"):
            validate_solx_fib_contract(document)

    def test_a_dropped_input_case_is_refused_before_the_byte_comparison(self):
        document = workload_document("fib_rmsnorm_h512", rows=8, columns=512, backend="triton-b300")
        document["cases"] = document["cases"][:1]
        with self.assertRaisesRegex(ValueError, "required input cases differ"):
            validate_solx_fib_contract(document)

    def test_a_nonpositive_batch_extent_is_refused_by_the_abi_bound(self):
        with self.assertRaisesRegex(ValueError, "BF16 buffer ABI"):
            workload_document("fib_rmsnorm_h512", rows=0, columns=512, backend="triton-b300")


class StarterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_the_launcher_produces_a_lowering_eligible_starter_for_every_task(self):
        for task in launchable_tasks():
            _, source = create_task(task, backend="triton-b300", rows=8,
                                    columns=SPECS[task]["hidden"])
            assessment = self.compiler.assess(parse(source, filename="starter.py").document)
            self.assertTrue(assessment.lowering_eligible,
                            (task, [finding.code for finding in assessment.findings]))
            self.assertEqual(assessment.findings, (), task)

    def test_partitioned_starters_lower_with_one_writer_and_full_output_loop(self):
        for task in PARTITIONED_TASKS:
            width = SPECS[task]["hidden"]
            _, source = create_task(task, backend="triton-b300", rows=2, columns=width)
            document = parse(source, filename="starter.py").document
            stores = [op for op in document["operations"] if op["kind"] == "store"]
            self.assertEqual(len(stores), 1, task)
            assessment = self.compiler.assess(document)
            self.assertEqual(assessment.findings, (), task)
            lowered = self.compiler.lower(assessment)
            self.assertEqual(lowered.target, "sm_103a")
            self.assertIn(f'mean_square = square_sum / {float(width)!r}', source)
            self.assertIn(f'tile={width & -width}', source)
            self.assertIn('square_sum = ' + ' + '.join(
                f'sum_{i}' for i in range(len(row_spans(width)))), source)

    def test_the_starter_keeps_the_bf16_abi_and_widens_only_in_between(self):
        for task in launchable_tasks():
            workload = _tiny(task, rows=8)
            schedule = parse(starter_source(workload), filename="starter.py").document
            dtypes = {buffer["name"]: buffer["dtype"] for buffer in schedule["buffers"]}
            self.assertEqual((dtypes["x"], dtypes["weight"], dtypes["out"]),
                             ("bf16", "bf16", "bf16"), task)
            self.assertEqual(dtypes["values_0" if task in PARTITIONED_TASKS else "values"], "fp32", task)
            self.assertEqual(dtypes["narrowed"], "bf16", task)
            casts = [op["parameters"]["to"] for op in schedule["operations"] if op["kind"] == "cast"]
            self.assertEqual(casts[-1], "bf16", task)
            self.assertTrue(all(dtype == "fp32" for dtype in casts[:-1]), task)

    def test_the_starter_uses_the_task_s_own_epsilon(self):
        for task in launchable_tasks():
            source = starter_source(_tiny(task, rows=8))
            self.assertIn(f"mean_square + {SPECS[task]['epsilon']!r}", source, task)

    def test_the_starter_binds_the_exact_workload_it_was_generated_from(self):
        workload = _tiny("fib_rmsnorm_h512", rows=8)
        schedule = parse(starter_source(workload), filename="starter.py").document
        self.assertEqual(schedule["metadata"]["workload_contract_sha256"],
                         workload.canonical_sha256)


if __name__ == "__main__":
    unittest.main()
