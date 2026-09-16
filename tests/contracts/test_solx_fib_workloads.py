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
    ATOL, CASES, EPSILON, HIDDEN_SIZE, RTOL, TASKS,
    materialize_case, reference_outputs, validate_solx_fib_contract, workload_document,
)
from open_cake_ir.tasks.tiles.workload import _round
from open_cake_ir.tasks.workloads import create_task, load_workload

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/workloads/solx-fib-rmsnorm-h4096-bf16-triton-b300-r170-v1.json"
TASK = "fib_rmsnorm_h4096"


def _tiny(rows=2, **kwargs):
    return WorkloadContract(workload_document(TASK, rows=rows, columns=4096, **kwargs))


class FrozenContractTests(unittest.TestCase):
    def test_committed_contract_loads_through_the_common_registry(self):
        workload = load_workload(CONTRACT)
        self.assertEqual(workload.workload_id, "solx-fib-rmsnorm-h4096-bf16-triton-b300-r170-v1")
        self.assertEqual(workload.document["operator"], TASKS[TASK][0])
        self.assertEqual(workload.target, "sm_103a")
        self.assertEqual(workload.case_ids, tuple(CASES))
        self.assertEqual(
            [(arg.name, arg.shape, arg.dtype, arg.mode) for arg in workload.tensor_abi("primary")],
            [("x", (170, 4096), "bf16", "input"), ("weight", (4096,), "bf16", "input"),
             ("out", (170, 4096), "bf16", "output")])

    def test_committed_bytes_are_the_generator_output(self):
        document = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(document, workload_document(TASK, rows=170, columns=4096,
                                                     backend="triton-b300"))

    def test_the_upstream_matched_ratio_gate_is_not_inherited(self):
        validation = json.loads(CONTRACT.read_text(encoding="utf-8"))["validation"]
        self.assertEqual(validation["comparison"], "elementwise_atol_rtol")
        self.assertTrue(validation["all_cases_required"])
        self.assertNotIn("required_matched_ratio", validation)
        # Every element is compared, and both bounds are tighter than the upstream 1e-2.
        self.assertLess(validation["rtol"], 1e-2)
        self.assertLess(validation["atol"], 1e-2)
        self.assertEqual((validation["atol"], validation["rtol"]), (ATOL, RTOL))

    def test_restricted_upstream_implementations_are_declared_not_assumed(self):
        provenance = json.loads(CONTRACT.read_text(encoding="utf-8"))["provenance"]
        restricted = [entry for entry in provenance if entry["kind"] == "restricted_artifact"]
        self.assertTrue(restricted)
        for entry in restricted:
            self.assertEqual(entry["role"], "complete_low_level_target_implementation")
            self.assertIn("not_readable_by_clean_start", entry["scope"])
        upstream, = [entry for entry in provenance if entry["kind"] == "solx_pack_task"]
        self.assertEqual(upstream["upstream"], "flashinfer-bench")
        self.assertIn("semantic_source_only", upstream["scope"])
        # The remaining upstream batch sizes are recorded, not claimed as examined.
        self.assertEqual(upstream["workload_count"], 14)


class OracleTests(unittest.TestCase):
    def test_reference_matches_the_real_valued_definition_on_a_hand_computation(self):
        workload = _tiny(rows=2)
        inputs = materialize_case(workload, "primary")
        outputs = reference_outputs(workload, "primary", inputs)["out"]
        width = 4096
        for row in range(2):
            values = inputs["x"][row * width:(row + 1) * width]
            inverse = 1.0 / math.sqrt(math.fsum(v * v for v in values) / width + EPSILON)
            expected = [_round(v * inverse * w, "bf16")
                        for v, w in zip(values, inputs["weight"])]
            self.assertEqual(outputs[row * width:(row + 1) * width], expected)

    def test_every_declared_case_is_finite_and_exactly_representable_in_bf16(self):
        workload = _tiny(rows=2)
        for case_id in workload.case_ids:
            inputs = materialize_case(workload, case_id)
            for name, vector in inputs.items():
                self.assertTrue(all(_round(v, "bf16") == v for v in vector), (case_id, name))
            outputs = reference_outputs(workload, case_id, inputs)["out"]
            self.assertEqual(len(outputs), 2 * 4096)
            self.assertTrue(all(math.isfinite(v) and _round(v, "bf16") == v for v in outputs), case_id)

    def test_epsilon_keeps_an_all_zero_row_finite_and_zero(self):
        workload = _tiny(rows=1)
        inputs = materialize_case(workload, "zeros")
        self.assertTrue(all(v == 0.0 for v in inputs["x"]))
        self.assertEqual(set(reference_outputs(workload, "zeros", inputs)["out"]), {0.0})

    def test_materialization_is_deterministic(self):
        workload = _tiny(rows=2)
        self.assertEqual(materialize_case(workload, "primary"),
                         materialize_case(workload, "primary"))

    def test_oracle_refuses_an_input_outside_the_declared_domain(self):
        workload = _tiny(rows=1)
        inputs = materialize_case(workload, "primary")
        inputs["x"][0] = 512.0
        with self.assertRaisesRegex(ValueError, "bounded, finite bf16"):
            reference_outputs(workload, "primary", inputs)

    def test_oracle_refuses_a_value_that_is_not_representable_in_bf16(self):
        workload = _tiny(rows=1)
        inputs = materialize_case(workload, "primary")
        inputs["x"][0] = 1.0000001
        with self.assertRaisesRegex(ValueError, "bounded, finite bf16"):
            reference_outputs(workload, "primary", inputs)

    def test_oracle_refuses_an_incomplete_input_abi(self):
        workload = _tiny(rows=1)
        inputs = materialize_case(workload, "primary")
        del inputs["weight"]
        with self.assertRaisesRegex(ValueError, "complete input ABI"):
            reference_outputs(workload, "primary", inputs)


class RefusalTests(unittest.TestCase):
    def test_another_hidden_size_is_a_different_upstream_task(self):
        with self.assertRaisesRegex(ValueError, "different upstream task"):
            workload_document(TASK, rows=8, columns=2048, backend="triton-b300")
        self.assertEqual(HIDDEN_SIZE[TASK], 4096)

    def test_a_route_that_cannot_name_bf16_is_refused_by_dtype(self):
        with self.assertRaisesRegex(ValueError, r"cannot name dtype 'bf16'"):
            workload_document(TASK, rows=8, columns=4096, backend="metal-m1-pro")

    def test_a_target_that_does_not_declare_cast_is_refused_by_operation_kind(self):
        # gfx938 declares load/elementwise/reduce/store only, so this BF16 ABI has no
        # widening or narrowing conversion there. The refusal names that, not the dtype.
        with self.assertRaisesRegex(ValueError, r"does not admit operation kind 'cast'"):
            workload_document(TASK, rows=8, columns=4096, backend="triton-dcu")

    def test_a_valid_but_different_backend_is_admitted(self):
        document = workload_document(TASK, rows=8, columns=4096, backend="triton-b200")
        self.assertEqual(document["semantics"]["target"], "sm_100a")
        validate_solx_fib_contract(document)

    def test_a_changed_epsilon_is_refused_by_the_frozen_contract_guard(self):
        document = workload_document(TASK, rows=8, columns=4096, backend="triton-b300")
        document["semantics"]["epsilon"] = 1e-6
        with self.assertRaisesRegex(ValueError, "frozen semantic/ABI/input/validation contract differs"):
            validate_solx_fib_contract(document)

    def test_a_loosened_tolerance_is_refused_by_the_frozen_contract_guard(self):
        document = workload_document(TASK, rows=8, columns=4096, backend="triton-b300")
        document["validation"]["rtol"] = 1e-2
        with self.assertRaisesRegex(ValueError, "frozen semantic/ABI/input/validation contract differs"):
            validate_solx_fib_contract(document)

    def test_a_dropped_input_case_is_refused_before_the_byte_comparison(self):
        document = workload_document(TASK, rows=8, columns=4096, backend="triton-b300")
        document["cases"] = document["cases"][:1]
        with self.assertRaisesRegex(ValueError, "required input cases differ"):
            validate_solx_fib_contract(document)

    def test_a_nonpositive_batch_extent_is_refused_by_the_abi_bound(self):
        with self.assertRaisesRegex(ValueError, "BF16 buffer ABI"):
            workload_document(TASK, rows=0, columns=4096, backend="triton-b300")


class StarterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_the_launcher_produces_a_lowering_eligible_starter_for_this_task(self):
        document, source = create_task(TASK, backend="triton-b300", rows=8, columns=4096)
        assessment = self.compiler.assess(parse(source, filename="starter.py").document)
        self.assertTrue(assessment.lowering_eligible,
                        [finding.code for finding in assessment.findings])
        self.assertEqual(assessment.findings, ())

    def test_the_starter_keeps_the_bf16_abi_and_widens_only_in_between(self):
        workload = _tiny(rows=8)
        schedule = parse(starter_source(workload), filename="starter.py").document
        dtypes = {buffer["name"]: buffer["dtype"] for buffer in schedule["buffers"]}
        self.assertEqual((dtypes["x"], dtypes["weight"], dtypes["out"]), ("bf16", "bf16", "bf16"))
        self.assertEqual(dtypes["values"], "fp32")
        self.assertEqual(dtypes["narrowed"], "bf16")
        casts = [op["parameters"]["to"] for op in schedule["operations"] if op["kind"] == "cast"]
        self.assertEqual(casts, ["fp32", "fp32", "bf16"])

    def test_the_starter_binds_the_exact_workload_it_was_generated_from(self):
        workload = _tiny(rows=8)
        schedule = parse(starter_source(workload), filename="starter.py").document
        self.assertEqual(schedule["metadata"]["workload_contract_sha256"],
                         workload.canonical_sha256)


if __name__ == "__main__":
    unittest.main()
