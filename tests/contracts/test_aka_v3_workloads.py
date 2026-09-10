from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.aka_v3 import workload as aka_v3
from open_cake_ir.tasks.workloads import load_workload


class AkaV3WorkloadTests(unittest.TestCase):
    def test_selected_contracts_are_loadable_and_keep_historical_boundary(self) -> None:
        expected = {
            "aka-residual-layernorm-fp32-triton-b200-v1": "aka_residual_layernorm_fp32",
            "aka-gemm-nt-bias-fp32-triton-b200-v1": "aka_gemm_nt_bias_fp32",
            "aka-row-gather-fp32-triton-b200-v1": "aka_row_gather_fp32",
            "aka-histogram-fp32-triton-b200-v1": "aka_histogram_fp32",
            "aka-max-pool1d-nwc-fp32-triton-b200-v1": "aka_max_pool1d_nwc_fp32",
            "aka-momentum-sgd-fp32-triton-b200-v1": "aka_momentum_sgd_fp32",
        }
        for workload_id, operator in expected.items():
            source = ROOT / "contracts/workloads" / f"{workload_id}.json"
            workload = load_workload(source)
            self.assertEqual(workload.document["operator"], operator)
            self.assertEqual(workload.document["semantics"]["target"], "sm_100a")
            self.assertIn("historical_B200_evidence_does_not_transfer", workload.document["provenance"][0]["scope"])

    def test_cpu_oracles_cover_complete_declared_output_abi(self) -> None:
        for task_name in aka_v3.TASKS:
            workload = WorkloadContract(aka_v3.workload_document(task_name))
            case_id = workload.case_ids[-1]
            inputs = aka_v3.materialize_case(workload, case_id)
            outputs = aka_v3.reference_outputs(workload, case_id, inputs)
            expected = {arg.name for arg in workload.tensor_abi(case_id) if arg.mode == "output"}
            self.assertEqual(set(outputs), expected)
            for argument in workload.tensor_abi(case_id):
                if argument.mode == "output":
                    self.assertEqual(len(outputs[argument.name]), __import__("math").prod(argument.shape))

    def test_gather_boundary_and_repetition_are_public_contract_cases(self) -> None:
        workload = WorkloadContract(aka_v3.workload_document("row_gather"))
        boundary = aka_v3.materialize_case(workload, "boundary_indices")["output_row_to_input_row"]
        repeated = aka_v3.materialize_case(workload, "repeated_indices")["output_row_to_input_row"]
        self.assertEqual(set(boundary), {0, 31})
        self.assertEqual(set(repeated), {16})

    def test_gather_does_not_wrap_a_negative_or_oversized_index(self) -> None:
        workload = WorkloadContract(aka_v3.workload_document("row_gather"))
        inputs = aka_v3.materialize_case(workload, "primary")
        inputs["output_row_to_input_row"][0] = -1
        with self.assertRaisesRegex(ValueError, "within the declared source rows"):
            aka_v3.reference_outputs(workload, "primary", inputs)
        inputs["output_row_to_input_row"][0] = 32
        with self.assertRaisesRegex(ValueError, "within the declared source rows"):
            aka_v3.reference_outputs(workload, "primary", inputs)

    def test_histogram_endpoints_and_out_of_range_values_have_distinct_contract_effects(self) -> None:
        workload = WorkloadContract(aka_v3.workload_document("histogram"))
        lower = aka_v3.reference_outputs(workload, "lower_edge", aka_v3.materialize_case(workload, "lower_edge"))["counts"]
        upper = aka_v3.reference_outputs(workload, "upper_edge", aka_v3.materialize_case(workload, "upper_edge"))["counts"]
        outside = aka_v3.reference_outputs(workload, "out_of_range", aka_v3.materialize_case(workload, "out_of_range"))["counts"]
        self.assertEqual(lower[0], 1024.0)
        self.assertEqual(upper[-1], 1024.0)
        self.assertEqual(sum(outside), 0.0)

    def test_pooling_excludes_padding_and_momentum_exposes_both_state_paths(self) -> None:
        pool = WorkloadContract(aka_v3.workload_document("max_pool1d"))
        inputs = aka_v3.materialize_case(pool, "negative_only")
        self.assertTrue(all(value < 0.0 for value in aka_v3.reference_outputs(pool, "negative_only", inputs)["output"]))
        momentum = WorkloadContract(aka_v3.workload_document("momentum_sgd"))
        state = aka_v3.materialize_case(momentum, "primary")
        off = aka_v3.reference_outputs(momentum, "primary", state)
        state["nesterov"] = [1]
        on = aka_v3.reference_outputs(momentum, "primary", state)
        self.assertNotEqual(off["param_out"], on["param_out"])
        self.assertNotEqual(off["moment_out"], [])

    def test_wrong_target_or_historical_provenance_is_rejected(self) -> None:
        document = aka_v3.workload_document("gemm_nt_bias")
        document["semantics"]["target"] = "sm_103a"
        with self.assertRaisesRegex(ValueError, "identity, target"):
            aka_v3.validate_aka_v3_contract(document)
        document = aka_v3.workload_document("gemm_nt_bias")
        document["provenance"][0]["scope"] = "current B200 qualification"
        with self.assertRaisesRegex(ValueError, "frozen semantic"):
            aka_v3.validate_aka_v3_contract(document)
