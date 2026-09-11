from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.deepseek_v4 import workload as dsv4
from open_cake_ir.tasks.workloads import load_workload


class DeepSeekV4WorkloadTests(unittest.TestCase):
    def test_official_source_bound_contracts_load_at_their_exact_b200_target(self) -> None:
        expected = {
            "deepseek-v4-csa-indexer-topk-fp32-triton-b200-v1": "deepseek_v4_csa_indexer_topk_fp32",
            "deepseek-v4-moe-gate-fp32-triton-b200-v1": "deepseek_v4_moe_gate_fp32",
        }
        for workload_id, operator in expected.items():
            workload = load_workload(ROOT / "contracts/workloads" / f"{workload_id}.json")
            self.assertEqual(workload.document["operator"], operator)
            self.assertEqual(workload.document["semantics"]["target"], "sm_100a")
            self.assertEqual(workload.document["provenance"][0]["revision"], dsv4.MODEL_REVISION)

    def test_csa_steady_state_returns_exactly_1024_unique_indices_per_query(self) -> None:
        workload = WorkloadContract(dsv4.workload_document("csa_indexer_topk"))
        outputs = dsv4.reference_outputs(workload, "boundary", dsv4.materialize_case(workload, "boundary"))
        shape = workload.case("boundary")["shape"]
        self.assertEqual(len(outputs["indices"]), shape["Q"] * shape["K"])
        for row in range(shape["Q"]):
            indices = outputs["indices"][row * shape["K"]:(row + 1) * shape["K"]]
            self.assertEqual(len(set(indices)), shape["K"])
            self.assertTrue(all(0 <= index < shape["S"] for index in indices))

    def test_csa_refuses_score_ties_in_its_deterministic_standalone_boundary(self) -> None:
        workload = WorkloadContract(dsv4.workload_document("csa_indexer_topk"))
        inputs = dsv4.materialize_case(workload, "primary")
        inputs["index_scores"][0] = inputs["index_scores"][1]
        with self.assertRaisesRegex(ValueError, "ambiguous score ties"):
            dsv4.reference_outputs(workload, "primary", inputs)

    def test_moe_gate_preserves_six_unique_ids_and_route_weight_mass(self) -> None:
        workload = WorkloadContract(dsv4.workload_document("moe_gate"))
        inputs = dsv4.materialize_case(workload, "primary")
        outputs = dsv4.reference_outputs(workload, "primary", inputs)
        shape = workload.case("primary")["shape"]
        for row in range(shape["T"]):
            begin = row * shape["K"]
            self.assertEqual(len(set(outputs["expert_ids"][begin:begin + shape["K"]])), shape["K"])
            self.assertTrue(math.isclose(sum(outputs["weights"][begin:begin + shape["K"]]), 2.5, rel_tol=2e-6, abs_tol=2e-6))

    def test_moe_selection_bias_changes_ids_but_not_the_weight_formula_source(self) -> None:
        workload = WorkloadContract(dsv4.workload_document("moe_gate"))
        base = dsv4.materialize_case(workload, "primary")
        baseline = dsv4.reference_outputs(workload, "primary", base)
        changed = {name: list(values) for name, values in base.items()}
        changed["selection_bias"] = dsv4.materialize_case(workload, "bias_changes_selection")["selection_bias"]
        biased = dsv4.reference_outputs(workload, "primary", changed)
        self.assertNotEqual(baseline["expert_ids"], biased["expert_ids"])
        self.assertTrue(all(weight > 0.0 for weight in biased["weights"]))
