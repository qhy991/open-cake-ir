"""The B300 prefill successor has its own exact stateful task boundary."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.tasks.kda_prefill.contract import ROWS, case_lengths, validate_contract
from open_cake_ir.tasks.workloads import load_workload


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "contracts/workloads/cake-kda-prefill-b300-v1.json"
CATALOG = ROOT / "experiments/flashinfer_rewrites/catalog.json"


class KdaPrefillB300Contract(unittest.TestCase):
    def test_six_benchmark_rows_and_two_guardrails_keep_exact_state_effects(self) -> None:
        workload = load_workload(WORKLOAD)
        self.assertEqual(workload.target, "sm_103a")
        self.assertEqual(workload.case_ids, tuple(row[0] for row in ROWS))
        self.assertEqual(case_lengths(workload.document, "h64_fixed8192"), (8192,))
        self.assertEqual(case_lengths(workload.document, "h96_mixed"),
                         (1300, 547, 2048, 963, 271, 3063))
        self.assertEqual(workload.document["tensors"]["final_state"]["alias_of"],
                         "initial_state")
        self.assertEqual(workload.document["tensors"]["cu_seqlens"]["dtype"], "int64")
        self.assertEqual(workload.document["validation"]["case_tiers"]["benchmark"],
                         [row[0] for row in ROWS[2:]])
        original = next(task for task in json.loads(CATALOG.read_text())["tasks"]
                        if task["id"] == "027_cake_kda_prefill")
        self.assertEqual(original["assessment"]["targets"], ["sm_100a"])
        self.assertEqual(original["status"], "blocked")

    def test_wrong_target_state_alias_lengths_and_quality_gate_are_refused(self) -> None:
        source = json.loads(WORKLOAD.read_text())
        mutations = (
            ("target", "target, ABI or state semantics",
             lambda doc: doc["semantics"].__setitem__("target", "sm_100a")),
            ("alias", "tensor final_state",
             lambda doc: doc["tensors"]["final_state"].__setitem__(
                 "alias_of", "out")),
            ("lengths", "sequence lengths differ",
             lambda doc: doc["semantics"]["sequence_lengths"].__setitem__(
                 "h96_mixed", [1300, 547, 2048, 963, 271, 3062])),
            ("abi", "target, ABI or state semantics",
             lambda doc: doc["semantics"]["candidate_abi"].remove("initial_state")),
            ("tolerance", "correctness gates differ",
             lambda doc: doc["validation"].__setitem__("atol", 0.02)),
            ("cache", "measurement boundary differs",
             lambda doc: doc["validation"]["measurement"].__setitem__(
                 "cache", "warm_L2")),
        )
        for label, message, mutate in mutations:
            with self.subTest(label=label):
                changed = deepcopy(source)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, message):
                    validate_contract(changed)


if __name__ == "__main__":
    unittest.main()
