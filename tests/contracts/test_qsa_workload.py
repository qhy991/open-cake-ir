from __future__ import annotations

import json
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.tasks.workloads import load_workload

from open_cake_ir.evaluation import WorkloadContract  # noqa: E402


class QsaWorkloadContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ROOT / "contracts/workloads/qsa-prefill-t32768-v1.json"

    def test_qsa_contract_freezes_target_and_non_checkpoint_boundary(self) -> None:
        workload = load_workload(self.path)

        self.assertEqual(
            workload.case_ids,
            ("conformance_t4096", "target_t32768"),
        )
        self.assertEqual(workload.case("target_t32768")["shape"], {"B": 1, "T": 32768})
        semantics = workload.document["semantics"]
        self.assertEqual(
            semantics["relu_reduction_order"],
            "relu_before_sum_over_index_heads",
        )
        self.assertFalse(semantics["checkpoint_configuration_claimed"])

    def test_qsa_oracle_source_is_the_bound_external_implementation(self) -> None:
        workload = load_workload(self.path)
        source = ROOT / "src/open_cake_ir/tasks/qsa/evaluation.py"

        self.assertEqual(
            sha256(source.read_bytes()).hexdigest(),
            workload.document["oracle"]["source_sha256"],
        )

    def test_qsa_relu_order_and_geometry_fail_closed(self) -> None:
        original = json.loads(self.path.read_text(encoding="utf-8"))
        mutations = (
            lambda document: document["semantics"].update(
                {"relu_reduction_order": "sum_heads_then_relu"}
            ),
            lambda document: document["semantics"].update(
                {"checkpoint_configuration_claimed": True}
            ),
            lambda document: document["semantics"]["task_declared_geometry"].update(
                {"token_budget": 4096}
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                document = json.loads(json.dumps(original))
                mutate(document)
                path = Path(directory) / "workload.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "QSA workload semantics differ"):
                    load_workload(path)


if __name__ == "__main__":
    unittest.main()
