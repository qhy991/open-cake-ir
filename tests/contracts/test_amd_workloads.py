"""Frozen AMD operator Workloads remain separate from Compiler routes."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from open_cake_ir.tasks.workloads import load_workload  # noqa: E402
from open_cake_ir.evaluation import WorkloadContract


ROOT = Path(__file__).resolve().parents[2]
SWIGLU = ROOT / "contracts/workloads/swiglu-fp32-v1.json"
RMSNORM_V1 = ROOT / "contracts/workloads/llama-rmsnorm-mul-fp32-v1.json"
RMSNORM_V2 = ROOT / "contracts/workloads/llama-rmsnorm-mul-fp32-v2.json"


class AmdWorkloadTests(unittest.TestCase):
    def test_swiglu_owns_two_correctness_distributions(self) -> None:
        workload = load_workload(SWIGLU)

        self.assertEqual(workload.workload_id, "swiglu-fp32-independent-v1")
        self.assertEqual(
            workload.case_ids,
            ("seeded_random", "signed_saturation"),
        )

    def test_llama_v1_preserves_the_historical_provenance(self) -> None:
        workload = load_workload(RMSNORM_V1)

        provenance = json.loads(RMSNORM_V1.read_text(encoding="utf-8"))["provenance"]
        self.assertEqual(
            provenance[0]["revision"],
            "f280b26983ad0fdb705a0d9ebf0503e76f2899b0",
        )

    def test_llama_v2_retains_its_pinned_provenance(self) -> None:
        workload = load_workload(RMSNORM_V2)
        document = json.loads(RMSNORM_V2.read_text(encoding="utf-8"))

        self.assertEqual(
            workload.workload_id,
            "llama-rmsnorm-mul-fp32-independent-v2",
        )
        self.assertEqual(
            workload.case_ids,
            ("seeded_random", "reduction_rsqrt_stress"),
        )
        self.assertEqual(
            document["provenance"][0]["revision"],
            "eb25b7263e1604b4382295563f5a924002d6f87c",
        )


    def test_semantics_cases_oracle_and_tolerance_drift_are_rejected(self) -> None:
        mutations = {
            "formula": lambda d: d["semantics"].__setitem__("definition", "x"),
            "oracle": lambda d: d["oracle"].__setitem__("kind", "candidate_output"),
            "tolerance": lambda d: d["validation"].__setitem__("atol", 1.0),
            "case": lambda d: d["cases"][0]["shape"].__setitem__("D", 64),
            "distribution": lambda d: d["cases"][0].__setitem__("mode", "zeros"),
        }
        for source in (SWIGLU, RMSNORM_V1, RMSNORM_V2):
            original = json.loads(source.read_text())
            for label, mutate in mutations.items():
                with self.subTest(source=source.name, drift=label):
                    document = copy.deepcopy(original)
                    mutate(document)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "workload.json"
                        path.write_text(json.dumps(document), encoding="utf-8")
                        with self.assertRaises(ValueError):
                            load_workload(path)

if __name__ == "__main__":
    unittest.main()
