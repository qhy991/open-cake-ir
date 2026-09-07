"""Frozen AMD operator Workloads remain separate from Compiler routes."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from open_cake_ir.evaluation import WorkloadContract


ROOT = Path(__file__).resolve().parents[2]
SWIGLU = ROOT / "contracts/workloads/swiglu-fp32-v1.json"
RMSNORM_V1 = ROOT / "contracts/workloads/llama-rmsnorm-mul-fp32-v1.json"
RMSNORM_V2 = ROOT / "contracts/workloads/llama-rmsnorm-mul-fp32-v2.json"


class AmdWorkloadTests(unittest.TestCase):
    def test_swiglu_owns_two_correctness_distributions(self) -> None:
        workload = WorkloadContract.load(SWIGLU)

        self.assertEqual(workload.workload_id, "swiglu-fp32-independent-v1")
        self.assertEqual(
            workload.case_ids,
            ("seeded_random", "signed_saturation"),
        )

    def test_llama_v1_preserves_the_historical_provenance(self) -> None:
        workload = WorkloadContract.load(RMSNORM_V1)

        provenance = json.loads(RMSNORM_V1.read_text(encoding="utf-8"))["provenance"]
        self.assertEqual(
            provenance[0]["revision"],
            "f280b26983ad0fdb705a0d9ebf0503e76f2899b0",
        )

    def test_llama_v2_pins_latest_without_claiming_an_amd_kernel_delta(self) -> None:
        workload = WorkloadContract.load(RMSNORM_V2)
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



if __name__ == "__main__":
    unittest.main()
