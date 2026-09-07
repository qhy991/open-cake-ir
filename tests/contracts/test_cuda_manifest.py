from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.tasks.flash_kmeans.cuda_manifest import CudaLaunchManifest


class CudaManifestFeedbackTest(unittest.TestCase):
    def test_limit_feedback_names_the_observed_value(self) -> None:
        document = {
            "schema_version": 1,
            "abi": "flash_kmeans_assign_v1",
            "target": "sm_100a",
            "kernel_name": "too_much_shared_memory",
            "grid": [1, 1, 1],
            "block": [32, 1, 1],
            "dynamic_shared_memory_bytes": 262_144,
        }

        with self.assertRaisesRegex(
            ValueError,
            r"dynamic_shared_memory_bytes must be an integer in \[0, 232448\]; got 262144",
        ):
            CudaLaunchManifest.from_dict(document)


if __name__ == "__main__":
    unittest.main()
