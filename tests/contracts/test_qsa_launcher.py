from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from launch_qsa_seed_experiment import _task  # noqa: E402


class QsaLauncherContractTest(unittest.TestCase):
    def _runtime(self) -> dict[str, object]:
        return {
            "judge": {
                "python": "/runtime/python",
                "nvcc": "/cuda/nvcc",
                "cuobjdump": "/cuda/cuobjdump",
            }
        }

    def test_component_timing_has_a_distinct_task_and_explicit_evaluator_flag(self) -> None:
        task = _task(
            remote_root="/remote/open-cake",
            runtime=self._runtime(),
            executor=SimpleNamespace(executor_id="executor-v1", canonical_sha256="a" * 64),
            protocol="seed",
            component_timing=True,
        )

        self.assertEqual(
            task["task_id"], "open-cake-qsa-prefill-t32768-seed-component-v1"
        )
        for stage in task["stages"]:
            self.assertIn("--component-timing", stage["judge"]["command"])

    def test_component_timing_cannot_masquerade_as_a_profile_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires the seed protocol"):
            _task(
                remote_root="/remote/open-cake",
                runtime=self._runtime(),
                executor=SimpleNamespace(
                    executor_id="executor-v1", canonical_sha256="a" * 64
                ),
                protocol="profile",
                component_timing=True,
            )


if __name__ == "__main__":
    unittest.main()
