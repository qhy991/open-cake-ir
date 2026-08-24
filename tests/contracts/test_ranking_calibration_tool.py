from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.calibrate_ranking_at_scale import DEFAULT_TILES, VARIANTS  # noqa: E402
from tools.kernel_cases import ORACLES, global_shapes  # noqa: E402

from open_cake_ir.compiler import Compiler  # noqa: E402


class RankingCalibrationInstrumentTests(unittest.TestCase):
    _SCHEDULES = {
        "flash_kmeans_b32_smoke": "corpus/schedules/flash-kmeans-b32-smoke-v2.json",
        "rmsnorm_b8_smoke": "corpus/schedules/rmsnorm-b8-smoke.json",
        "gemm_bias_b1_smoke": "corpus/schedules/gemm-bias-b1-smoke.json",
    }

    def setUp(self) -> None:
        self.compiler = Compiler.load(ROOT, "compiler/revision.lock.json")

    def test_every_sweep_has_one_oracle_and_one_tile_domain(self) -> None:
        self.assertEqual(set(VARIANTS), set(DEFAULT_TILES))
        self.assertLessEqual(set(VARIANTS), set(ORACLES))
        self.assertEqual(set(VARIANTS), set(self._SCHEDULES))

    def test_default_sweeps_preserve_one_workload_and_reach_lowering(self) -> None:
        for profile, relative in self._SCHEDULES.items():
            with self.subTest(profile=profile):
                base = json.loads((ROOT / relative).read_text(encoding="utf-8"))
                global_views = []
                lowering_eligible = 0
                refused = 0
                for tile in DEFAULT_TILES[profile]:
                    document = VARIANTS[profile](base, 512, tile, 128, 4)
                    global_views.append(global_shapes(document))
                    assessment = self.compiler.assess(document)
                    if assessment.lowering_eligible:
                        self.compiler.lower(assessment)
                        lowering_eligible += 1
                    else:
                        refused += 1

                self.assertTrue(all(view == global_views[0] for view in global_views))
                self.assertGreater(lowering_eligible, 0)
                self.assertGreater(refused, 0)


if __name__ == "__main__":
    unittest.main()
