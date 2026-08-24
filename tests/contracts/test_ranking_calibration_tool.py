from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.calibrate_ranking_at_scale import (  # noqa: E402
    DEFAULT_TILES,
    EXTENT_SYMBOLS,
    VARIANTS,
    main,
)
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
        self.assertEqual(set(VARIANTS), set(EXTENT_SYMBOLS))
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

    def test_existing_output_is_refused_before_importing_gpu_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "measurement.json"
            output.write_text("historical\n", encoding="utf-8")
            with patch.object(
                sys,
                "argv",
                [
                    "calibrate_ranking_at_scale.py",
                    "--observed-at",
                    "2026-08-24T12:00:00Z",
                    "--out",
                    str(output),
                ],
            ):
                with self.assertRaisesRegex(SystemExit, "already exists"):
                    main()
            self.assertEqual(output.read_text(encoding="utf-8"), "historical\n")

    def test_retained_declared_domains_are_complete_and_reproducible(self) -> None:
        prose = (ROOT / "docs/ANALYSIS_CALIBRATION.md").read_text(encoding="utf-8")
        for name in (
            "gemm-b200-ranking-m512-v5.json",
            "flash-kmeans-b200-ranking-n512-v5.json",
        ):
            with self.subTest(name=name):
                record = json.loads(
                    (ROOT / "evidence/calibration" / name).read_text(encoding="utf-8")
                )
                self.assertEqual(record["schema_version"], 3)
                self.assertEqual(
                    len(record["rows"]) + len(record["excluded"]),
                    record["evaluation_domain"]["candidate_count"],
                )
                self.assertFalse(
                    any(
                        item["disposition"] == "incorrect"
                        for item in record["excluded"]
                    )
                )
                self.assertEqual(
                    {item["role"] for item in record["evaluation_sources"]},
                    {"driver", "oracle"},
                )
                for source in record["evaluation_sources"]:
                    self.assertEqual(
                        hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(),
                        source["raw_sha256"],
                    )
                repetitions = record["protocol"]["timing_samples_per_candidate"]
                for row in record["rows"]:
                    self.assertEqual(len(row["timing_samples_ms"]), repetitions)
                    self.assertTrue(
                        math.isclose(
                            statistics.median(row["timing_samples_ms"]),
                            row["median_ms"],
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    )

                revision_id = record["compiler_revision"]["revision_id"]
                release_name = revision_id.rsplit("-", 1)[-1]
                revision_path = ROOT / f"compiler/releases/{release_name}/revision.lock.json"
                if not revision_path.exists():
                    revision_path = ROOT / "compiler/revision.lock.json"
                revision = json.loads(revision_path.read_text(encoding="utf-8"))
                canonical = json.dumps(
                    revision,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
                self.assertEqual(revision["revision_id"], revision_id)
                self.assertEqual(
                    hashlib.sha256(canonical).hexdigest(),
                    record["compiler_revision"]["revision_sha256"],
                )

                ranked = sorted(
                    (
                        row
                        for row in record["rows"]
                        if row["ranked_by_hypothesis"]
                    ),
                    key=lambda row: (
                        -row["ctas"]
                        / (
                            row["ctas_per_multiprocessor_upper_bound"]
                            * row["multiprocessor_count"]
                        ),
                        -row["ctas_per_multiprocessor_upper_bound"],
                        row["schedule_id"],
                    ),
                )
                best = min(row["median_ms"] for row in record["rows"])
                top_one_regret = (ranked[0]["median_ms"] / best - 1) * 100
                self.assertIn(f"{top_one_regret:.2f}%", prose)

if __name__ == "__main__":
    unittest.main()
