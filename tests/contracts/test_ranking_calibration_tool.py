from __future__ import annotations

import hashlib
import importlib.util
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

# These instruments and their source assertions belong to the frozen v6/v7/v8
# Compiler environment. Retiring both flat modules does not migrate that history.
_REQUIRES_PINNED_GIT = (
    importlib.util.find_spec("open_cake_ir.compiler.analysis") is None
    and importlib.util.find_spec("open_cake_ir.compiler.ranking") is None
)
if not _REQUIRES_PINNED_GIT:
    from tools.calibrate_ranking_at_scale import (  # noqa: E402
        DEFAULT_TILES,
        EXTENT_SYMBOLS,
        VARIANTS,
        main,
    )
    from tools.calibrate_gemm_ranking_interleaved import (  # noqa: E402
        _check as evaluate_interleaved,
    )
    from tools.check_ranking_calibration import evaluate  # noqa: E402
    from tools.kernel_cases import ORACLES, global_shapes  # noqa: E402

from open_cake_ir.compiler import Compiler  # noqa: E402


@unittest.skipIf(_REQUIRES_PINNED_GIT,
                 "requires pinned historical Git for ranking calibration v6/v7/v8")
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
                # Refusal is revision-specific. Frozen calibration records own the
                # historical rejected rows; a later Compiler may admit every default.
                self.assertEqual(
                    lowering_eligible + refused,
                    len(DEFAULT_TILES[profile]),
                )

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

    def test_preregistered_candidate_set_decision_is_historical_after_ir_migration(self) -> None:
        plan = ROOT / "contracts/calibrations/gemm-b200-ranking-m512-v6.json"
        retained = (
            ROOT / "evidence/calibration/gemm-b200-ranking-m512-v6-decision.json"
        )

        with self.assertRaisesRegex(ValueError, "calibration Schedule differs"):
            evaluate(ROOT, plan)
        replayed = json.loads(retained.read_text(encoding="utf-8"))
        self.assertFalse(replayed["decision"]["all_repetitions_passed"])
        self.assertEqual(
            [item["candidate_set_count"] for item in replayed["repetitions"]],
            [2300, 2300],
        )
        self.assertTrue(
            all(
                item["maximum_survivor_regret_ratio"] > 1.05
                for item in replayed["repetitions"]
            )
        )
        revision = json.loads(
            (ROOT / "compiler/revision.lock.json").read_text(encoding="utf-8")
        )
        self.assertEqual(revision["calibration_coverage"], [])

    def test_interleaved_successor_is_retained_but_not_rebound_to_current_ir(self) -> None:
        v7 = ROOT / "contracts/calibrations/gemm-b200-ranking-m512-v7.json"
        with self.assertRaisesRegex(ValueError, "calibration Schedule differs"):
            evaluate_interleaved(v7)

        v8 = ROOT / "contracts/calibrations/gemm-b200-ranking-m512-v8.json"
        retained = (
            ROOT / "evidence/calibration/gemm-b200-ranking-m512-v8-decision.json"
        )
        with self.assertRaisesRegex(ValueError, "calibration Schedule differs"):
            evaluate_interleaved(v8)
        replayed = json.loads(retained.read_text(encoding="utf-8"))
        self.assertFalse(replayed["decision"]["all_repetitions_passed"])
        self.assertEqual(
            [item["decisive_candidate_set_count"] for item in replayed["repetitions"]],
            [1450, 1450],
        )
        self.assertEqual(
            [item["abstained_candidate_set_count"] for item in replayed["repetitions"]],
            [850, 850],
        )
        self.assertGreater(
            replayed["repetitions"][0]["maximum_decisive_survivor_regret_ratio"],
            1.05,
        )
        self.assertLessEqual(
            replayed["repetitions"][1]["maximum_decisive_survivor_regret_ratio"],
            1.05,
        )
        revision = json.loads(
            (ROOT / "compiler/revision.lock.json").read_text(encoding="utf-8")
        )
        self.assertEqual(revision["calibration_coverage"], [])

if __name__ == "__main__":
    unittest.main()
