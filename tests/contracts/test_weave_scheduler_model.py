"""Work-conservation and dependency probes for the Weave scheduling reference."""
from __future__ import annotations

import unittest

from experiments.weave.scheduler_model import (
    Calibration, RoutedVolume, Tile, choose_plan, run_chunks,
)


class WeaveSchedulerModel(unittest.TestCase):
    def setUp(self) -> None:
        self.curves = Calibration(
            bandwidth_by_comm_sms={1: 10, 2: 18, 3: 25, 4: 29},
            flops_by_compute_sms={1: 30, 2: 54, 3: 72, 4: 88},
            chunk_efficiency={1: 1, 2: .98, 4: .85},
            overlap_ratio=.5,
        )

    def test_joint_search_changes_with_routed_compute_intensity(self) -> None:
        communication_heavy = choose_plan(
            5, RoutedVolume(10, 100, 100, 32, 1), self.curves, tile_flops=4096)
        compute_heavy = choose_plan(
            5, RoutedVolume(10, 100, 100, 32, 64), self.curves, tile_flops=4096)
        self.assertEqual((communication_heavy.comm_sms, communication_heavy.chunks), (3, 2))
        self.assertEqual((compute_heavy.comm_sms, compute_heavy.chunks), (1, 1))
        for plan in (communication_heavy, compute_heavy):
            self.assertGreaterEqual(plan.predicted_seconds, plan.communication_seconds)
            self.assertGreaterEqual(plan.predicted_seconds,
                                    plan.compute_seconds + plan.exposed_tail_seconds)
            self.assertGreaterEqual(plan.steal_tiles_per_sm_estimate, 0)

    def test_calibration_is_a_snapshot_and_missing_split_refuses(self) -> None:
        observed = {1: 10., 2: 18.}
        curves = Calibration(observed, {3: 72., 4: 88.}, {1: 1.}, .25)
        observed[1] = 999.
        self.assertEqual(curves.bandwidth_by_comm_sms[1], 10.)
        with self.assertRaises(ValueError):
            choose_plan(3, RoutedVolume(1, 1, 1, 32, 8), curves, tile_flops=128)
        with self.assertRaises(ValueError):
            RoutedVolume(0, 1, 2, 32, 8)
        with self.assertRaises(ValueError):
            Calibration({1: 0.}, {1: 1.}, {1: 1.}, .25)

    def test_all_tiles_complete_once_under_different_worker_orders(self) -> None:
        chunks = ((3, 2, 2, 2), (2, 3, 2, 2))
        expected = {Tile(chunk, phase, index)
                    for chunk, counts in enumerate(chunks)
                    for phase, count in zip(("dispatch", "gemm0", "gemm1", "combine"),
                                            counts, strict=True)
                    for index in range(count)}
        for seed in range(20):
            events = run_chunks(6, 2, chunks, max_stolen=3, seed=seed)
            self.assertEqual(len(events), len(expected))
            self.assertEqual({event.tile for event in events}, expected)
            completed = set()
            for event in events:
                tile = event.tile
                if tile.phase != "dispatch":
                    predecessor = {"gemm0": "dispatch", "gemm1": "gemm0",
                                   "combine": "gemm1"}[tile.phase]
                    count = chunks[tile.chunk][("dispatch", "gemm0", "gemm1", "combine").index(predecessor)]
                    self.assertTrue(all(Tile(tile.chunk, predecessor, i) in completed
                                        for i in range(count)))
                if event.stolen:
                    self.assertEqual(event.worker, "comm")
                    self.assertIn(tile.phase, {"gemm0", "gemm1"})
                    self.assertTrue(all(done.phase != "dispatch" for done in expected - completed))
                    self.assertFalse(any(done.phase == "combine" for done in completed))
                completed.add(tile)
        chosen = run_chunks(6, 2, chunks, max_stolen=3, seed=1)
        self.assertTrue(any(event.stolen for event in chosen))
        self.assertTrue(any(event.worker == "compute" and event.tile.phase == "combine"
                            for event in chosen))

    def test_zero_remote_work_and_bad_worker_split(self) -> None:
        events = run_chunks(4, 1, ((0, 2, 1, 0), (0, 1, 1, 0)),
                            max_stolen=2, seed=8)
        self.assertEqual(len(events), 5)
        self.assertTrue(all(event.tile.phase in {"gemm0", "gemm1"} for event in events))
        with self.assertRaises(ValueError):
            run_chunks(4, 4, ((1, 1, 1, 1),), max_stolen=1, seed=0)
        with self.assertRaises(ValueError):
            run_chunks(4, 1, ((1, -1, 1, 1),), max_stolen=1, seed=0)


if __name__ == "__main__":
    unittest.main()
