from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402


class StreamingTopKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.schedule = ROOT / "corpus/schedules/top-k-streaming-b8-smoke.json"

    def _document(self, *, source_tiles_per_merge: int | None = None) -> dict:
        document = json.loads(self.schedule.read_text(encoding="utf-8"))
        if source_tiles_per_merge is not None:
            next(
                operation
                for operation in document["operations"]
                if operation["id"] == "select_blocks"
            )["parameters"]["source_tiles_per_merge"] = source_tiles_per_merge
        return document

    def _lower(self, document: dict) -> str:
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        source = self.compiler.lower(assessment).source
        ast.parse(source)
        return source

    def test_top_k_state_is_initialized_once_and_merged_inside_the_loop(self) -> None:
        assessment = self.compiler.assess_file(self.schedule)

        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        source = self.compiler.lower(assessment).source
        loop = source.index("for score_start in tl.range")
        self.assertLess(source.index('top_values = tl.full((8,), float("-inf")'), loop)
        self.assertLess(source.index("top_indices = tl.full((8,), 2147483647"), loop)
        self.assertGreater(source.index("select_blocks_previous_values = top_values"), loop)
        self.assertIn(
            "select_blocks_source_positions = score_start + tl.arange(0, 8)",
            source,
        )
        self.assertIn(
            "select_blocks_ranked_keys = tl.topk(select_blocks_combined_keys, 8)",
            source,
        )
        self.assertNotIn("select_blocks_source_candidates_", source)
        self.assertNotIn("select_blocks_pending_keys", source)
        self.assertNotIn("# CAKE_FLUSH:select_blocks", source)

    def test_two_tile_cadence_skips_the_first_merge_and_flushes_an_odd_tail(self) -> None:
        source = self._lower(self._document(source_tiles_per_merge=2))
        loop = source.index("for score_start in tl.range")

        self.assertLess(
            source.index("select_blocks_pending_keys = tl.zeros((8,), tl.uint64)"),
            loop,
        )
        self.assertIn(
            "select_blocks_source_valid = "
            "select_blocks_source_positions < N_SCORE_LOOP",
            source,
        )
        self.assertIn(
            "        if ((score_start // BLOCK_SCORE_LOOP) & 1) != 0:\n"
            "            select_blocks_pair_keys = tl.cat("
            "select_blocks_pending_keys, select_blocks_source_keys, can_reorder=False)",
            source,
        )
        self.assertIn(
            "        else:\n"
            "            select_blocks_pending_keys = select_blocks_source_keys",
            source,
        )
        self.assertIn(
            "            select_blocks_pair_ranked_keys = "
            "tl.topk(select_blocks_pair_combined_keys, 8)",
            source,
        )
        flush = source.index("# CAKE_FLUSH:select_blocks")
        self.assertGreater(flush, loop)
        self.assertIn(
            "select_blocks_source_tile_count = "
            "(N_SCORE_LOOP + BLOCK_SCORE_LOOP - 1) // BLOCK_SCORE_LOOP",
            source,
        )
        self.assertIn(
            "    if (select_blocks_source_tile_count & 1) != 0:\n"
            "        select_blocks_flush_zero_keys = tl.zeros((8,), tl.uint64)",
            source,
        )
        self.assertIn(
            "        select_blocks_flush_ranked_keys = "
            "tl.topk(select_blocks_flush_combined_keys, 8)",
            source,
        )
        # There is one textual call under the pair branch and one under the odd-tail
        # branch. There is no unconditional fallback that would execute on first tiles.
        self.assertEqual(source.count("tl.topk("), 2)

    def test_two_tile_static_partial_tail_uses_the_selection_stop(self) -> None:
        document = self._document(source_tiles_per_merge=2)
        next(buffer for buffer in document["buffers"] if buffer["name"] == "scores")[
            "shape"
        ][1] = 30

        source = self._lower(document)

        self.assertIn(
            "select_blocks_source_valid = "
            "select_blocks_source_positions < N_SCORE_LOOP",
            source,
        )
        self.assertIn(
            "select_blocks_source_tile_count = "
            "(N_SCORE_LOOP + BLOCK_SCORE_LOOP - 1) // BLOCK_SCORE_LOOP",
            source,
        )

    def test_two_tile_dynamic_stop_owns_both_validity_and_odd_flush(self) -> None:
        document = self._document(source_tiles_per_merge=2)
        document["tile_loops"][0]["stop"] = {
            "program": "batch",
            "add": 1,
            "floor_div": 1,
        }

        source = self._lower(document)
        stop = "tl.minimum(tl.maximum((batch + 1), 0), N_SCORE_LOOP)"

        self.assertIn(
            f"select_blocks_source_valid = select_blocks_source_positions < {stop}",
            source,
        )
        self.assertIn(
            f"select_blocks_source_tile_count = "
            f"({stop} + BLOCK_SCORE_LOOP - 1) // BLOCK_SCORE_LOOP",
            source,
        )

    @staticmethod
    def _ordered_prefix(
        candidates: list[tuple[float, int]], k: int
    ) -> list[tuple[float, int]]:
        return sorted(candidates, key=lambda item: (-item[0], item[1]))[:k]

    @classmethod
    def _paired_reference(
        cls, values: list[float], *, stop: int, tile: int, k: int
    ) -> tuple[list[float], list[int]]:
        state: list[tuple[float, int]] = []
        pending: list[tuple[float, int]] = []
        trip_count = (stop + tile - 1) // tile
        for trip, start in enumerate(range(0, stop, tile)):
            current = [
                (values[position], position)
                for position in range(start, min(start + tile, stop))
            ]
            if trip & 1:
                state = cls._ordered_prefix(state + pending + current, k)
            else:
                pending = current
        if trip_count & 1:
            state = cls._ordered_prefix(state + pending, k)
        padded = state + [(float("-inf"), -1)] * (k - len(state))
        return [item[0] for item in padded], [item[1] for item in padded]

    def test_two_tile_reference_matches_direct_order_for_every_tail_kind(self) -> None:
        values = [
            float("-inf"),
            3.0,
            3.0,
            -0.0,
            0.0,
            8.0,
            -4.0,
            8.0,
            1.0,
            float("-inf"),
            6.0,
            6.0,
            -2.0,
        ]
        tile, k = 4, 8

        for stop in (0, 1, 4, 5, 8, 9, len(values)):
            with self.subTest(stop=stop):
                observed_values, observed_indices = self._paired_reference(
                    values, stop=stop, tile=tile, k=k
                )
                expected = self._ordered_prefix(
                    [(values[index], index) for index in range(stop)], k
                )
                expected += [(float("-inf"), -1)] * (k - len(expected))
                self.assertEqual(observed_indices, [item[1] for item in expected])
                self.assertEqual(observed_values, [item[0] for item in expected])

    def test_across_loop_without_a_loop_fails_locally(self) -> None:
        assessment = self.compiler.assess_file(
            ROOT / "corpus/schedules/top-k-b8-smoke-across-loop-drift.json"
        )

        self.assertFalse(assessment.accepted)
        self.assertIn(
            "TOP_K_LOOP_REQUIRED",
            [finding.code for finding in assessment.findings],
        )

    def test_score_state_shape_must_equal_the_loop_tile(self) -> None:
        document = json.loads(self.schedule.read_text(encoding="utf-8"))
        next(buffer for buffer in document["buffers"] if buffer["name"] == "score_tile")[
            "shape"
        ] = [4]

        assessment = self.compiler.assess(document)

        self.assertFalse(assessment.accepted)
        self.assertIn(
            "TOP_K_LOOP_TILE_MISMATCH",
            [finding.code for finding in assessment.findings],
        )


if __name__ == "__main__":
    unittest.main()
