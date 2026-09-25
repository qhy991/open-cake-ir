"""Counterexamples for temporal expert tile ownership and terminal flush."""
import unittest

import numpy as np

from tile_packing_analysis import analyze


class TilePackingAnalysisTest(unittest.TestCase):
    def test_chunk_arrivals_publish_full_tiles_and_flush_each_partial_expert(self):
        ids = [np.array([[0, 1], [0, 2], [3, 1]], dtype=np.int32),
               np.array([[2, 3], [0, 1], [2, 3]], dtype=np.int32)]
        plan = analyze(ids, experts=4, tile_rows=2, source_chunk_tokens=1)
        summary = plan['summary']
        self.assertEqual(summary['routes'], 12)
        self.assertEqual(summary['tile_tasks'], 8)
        self.assertEqual(summary['terminal_partial_tiles'], 4)
        self.assertEqual(summary['owner_routes'], [6, 6])
        self.assertEqual(summary['owner_tile_tasks'], [4, 4])
        self.assertEqual(summary['direct_per_source_chunk_tile_tasks'], 12)
        rows = [tuple(row) for task in plan['tasks'] for row in task['rows']]
        self.assertEqual(set(rows), {(r, t, k) for r in range(2)
                                     for t in range(3) for k in range(2)})
        self.assertEqual(len(rows), len(set(rows)))

    def test_empty_experts_do_not_consume_tasks_and_full_bins_need_no_flush(self):
        ids = [np.zeros((2, 1), dtype=np.int32),
               np.zeros((2, 1), dtype=np.int32)]
        plan = analyze(ids, experts=4, tile_rows=2, source_chunk_tokens=2)
        self.assertEqual(plan['summary']['owner_tile_tasks'], [2, 0])
        self.assertEqual(plan['summary']['terminal_partial_tiles'], 0)
        self.assertEqual([task['tile_index'] for task in plan['tasks']], [0, 1])

    def test_wave_flush_advances_work_and_accounts_for_extra_padding(self):
        ids = [np.zeros((3, 1), dtype=np.int32),
               np.zeros((3, 1), dtype=np.int32)]
        baseline = analyze(ids, experts=4, tile_rows=4, source_chunk_tokens=1)
        early = analyze(ids, experts=4, tile_rows=4,
                        source_chunk_tokens=1, early_flush_min_rows=2)
        self.assertEqual(baseline['summary']['tile_tasks'], 2)
        self.assertEqual(early['summary']['tile_tasks'], 3)
        self.assertEqual(early['summary']['early_partial_tiles'], 2)
        self.assertEqual(early['summary']['terminal_partial_tiles'], 1)
        self.assertEqual(early['summary']['tile_padding_rows'], 6)
        self.assertEqual([task['published_after'] for task in early['tasks']],
                         [['wave_end', 0], ['wave_end', 1], 'all_dispatch_done'])
        self.assertEqual(sum(task['valid_rows'] for task in early['tasks']), 6)

    def test_repeated_expert_within_one_token_is_refused(self):
        ids = [np.array([[0, 0]], dtype=np.int32),
               np.array([[1, 2]], dtype=np.int32)]
        with self.assertRaisesRegex(ValueError, 'repeats an expert'):
            analyze(ids, experts=4, tile_rows=2, source_chunk_tokens=1)

    def test_early_threshold_cannot_exceed_tile_capacity(self):
        ids = [np.array([[0]], dtype=np.int32),
               np.array([[1]], dtype=np.int32)]
        with self.assertRaisesRegex(ValueError, 'tile rows'):
            analyze(ids, experts=4, tile_rows=2, source_chunk_tokens=1,
                    early_flush_min_rows=2)


if __name__ == '__main__':
    unittest.main()
