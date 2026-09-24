"""Per-rank c/K/steal controls cover skew and uneven token chunks."""
from __future__ import annotations

from dataclasses import replace
import unittest

from experiments.weave.dispatch_ledger import DispatchLedger, dispatch_ledger
from experiments.weave.rank_plan import Chunk, rank_plans


class WeaveRankPlan(unittest.TestCase):
    def test_spatial_split_and_temporal_thresholds_follow_routed_work(self):
        shape = {'R': 4, 'T': 8, 'E': 8, 'K': 2}
        ids = [[[0, 1] for _ in range(8)] for _ in range(4)]
        ledger = dispatch_ledger(shape, ids)
        plans = rank_plans(shape, ledger, sm_count=148,
                           communication_ctas=(12, 36, 120, 147),
                           chunks=(2, 4, 1, 8),
                           steal_budgets=(24, 0, 0, 0))
        self.assertEqual([(plan.communication_ctas, plan.compute_ctas)
                          for plan in plans], [(12, 136), (36, 112), (120, 28), (147, 1)])
        self.assertEqual(plans[0].chunks, (Chunk(0, 4, 8), Chunk(4, 8, 8)))
        self.assertEqual(plans[1].chunks,
                         (Chunk(0, 2, 4), Chunk(2, 4, 4),
                          Chunk(4, 6, 4), Chunk(6, 8, 4)))
        self.assertEqual((plans[0].inbound_payload_slots,
                          plans[0].compute_task_slots), (24, 64))
        self.assertEqual(sum(chunk.expected_contributions for chunk in plans[3].chunks), 16)

    def test_tail_t7_k2_needs_unequal_chunk_completion_counts(self):
        shape = {'R': 4, 'T': 7, 'E': 8, 'K': 2}
        ids = [[[0, 1] for _ in range(7)] for _ in range(4)]
        plans = rank_plans(shape, dispatch_ledger(shape, ids), sm_count=148,
                           communication_ctas=(120, 12, 12, 12),
                           chunks=(2, 3, 7, 1),
                           steal_budgets=(20, 0, 0, 0))
        self.assertEqual(plans[0].chunks, (Chunk(0, 4, 8), Chunk(4, 7, 6)))
        self.assertEqual(plans[1].chunks,
                         (Chunk(0, 3, 6), Chunk(3, 5, 4), Chunk(5, 7, 4)))
        self.assertEqual((plans[0].inbound_payload_slots,
                          plans[0].compute_task_slots), (21, 56))

    def test_invalid_controls_and_missing_return_refuse(self):
        shape = {'R': 4, 'T': 8, 'E': 8, 'K': 2}
        ids = [[[0, 1] for _ in range(8)] for _ in range(4)]
        ledger = dispatch_ledger(shape, ids)
        controls = {'sm_count': 148, 'communication_ctas': (12, 12, 12, 12),
                    'chunks': (2, 2, 2, 2), 'steal_budgets': (0, 0, 0, 0)}
        for change, reason in (({'communication_ctas': (0, 12, 12, 12)}, 'nonempty'),
                               ({'communication_ctas': (148, 12, 12, 12)}, 'nonempty'),
                               ({'chunks': (9, 2, 2, 2)}, 'chunk count'),
                               ({'steal_budgets': (0, 1, 0, 0)}, 'steal budget')):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, reason):
                rank_plans(shape, ledger, **(controls | change))
        bad = [list(rows) for rows in ledger.tasks_by_rank]
        bad[0][0] = replace(bad[0][0], route=bad[0][1].route)
        incomplete = DispatchLedger(ledger.payloads_by_rank,
                                    tuple(tuple(rows) for rows in bad))
        with self.assertRaisesRegex(ValueError, 'one complete return domain'):
            rank_plans(shape, incomplete, **controls)


if __name__ == '__main__':
    unittest.main()
