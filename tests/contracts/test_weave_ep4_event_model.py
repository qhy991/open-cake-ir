"""Bounded EP4 interleavings conserve routes, allow stealing and chunk overlap."""
from __future__ import annotations

from dataclasses import replace
import random
import unittest

from experiments.weave.dispatch_ledger import dispatch_ledger
from experiments.weave.ep4_event_model import simulate_ep4
from experiments.weave.rank_plan import rank_plans


class WeaveEP4EventModel(unittest.TestCase):
    def test_skewed_c_k_steal_plans_complete_once_and_allow_early_combine(self):
        shape = {'R': 4, 'T': 8, 'E': 8, 'K': 2}
        ids = [[[0, 1] for _ in range(8)] for _ in range(4)]
        ledger = dispatch_ledger(shape, ids)
        plans = rank_plans(shape, ledger, sm_count=148,
                           communication_ctas=(147, 12, 12, 12),
                           chunks=(2, 2, 2, 2),
                           steal_budgets=(64, 0, 0, 0))
        traces = [simulate_ep4(shape, ledger, plans, seed=seed)
                  for seed in range(20)]
        self.assertTrue(all(trace.completed for trace in traces))
        self.assertTrue(any(trace.stolen_by_rank[0] > 0 for trace in traces))
        self.assertTrue(any(trace.first_combine_compute_done[0] < 64
                            for trace in traces))
        for trace in traces:
            with self.subTest(stolen=trace.stolen_by_rank):
                self.assertEqual(len(trace.events), 160)
                self.assertLessEqual(trace.stolen_by_rank[0], 64)
                self.assertEqual(trace.computed_tasks, 64)
                self.assertEqual(trace.combined_tokens, 32)
                compute = [event for event in trace.events
                           if event.phase in ('compute', 'steal')]
                self.assertEqual({(event.source_rank, event.token, event.route)
                                  for event in compute},
                                 {(source, token, route)
                                  for source in range(4) for token in range(8)
                                  for route in range(2)})
                for rank in range(4):
                    first_steal = next((index for index, event in enumerate(trace.events)
                                        if event.rank == rank and event.stolen), None)
                    if first_steal is not None:
                        self.assertEqual(sum(event.phase == 'dispatch'
                                             and event.rank == rank
                                             for event in trace.events[:first_steal]), 16)

    def test_t7_k2_tail_completes_with_nonuniform_thresholds(self):
        shape = {'R': 4, 'T': 7, 'E': 8, 'K': 2}
        ids = [[[0, 1] for _ in range(7)] for _ in range(4)]
        ledger = dispatch_ledger(shape, ids)
        plans = rank_plans(shape, ledger, sm_count=148,
                           communication_ctas=(120, 12, 36, 147),
                           chunks=(2, 3, 7, 1),
                           steal_budgets=(56, 0, 0, 0))
        self.assertEqual([chunk.expected_contributions
                          for chunk in plans[0].chunks], [8, 6])
        for seed in range(12):
            trace = simulate_ep4(shape, ledger, plans, seed=seed)
            self.assertTrue(trace.completed, (seed, trace.blocked_reason))
            self.assertEqual((trace.computed_tasks, trace.combined_tokens), (56, 28))

    def test_mixed_routing_and_malformed_chunk_refusal(self):
        shape = {'R': 4, 'T': 8, 'E': 8, 'K': 2}
        rng = random.Random(281)
        for seed in range(8):
            ids = [[rng.sample(range(8), 2) for _ in range(8)]
                   for _ in range(4)]
            ledger = dispatch_ledger(shape, ids)
            plans = rank_plans(shape, ledger, sm_count=148,
                               communication_ctas=(12, 36, 120, 147),
                               chunks=(1, 2, 4, 8),
                               steal_budgets=ledger.task_slots)
            trace = simulate_ep4(shape, ledger, plans, seed=seed)
            self.assertTrue(trace.completed, (seed, trace.blocked_reason))
        bad_chunk = replace(plans[0].chunks[0], expected_contributions=17)
        bad_plan = replace(plans[0], chunks=(bad_chunk,))
        with self.assertRaisesRegex(ValueError, 'complete bounded rank plan'):
            simulate_ep4(shape, ledger, (bad_plan, *plans[1:]), seed=0)


if __name__ == '__main__':
    unittest.main()
