"""Deduplicated remote payloads remain distinct from expert work claims."""
from __future__ import annotations

import random
import unittest

from experiments.weave.dispatch_ledger import dispatch_ledger
from open_cake_ir.tasks.weave_ep.workload import routed_volume_values


SHAPE = {'R': 4, 'T': 8, 'E': 8, 'K': 2, 'H': 16, 'I': 32}


class WeaveDispatchLedger(unittest.TestCase):
    def test_skew_shares_payload_but_keeps_both_expert_tasks(self):
        ids = [[[0, 1] for _ in range(8)] for _ in range(4)]
        ledger = dispatch_ledger(SHAPE, ids)
        self.assertEqual(ledger.payload_slots, (24, 0, 0, 0))
        self.assertEqual(ledger.task_slots, (64, 0, 0, 0))
        self.assertEqual(ledger.payload_slots[0],
                         routed_volume_values(SHAPE, ids)[0].unique_remote_in_tokens)
        remote = [task for task in ledger.tasks_by_rank[0] if task.source_rank != 0]
        self.assertEqual(len(remote), 48)
        for source in (1, 2, 3):
            for token in range(8):
                pair = [task for task in remote
                        if (task.source_rank, task.token) == (source, token)]
                self.assertEqual({task.route for task in pair}, {0, 1})
                self.assertEqual(len({task.remote_payload_slot for task in pair}), 1)

    def test_local_remote_and_tail_capacity(self):
        local = [[[rank * 2, rank * 2 + 1] for _ in range(8)]
                 for rank in range(4)]
        local_ledger = dispatch_ledger(SHAPE, local)
        self.assertEqual(local_ledger.payload_slots, (0, 0, 0, 0))
        self.assertEqual(local_ledger.task_slots, (16, 16, 16, 16))
        self.assertTrue(all(task.remote_payload_slot is None
                            for rank_tasks in local_ledger.tasks_by_rank
                            for task in rank_tasks))
        remote = [[[(rank + 1) % 4 * 2, (rank + 2) % 4 * 2]
                   for _ in range(8)] for rank in range(4)]
        remote_ledger = dispatch_ledger(SHAPE, remote)
        self.assertEqual(remote_ledger.payload_slots, (16, 16, 16, 16))
        self.assertEqual(remote_ledger.task_slots, (16, 16, 16, 16))
        self.assertTrue(all(task.remote_payload_slot is not None
                            for rank_tasks in remote_ledger.tasks_by_rank
                            for task in rank_tasks))
        tail = dict(SHAPE, T=7)
        skew_tail = [[[0, 1] for _ in range(7)] for _ in range(4)]
        self.assertEqual(dispatch_ledger(tail, skew_tail).payload_slots,
                         (21, 0, 0, 0))
        self.assertEqual(dispatch_ledger(tail, skew_tail).task_slots,
                         (56, 0, 0, 0))

    def test_every_route_and_remote_payload_has_one_owner(self):
        rng = random.Random(1809)
        for _ in range(12):
            ids = [[rng.sample(range(8), 2) for _ in range(8)]
                   for _ in range(4)]
            ledger = dispatch_ledger(SHAPE, ids)
            volumes = routed_volume_values(SHAPE, ids)
            self.assertEqual(ledger.payload_slots,
                             tuple(row.unique_remote_in_tokens for row in volumes))
            self.assertEqual(ledger.task_slots,
                             tuple(row.local_routes + row.remote_in_routes
                                   for row in volumes))
            tasks = [task for rank_tasks in ledger.tasks_by_rank for task in rank_tasks]
            self.assertEqual(len(tasks), 64)
            self.assertEqual({(task.source_rank, task.token, task.route)
                              for task in tasks},
                             {(source, token, route)
                              for source in range(4) for token in range(8)
                              for route in range(2)})
            for destination, rank_tasks in enumerate(ledger.tasks_by_rank):
                payloads = ledger.payloads_by_rank[destination]
                self.assertEqual(len({(payload.source_rank, payload.token)
                                      for payload in payloads}), len(payloads))
                for task in rank_tasks:
                    self.assertEqual(task.destination_rank, destination)
                    self.assertEqual(task.expert // 2, destination)
                    if task.remote_payload_slot is None:
                        self.assertEqual(task.source_rank, destination)
                    else:
                        payload = payloads[task.remote_payload_slot]
                        self.assertEqual((payload.source_rank, payload.token,
                                          payload.destination_rank),
                                         (task.source_rank, task.token, destination))

    def test_bad_routes_and_geometry_refuse_before_reservation(self):
        valid = [[[0, 1] for _ in range(8)] for _ in range(4)]
        for changed in ({'R': 4, 'T': 8, 'E': 7, 'K': 2},
                        {'R': 4, 'T': 8, 'E': 8, 'K': 9}):
            with self.assertRaisesRegex(ValueError, 'geometry'):
                dispatch_ledger(changed, valid)
        duplicate = [[[0, 1] for _ in range(8)] for _ in range(4)]
        duplicate[1][2] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'distinct'):
            dispatch_ledger(SHAPE, duplicate)
        bad_id = [[[0, 1] for _ in range(8)] for _ in range(4)]
        bad_id[2][3] = [-1, 1]
        with self.assertRaisesRegex(ValueError, 'in range'):
            dispatch_ledger(SHAPE, bad_id)


if __name__ == '__main__':
    unittest.main()
