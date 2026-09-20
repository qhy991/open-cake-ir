"""The uncalibrated structural scorer cannot reorder a new candidate set."""
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

from open_cake_ir.lab.candidate_filter import _build_filter_candidates


class DefaultCandidateOrderingTests(unittest.TestCase):
    def test_builds_every_member_before_retaining_provider_order_among_survivors(self):
        payloads = (b'first', b'rejected', b'last')
        observed = []
        def build(submission):
            observed.append(submission.payload)
            return SimpleNamespace(disposition='rejected' if submission.payload == b'rejected' else 'launchable',
                                   semantic_sha256=None, empirical_cost=None)
        ledger = Mock()
        def append(kind, row):
            self.assertEqual(tuple(observed), payloads)
            self.assertEqual(kind, 'candidate_set_filtered')
            self.assertEqual((row['submitted'], row['launchable']), (3, 2))
        ledger.append.side_effect = append
        built, order, applied, rows, summary = _build_filter_candidates(
            empirical_enabled=False, environment=SimpleNamespace(media_type='text/x-cuda', build=build),
            ledger=ledger, candidate_payloads=payloads, turn_number=1)
        self.assertEqual([built[i][0].payload for i in order], [b'first', b'last', b'rejected'])
        self.assertFalse(applied)
        self.assertIsNone(summary)
        self.assertTrue(all(row['cost'] is None for row in rows))
        ledger.append.assert_called_once()
