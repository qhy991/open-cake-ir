"""Bounded diagnostics and their retained rejection relation, without a build."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

from open_cake_ir.lab.diagnoses import (
    findings_feedback, validate_findings_feedback, rejected_candidate_feedback, rejected_peer_feedback,
)
from open_cake_ir.lab.replay.refusals import ReplayRefusal
from open_cake_ir.lab.replay.selection import _replay_candidate_selection


class FeedbackDiagnosticsTests(unittest.TestCase):
    def test_history_and_peer_rejections_share_one_field_projection(self):
        feedback = {'stage': 'assessment', 'findings': [
            {'code': 'LOCAL_CONTRACT', 'path': 'operations[1]', 'severity': 'blocking',
             'blocks_lowering': True, 'message': 'x' * 513, 'private_source': 'hidden'},
            {'code': 'ADVISORY', 'blocks_lowering': False, 'message': 'not a blocking finding'},
        ]}
        row = rejected_candidate_feedback('a' * 64, feedback, arm='open_cake')
        peers = rejected_peer_feedback([(SimpleNamespace(sha256='a' * 64),
            SimpleNamespace(disposition='rejected', feedback=feedback))], arm='open_cake')
        self.assertEqual(peers, [row])
        self.assertEqual(row['findings'], findings_feedback(feedback['findings'], blocking_only=True)['findings'])
        self.assertTrue(row['text_truncated'])
        self.assertEqual(len(row['findings']), 1)
        self.assertNotIn('private_source', row['findings'][0])

    def test_projection_preserves_field_types_and_reports_omissions(self):
        finding = {"code": "LOCAL_CONTRACT", "path": "operations[1]",
                   "category": "hardware_conformance", "severity": "error",
                   "message": "x" * 513, "blocks_acceptance": False,
                   "blocks_lowering": True,
                   "source_location": {"line": 2, "column": 3},
                   "raw_source": "unexposed", "artifacts": ["unexposed"]}
        projected = findings_feedback([finding] * 9)
        validate_findings_feedback(projected)
        self.assertEqual(len(projected["findings"]), 8)
        self.assertEqual(projected["omitted_findings"], 1)
        self.assertIs(projected["text_truncated"], True)
        row = projected["findings"][0]
        self.assertEqual(row["message"], "x" * 512)
        self.assertIs(row["blocks_acceptance"], False)
        self.assertIs(row["blocks_lowering"], True)
        self.assertEqual(row["source_location"], {"line": 2, "column": 3})
        self.assertNotIn("raw_source", row)
        self.assertNotIn("artifacts", row)

    def test_retained_findings_refuse_wrong_field_types_and_unknown_payloads(self):
        malformed = [
            {"code": True}, {"message": False}, {"path": 1},
            {"blocks_acceptance": "false"}, {"blocks_lowering": "true"},
            {"blocks_lowering": 1}, {"raw_source": "unexposed"},
            {"source_location": {"line": True}},
        ]
        for finding in malformed:
            with self.subTest(finding=finding), self.assertRaises(ValueError):
                validate_findings_feedback({"findings": [finding],
                    "omitted_findings": 0, "text_truncated": False})
        self.assertEqual(findings_feedback(malformed)["findings"], [{}] * len(malformed))

    @staticmethod
    def rejected_replay(diagnostics):
        identity = "a" * 64
        original = {"code": "LOCAL_CONTRACT", "path": "operations[1]",
                    "blocks_acceptance": False, "blocks_lowering": True}
        events = [
            {"kind": "candidate_set_filtered", "payload": {
                "turn": 1, "submitted": 1, "launchable": 0,
                "order": [{"candidate_sha256": identity, "disposition": "rejected",
                           "cost": None, "semantic_sha256": None,
                           "diagnostics": diagnostics}]}},
            {"kind": "candidate_selected", "payload": {
                "turn": 1, "candidate_sha256": identity,
                "qualified_search_candidates": [], "reason": "all_candidates_rejected"}},
        ]
        return _replay_candidate_selection(
            candidate_set_turns={1}, cumulative_by_turn={1: 10},
            empirical_selection=None, events=events, fault_turn=None, launchables={},
            lock=SimpleNamespace(document={"evaluation_protocol": {"searches_per_turn": 1}}),
            provider_candidate_bytes={}, provider_candidates_by_turn={1: (identity,)},
            receipt_order=(), receipts={},
            rejected={(1, identity): {"feedback": {"stage": "assessment", "findings": [original]}}},
        )

    def test_rejected_filter_diagnostics_must_project_original_rejection(self):
        original = {"code": "LOCAL_CONTRACT", "path": "operations[1]",
                    "blocks_acceptance": False, "blocks_lowering": True}
        diagnostics = findings_feedback([original])
        observations, _, _ = self.rejected_replay(diagnostics)
        self.assertEqual(len(observations), 1)
        self.assertFalse(observations[0].search_qualified)
        for change in ("finding", "omitted_findings", "text_truncated"):
            forged = copy.deepcopy(diagnostics)
            if change == "finding":
                forged["findings"][0]["code"] = "DIFFERENT_CONTRACT"
            elif change == "omitted_findings":
                forged["omitted_findings"] = 1
            else:
                forged["text_truncated"] = True
            # Each forgery is structurally valid: the rejection relation refuses it.
            validate_findings_feedback(forged)
            with self.subTest(change=change), self.assertRaisesRegex(
                    ReplayRefusal, "bounded projection of retained rejection feedback"):
                self.rejected_replay(forged)


if __name__ == "__main__":
    unittest.main()
