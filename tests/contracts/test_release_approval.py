from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import CompilerError  # noqa: E402
from open_cake_ir.compiler.release import build_gate_report, build_release, verify_release  # noqa: E402


class ReleaseApprovalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="cake-approval-contract-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.record_temporary = tempfile.TemporaryDirectory(prefix="cake-review-record-fixture-")
        cls.addClassCleanup(cls.record_temporary.cleanup)
        cls.record_path = Path(cls.record_temporary.name).resolve() / "review.json"
        cls.project = Path(cls.temporary.name).resolve()
        sources = json.loads((ROOT / "compiler/source_set.json").read_text())["paths"]
        for relative in set(sources) | {
            "compiler/revision.json", "compiler/source_set.json", "corpus/manifest.json"
        }:
            destination = cls.project / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        gate = build_gate_report(
            cls.project,
            cls.project / "compiler/revision.json",
            cls.project / "compiler/source_set.json",
        )
        (cls.project / "compiler/corpus-gate-report.json").write_text(json.dumps(gate.document))
        cls.approval = {
            "schema_version": 3,
            "decision": "approved",
            "gate_report": {
                "path": "compiler/corpus-gate-report.json",
                "canonical_sha256": gate.canonical_sha256,
            },
            "reviewer": {
                "kind": "agent_session",
                "model": "gpt-6-astra",
                "session_id": "fixture-review-session",
                "author_session_id": "fixture-author-session",
            },
            "review_record": str(cls.record_path),
            "approval_basis": f"Synthetic independent review fixture: {cls.record_path}",
        }

    def build(self, approval: dict[str, object], *, write_record: bool = True):
        if write_record:
            self.record_path.write_text(json.dumps({
                "schema_version": 1, "decision": approval["decision"],
                "gate_report": approval["gate_report"], "reviewer": approval["reviewer"],
                "reviewed_commit": "c" * 40,
                "launcher_record": "synthetic CPU test fixture; no model execution",
                "review_basis": approval["approval_basis"],
            }))
        path = self.project / "compiler/release-approval.json"
        path.write_text(json.dumps(approval))
        return build_release(
            self.project,
            self.project / "compiler/revision.json",
            self.project / "compiler/source_set.json",
            self.project / "compiler/corpus-gate-report.json",
            path,
        )

    def test_each_owner_allowed_model_can_approve_an_independent_session(self) -> None:
        for model in ("opus-5", "fable-5", "fable-5.1", "gpt-6-astra"):
            with self.subTest(model=model):
                approval = copy.deepcopy(self.approval)
                approval["reviewer"]["model"] = model
                release = self.build(approval)
                self.assertEqual(release.document["state"], "released")
                self.assertTrue(release.verify(self.project))

    def test_human_review_remains_available(self) -> None:
        approval = copy.deepcopy(self.approval)
        approval["reviewer"] = {"kind": "human", "name": "Fixture human reviewer"}
        approval.pop("review_record")
        self.assertEqual(self.build(approval).document["state"], "released")

    def test_new_construction_requires_exact_record_but_retained_verification_does_not(self) -> None:
        approval = copy.deepcopy(self.approval)
        release = self.build(approval)
        released = self.project / "compiler/retained-test-release.json"
        released.write_text(json.dumps(release.document, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False) + "\n")
        self.record_path.unlink()
        self.assertTrue(verify_release(self.project, self.project / "compiler/revision.json",
            self.project / "compiler/source_set.json", self.project / "compiler/corpus-gate-report.json",
            self.project / "compiler/release-approval.json", released))
        with self.assertRaisesRegex(CompilerError, "review_record is unavailable"):
            self.build(approval, write_record=False)

    def test_missing_stale_or_mismatched_review_record_is_refused(self) -> None:
        for field, replacement in (
            ("decision", "rejected"), ("gate_report", {}), ("reviewer", {}),
            ("reviewed_commit", ""), ("launcher_record", ""), ("review_basis", "stale"),
        ):
            with self.subTest(field=field):
                self.build(copy.deepcopy(self.approval))
                record = json.loads(self.record_path.read_text())
                record[field] = replacement
                self.record_path.write_text(json.dumps(record))
                with self.assertRaisesRegex(CompilerError, "review_record declarations differ"):
                    self.build(copy.deepcopy(self.approval), write_record=False)
        approval = copy.deepcopy(self.approval)
        approval.pop("review_record")
        with self.assertRaisesRegex(CompilerError, "absolute external review_record"):
            self.build(approval)

    def test_final_identity_distinguishes_independent_approvals_of_the_same_draft(self) -> None:
        first = self.build(copy.deepcopy(self.approval))
        other = copy.deepcopy(self.approval)
        other["reviewer"]["session_id"] = "another-independent-fixture-review"
        second = self.build(other)
        self.assertEqual(first.document["corpus_gate"], second.document["corpus_gate"])
        self.assertNotEqual(first.document["revision_id"], second.document["revision_id"])
        self.assertEqual(self.build(other).document["revision_id"], second.document["revision_id"])

    def test_same_reviewer_may_issue_a_new_record_for_a_new_gate(self) -> None:
        first = self.build(copy.deepcopy(self.approval))
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve() / "project"
            shutil.copytree(self.project, project)
            source = project / "compiler/AUTHORING_CONTRACT.md"
            source.write_text(source.read_text() + "\nSuccessor CPU review fixture.\n")
            gate = build_gate_report(project, project / "compiler/revision.json",
                                     project / "compiler/source_set.json")
            (project / "compiler/corpus-gate-report.json").write_text(json.dumps(gate.document))
            record_path = Path(directory).resolve() / "second-review.json"
            approval = copy.deepcopy(self.approval)
            approval["gate_report"]["canonical_sha256"] = gate.canonical_sha256
            approval["review_record"] = str(record_path)
            approval["approval_basis"] = f"Synthetic second exact review: {record_path}"
            record_path.write_text(json.dumps({
                "schema_version": 1, "decision": "approved", "gate_report": approval["gate_report"],
                "reviewer": approval["reviewer"], "reviewed_commit": "d" * 40,
                "launcher_record": "synthetic second review; no model execution",
                "review_basis": approval["approval_basis"],
            }))
            approval_path = project / "compiler/release-approval.json"
            approval_path.write_text(json.dumps(approval))
            second = build_release(project, project / "compiler/revision.json",
                project / "compiler/source_set.json", project / "compiler/corpus-gate-report.json",
                approval_path)
            self.assertNotEqual(first.document["revision_id"], second.document["revision_id"])

    def test_unapproved_models_and_missing_provenance_are_refused(self) -> None:
        for model in ("gpt-5.6-sol", "fable-5.2", "gpt-6-astra-auto", "", None, []):
            with self.subTest(model=model):
                approval = copy.deepcopy(self.approval)
                approval["reviewer"]["model"] = model
                with self.assertRaisesRegex(CompilerError, "model is not allowed"):
                    self.build(approval)
        for field in ("model", "session_id", "author_session_id"):
            with self.subTest(missing=field):
                approval = copy.deepcopy(self.approval)
                del approval["reviewer"][field]
                with self.assertRaisesRegex(CompilerError, "reviewer fields differ"):
                    self.build(approval)

    def test_self_review_and_ambiguous_session_ids_are_refused(self) -> None:
        for session in ("fixture-author-session", "FIXTURE-AUTHOR-SESSION", "", " ", " padded "):
            with self.subTest(session=session):
                approval = copy.deepcopy(self.approval)
                approval["reviewer"]["session_id"] = session
                with self.assertRaises(CompilerError):
                    self.build(approval)

    def test_malformed_or_legacy_reviewer_cannot_bypass_the_policy(self) -> None:
        for reviewer in (
            "unstructured agent reviewer",
            {"kind": "human", "name": " "},
            {"kind": "agent_session", "model": "gpt-6-astra"},
            {"kind": "unknown", "name": "reviewer"},
        ):
            with self.subTest(reviewer=reviewer):
                approval = copy.deepcopy(self.approval)
                approval["reviewer"] = reviewer
                with self.assertRaises(CompilerError):
                    self.build(approval)
        approval = copy.deepcopy(self.approval)
        approval.update(schema_version=1, reviewer="historical fixture reviewer")
        with self.assertRaisesRegex(CompilerError, "approval differs"):
            self.build(approval)

    def test_an_allowed_reviewer_still_requires_an_affirmative_exact_gate_decision(self) -> None:
        for decision in ("request-changes", "rejected"):
            with self.subTest(decision=decision):
                approval = copy.deepcopy(self.approval)
                approval["decision"] = decision
                with self.assertRaisesRegex(CompilerError, "approval differs"):
                    self.build(approval)
        approval = copy.deepcopy(self.approval)
        approval["gate_report"]["canonical_sha256"] = "0" * 64
        with self.assertRaisesRegex(CompilerError, "does not bind this Gate"):
            self.build(approval)
        approval = copy.deepcopy(self.approval)
        approval["approval_basis"] = " "
        with self.assertRaisesRegex(CompilerError, "basis differs"):
            self.build(approval)


if __name__ == "__main__":
    unittest.main()
