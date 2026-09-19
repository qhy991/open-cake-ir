"""A passing Corpus Gate is evidence only about the Targets it examined."""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.compiler.corpus import CorpusCaseReport, CorpusGateReport  # noqa: E402


def _case(case_id: str, target: str, *, matched: bool = True) -> CorpusCaseReport:
    return CorpusCaseReport(
        case_id=case_id,
        schedule_path=f"corpus/schedules/{case_id}.json",
        target=target,
        expected_accepted=True,
        expected_lowering_eligible=True,
        expected_finding_codes=(),
        expected_schedule_sha256="0" * 64,
        expected_lowering_source_sha256=None,
        observed_accepted=True,
        observed_lowering_eligible=True,
        observed_finding_codes=(),
        observed_schedule_sha256="0" * 64,
        observed_lowering_source_sha256=None,
        matched=matched,
    )


class CoverageDerivationContracts(unittest.TestCase):
    def test_declared_targets_split_into_examined_and_unexamined(self):
        report = CorpusGateReport(
            corpus_id="fixture",
            compiler_revision_id="fixture",

            passed=True,
            cases=(_case("a", "alpha"), _case("b", "alpha"), _case("c", "beta")),
            declared_targets=("alpha", "beta", "gamma"),
        )
        self.assertEqual(dict(report.examined_targets), {"alpha": 2, "beta": 1})
        self.assertEqual(report.unexamined_targets, ("gamma",))
        self.assertEqual(
            set(report.examined_targets) | set(report.unexamined_targets),
            set(report.declared_targets),
        )
        self.assertFalse(set(report.examined_targets) & set(report.unexamined_targets))

    def test_a_refusal_case_is_not_counted_as_coverage(self):
        """A case naming an undeclared Target exercises the refusal, not the Target."""

        report = CorpusGateReport(
            "fixture", "fixture", True,
            (_case("a", "alpha"), _case("refusal", "delta")),
            ("alpha", "beta"),
        )
        self.assertEqual(dict(report.examined_targets), {"alpha": 1})
        self.assertEqual(dict(report.undeclared_case_targets), {"delta": 1})
        self.assertNotIn("delta", report.examined_targets)
        self.assertIn("beta", report.unexamined_targets)

    def test_coverage_reports_absence_and_never_blocks_the_gate(self):
        """Unexamined Targets are reported, not repaired and not failed.

        Failing here would pressure an author to mint cases to empty the list, which
        reports a coverage the Corpus does not have.
        """

        report = CorpusGateReport(
            "fixture", "fixture", True, (_case("a", "alpha"),), ("alpha", "beta"),
        )
        self.assertTrue(report.passed)
        self.assertEqual(report.unexamined_targets, ("beta",))

    def test_a_gate_built_without_declared_targets_claims_no_coverage(self):
        report = CorpusGateReport("fixture", "fixture", True, (_case("a", "alpha"),))
        self.assertEqual(dict(report.examined_targets), {})
        self.assertEqual(report.unexamined_targets, ())
        self.assertEqual(dict(report.undeclared_case_targets), {"alpha": 1})


class LiveCorpusContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = Compiler.load(ROOT, ROOT / "compiler/revision.json").check_corpus()

    def test_every_case_names_the_target_its_schedule_declares(self):
        manifest = json.loads((ROOT / "corpus/manifest.json").read_text(encoding="utf-8"))
        declared_by_path = {}
        for case in manifest["cases"]:
            path = ROOT / case["schedule"]
            if path.suffix == ".json":
                declared_by_path[case["case_id"]] = json.loads(
                    path.read_text(encoding="utf-8")
                )["target"]
        self.assertTrue(declared_by_path)
        for case in self.report.cases:
            if case.case_id in declared_by_path:
                self.assertEqual(case.target, declared_by_path[case.case_id], case.case_id)

    def test_the_declared_target_set_is_the_targets_directory(self):
        declared = {path.stem for path in (ROOT / "compiler/targets").glob("*.json")}
        self.assertEqual(set(self.report.declared_targets), declared)

    def test_unexamined_is_derived_from_the_manifest_not_asserted(self):
        counted = {case.target for case in self.report.cases}
        self.assertEqual(
            set(self.report.unexamined_targets),
            {target for target in self.report.declared_targets if target not in counted},
        )




if __name__ == "__main__":
    unittest.main()
