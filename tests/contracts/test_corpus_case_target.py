"""A corpus case may name the exact target it assesses its Schedule on.

F-2026-09-14-004 left two declared Targets with zero cases while the full Gate passed.
The cause is structural rather than an omission: a Schedule embeds its own target, so
covering a second target meant copying the whole document. A case may now name the
target instead, which makes coverage a case rather than a duplicated Schedule.

Adding the cases themselves is a separate, reviewed act: these hold the mechanism, and
assert that it changes nothing about the Schedules that do not use it.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.corpus import check_corpus
from open_cake_ir.compiler.errors import CompilerError

ROOT = Path(__file__).resolve().parents[2]


class CorpusCaseTarget(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def _corpus(self, cases, directory: Path) -> Path:
        path = directory / "corpus.json"
        path.write_text(json.dumps(
            {"schema_version": 1, "corpus_id": "fixture", "state": "draft", "cases": cases}))
        return path

    def _case(self, **overrides) -> dict:
        case = {
            "case_id": "wave64-on-a-cuda-target",
            "schedule": "corpus/schedules/gfx938-wave64-thread-extent-refusal.json",
            "target": "sm_100a",
            "expected": {"accepted": True, "lowering_eligible": True, "finding_codes": [],
                         "schedule_sha256": "0" * 64, "lowering_source_sha256": None},
        }
        case.update(overrides)
        return case

    def test_one_schedule_is_assessed_on_the_target_its_case_names(self) -> None:
        """The same Schedule, two verdicts, separated only by the named target.

        32 slots is 2048 threads on gfx938's 64-lane width and 1024 on sm_100a's 32.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            report = check_corpus(self.compiler, self._corpus([self._case()], Path(directory)))
        case = report.cases[0]
        self.assertEqual(case.target, "sm_100a")
        self.assertTrue(case.observed_accepted, case.observed_finding_codes)
        self.assertNotIn("TARGET_THREAD_LIMIT", case.observed_finding_codes)

        with tempfile.TemporaryDirectory() as directory:
            declared = check_corpus(
                self.compiler,
                self._corpus([{k: v for k, v in self._case().items() if k != "target"}],
                             Path(directory)))
        self.assertEqual(declared.cases[0].target, "gfx938")
        self.assertFalse(declared.cases[0].observed_accepted)
        self.assertIn("TARGET_THREAD_LIMIT", declared.cases[0].observed_finding_codes)

    def test_a_named_target_is_refused_where_it_would_state_nothing(self) -> None:
        import tempfile

        for overrides, expected in (
            ({"target": "gfx938"}, "already declares"),
            ({"target": ""}, "corpus.cases"),
            ({"schedule": "examples/python/fma.py"}, "only a JSON Schedule"),
        ):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(CompilerError, expected):
                    check_corpus(self.compiler,
                                 self._corpus([self._case(**overrides)], Path(directory)))

    def test_the_declared_corpus_still_matches_every_retained_expectation(self) -> None:
        """The mechanism changes no verdict it is not used by."""
        report = self.compiler.check_corpus()
        self.assertTrue(report.passed, [c.case_id for c in report.cases if not c.matched])
        # 151 until the two gfx938 mma cases landed (F-2026-09-17-009), then 153 until
        # the two gfx938 tanh cases, then 155 until the fp32 contraction, then 156 until the tf32 and fp8 ones. The count is pinned so a case cannot appear or
        # vanish without someone saying why; moving it is the saying-why.
        self.assertEqual(report.case_count, 158)


if __name__ == "__main__":
    unittest.main()
