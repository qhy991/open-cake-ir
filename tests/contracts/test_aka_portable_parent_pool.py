from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.audit_aka_corpus import SourceRecord  # noqa: E402
from tools.run_aka_portable_parent_pool import (  # noqa: E402
    PortablePoolError,
    select_new_entries,
    select_requested_entries,
)
from tools.run_aka_portable_parents_codex import PortableEntry  # noqa: E402


def entry(index: int, *, ready: bool = True) -> PortableEntry:
    source = SourceRecord(
        relative_path="categories/data_movement_and_layout/copy/analysis.jsonl",
        line_number=index,
        category="data_movement_and_layout",
        operator="copy",
        task="analysis",
        primary_field="input",
        relation="analyze",
        fields={
            "instruction": "Analyze one copy.",
            "input": "__global__ void copy() {}",
            "reasoning": "Visible source only.",
            "output": "One review.",
        },
        record_format="aka_v1_operator_sft",
    )
    record = {
        "case_id": f"portable-{index:03d}",
        "derived_parent_id": f"derived-{index:03d}",
        "provenance": {
            "source_selection": {
                "path": source.relative_path,
                "line": index,
                "parent_field": "input",
            }
        },
    }
    return PortableEntry(
        queue_case_id=f"case-{index:06d}",
        source_record=source,
        portable_record=record,
        review_ready=ready,
        blocker=None if ready else "source_field_override_required",
    )


class AkaPortableParentPoolTests(unittest.TestCase):
    def test_selects_exact_new_hundred_in_current_order(self) -> None:
        prior = [entry(index) for index in range(1, 51)]
        current = [entry(index) for index in range(1, 151)]

        selected, blocked = select_new_entries(current, prior, expected_count=100)

        self.assertEqual(len(selected), 100)
        self.assertEqual(blocked, [])
        self.assertEqual(selected[0].queue_case_id, "case-000051")
        self.assertEqual(selected[-1].queue_case_id, "case-000150")

    def test_refuses_blocked_or_wrong_size_delta(self) -> None:
        prior = [entry(index) for index in range(1, 3)]
        with self.assertRaisesRegex(PortablePoolError, "expected 3"):
            select_new_entries(
                [entry(index) for index in range(1, 4)],
                prior,
                expected_count=3,
            )
        with self.assertRaisesRegex(PortablePoolError, "source-field blockers"):
            select_new_entries(
                [entry(1), entry(2), entry(3, ready=False)],
                prior,
                expected_count=1,
            )

        selected, blocked = select_new_entries(
            [entry(1), entry(2), entry(3, ready=False)],
            prior,
            expected_count=1,
            expected_review_ready_count=0,
        )
        self.assertEqual(selected, [])
        self.assertEqual([item.queue_case_id for item in blocked], ["case-000003"])

        with self.assertRaisesRegex(PortablePoolError, "expected 1 review-ready"):
            select_new_entries(
                [entry(1), entry(2), entry(3, ready=False)],
                prior,
                expected_count=1,
                expected_review_ready_count=1,
            )

    def test_explicit_recovery_subset_is_exact_and_preserves_delta_order(self) -> None:
        current = [entry(index) for index in range(1, 6)]

        selected = select_requested_entries(
            current, ["case-000004", "case-000002"]
        )

        self.assertEqual(
            [item.queue_case_id for item in selected],
            ["case-000002", "case-000004"],
        )
        with self.assertRaisesRegex(PortablePoolError, "duplicates"):
            select_requested_entries(
                current, ["case-000002", "case-000002"]
            )
        with self.assertRaisesRegex(PortablePoolError, "outside"):
            select_requested_entries(current, ["case-999999"])


if __name__ == "__main__":
    unittest.main()
