"""Every findings record parses, and every field is the type the contract states.

The gap this closes, measured: F-2026-09-17-011's `observation` spent five commits as a
2133-element array of single characters, because a trailing comma turned an assignment
into a tuple and the next `+=` iterated a string into it. The record stayed valid JSON
the whole time, so nothing noticed -- the ledger is the curation unit between campaigns
and Revisions under AGENTS.md, and nothing validated it at all.

These check shape, not content. A finding's argument is for a reader; its field types are
for whatever reads the directory.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
FINDINGS = ROOT / "findings"

# Every field that is a string in all 73 records today. `backend`, `target`, `evidence`
# and `verified_by` are deliberately absent: the ledger carries more than one shape for
# each, and a test that pinned one record's shape would refuse the others rather than
# catch anything.
_TEXT = ("id", "kind", "date", "title", "compiler_revision_id", "executor_revision",
         "observation", "proposed_change", "decision")
_OPTIONAL_TEXT = ("implemented_in",)


def _records():
    for path in sorted(FINDINGS.glob("*.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


class FindingRecords(unittest.TestCase):
    def test_the_directory_is_not_empty(self) -> None:
        """Otherwise every check below passes by having nothing to check."""
        self.assertGreater(len(list(FINDINGS.glob("*.json"))), 20)

    def test_every_record_is_a_json_object(self) -> None:
        for path, record in _records():
            with self.subTest(finding=path.name):
                self.assertIsInstance(record, dict)

    def test_every_narrative_field_is_a_string(self) -> None:
        for path, record in _records():
            for field in _TEXT:
                with self.subTest(finding=path.name, field=field):
                    self.assertIn(field, record)
                    self.assertIsInstance(
                        record[field], str,
                        f"{path.name}.{field} is {type(record[field]).__name__}; a list "
                        "here is the trailing-comma tuple defect this test exists for")
            for field in _OPTIONAL_TEXT:
                if record.get(field) is not None:
                    with self.subTest(finding=path.name, field=field):
                        self.assertIsInstance(record[field], str)

    def test_no_narrative_field_is_a_sequence_of_single_characters(self) -> None:
        """The specific shape the corruption took, named so a reader recognises it."""
        for path, record in _records():
            for field in _TEXT + _OPTIONAL_TEXT:
                value = record.get(field)
                if isinstance(value, (list, tuple)) and value:
                    with self.subTest(finding=path.name, field=field):
                        self.fail(f"{path.name}.{field} is a sequence of "
                                  f"{len(value)} items, the first {type(value[0]).__name__}")

    def test_every_record_carries_evidence(self) -> None:
        """Its shape varies across the ledger; that it is present and non-empty does not."""
        for path, record in _records():
            with self.subTest(finding=path.name):
                evidence = record.get("evidence")
                self.assertIsInstance(evidence, (dict, list))
                self.assertTrue(evidence, f"{path.name} carries an empty evidence field")

    def test_the_id_matches_the_filename(self) -> None:
        for path, record in _records():
            with self.subTest(finding=path.name):
                self.assertEqual(record["id"], "F-" + path.name[:len("2026-09-17-011")])

    def test_schema_version_is_an_integer(self) -> None:
        for path, record in _records():
            with self.subTest(finding=path.name):
                self.assertIsInstance(record.get("schema_version"), int)

    def test_ids_are_unique(self) -> None:
        """Two records claiming one id is how an index stops tracking one of them."""
        seen: dict[str, str] = {}
        for path, record in _records():
            with self.subTest(finding=path.name):
                self.assertNotIn(record["id"], seen,
                                 f"{path.name} and {seen.get(record['id'])} share an id")
                seen[record["id"]] = path.name


if __name__ == "__main__":
    unittest.main()
