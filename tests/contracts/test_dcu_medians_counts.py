"""The counts in F-2026-09-18-002 and -005 derive from the retained medians.

The gap this closes: those counts were wrong three times running -- seven, then eleven,
then a partition that did not say what it partitioned -- and each correction edited the
previous sentence instead of recounting. A count asserted in prose has no way to be
wrong out loud. These derive the same numbers from
`findings/data/2026-09-18-dcu-confirmatory-medians.json` and check the records against
them, so the next edit that changes a number without the data behind it fails here.

The medians themselves cannot be re-measured without the device; what this guards is the
arithmetic done on them, which is where every one of those errors was.

What it catches, checked by mutation rather than asserted: moving a median off the floor
(3 failures), adding a task to the data (2), changing a count word in a record (1),
reintroducing a retracted phrase (1), and changing a winner's ratio (1). What it does not
catch, deliberately: changing a separated task's median without changing which group it
falls in -- that is a claim about the measurement, and no test here can check a number the
device produced.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "findings/data/2026-09-18-dcu-confirmatory-medians.json"
FINDINGS = ROOT / "findings"


def _record(stem: str) -> dict:
    path = next(FINDINGS.glob(f"{stem}-*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


class DcuMedianCounts(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(DATA.read_text(encoding="utf-8"))
        self.floor = self.data["floor_ms"]
        self.medians = self.data["medians_ms"]

    def _at_floor(self):
        return sorted(t for t, m in self.medians.items()
                      if m["candidate"] == self.floor and m["baseline"] == self.floor)

    def _separated(self):
        return sorted(t for t, m in self.medians.items()
                      if not (m["candidate"] == self.floor and m["baseline"] == self.floor))

    def test_twelve_runs_sit_at_the_floor_with_both_arms(self) -> None:
        self.assertEqual(len(self._at_floor()), 12)

    def test_fourteen_runs_separated_and_one_arm_on_the_floor_still_separates(self) -> None:
        separated = self._separated()
        self.assertEqual(len(separated), 14)
        # The rule that decides the split, pinned because a reader could not recover it
        # from the record: prelu and swiglu each have one arm exactly on the floor and
        # both are separated, rather than one of each.
        self.assertIn("prelu", separated)
        self.assertIn("swiglu", separated)

    def test_the_two_groups_partition_the_collection_and_name_what_is_absent(self) -> None:
        self.assertEqual(len(self._at_floor()) + len(self._separated()), len(self.medians))
        self.assertEqual(len(self.medians), 26)
        # 27 qualified; softsign's medians are not in the collection and the data says so.
        self.assertEqual(sorted(self.data["absent"]), ["softsign"])

    def test_three_candidates_beat_their_baseline_past_materiality(self) -> None:
        ratio = self.data["materiality_ratio"]
        ahead = {t: round(m["baseline"] / m["candidate"], 3)
                 for t, m in self.medians.items() if m["baseline"] > m["candidate"]}
        material = {t: r for t, r in ahead.items() if r > ratio}
        self.assertEqual(len(ahead), 5, ahead)
        self.assertEqual(sorted(material),
                         ["momentum_sgd", "pairwise_sqdist", "per_channel_moments"])
        self.assertEqual(material["per_channel_moments"], 1.405)

    def test_the_records_state_the_numbers_the_data_gives(self) -> None:
        two = json.dumps(_record("2026-09-18-002"))
        five = json.dumps(_record("2026-09-18-005"))
        self.assertIn("Twelve DCU tasks", two)
        self.assertIn("Fourteen runs separated", two)
        # The counts each record retracted must not reappear in either.
        for retracted in ("eleven other tasks", "all seven passed", "the other seven"):
            self.assertNotIn(retracted, two)
            self.assertNotIn(retracted, five)


if __name__ == "__main__":
    unittest.main()
