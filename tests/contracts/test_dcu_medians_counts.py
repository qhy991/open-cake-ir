"""Every DCU count, enumeration and quoted median derives from the retained table.

The gap this closes, and it has been closed badly once already. The floor count in
F-2026-09-18-002 was asserted in prose twice before it was right -- seven, then eleven --
and a second count in the same record, how many runs rose above the floor, was wrong as
well; each correction edited the previous sentence instead of recounting. A first version
of this file was written to stop that and did not: it asserted the literals `12` and `14`
and the English strings "Twelve DCU tasks" and "Fourteen runs separated", which is a third
hand-written copy of the same fact rather than a derivation. An independent reviewer
mutated it and five drift modes passed -- an enumeration naming a task the record puts in
the other group, a ratio changed in -005, a per-task median pair changed in -002, and
either record's count word changed.

Nothing here is written down twice. Counts are `len()` of a group computed from
`findings/data/2026-09-18-dcu-confirmatory-medians.json`, rendered through `WORDS`, and
looked for inside the record's own phrase; enumerations are compared set-wise; and every
median pair and ratio the prose quotes is checked against the table it claims to quote.

What this cannot check is whether the table matches the device. That is what the table's
per-row `workspace` and `receipt` are for: a reader with bw1100 can re-read any single row
from the receipt it names. Before those existed a mis-transcribed row was unfalsifiable
from inside the checkout, and two rows did in fact differ from the device -- not
mis-transcribed, but silently taken from the first of two qualified runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "findings/data/2026-09-18-dcu-confirmatory-medians.json"
FINDINGS = ROOT / "findings"

#: Enough to name any count these records make. A derived count is rendered through this
#: and looked for in the prose, so the table and the prose cannot disagree in words either.
WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
         13: "thirteen", 14: "fourteen", 15: "fifteen", 26: "twenty-six",
         27: "twenty-seven", 30: "thirty"}


def _record(stem: str) -> dict:
    return json.loads(next(FINDINGS.glob(f"{stem}-*.json")).read_text(encoding="utf-8"))


class DcuMedianCounts(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(DATA.read_text(encoding="utf-8"))
        self.floor = self.data["floor_ms"]
        self.rows = self.data["medians_ms"]
        self.two, self.five = _record("2026-09-18-002"), _record("2026-09-18-005")
        self.two_text = json.dumps(self.two)
        self.five_text = json.dumps(self.five)
        index = (FINDINGS / "README.md").read_text(encoding="utf-8").splitlines()
        self.index = {i: next(line for line in index if line.startswith(f"- {i} "))
                      for i in ("F-2026-09-18-002", "F-2026-09-18-005")}

    # -- the groups, derived once ---------------------------------------------------
    def _at_floor(self) -> set[str]:
        return {t for t, r in self.rows.items()
                if r["candidate"] == self.floor and r["baseline"] == self.floor}

    def _separated(self) -> set[str]:
        return set(self.rows) - self._at_floor()

    def _ahead(self) -> dict[str, float]:
        return {t: round(r["baseline"] / r["candidate"], 3)
                for t, r in self.rows.items() if r["baseline"] > r["candidate"]}

    def _material(self) -> dict[str, float]:
        return {t: v for t, v in self._ahead().items()
                if v > self.data["materiality_ratio"]}

    def _word(self, count: int) -> str:
        return WORDS[count]

    # -- counts, inside the records' own phrases -------------------------------------
    def test_every_stated_count_is_the_count_the_table_gives(self) -> None:
        floor, separated, total = (self._word(len(self._at_floor())),
                                   self._word(len(self._separated())),
                                   self._word(len(self.rows)))
        ahead, material = self._word(len(self._ahead())), self._word(len(self._material()))
        for label, pattern, text in (
            ("002 title", rf"{floor} DCU tasks", self.two["title"]),
            ("002 retained floor", rf"{floor} tasks whose confirmatory",
             self.two["evidence"]["retained"]),
            ("002 retained separated", rf"{separated} runs that separated",
             self.two["evidence"]["retained"]),
            ("002 retained total", rf"{total} qualified runs",
             self.two["evidence"]["retained"]),
            ("002 enumeration heading", rf"{separated} runs separated:", self.two_text),
            ("002 recount", rf"gives {separated} separated", self.two["observation"]),
            ("002 arithmetic", rf"{floor} plus {separated} is that {total}",
             self.two["observation"]),
            ("005 title", rf"{material} DCU campaigns", self.five["title"]),
            ("005 confirmed", rf"{material} campaigns confirmed", self.five["observation"]),
            ("005 nominally ahead", rf"{ahead} candidates are nominally ahead",
             self.five["observation"]),
            ("005 cites the floor count", rf"{floor} other tasks report", self.five_text),
            ("index 002 floor", rf"{floor} DCU tasks", self.index["F-2026-09-18-002"]),
            ("index 002 separated", rf"{separated} runs separated",
             self.index["F-2026-09-18-002"]),
            ("index 002 total", rf"{total} qualified runs", self.index["F-2026-09-18-002"]),
            ("index 005", rf"{material} DCU campaigns", self.index["F-2026-09-18-005"]),
        ):
            with self.subTest(claim=label):
                self.assertRegex(text.lower(), pattern.lower(),
                                 f"{label}: the table gives {pattern!r}")

    # -- enumerations, compared set-wise --------------------------------------------
    def _named(self, text: str, within: set[str]) -> set[str]:
        return {t for t in within if re.search(rf"\b{re.escape(t)}\b", text)}

    def test_the_enumerations_are_the_groups_the_table_gives(self) -> None:
        at_floor, separated = self._at_floor(), self._separated()
        for label, listing, expected, other in (
            ("at the floor", self.two["evidence"]["retained"], at_floor, separated),
            ("separated", [s for s in self.two["evidence"]["sites"]
                           if "runs separated:" in s][0], separated, at_floor),
        ):
            with self.subTest(enumeration=label):
                names = listing.split("separated:")[-1] if label == "separated" else listing
                self.assertEqual(self._named(names, expected), expected,
                                 f"the {label} enumeration omits a task the table puts there")
                intruders = self._named(names, other)
                self.assertEqual(intruders, set(),
                                 f"the {label} enumeration names {sorted(intruders)}, "
                                 "which the table puts in the other group")

    def test_nothing_is_reported_absent_that_the_table_holds(self) -> None:
        for task in self.data.get("absent", {}):
            self.assertNotIn(task, self.rows)
        live = r"(?:is|are)(?: the one| still)? absent|counted in neither group"
        for text in (self.two_text, self.index["F-2026-09-18-002"]):
            for task in self.rows:
                self.assertNotRegex(
                    text, rf"{re.escape(task)}[^\"]{{0,60}}(?:{live})",
                    f"{task} is in the table and cannot also be reported absent. "
                    "Narrating that it was absent before its receipt was read is fine; "
                    "this forbids only the claim still standing.")

    # -- every median pair and ratio the prose quotes --------------------------------
    _PAIRS = (
        re.compile(r"(?P<task>[a-z_]+) is candidate (?P<c>\d\.\d{6}) against baseline "
                   r"(?P<b>\d\.\d{6})(?: \((?P<r>\d\.\d+)\))?"),
        re.compile(r"(?P<task>[a-z_]+): candidate (?P<c>\d\.\d{6})(?: ms)?, baseline "
                   r"(?P<b>\d\.\d{6})(?:, speedup (?P<r>\d\.\d+))?"),
        re.compile(r"(?P<task>[a-z_]+) \((?P<c>\d\.\d{6})/(?P<b>\d\.\d{6})\)"),
        re.compile(r"(?P<task>[a-z_]+) (?P<c>\d\.\d{6}) against (?P<b>\d\.\d{6})"
                   r"(?: \((?P<r>\d\.\d+)\))?"),
    )

    def test_every_quoted_median_pair_and_ratio_matches_the_table(self) -> None:
        reruns = {(w["candidate"], w["baseline"])
                  for runs in self.data["excluded_reruns"].values() for w in runs}
        found = 0
        for text in (self.two_text, self.five_text):
            for pattern in self._PAIRS:
                for match in pattern.finditer(text):
                    task = match.group("task")
                    if task not in self.rows:
                        continue
                    pair = (float(match.group("c")), float(match.group("b")))
                    if pair in reruns:      # a named re-run, quoted as such
                        continue
                    found += 1
                    row = self.rows[task]
                    with self.subTest(task=task, quote=match.group(0)):
                        self.assertEqual(pair, (row["candidate"], row["baseline"]))
                        if match.groupdict().get("r"):
                            self.assertEqual(float(match.group("r")),
                                             round(row["baseline"] / row["candidate"], 3))
        # A phrasing change that stops the patterns matching would otherwise pass here in
        # silence, which is the exact shape of the defect this file exists for.
        self.assertGreaterEqual(found, 8, "the records quote fewer median pairs than "
                                          "expected; if the phrasing changed, teach the "
                                          "patterns rather than lowering this")

    def test_the_material_winners_are_the_tables(self) -> None:
        material = self._material()
        for task, ratio in material.items():
            with self.subTest(task=task):
                self.assertIn(task, self.five_text, "005 does not name a material winner")
                self.assertIn(f"{ratio}", self.five_text,
                              f"005 does not quote {task}'s ratio {ratio}")
        for task in set(self._ahead()) - set(material):
            with self.subTest(excluded=task):
                self.assertIn(task, self.five_text,
                              "005 does not name a nominal winner it excludes")

    # -- the re-runs the row-selection rule drops ------------------------------------
    def test_a_dropped_rerun_that_reads_better_is_named_where_it_matters(self) -> None:
        """The rule keeps the first qualified run. Where that discards the better
        reading, the record that would have reported it says so."""
        for task, runs in self.data["excluded_reruns"].items():
            kept = self.rows[task]
            for run in runs:
                better = (run["baseline"] / run["candidate"]
                          > kept["baseline"] / kept["candidate"])
                if better and run["baseline"] / run["candidate"] > self.data["materiality_ratio"]:
                    with self.subTest(task=task, workspace=run["workspace"]):
                        self.assertIn(run["workspace"], self.five_text,
                                      f"{task}'s discarded re-run clears materiality and "
                                      "is not named in F-2026-09-18-005")

    # -- provenance, which is what makes a single row falsifiable --------------------
    def test_every_row_names_the_run_and_receipt_it_came_from(self) -> None:
        for task, row in self.rows.items():
            with self.subTest(task=task):
                self.assertTrue(row["workspace"].startswith(f"{task}-"), row["workspace"])
                self.assertRegex(row["receipt"], r"^objects/sha256/[0-9a-f]{2}/[0-9a-f]{64}$")
        workspaces = [r["workspace"] for r in self.rows.values()]
        self.assertEqual(len(set(workspaces)), len(workspaces))

    # -- the stamp, derived from the runs rather than written beside them ------------
    def test_each_record_stamps_the_revisions_its_campaigns_actually_ran_at(self) -> None:
        """F-2026-09-18-002 was stamped at a commit no campaign it cites ran at, and nine
        of them ran after it. F-2026-09-18-005 was stamped twice at commits that postdate
        all three of its campaigns, the second time inside the sentence correcting the
        first. ADR 0065 makes the stamp so retained evidence replays against the tools
        that produced it, which is a per-campaign fact -- so it is derived here."""
        for label, record, tasks in (
            ("F-2026-09-18-002", self.two, set(self.rows)),
            ("F-2026-09-18-005", self.five,
             {"per_channel_moments", "momentum_sgd", "pairwise_sqdist"}),
        ):
            expected = {self.rows[t]["compiler_revision"] for t in tasks}
            stamped = set(re.findall(r"open-cake-ir@[0-9a-f]{8}",
                                     record["compiler_revision_id"]))
            with self.subTest(record=label):
                self.assertEqual(stamped, expected,
                                 f"{label} stamps {sorted(stamped)}; its campaigns ran at "
                                 f"{sorted(expected)}")
                for revision in expected:
                    self.assertIn(revision.split("@")[1], record["executor_revision"],
                                  f"{label}'s executor_revision omits {revision}")

    def test_every_row_names_the_revision_its_campaign_ran_at(self) -> None:
        for task, row in self.rows.items():
            with self.subTest(task=task):
                self.assertRegex(row["compiler_revision"], r"^open-cake-ir@[0-9a-f]{8}$")
        for runs in self.data["excluded_reruns"].values():
            for run in runs:
                with self.subTest(workspace=run["workspace"]):
                    self.assertRegex(run["compiler_revision"], r"^open-cake-ir@[0-9a-f]{8}$")

    def test_the_two_groups_partition_the_collection(self) -> None:
        self.assertEqual(len(self._at_floor()) + len(self._separated()), len(self.rows))
        self.assertEqual(self._at_floor() & self._separated(), set())

    def test_one_arm_on_the_floor_still_separates(self) -> None:
        """The rule that decides the split, pinned by the two runs that exercise it."""
        for task in ("prelu", "swiglu"):
            with self.subTest(task=task):
                row = self.rows[task]
                self.assertTrue(row["candidate"] == self.floor
                                or row["baseline"] == self.floor)
                self.assertIn(task, self._separated())

    def test_no_retracted_phrasing_stands_as_a_live_claim(self) -> None:
        # Each entry is the LIVE form of a claim that was retracted. A record may quote a
        # retracted sentence while retracting it -- that is how the correction is legible --
        # so these must not match the narration of the correction, only the claim itself.
        for retracted in ("all seven passed", "eleven other tasks", "the other seven",
                          "wrong three times", "twenty-six qualified runs",
                          "softsign the one qualified run absent",
                          "here names 2e901f53"):
            for text in (self.two_text, self.five_text,
                         *self.index.values(), DATA.read_text(encoding="utf-8")):
                self.assertNotIn(retracted, text)


if __name__ == "__main__":
    unittest.main()
