"""Every number in the DCU records is checked against the retained medians table.

The gap this closes, and it has now been closed badly twice. The floor count in
F-2026-09-18-002 was asserted in prose twice before it was right -- seven, then eleven --
and a second count in the same record, how many runs rose above the floor, was wrong as
well. The first guard written to stop that asserted the literals `12` and `14` and the
strings "Twelve DCU tasks" and "Fourteen runs separated", which is a third hand-written
copy rather than a derivation; a reviewer escaped it with five mutations. The second guard
checked fifteen chosen phrase sites and called that "every count": a reviewer changed
seven independent numbers across the two records in one mutation and the suite stayed
green, and every occurrence of a ratio but one was unanchored.

So nothing here is a chosen site.

* Every decimal literal in either record must be a number the table derives -- a median, a
  ratio `baseline / candidate` for any row or excluded re-run, the floor, the materiality
  ratio -- or one of the `FOOTPRINT_SWEEP` constants, which come from separate evidence and
  are listed with that reason. This is exhaustive: a changed digit anywhere fails.
* Every `<count word> <noun>` phrase must name a quantity that is derived *for that noun*.
  Thirteen operators, fourteen runs, twenty-seven tasks, thirty runs, three revisions: each
  is checked against the group it counts, so "nine operators" fails even though nine is a
  derived quantity somewhere else in the record.
* The enumerations are compared set-wise, the row-selection rule is recomputed from the
  workspace timestamps rather than believed, and each record's stamp is derived from the
  revisions its own campaigns ran at.

What no test in this checkout can do is verify the table against the device. That is what
each row's `workspace` and `receipt` are for, and what
`tools/read_dcu_confirmatory_medians.py` is for: the table is regenerated from sealed
evidence rather than edited, so a reader with bw1100 can reproduce it or refute one row.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "findings/data/2026-09-18-dcu-confirmatory-medians.json"
FINDINGS = ROOT / "findings"

WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
         13: "thirteen", 14: "fourteen", 15: "fifteen", 26: "twenty-six",
         27: "twenty-seven", 30: "thirty"}
NUMBER = {word: value for value, word in WORDS.items()}

#: Decimals these records quote that the medians table cannot derive, with why. All are
#: from the footprint sweep in F-2026-09-18-002's own evidence -- megabytes moved and
#: microseconds taken by a synthetic kernel at sizes the tasks do not run at -- which is
#: separate evidence from the confirmatory receipts this table holds.
FOOTPRINT_SWEEP = {0.25, 0.5, 1.0, 1.004, 1.008, 1.5, 1.504, 2.4, 2.399, 2.559, 2.719,
                   6.719, 8.0, 153.759}

#: Counts these records state while narrating what they retract. Each was a live claim in
#: an earlier version and is now quoted inside its own correction, so the prose has to be
#: able to say it. None of them may stand as a current count: the positive checks below
#: pin the live ones, and no value here is what any group actually holds today.
RETRACTED_COUNTS = {3, 7, 10, 11, 12, 26}


def _record(stem: str) -> dict:
    return json.loads(next(FINDINGS.glob(f"{stem}-*.json")).read_text(encoding="utf-8"))


class DcuMedianCounts(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(DATA.read_text(encoding="utf-8"))
        self.floor = self.data["floor_ms"]
        self.rows = self.data["medians_ms"]
        self.reruns = self.data["excluded_reruns"]
        self.two, self.five = _record("2026-09-18-002"), _record("2026-09-18-005")
        self.three = _record("2026-09-18-003")
        self.two_text = json.dumps(self.two, ensure_ascii=False)
        self.five_text = json.dumps(self.five, ensure_ascii=False)
        index = (FINDINGS / "README.md").read_text(encoding="utf-8").splitlines()
        self.index = {i: next(line for line in index if line.startswith(f"- {i} "))
                      for i in ("F-2026-09-18-002", "F-2026-09-18-005")}

    # -- the groups, derived once ---------------------------------------------------
    def _at_floor(self) -> set[str]:
        return {t for t, r in self.rows.items()
                if r["candidate"] == self.floor and r["baseline"] == self.floor}

    def _separated(self) -> set[str]:
        return set(self.rows) - self._at_floor()

    def _ratios(self) -> dict[str, float]:
        return {t: round(r["baseline"] / r["candidate"], 3) for t, r in self.rows.items()}

    def _ahead(self) -> dict[str, float]:
        return {t: v for t, v in self._ratios().items() if v > 1}

    def _material(self) -> dict[str, float]:
        return {t: v for t, v in self._ahead().items()
                if v > self.data["materiality_ratio"]}

    def _revisions(self, tasks=None) -> set[str]:
        return {self.rows[t]["compiler_revision"] for t in (tasks or self.rows)}

    # -- every decimal, exhaustively -------------------------------------------------
    def _derived_decimals(self) -> set[float]:
        values = {self.floor, self.data["materiality_ratio"]}
        for row in self.rows.values():
            values |= {row["candidate"], row["baseline"],
                       round(row["baseline"] / row["candidate"], 3)}
        for runs in self.reruns.values():
            for run in runs:
                values |= {run["candidate"], run["baseline"],
                           round(run["baseline"] / run["candidate"], 3)}
        return values

    def test_every_decimal_either_derives_or_is_a_listed_constant(self) -> None:
        derived = self._derived_decimals()
        for label, text in (("002", self.two_text), ("005", self.five_text)):
            for literal in re.findall(r"(?<![\w.:])\d+\.\d+(?![\w])", text):
                value = float(literal)
                with self.subTest(record=label, literal=literal):
                    self.assertTrue(
                        value in derived or value in FOOTPRINT_SWEEP,
                        f"{label} quotes {literal}, which is neither a median, a ratio, "
                        "the floor, the materiality ratio, nor a listed footprint-sweep "
                        "constant. If it is a real new measurement, add it to the table or "
                        "to FOOTPRINT_SWEEP with its reason.")

    # -- every counted noun, against the group it counts ------------------------------
    def _noun_groups(self) -> dict[str, set[int]]:
        floor, separated = len(self._at_floor()), len(self._separated())
        tasks, runs = len(self.rows), self.data["qualified_runs"]
        ahead, material = len(self._ahead()), len(self._material())
        revisions = len(self._revisions())
        per_revision = {sum(1 for r in self.rows.values()
                            if r["compiler_revision"] == revision)
                        for revision in self._revisions()}
        return {
            "operators": {floor},
            "results": {floor, material},
            "candidates": {material, ahead, ahead - material},
            "campaigns": {tasks, material, revisions, *per_revision},
            "tasks": {floor, tasks, len(self.reruns)},
            "runs": {floor, separated, runs},
            "rows": {tasks},
            "revisions": {revisions},
        }

    #: Quantity phrases that carry no noun, because the noun is two sentences back. Each
    #: names the group it refers to, so the number is still checked against the table; what
    #: the list adds is that a NEW bare phrase fails until it is classified here. Three of
    #: these -- "the count is X", "for those X", "six of the X" -- are mutations a reviewer
    #: used to change the floor count in prose while every other check passed.
    BARE = (
        (r"the count is ([\w-]+)", "floor"),
        (r"for those ([\w-]+) the comparison", "floor"),
        (r"six of the ([\w-]+)[:,]", "floor"),
        (r"to ([\w-]+) over all", "floor"),
        (r"all ([\w-]+) passed their", "floor"),
        (r"([\w-]+) is the fourth value", "floor"),
        (r"seven, eleven, twelve, ([\w-]+), which is what git shows", "floor"),
        (r"gives ([\w-]+) separated", "separated"),
        (r"named in the ([\w-]+)\.", "separated"),
        (r"over all ([\w-]+)", "tasks"),
    )
    #: The one construction that states the partition, checked as a whole.
    PARTITION = r"([\w-]+) plus ([\w-]+) is that ([\w-]+)"

    def _quantities(self) -> dict[str, int]:
        return {"floor": len(self._at_floor()), "separated": len(self._separated()),
                "tasks": len(self.rows), "runs": self.data["qualified_runs"]}

    def test_every_bare_quantity_phrase_names_the_count_the_table_gives(self) -> None:
        quantities = self._quantities()
        for label, text in (("002", self.two_text), ("005", self.five_text)):
            for pattern, group in self.BARE:
                for match in re.finditer(pattern, text, re.I):
                    word = match.group(1).lower()
                    with self.subTest(record=label, phrase=match.group(0)):
                        self.assertIn(word, NUMBER, "not a count word")
                        self.assertEqual(NUMBER[word], quantities[group],
                                         f"{label} says {match.group(0)!r}; the table "
                                         f"gives {quantities[group]} for the {group} group")
            for match in re.finditer(self.PARTITION, text, re.I):
                with self.subTest(record=label, phrase=match.group(0)):
                    self.assertEqual(
                        [NUMBER[w.lower()] for w in match.groups()],
                        [quantities["floor"], quantities["separated"], quantities["tasks"]],
                        f"{label}'s partition sentence does not add up to the table")

    #: Words that could name one of these groups. Every occurrence of one of these in either
    #: record must be accounted for by a checked phrase or be a value the record retracts --
    #: otherwise it is a quantity nothing looks at, which is how this record went wrong three
    #: times. Smaller words are left out: they are overwhelmingly ordinary prose ("one more",
    #: "two operands"), and the live counts they do carry are checked by noun above.
    GROUP_WORDS = ("nine", "eleven", "twelve", "thirteen", "fourteen", "twenty-six",
                   "twenty-seven", "thirty")

    def test_no_group_sized_number_goes_unchecked(self) -> None:
        """The completeness half. Without it the two checks above are chosen sites again:
        a reviewer moved the floor count in four sentences none of them looked at."""
        nouns = tuple(self._noun_groups())
        for label, text in (("002", self.two_text), ("005", self.five_text)):
            covered = set()
            for pattern, _ in self.BARE:
                for match in re.finditer(pattern, text, re.I):
                    covered.add(match.span(1))
            for match in re.finditer(self.PARTITION, text, re.I):
                covered.update(match.span(i) for i in (1, 2, 3))
            for noun in nouns:
                for match in re.finditer(rf"((?:[\w-]+\s+){{0,2}}){noun}\b", text):
                    start = match.start(1)
                    for token in re.finditer(r"[\w-]+", match.group(1)):
                        covered.add((start + token.start(), start + token.end()))
            for word in self.GROUP_WORDS:
                for match in re.finditer(rf"\b{word}\b", text, re.I):
                    if match.span() in covered or NUMBER[word] in RETRACTED_COUNTS:
                        continue
                    self.fail(
                        f"{label} states {word!r} where nothing checks it: "
                        f"...{text[max(0, match.start() - 60):match.end() + 30]!r}... "
                        "Give it a noun this file counts, or add its construction to BARE.")

    def test_every_counted_noun_names_a_quantity_derived_for_that_noun(self) -> None:
        groups = self._noun_groups()
        for label, text in (("002", self.two_text), ("005", self.five_text)):
            for noun, allowed in groups.items():
                for match in re.finditer(rf"((?:[\w-]+\s+){{0,2}}){noun}\b", text):
                    tokens = match.group(1).lower().split()
                    named = [t for t in tokens if t in NUMBER or t == "hundred"]
                    if not named:
                        continue
                    with self.subTest(record=label, phrase=match.group(0).strip()):
                        self.assertNotIn("hundred", named,
                                         "an unrecognised quantity stands before a noun "
                                         "this record counts")
                        value = NUMBER[named[-1]]
                        self.assertTrue(
                            value in allowed or value in RETRACTED_COUNTS,
                            f"{label} says {match.group(0).strip()!r}; the table gives "
                            f"{sorted(allowed)} for {noun}")

    # -- the live counts, in the records' own phrases ---------------------------------
    def test_each_record_states_the_count_the_table_gives(self) -> None:
        floor, separated = (WORDS[len(self._at_floor())], WORDS[len(self._separated())])
        tasks, runs = WORDS[len(self.rows)], WORDS[self.data["qualified_runs"]]
        ahead, material = WORDS[len(self._ahead())], WORDS[len(self._material())]
        for label, pattern, text in (
            ("002 title", rf"{floor} DCU tasks", self.two["title"]),
            ("002 retained floor", rf"{floor} tasks whose confirmatory",
             self.two["evidence"]["retained"]),
            ("002 retained separated", rf"{separated} runs that separated",
             self.two["evidence"]["retained"]),
            ("002 retained tasks", rf"{tasks} qualified tasks",
             self.two["evidence"]["retained"]),
            ("002 retained runs", rf"{runs} runs reached a qualified endpoint",
             self.two["evidence"]["retained"]),
            ("002 enumeration heading", rf"{separated} runs separated:", self.two_text),
            ("002 recount", rf"gives {separated} separated", self.two["observation"]),
            ("002 arithmetic", rf"{floor} plus {separated} is that {tasks}",
             self.two["observation"]),
            ("005 title", rf"{material} DCU campaigns", self.five["title"]),
            ("005 confirmed", rf"{material} campaigns confirmed", self.five["observation"]),
            ("005 nominally ahead", rf"{ahead} candidates are nominally ahead",
             self.five["observation"]),
            ("005 cites the floor count", rf"{floor} other tasks report", self.five_text),
            ("index 002 floor", rf"{floor} DCU tasks", self.index["F-2026-09-18-002"]),
            ("index 002 separated", rf"{separated} runs separated",
             self.index["F-2026-09-18-002"]),
            ("index 002 tasks", rf"{tasks} qualified tasks",
             self.index["F-2026-09-18-002"]),
            ("index 005", rf"{material} DCU campaigns", self.index["F-2026-09-18-005"]),
        ):
            with self.subTest(claim=label):
                self.assertRegex(text.lower(), pattern.lower(),
                                 f"{label}: the table gives {pattern!r}")

    # -- enumerations, compared set-wise ---------------------------------------------
    def _named(self, text: str, within: set[str]) -> set[str]:
        return {t for t in within if re.search(rf"\b{re.escape(t)}\b", text)}

    def test_the_enumerations_are_the_groups_the_table_gives(self) -> None:
        at_floor, separated = self._at_floor(), self._separated()
        listing = [s for s in self.two["evidence"]["sites"] if "runs separated:" in s][0]
        for label, names, expected, other in (
            ("at the floor", self.two["evidence"]["retained"], at_floor, separated),
            ("separated", listing.split("separated:")[-1], separated, at_floor),
        ):
            with self.subTest(enumeration=label):
                self.assertEqual(self._named(names, expected), expected,
                                 f"the {label} enumeration omits a task the table puts there")
                intruders = self._named(names, other)
                self.assertEqual(intruders, set(),
                                 f"the {label} enumeration names {sorted(intruders)}, "
                                 "which the table puts in the other group")

    def test_nothing_is_reported_absent_that_the_table_holds(self) -> None:
        live = r"(?:is|are)(?: the one| still)? absent|counted in neither group"
        for text in (self.two_text, self.index["F-2026-09-18-002"]):
            for task in self.rows:
                self.assertNotRegex(
                    text, rf"{re.escape(task)}[^\"]{{0,60}}(?:{live})",
                    f"{task} is in the table and cannot also be reported absent. Narrating "
                    "that it was absent before its receipt was read is fine; this forbids "
                    "only the claim still standing.")

    # -- every quoted median pair, and every occurrence of every ratio -----------------
    _PAIRS = (
        re.compile(r"(?P<task>[a-z_]+) is candidate (?P<c>\d\.\d{6}) against baseline "
                   r"(?P<b>\d\.\d{6})(?: \((?P<r>\d\.\d+)\))?"),
        re.compile(r"(?P<task>[a-z_]+): candidate (?P<c>\d\.\d{6})(?: ms)?, baseline "
                   r"(?P<b>\d\.\d{6})(?:, speedup (?P<r>\d\.\d+))?"),
        re.compile(r"(?P<task>[a-z_]+) \((?P<c>\d\.\d{6})/(?P<b>\d\.\d{6})\)"),
        re.compile(r"(?P<task>[a-z_]+) (?P<c>\d\.\d{6}) against (?P<b>\d\.\d{6})"
                   r"(?: \((?P<r>\d\.\d+)\))?"),
    )
    #: How many pairs the two records quote in the shapes above. Pinned exactly rather than
    #: as a floor: a floor lets a rephrasing silently stop matching, which is the shape of
    #: the defect this file exists for, and a reviewer used exactly that slack.
    QUOTED_PAIRS = 9

    def test_every_quoted_median_pair_and_ratio_matches_the_table(self) -> None:
        rerun_pairs = {(w["candidate"], w["baseline"])
                       for runs in self.reruns.values() for w in runs}
        found = 0
        for text in (self.two_text, self.five_text):
            for pattern in self._PAIRS:
                for match in pattern.finditer(text):
                    task = match.group("task")
                    if task not in self.rows:
                        continue
                    pair = (float(match.group("c")), float(match.group("b")))
                    if pair in rerun_pairs:
                        continue
                    found += 1
                    row = self.rows[task]
                    with self.subTest(task=task, quote=match.group(0)):
                        self.assertEqual(pair, (row["candidate"], row["baseline"]))
                        if match.groupdict().get("r"):
                            self.assertEqual(float(match.group("r")),
                                             round(row["baseline"] / row["candidate"], 3))
        self.assertEqual(found, self.QUOTED_PAIRS,
                         "the records quote a different number of median pairs than this "
                         "file expects; if the phrasing changed, teach the patterns and "
                         "move the pin -- do not lower it")

    def test_every_occurrence_of_a_winning_ratio_is_the_tables(self) -> None:
        """Not a presence check: 1.405 occurs five times in F-2026-09-18-005, and a guard
        that asks only whether it appears passes when four of them change."""
        for task, ratio in self._material().items():
            row = self.rows[task]
            with self.subTest(task=task):
                self.assertIn(task, self.five_text, "005 does not name a material winner")
                occurrences = re.findall(rf"(?<![\d.]){re.escape(f'{ratio}')}(?![\d])",
                                         self.five_text)
                self.assertTrue(occurrences, f"005 never quotes {task}'s ratio {ratio}")
                self.assertEqual(round(row["baseline"] / row["candidate"], 3), ratio)
        for task in set(self._ahead()) - set(self._material()):
            with self.subTest(excluded=task):
                self.assertIn(task, self.five_text,
                              "005 does not name a nominal winner it excludes")

    # -- the row-selection rule, recomputed rather than believed ----------------------
    def test_each_row_is_the_first_qualified_run_of_its_task(self) -> None:
        stamp = re.compile(r"-(\d{8}-\d{6})$")
        for task, row in self.rows.items():
            kept = stamp.search(row["workspace"])
            with self.subTest(task=task):
                self.assertIsNotNone(kept, row["workspace"])
                for run in self.reruns.get(task, []):
                    dropped = stamp.search(run["workspace"])
                    self.assertIsNotNone(dropped, run["workspace"])
                    self.assertLess(kept.group(1), dropped.group(1),
                                    f"{task}'s kept run is not the first: the data file "
                                    "says each row is the task's first qualified run")

    def test_a_dropped_rerun_that_reads_better_is_named_where_it_matters(self) -> None:
        for task, runs in self.reruns.items():
            kept = self.rows[task]
            for run in runs:
                ratio = run["baseline"] / run["candidate"]
                if (ratio > kept["baseline"] / kept["candidate"]
                        and ratio > self.data["materiality_ratio"]):
                    with self.subTest(task=task, workspace=run["workspace"]):
                        self.assertIn(run["workspace"], self.five_text,
                                      f"{task}'s discarded re-run clears materiality and "
                                      "is not named in F-2026-09-18-005")

    # -- the stamps, derived from the runs -------------------------------------------
    def test_each_record_stamps_the_revisions_its_campaigns_actually_ran_at(self) -> None:
        """F-2026-09-18-002 was stamped at a commit no campaign it cites ran at; -005 at
        one postdating all three of its own, inside the sentence correcting the previous
        wrong stamp; -003 at a third, 1h41m after the sweep it cites began. ADR 0065 makes
        the stamp so retained evidence replays against the tools that produced it, which is
        a per-campaign fact -- so it is derived here rather than read."""
        for label, record, tasks in (
            ("F-2026-09-18-002", self.two, set(self.rows)),
            ("F-2026-09-18-005", self.five,
             {"per_channel_moments", "momentum_sgd", "pairwise_sqdist"}),
            # -003's eight tasks include two that never qualified, so they have no row; the
            # six that did are what the table can speak for, and they span the same three.
            ("F-2026-09-18-003", self.three,
             {"softsign", "per_channel_moments", "softmax",
              "layernorm_gamma_beta_backward", "channel_absmax_scale", "attention_decode"}),
        ):
            expected = self._revisions(tasks)
            stamped = set(re.findall(r"open-cake-ir@[0-9a-f]{8}",
                                     record["compiler_revision_id"]))
            with self.subTest(record=label):
                self.assertTrue(expected <= stamped,
                                f"{label} stamps {sorted(stamped)}; its campaigns ran at "
                                f"{sorted(expected)}")
                for revision in expected:
                    self.assertIn(revision.split("@")[1], record["executor_revision"],
                                  f"{label}'s executor_revision omits {revision}")

    def test_every_row_names_the_run_receipt_and_revision_it_came_from(self) -> None:
        for task, row in self.rows.items():
            with self.subTest(task=task):
                self.assertTrue(row["workspace"].startswith(f"{task}-"), row["workspace"])
                self.assertRegex(row["receipt"], r"^objects/sha256/[0-9a-f]{2}/[0-9a-f]{64}$")
                self.assertRegex(row["compiler_revision"], r"^open-cake-ir@[0-9a-f]{8}$")
        workspaces = [r["workspace"] for r in self.rows.values()]
        self.assertEqual(len(set(workspaces)), len(workspaces))

    def test_the_table_says_how_many_runs_and_how_many_tasks(self) -> None:
        """One word cannot carry both: thirty runs produced twenty-seven rows, and reading
        `qualified runs` as twenty-seven is what made the two records disagree."""
        self.assertEqual(self.data["qualified_tasks"], len(self.rows))
        self.assertEqual(self.data["qualified_runs"],
                         len(self.rows) + sum(len(r) for r in self.reruns.values()))
        self.assertGreater(self.data["qualified_runs"], self.data["qualified_tasks"])

    def test_the_table_names_the_instrument_that_regenerates_it(self) -> None:
        """The `collected` field named a source that could not produce the file twice: a
        sweep pair covering eight of its twenty-seven tasks, then a script in no commit."""
        tool = self.data["generated_by"]
        self.assertTrue((ROOT / tool).is_file(), f"{tool} is named but not in the checkout")
        self.assertIn(tool, self.data["collected"])

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
        for retracted in ("all seven passed", "eleven other tasks", "wrong three times",
                          "twenty-six qualified runs", "twenty-seven qualified runs",
                          "softsign the one qualified run absent", "here names 2e901f53",
                          "fourth time this number has moved"):
            for text in (self.two_text, self.five_text,
                         *self.index.values(), DATA.read_text(encoding="utf-8")):
                self.assertNotIn(retracted, text)


if __name__ == "__main__":
    unittest.main()
