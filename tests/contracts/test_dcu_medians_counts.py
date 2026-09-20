"""Numbers in the DCU records are checked against the retained table by association.

Three earlier versions of this file were each escaped by a reviewer, and each failure was
the same one at a deeper level. The first pinned literals (`12`, `14`, "Twelve DCU tasks")
-- a third hand-written copy of the fact it guarded. The second checked fifteen chosen
phrase sites and was described as checking every count; seven numbers changed across two
records without failing it. The third made the decimal check exhaustive but tested only
*set membership*, so any permutation of legal values passed: swapping `rmsnorm 1.004,
layernorm 1.008`, or `momentum_sgd and pairwise_sqdist are 1.057 and 1.054`, left the
suite green, and a ratio check whose docstring promised otherwise reduced to a tautology.
45 of 60 mutations escaped that version.

So this one checks *which number belongs to which task*.

* Every decimal in the records must be accounted for, and accounted for in one of three
  ways: it is unique to one task in the table -- a median, a ratio, or a footprint derived
  from that task's own Workload Contract -- in which case the nearest task name must be
  that task; or it appears in one of the `CONSTRUCTIONS`, which say what a phrase's number
  has to be; or it is a listed constant from evidence this table does not hold. Anything
  else fails as an unchecked number, which is what forces new prose through this file.
* Counts are checked against the group they count, including counts written as digits,
  and a retracted value is allowed only inside a sentence that marks it as retracted --
  not, as before, anywhere at all.
* Stamps are compared both ways against the revisions the cited campaigns ran at, which
  the table now records for every campaign on the host rather than only the qualified ones.

What no test here can do is verify the table against the device. That is what each row's
`workspace` and `receipt` are for, and what `tools/read_dcu_confirmatory_medians.py` is
for: the table is regenerated from sealed evidence rather than edited.

What this file does NOT check is stated plainly, because claiming otherwise is the defect
it exists for: F-2026-09-18-003's turn-budget counts (21 and 25 at two and four turns, its
event numbers, its error bounds) come from that record's own sweep ledgers, which this
table does not hold. Only its 27, its 30 and its stamp are checked here.
"""

from __future__ import annotations

import json
import re
from math import prod
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "findings/data/2026-09-18-dcu-confirmatory-medians.json"
FINDINGS = ROOT / "findings"

WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
         13: "thirteen", 14: "fourteen", 15: "fifteen", 26: "twenty-six",
         27: "twenty-seven", 28: "twenty-eight", 29: "twenty-nine", 30: "thirty",
         67: "sixty-seven"}
NUMBER = {word: value for value, word in WORDS.items()}
BYTES_PER = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8e4m3": 1, "i32": 4, "int32": 4, "i64": 8}

#: Decimals from evidence this table does not hold, each with the record that owns it.
#: A value here is exempt from association, so the list is kept short and specific.
DECLARED = {
    # The sweep's plateau, quoted in F-2026-09-18-002's observation. The sweep's own
    # numbers are NOT listed here: they are covered where they are written, as ordered
    # pairs, because exempting their values everywhere also exempted the 1.000 and 1.500
    # MiB footprints and let the pair carrying this record's argument be swapped.
    2.4,
    # F-2026-09-18-003's validation error bounds for gemm_silu, from its own receipts.
    0.5, 100.8,
}
#: The sweep now lives in the data file with its instrument named, so this file no longer
#: keeps a second copy of it to compare against the first.
#: Words marking a sentence as narrating a retracted claim. A retracted count is admitted
#: only inside one of these; an earlier version admitted them anywhere, which let a live
#: count be wrong whenever its value happened to be 3, 7, 10, 11, 12 or 26.
RETRACTION = ("it said", "record said", "version said", "version of this record",
              "an earlier reading", "asserted as", "stood at", "has been wrong",
              "have been wrong", "was wrong", "were wrong", "which was false", "retract",
              "moved the count", "has taken", "until it was read", "would have meant",
              "is withdrawn", "argued from bytes")
RETRACTED_COUNTS = {3, 7, 10, 11, 12, 26}
#: Tasks whose footprint F-2026-09-18-002 quotes, derived from their Workload Contracts.
FOOTPRINT_TASKS = ("silu", "selu", "rmsnorm", "layernorm", "softmax_backward",
                   "rmsnorm_input_gradient")


def _text(value) -> str:
    """Every string in a record, as plain text -- not JSON, so sentences split sanely."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return "\n".join(_text(v) for v in value)
    return ""


#: Every spelling a count can take here, longest first so "twenty-seven" beats "seven".
_NUMBER_WORDS = "|".join(sorted(WORDS.values(), key=len, reverse=True))


def _record(stem: str) -> dict:
    return json.loads(next(FINDINGS.glob(f"{stem}-*.json")).read_text(encoding="utf-8"))


def _footprint_mib(task: str) -> float:
    from open_cake_ir.tasks.workloads import create_task
    document, _ = create_task(task, backend="triton-dcu", rows=128, columns=1024)
    extents = next(c for c in document["cases"] if c["case_id"] == "primary")["shape"]
    return sum(prod(extents[d] if isinstance(d, str) else int(d) for d in spec["shape"])
               * BYTES_PER[spec["dtype"]]
               for spec in document["tensors"].values()) / 2 ** 20


class DcuMedianCounts(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.data = json.loads(DATA.read_text(encoding="utf-8"))
        self.floor = self.data["floor_ms"]
        self.rows = self.data["medians_ms"]
        self.reruns = self.data["excluded_reruns"]
        self.two, self.three, self.five = (_record("2026-09-18-002"),
                                           _record("2026-09-18-003"),
                                           _record("2026-09-18-005"))
        index = (FINDINGS / "README.md").read_text(encoding="utf-8").splitlines()
        self.index = {i: next(line for line in index if line.startswith(f"- {i} "))
                      for i in ("F-2026-09-18-002", "F-2026-09-18-003", "F-2026-09-18-005")}
        # Everything this file reads as prose. The data file's own hand-written fields are
        # in here too: `row_selection` states counts, and nothing else looked at them.
        data_prose = "\n".join(self.data[k] for k in
                               ("what", "collected", "row_selection", "receipt_paths"))
        # Prose whose numbers this table can speak for. Counts are checked across all of
        # it.
        self.prose = {
            "002": _text(self.two), "005": _text(self.five), "data": data_prose,
            "index-002": self.index["F-2026-09-18-002"],
            "index-005": self.index["F-2026-09-18-005"],
        }
        # F-2026-09-18-003 is scoped deliberately. Its subject is turn budget, and its
        # counts -- 21 and 25 tasks at two and four turns, event indices, error bounds --
        # come from its own sweep ledgers, which this table does not hold. Checking them
        # here would mean hand-writing them a second time, which is the defect this file
        # exists for. What IS checked is every claim it makes about THIS table, plus its
        # stamp; `THREE_CLAIMS` is that list, and the decimal rule runs over it too.
        self.three_text = _text(self.three)
        self.index_three = self.index["F-2026-09-18-003"]

    # -- derived groups --------------------------------------------------------------
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

    def _revisions(self, tasks) -> set[str]:
        return {self.rows[t]["compiler_revision"] for t in tasks}

    def _campaign_revisions(self, tasks) -> set[str]:
        return {revision for workspace, revision in self.data["campaign_revisions"].items()
                if workspace.rsplit("-", 2)[0] in tasks}

    # -- which value belongs to which task -------------------------------------------
    def _owners(self) -> dict[float, set[str]]:
        owners: dict[float, set[str]] = {}
        for task, row in self.rows.items():
            for value in (row["candidate"], row["baseline"],
                          round(row["baseline"] / row["candidate"], 3)):
                owners.setdefault(value, set()).add(task)
        for task, runs in self.reruns.items():
            for run in runs:
                for value in (run["candidate"], run["baseline"],
                              round(run["baseline"] / run["candidate"], 3)):
                    owners.setdefault(value, set()).add(task)
        for task in FOOTPRINT_TASKS:
            owners.setdefault(round(_footprint_mib(task), 3), set()).add(task)
        return owners

    def _nearest_task(self, text: str, at: int) -> str | None:
        best, distance = None, 10 ** 9
        for task in self.rows:
            for match in re.finditer(rf"\b{re.escape(task)}\b", text):
                gap = min(abs(match.start() - at), abs(match.end() - at))
                if gap < distance:
                    best, distance = task, gap
        return best if distance <= 140 else None

    #: Phrases that say what their number must be. `where` is a key of `_quantities`, or
    #: "pairs" for a construction carrying its own task/value correspondence.
    def _constructions(self):
        return (
            (r"identical (?P<v>\d\.\d{6}) ms", "floor"),
            (r"both (?P<v>\d\.\d{6}), speedup", "floor"),
            (r"both reporting (?P<v>\d\.\d{6})", "floor"),
            (r"both arms read (?P<v>\d\.\d{6})", "floor"),
            (r"both arms (?P<v>\d\.\d{6}), so it is at", "floor"),
            (r"both arms on (?P<v>\d\.\d{6}) is", "floor"),
            (r"both take (?P<v>\d\.\d{6}) ms", "floor"),
            (r"both report (?P<v>\d\.\d{6}), and", "floor"),
            (r"are above (?P<v>\d\.\d{6}), the value", "floor"),
            (r"this floor is (?P<v>\d\.\d{6}) ms", "floor"),
            (r"in the group reads (?P<v>\d\.\d{6}) ms", "floor"),
            (r"speedup (?P<v>\d\.\d{3}), classification close_null", "unit_ratio"),
            (r"declared (?P<v>\d\.\d+) materiality", "materiality"),
            (r"materiality_ratio is (?P<v>\d\.\d+)", "materiality"),
            (r"materiality ratio of (?P<v>\d\.\d+)", "materiality"),
            (r"one by (?P<v>\d\.\d{3})", "best_ratio"),
            (r"so (?P<v>\d\.\d{3}) says the author", "best_ratio"),
            (r"whether (?P<v>\d\.\d{3}) is a property", "best_ratio"),
            (r"per_channel_moments by (?P<v>\d\.\d{3})", "best_ratio"),
        )

    def _quantities(self) -> dict[str, float]:
        return {"floor": self.floor, "unit_ratio": 1.0,
                "materiality": self.data["materiality_ratio"],
                "best_ratio": max(self._material().values())}

    def test_every_decimal_is_checked_and_belongs_to_the_task_it_sits_beside(self) -> None:
        owners, quantities = self._owners(), self._quantities()
        scanned = dict(self.prose, **{"003": self.three_text,
                                      "index-003": self.index_three})
        for label, text in scanned.items():
            covered: set[tuple[int, int]] = set()
            for pattern, where in self._constructions():
                for match in re.finditer(pattern, text):
                    covered.add(match.span("v"))
                    with self.subTest(record=label, phrase=match.group(0)):
                        self.assertEqual(float(match.group("v")), quantities[where],
                                         f"{label}: {match.group(0)!r} must state the "
                                         f"{where} the table gives")
            # "A and B are X and Y" -- the one shape the nearest-name rule gets wrong,
            # and the one a reviewer used to swap two ratios without failing anything.
            for match in re.finditer(r"(?P<m1>\d+\.\d{2}) MiB and (?P<m2>\d+\.\d{2}) MiB",
                                     text):
                covered.update((match.span("m1"), match.span("m2")))
                with self.subTest(record=label, phrase=match.group(0)):
                    for group in ("m1", "m2"):
                        self.assertIn(float(match.group(group)), dict(self._sweep()),
                                      "not a size this sweep sampled")
            for match in re.finditer(r"(?:move|at) (?P<a>\d\.\d{3}) and (?P<b>\d\.\d{3}) MiB",
                                     text):
                covered.update((match.span("a"), match.span("b")))
                with self.subTest(record=label, phrase=match.group(0)):
                    self.assertEqual(
                        [float(match.group("a")), float(match.group("b"))],
                        [round(_footprint_mib("silu"), 3),
                         round(_footprint_mib("softmax_backward"), 3)],
                        "the bracketing pair is silu against softmax_backward")
            for match in re.finditer(r"(?P<a>[a-z_]+) and (?P<b>[a-z_]+) are "
                                     r"(?P<x>\d\.\d{3}) and (?P<y>\d\.\d{3})", text):
                covered.update((match.span("x"), match.span("y")))
                for task, value in ((match.group("a"), match.group("x")),
                                    (match.group("b"), match.group("y"))):
                    if task in self.rows:
                        with self.subTest(record=label, task=task):
                            self.assertEqual(
                                float(value),
                                round(self.rows[task]["baseline"]
                                      / self.rows[task]["candidate"], 3),
                                f"{label}: {match.group(0)!r} pairs them the wrong way")
            # Sweep readings quoted away from the sweep site, each against the recorded
            # curve, and the grid step the record derives from the medians.
            sweep = dict(self._sweep())
            quantum_us = self.data["timer_quantum_ns"] / 1000
            for pattern, order in (
                (r"(?P<mib>\d+\.\d+) MiB reads (?P<us>\d+\.\d+) us", None),
                (r"(?P<us>\d+\.\d+) us at (?P<mib>\d+\.\d+) MiB", None),
            ):
                for match in re.finditer(pattern, text):
                    covered.update((match.span("mib"), match.span("us")))
                    with self.subTest(record=label, phrase=match.group(0)):
                        self.assertIn(float(match.group("mib")), sweep)
                        self.assertEqual(sweep[float(match.group("mib"))],
                                         float(match.group("us")),
                                         "that is not what the sweep read at that size")
            for pattern in (r"(?P<a>\d+\.\d+) us against (?P<b>\d+\.\d+) us",):
                for match in re.finditer(pattern, text):
                    covered.update((match.span("a"), match.span("b")))
                    with self.subTest(record=label, phrase=match.group(0)):
                        for group in ("a", "b"):
                            self.assertIn(float(match.group(group)), set(sweep.values()),
                                          "not a reading this sweep took")
            for match in re.finditer(r"which is (?P<n>\d+) grid steps", text):
                covered.add(match.span("n"))
                step = self.data["timer_quantum_ns"]
                residue = round(self.floor * 1e6) % step
                with self.subTest(record=label, phrase=match.group(0)):
                    self.assertEqual(int(match.group("n")),
                                     (round(self.floor * 1e6) - residue) // step,
                                     "that is not where the floor sits on the grid")
            for match in re.finditer(r"(?P<a2>\d+\.\d+) us against (?P<b2>\d+\.\d+) us, "
                                     r"(?P<n>[\w-]+) steps", text):
                step = self.data["timer_quantum_ns"]
                spread = round((float(match.group("b2")) - float(match.group("a2"))) * 1000)
                with self.subTest(record=label, phrase=match.group(0)):
                    self.assertEqual(NUMBER[match.group("n").lower()], spread // step,
                                     "that is not how many steps apart those readings are")
            for pattern in (r"these kernels cost (?P<v>\d+\.\d+) us",):
                for match in re.finditer(pattern, text):
                    covered.add(match.span("v"))
                    with self.subTest(record=label, phrase=match.group(0)):
                        self.assertEqual(float(match.group("v")), self.floor * 1000,
                                         "that is not the floor this table holds")
            for pattern in (r"a difference of (?P<v>\d+\.\d+) us",
                            r"that pair is (?P<v>\d+\.\d+) us",
                            r"observed synthetic-copy difference is (?P<v>\d+\.\d+) us",
                            r"differ by (?P<v>\d+\.\d+) us"):
                for match in re.finditer(pattern, text):
                    covered.add(match.span("v"))
                    with self.subTest(record=label, phrase=match.group(0)):
                        self.assertEqual(float(match.group("v")), quantum_us,
                                         "the record's own grid step is this")
            # The footprint sweep site: its numbers are checked in order by their own
            # test, so they are covered here rather than judged by nearest task name.
            for sweep in re.finditer(r"the footprint sweep on the same device:[^\n]*", text):
                for number in re.finditer(r"(?<![\w.:])\d+\.\d+(?![\w])", sweep.group()):
                    covered.add((sweep.start() + number.start(),
                                 sweep.start() + number.end()))
            for pattern in self._PAIRS:
                for match in pattern.finditer(text):
                    # Only a pair that actually names a task in the table is exempt here,
                    # because only then does the pair rule below check it. Exempting every
                    # match let "that run reads 0.006719 against 0.006399" through: the
                    # captured word was "reads", the pair rule skipped it as not-a-task,
                    # and these spans were already marked covered.
                    if match.group("task") not in self.rows:
                        continue
                    for group in ("c", "b", "r"):
                        if match.groupdict().get(group):
                            covered.add(match.span(group))
            for match in re.finditer(r"(?<![\w.:])\d+\.\d+(?![\w])", text):
                if match.span() in covered:
                    continue
                value = float(match.group())
                if value in DECLARED:
                    continue
                holders = owners.get(value)
                with self.subTest(record=label, literal=match.group(),
                                  near=text[max(0, match.start() - 50):match.end() + 20]):
                    self.assertIsNotNone(
                        holders,
                        f"{label} states {match.group()}, which is not a median, a ratio, "
                        "a derived footprint or a listed constant. Nothing checks it.")
                    # Association, for shared values as well as unique ones. Checking only
                    # the unique ones left the floor -- sixteen tasks hold it -- admissible
                    # anywhere, so a wholly invented sentence quoting it passed.
                    nearest = self._nearest_task(text, match.start())
                    self.assertIsNotNone(
                        nearest,
                        f"{label} states {match.group()} with no task named near it and no "
                        "construction covering it, so nothing says whose number it is")
                    self.assertIn(
                        nearest, holders,
                        f"{label}: {match.group()} belongs to {sorted(holders)}, but the "
                        f"task named beside it is {nearest}")

    def test_a_count_scoped_to_one_revision_is_that_revisions_count(self) -> None:
        """Not "a number the campaigns group is allowed": the count of rows at 4278caf2 is
        nine, at 0970a36e twelve, at 5151954e six, and a union of the three let any one
        stand for any other."""
        for label, text in self.prose.items():
            for match in re.finditer(r"the (?P<n>[\w-]+) campaigns at (?P<rev>[0-9a-f]{8})",
                                     text):
                revision = "open-cake-ir@" + match.group("rev")
                expected = sum(1 for row in self.rows.values()
                               if row["compiler_revision"] == revision)
                with self.subTest(record=label, phrase=match.group(0)):
                    self.assertIn(revision, set(self.data["campaign_revisions"].values()),
                                  "no campaign ran at that commit")
                    self.assertEqual(NUMBER[match.group("n").lower()], expected,
                                     f"{expected} rows carry {revision}")

    def _three_tasks(self) -> set[str]:
        """F-2026-09-18-003's eight tasks, read from the record rather than kept here.
        Two earlier copies of this set were hand-written in this file, which is the
        defect the file exists for."""
        retained = self.three["evidence"]["retained"]
        listed = retained.split("per-task logs:")[1].split("at four turns")[0]
        tasks = {name for name in re.findall(r"[a-z_]+", listed)
                 if name in self.data["campaign_revisions"]
                 or name in {w.rsplit("-", 2)[0]
                             for w in self.data["campaign_revisions"]}}
        self.assertEqual(len(tasks), 8, f"F-003 names {sorted(tasks)}, not eight tasks")
        return tasks

    def test_f003s_eight_tasks_ran_on_the_day_its_sweeps_ran(self) -> None:
        """Being a real task is not enough: swapping attention_decode for prelu kept the
        list eight names long and every name a task this host ran. Both of F-003's sweeps
        are on 2026-09-18, and prelu has campaigns only on the 17th."""
        campaigns = self.data["campaign_revisions"]
        days = set(re.findall(r"sweep-(\d{8})-", self.three["evidence"]["retained"]))
        self.assertTrue(days, "F-003 names no sweep this check can date")
        for task in self._three_tasks():
            ran_on = {w.split("-")[-2] for w in campaigns if w.rsplit("-", 2)[0] == task}
            with self.subTest(task=task):
                self.assertTrue(
                    ran_on & days,
                    f"F-003 lists {task} among the tasks its sweeps ran, but this host ran "
                    f"it only on {sorted(ran_on)}, not {sorted(days)}")

    def test_f003_claims_about_this_table_are_the_tables(self) -> None:
        """F-2026-09-18-003 is scoped: its turn-budget counts come from its own sweep
        ledgers, which this table does not hold, and hand-writing them here a second time
        is the defect this file exists for. Every claim it makes about THIS table is
        checked, and its decimals go through the same rule as the others."""
        counts = self._counts()
        for label, text in (("003", self.three_text), ("index-003", self.index_three)):
            for pattern, group in (
                (r"table records ([\w-]+) qualified runs", "runs"),
                (r"(\d+) at six", "tasks"),
                (r"twenty-five at four, ([\w-]+) at six", "tasks"),
                (r"can qualify, ([\w-]+) do", "tasks"),
                (rf"out of ({_NUMBER_WORDS})\b", "campaign_tasks"),
                (r"out of (\d+)\b", "campaign_tasks"),
                (r"of ([\w-]+) tasks that can qualify", "can_qualify"),
            ):
                for match in re.finditer(pattern, text, re.I):
                    token = match.group(1).lower()
                    value = int(token) if token.isdigit() else NUMBER.get(token)
                    with self.subTest(record=label, phrase=match.group(0)):
                        self.assertEqual(value, counts[group],
                                         f"{label} says {match.group(0)!r}; the table "
                                         f"gives {counts[group]} for {group}")
        self.assertTrue(self._campaign_revisions(self._three_tasks()),
                        "no campaign revision found for F-003's tasks")

    def test_a_quoted_start_time_is_the_workspace_it_names(self) -> None:
        """F-2026-09-18-005 dates its campaigns to argue about a stamp. The times are in
        the workspace names the table already holds, so they are read from there."""
        for label in ("002", "005"):
            for match in re.finditer(r"(?P<task>[a-z_]+) at (?P<h>\d{2}):(?P<m>\d{2})\b",
                                     self.prose[label]):
                task = match.group("task")
                if task not in self.rows:
                    continue
                started = self.rows[task]["workspace"].rsplit("-", 1)[-1]
                with self.subTest(record=label, phrase=match.group(0)):
                    self.assertEqual(match.group("h") + match.group("m"), started[:4],
                                     f"{task} started at {started[:2]}:{started[2:4]}")

    def test_a_revision_listed_in_prose_is_the_set_the_rows_carry(self) -> None:
        """The stamp fields are derived, but the prose repeats the commits beside them and
        nothing looked at those: swapping one for a retracted stamp passed."""
        rows = {r["compiler_revision"].split("@")[1] for r in self.rows.values()}
        for label in ("002", "005", "003"):
            text = self.prose.get(label) or self.three_text
            for match in re.finditer(r"ran at three revisions -- ([0-9a-f]{8}, [0-9a-f]{8} "
                                     r"and [0-9a-f]{8})", text):
                with self.subTest(record=label, phrase=match.group(0)):
                    self.assertEqual(set(re.findall(r"[0-9a-f]{8}", match.group(1))), rows)

    #: Spans a numeric construction above already judged, so the completeness rules do not
    #: report them as unchecked. Kept beside those constructions deliberately: adding one
    #: without adding it here makes the completeness rule fail, not pass.
    STEP_PHRASES = (r"which is (\d+) grid steps",
                    r"us, ([\w-]+) steps",
                    r"the ([\w-]+) at the floor are all",
                    r"the ([\w-]+) campaigns at [0-9a-f]{8}")

    def _step_spans(self, text: str) -> set[tuple[int, int]]:
        """Every span a construction above already judged. The noun rule and the two
        completeness rules all consult this, so a quantity checked against a narrower group
        than its noun's is not re-judged against the wider one, and a quantity no
        construction claims is still reported as unchecked."""
        spans = {match.span(1) for pattern in self.STEP_PHRASES
                 for match in re.finditer(pattern, text)}
        spans |= {match.span(1) for pattern, _ in self.BARE
                  for match in re.finditer(pattern, text, re.I)}
        return spans

    def test_no_group_sized_digit_goes_unchecked(self) -> None:
        """Counts written as digits rather than words. Inserting "Only 9 of the 13 were
        checked twice" into F-2026-09-18-002 passed every other rule here."""
        nouns = tuple(self._noun_groups())
        allowed = set(self._counts().values()) | set(RETRACTED_COUNTS) | {1, 2, 4, 5, 6, 8}
        for label, text in self.prose.items():
            covered = self._step_spans(text)
            for noun in nouns:
                for match in re.finditer(rf"((?:[\w-]+\s+){{0,3}}){noun}\b", text):
                    start = match.start(1)
                    for token in re.finditer(r"[\w-]+", match.group(1)):
                        covered.add((start + token.start(), start + token.end()))
            for match in re.finditer(r"(?<![\w.:@-])\d{1,2}(?![\w.:@-])", text):
                if match.span() in covered or int(match.group()) in allowed:
                    continue
                self.fail(f"{label} states the bare number {match.group()!r} where nothing "
                          f"checks it: ...{text[max(0, match.start() - 60):match.end() + 30]!r}")

    def _sweep(self):
        return [(float(a), float(b))
                for a, b in self.data["footprint_sweep"]["pairs_mib_us"]]

    def test_f003s_six_turn_workspaces_are_workspaces_this_host_holds(self) -> None:
        """Its six-turn site names three runs by timestamp; the table holds every campaign
        on the host, so the names are read against it rather than believed."""
        for match in re.finditer(r"(?P<task>[a-z_]+) (?:qualified )?at (?P<stamp>\d{6})\b",
                                 self.three_text):
            task, stamp = match.group("task"), match.group("stamp")
            if task not in {w.rsplit("-", 2)[0] for w in self.data["campaign_revisions"]}:
                continue
            with self.subTest(phrase=match.group(0)):
                self.assertIn(f"{task}-20260918-{stamp}", self.data["campaign_revisions"],
                              f"no campaign {task}-20260918-{stamp} on this host")

    def test_the_footprint_sweep_is_quoted_in_the_order_it_was_measured(self) -> None:
        sweep = [s for s in self.two["evidence"]["sites"] if "footprint sweep" in s]
        self.assertEqual(len(sweep), 1)
        numbers = [float(n) for n in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", sweep[0])]
        for mib, us in self._sweep():
            with self.subTest(mib=mib):
                self.assertIn(mib, numbers)
                self.assertIn(us, numbers)
                self.assertLess(numbers.index(mib), numbers.index(us) + 1)

    def test_the_sweep_is_named_as_a_probe_and_not_as_campaign_evidence(self) -> None:
        """It backs part of F-2026-09-18-002's argument and is the one quantity here that
        no sealed workspace holds, so the record must not read it as a task measurement."""
        sweep = self.data["footprint_sweep"]
        self.assertFalse(sweep["is_campaign_evidence"])
        self.assertTrue((ROOT / "tools/probe_dcu_shape_floor.py").is_file())
        self.assertIn("tools/probe_dcu_shape_floor.py", sweep["source"])
        self.assertIn("TRANSCRIBED", sweep["source"])

    def test_the_timer_grid_is_what_the_medians_say_it_is(self) -> None:
        """Every distinct median is an exact multiple of one step, offset by a constant.
        The step is what decides whether a shape can be measured at all, and this record
        argued from bytes for three versions without it."""
        from math import gcd
        values = sorted({round(r[k] * 1e6) for r in self.rows.values()
                         for k in ("candidate", "baseline")})
        step = 0
        for a in values:
            for b in values:
                step = gcd(step, abs(a - b))
        self.assertEqual(step, self.data["timer_quantum_ns"])
        self.assertEqual(len({v % step for v in values}), 1,
                         "the medians do not share one residue, so this is not a grid")
        text = _text(self.two)
        self.assertIn(f"{step} ns", text,
                      "F-2026-09-18-002 does not state the grid its medians lie on")
        # The pair the withdrawn argument rested on is one step apart in the sweep.
        sweep = dict(self._sweep())
        self.assertAlmostEqual(round((sweep[1.5] - sweep[1.0]) * 1000), step, places=6)

    def test_the_quoted_footprints_are_the_ones_their_contracts_give(self) -> None:
        """Derived from each task's own Workload Contract, not transcribed: tensors,
        extents and dtypes give the bytes, so a swapped pair cannot agree with both."""
        text = _text(self.two)
        for task in FOOTPRINT_TASKS:
            mib = round(_footprint_mib(task), 3)
            with self.subTest(task=task):
                self.assertRegex(
                    text, rf"{re.escape(task)}[^.]{{0,40}}?{mib:.3f}|{mib:.3f}[^.]{{0,30}}?"
                    rf"{re.escape(task)}",
                    f"F-2026-09-18-002 does not put {task} beside its {mib:.3f} MiB")

    # -- counts, against the group they count ----------------------------------------
    def _noun_groups(self) -> dict[str, set[int]]:
        floor, separated = len(self._at_floor()), len(self._separated())
        tasks, runs = len(self.rows), self.data["qualified_runs"]
        ahead, material = len(self._ahead()), len(self._material())
        revisions = len(self._revisions(self.rows))
        per_revision = {sum(1 for r in self.rows.values()
                            if r["compiler_revision"] == revision)
                        for revision in self._revisions(self.rows)}
        return {
            "operators": {floor}, "results": {floor, material},
            "candidates": {material, ahead, ahead - material},
            "campaigns": {tasks, material, revisions},
            "tasks": {floor, tasks, len(self.reruns)},
            "runs": {floor, separated, runs},
            "rows": {tasks}, "revisions": {revisions},
        }

    def _sentence(self, text: str, at: int) -> str:
        start = max(text.rfind(". ", 0, at), text.rfind("\n", 0, at)) + 1
        end = text.find(". ", at)
        return text[start:end if end != -1 else len(text)].lower()

    def _retracting(self, text: str, at: int) -> bool:
        sentence = self._sentence(text, at)
        return any(marker in sentence for marker in RETRACTION)

    def test_every_counted_noun_names_a_quantity_derived_for_that_noun(self) -> None:
        groups = self._noun_groups()
        for label, text in self.prose.items():
            judged = self._step_spans(text)
            for noun, allowed in groups.items():
                for match in re.finditer(rf"((?:[\w-]+\s+){{0,3}}){noun}\b", text):
                    # A count a construction above already judged against a narrower group
                    # than this noun's -- a per-revision count, say -- is not re-judged here
                    # against the union, which is what let one group's size stand for
                    # another's.
                    if any(start >= match.start(1) and end <= match.end(1)
                           for start, end in judged):
                        continue
                    tokens = match.group(1).lower().split()
                    named = [t for t in tokens
                             if (t in NUMBER and t != "one") or t == "hundred"
                             or t.isdigit()]
                    if not named:
                        continue
                    token = named[-1]
                    with self.subTest(record=label, phrase=match.group(0).strip()):
                        self.assertNotEqual(token, "hundred",
                                            "an unrecognised quantity stands before a "
                                            "noun this record counts")
                        value = int(token) if token.isdigit() else NUMBER[token]
                        if value in allowed:
                            continue
                        self.assertTrue(
                            value in RETRACTED_COUNTS
                            and self._retracting(text, match.start()),
                            f"{label} says {match.group(0).strip()!r}; the table gives "
                            f"{sorted(allowed)} for {noun}, and this sentence does not "
                            "mark the number as one the record retracts")

    #: Quantity phrases carrying no noun, each naming the group it refers to.
    BARE = (
        (r"the count is ([\w-]+)", "floor"),
        (r"the ([\w-]+) at the floor are all", "floor"),
        (r"those ([\w-]+) runs:", "floor"),
        (r"aggregates ([\w-]+) campaigns", "tasks"),
        (r"([\w-]+) results out of", "material"),
        (r"([\w-]+) are ahead by enough", "material"),
        (r"([\w-]+) of them across", "campaigns"),
        (r"across ([\w-]+) tasks", "campaign_tasks"),
        (r"the ([\w-]+) tasks that ran", "campaign_tasks"),
        (r"tasks and ([\w-]+) revisions", "all_revisions"),
        (r"so ([\w-]+) revisions in it are used by no row", "unused_revisions"),
        (r"the ([\w-]+) that qualified, the", "tasks"),
        (r"the ([\w-]+) that never did", "never_qualified"),
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
    PARTITION = r"([\w-]+) plus ([\w-]+) is that ([\w-]+)"

    def _counts(self) -> dict[str, int]:
        campaigns = self.data["campaign_revisions"]
        row_revisions = self._revisions(self.rows)
        return {"floor": len(self._at_floor()), "separated": len(self._separated()),
                "tasks": len(self.rows), "runs": self.data["qualified_runs"],
                "material": len(self._material()),
                # Everything the host ran, which is wider than the rows and is what a
                # record citing unqualified runs has to be checked against.
                "campaigns": len(campaigns),
                "campaign_tasks": len({w.rsplit("-", 2)[0] for w in campaigns}),
                "all_revisions": len(set(campaigns.values())),
                "unused_revisions": len(set(campaigns.values()) - row_revisions),
                "never_qualified": len({w.rsplit("-", 2)[0] for w in campaigns}
                                       - set(self.rows)),
                # Every task that ran except the one blocked by a tolerance no budget
                # reaches (gemm_bias, F-2026-09-18-004).
                "can_qualify": len({w.rsplit("-", 2)[0] for w in campaigns}) - 1}

    def test_every_bare_quantity_phrase_names_the_count_the_table_gives(self) -> None:
        counts = self._counts()
        for label, text in self.prose.items():
            for pattern, group in self.BARE:
                for match in re.finditer(pattern, text, re.I):
                    word = match.group(1).lower()
                    with self.subTest(record=label, phrase=match.group(0)):
                        self.assertIn(word, NUMBER, "not a count word")
                        self.assertEqual(NUMBER[word], counts[group],
                                         f"{label} says {match.group(0)!r}; the table "
                                         f"gives {counts[group]} for the {group} group")
            for match in re.finditer(self.PARTITION, text, re.I):
                with self.subTest(record=label, phrase=match.group(0)):
                    self.assertEqual(
                        [NUMBER[w.lower()] for w in match.groups()],
                        [counts["floor"], counts["separated"], counts["tasks"]],
                        f"{label}'s partition sentence does not add up to the table")

    GROUP_WORDS = ("nine", "eleven", "twelve", "thirteen", "fourteen", "twenty-six",
                   "twenty-seven", "twenty-eight", "twenty-nine", "thirty", "sixty-seven")

    def test_no_group_sized_number_goes_unchecked(self) -> None:
        nouns = tuple(self._noun_groups())
        for label, text in self.prose.items():
            covered = self._step_spans(text)
            for pattern, _ in self.BARE:
                for match in re.finditer(pattern, text, re.I):
                    covered.add(match.span(1))
            for match in re.finditer(self.PARTITION, text, re.I):
                covered.update(match.span(i) for i in (1, 2, 3))
            for noun in nouns:
                for match in re.finditer(rf"((?:[\w-]+\s+){{0,3}}){noun}\b", text):
                    start = match.start(1)
                    for token in re.finditer(r"[\w-]+", match.group(1)):
                        covered.add((start + token.start(), start + token.end()))
            for word in self.GROUP_WORDS:
                for match in re.finditer(rf"(?<![\w-]){word}(?![\w-])", text, re.I):
                    if match.span() in covered:
                        continue
                    if (NUMBER[word] in RETRACTED_COUNTS
                            and self._retracting(text, match.start())):
                        continue
                    self.fail(
                        f"{label} states {word!r} where nothing checks it: "
                        f"...{text[max(0, match.start() - 60):match.end() + 30]!r}... "
                        "Give it a noun this file counts, or add its construction to BARE.")

    # -- the live counts, in the records' own phrases ---------------------------------
    def test_each_record_states_the_count_the_table_gives(self) -> None:
        counts = self._counts()
        floor, separated = WORDS[counts["floor"]], WORDS[counts["separated"]]
        tasks, runs = WORDS[counts["tasks"]], WORDS[counts["runs"]]
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
            ("002 enumeration heading", rf"{separated} runs separated:", self.prose["002"]),
            ("002 recount", rf"gives {separated} separated", self.two["observation"]),
            ("002 arithmetic", rf"{floor} plus {separated} is that {tasks}",
             self.two["observation"]),
            ("005 title", rf"{material} DCU campaigns", self.five["title"]),
            ("005 confirmed", rf"{material} campaigns confirmed", self.five["observation"]),
            ("005 nominally ahead", rf"{ahead} candidates are nominally ahead",
             self.five["observation"]),
            ("005 cites the floor count", rf"{floor} other tasks report", self.prose["005"]),
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
                self.assertEqual(self._named(names, other), set(),
                                 f"the {label} enumeration names a task from the other group")

    def test_nothing_is_reported_absent_that_the_table_holds(self) -> None:
        live = r"(?:is|are)(?: the one| still)? absent|counted in neither group"
        for label in ("002", "index-002"):
            for task in self.rows:
                self.assertNotRegex(
                    self.prose[label], rf"{re.escape(task)}[^.]{{0,60}}(?:{live})",
                    f"{task} is in the table and cannot also be reported absent")

    # -- the quoted pairs -------------------------------------------------------------
    _PAIRS = (
        re.compile(r"(?P<task>[a-z_]+) is candidate (?P<c>\d\.\d{6}) against baseline "
                   r"(?P<b>\d\.\d{6})(?: \((?P<r>\d\.\d+)\))?"),
        re.compile(r"(?P<task>[a-z_]+): candidate (?P<c>\d\.\d{6})(?: ms)?, baseline "
                   r"(?P<b>\d\.\d{6})(?:, speedup (?P<r>\d\.\d+))?"),
        re.compile(r"(?P<task>[a-z_]+) \((?P<c>\d\.\d{6})/(?P<b>\d\.\d{6})\)"),
        re.compile(r"(?P<task>[a-z_]+) (?P<c>\d\.\d{6}) against (?P<b>\d\.\d{6})"
                   r"(?: \((?P<r>\d\.\d+)\))?"),
    )
    QUOTED_PAIRS = 9

    def test_every_quoted_median_pair_and_ratio_matches_the_table(self) -> None:
        rerun_pairs = {(w["candidate"], w["baseline"])
                       for runs in self.reruns.values() for w in runs}
        found = 0
        for label in ("002", "005"):
            for pattern in self._PAIRS:
                for match in pattern.finditer(self.prose[label]):
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

    def test_the_winners_and_the_excluded_are_the_tables(self) -> None:
        for task in self._material():
            with self.subTest(winner=task):
                self.assertIn(task, self.prose["005"])
        for task in set(self._ahead()) - set(self._material()):
            with self.subTest(excluded=task):
                self.assertIn(task, self.prose["005"],
                              "005 does not name a nominal winner it excludes")

    # -- the row-selection rule, recomputed ------------------------------------------
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
                                    f"{task}'s kept run is not the first")

    def test_a_dropped_rerun_that_reads_better_is_named_where_it_matters(self) -> None:
        for task, runs in self.reruns.items():
            kept = self.rows[task]
            for run in runs:
                ratio = run["baseline"] / run["candidate"]
                if (ratio > kept["baseline"] / kept["candidate"]
                        and ratio > self.data["materiality_ratio"]):
                    with self.subTest(task=task, workspace=run["workspace"]):
                        self.assertIn(run["workspace"], self.prose["005"],
                                      f"{task}'s discarded re-run clears materiality and "
                                      "is not named in F-2026-09-18-005")

    # -- the stamps, both ways --------------------------------------------------------
    def test_each_record_stamps_the_revisions_its_campaigns_actually_ran_at(self) -> None:
        """-002 was stamped at a commit no campaign it cites ran at; -005 at one
        postdating all three of its own, inside the sentence correcting the previous wrong
        stamp; -003 at a third, 1h41m after the sweep it cites began. Compared both ways:
        a stamp naming a revision none of its campaigns ran at is as wrong as one missing
        a revision they did, and an earlier version only checked the second."""
        for label, record, expected in (
            ("F-2026-09-18-002", self.two, self._revisions(self.rows)),
            ("F-2026-09-18-005", self.five,
             self._revisions({"per_channel_moments", "momentum_sgd", "pairwise_sqdist"})),
            # -003 is about turn budget, so it cites each task's runs at two, four and six
            # turns -- including those that qualified nothing and so have no row here.
            ("F-2026-09-18-003", self.three,
             self._campaign_revisions(self._three_tasks())),
        ):
            stamped = set(re.findall(r"open-cake-ir@[0-9a-f]{8}",
                                     record["compiler_revision_id"]))
            with self.subTest(record=label):
                self.assertEqual(stamped, expected,
                                 f"{label} stamps {sorted(stamped)}; its campaigns ran at "
                                 f"{sorted(expected)}")
                executors = set(re.findall(r"gfx938@([0-9a-f]{8})",
                                           record["executor_revision"]))
                self.assertEqual(executors, {r.split("@")[1] for r in expected},
                                 f"{label}'s executor_revision is not the same set")

    #: Commits these records name while narrating a stamp they retract. Each was a live
    #: stamp on some record; none may be presented as a commit a campaign ran at.
    RETRACTED_STAMPS = ("3fa16b8f", "2e901f53", "f91a862a", "2ab897a5", "9b51e3cd",
                        "e0682e1b")

    def _commit_time(self, sha: str) -> str:
        import subprocess
        out = subprocess.run(["git", "log", "-1", "--format=%ad",
                              "--date=format:%H:%M", sha],
                             cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else ""

    def test_every_commit_named_in_prose_is_one_these_records_can_name(self) -> None:
        """The stamp fields are derived, but the prose repeats commits beside them and
        nothing read those: swapping the commit in "a commit no campaign in the table ran
        at" for one that nine rows carry passed, leaving the sentence self-contradicting."""
        ran_at = {r.split("@")[1] for r in self.data["campaign_revisions"].values()}
        for label, text in dict(self.prose, **{"003": self.three_text}).items():
            for match in re.finditer(r"(?<![\w])(?P<sha>[0-9a-f]{8})(?![\w])", text):
                sha = match.group("sha")
                if sha.isdigit():          # a date, not a commit: 20260918 is eight hex
                    continue
                with self.subTest(record=label, sha=sha,
                                  near=text[max(0, match.start() - 60):match.end() + 40]):
                    self.assertTrue(
                        sha in ran_at or sha in self.RETRACTED_STAMPS,
                        f"{label} names commit {sha}, which no campaign ran at and which "
                        "is not one of the stamps these records retract")
            for match in re.finditer(r"open-cake-ir@(?P<sha>[0-9a-f]{8}), a commit no "
                                     r"campaign in the table ran at", text):
                with self.subTest(record=label, phrase=match.group(0)):
                    self.assertNotIn(match.group("sha"), ran_at,
                                     "that commit is one campaigns did run at")

    def test_a_commit_time_quoted_in_prose_is_the_time_git_gives(self) -> None:
        """The stamp arguments rest on these times and nothing read them: `2ab897a5 is
        07:30` could become `08:30`, and `2e901f53 is 08:10` could become `09:10`."""
        for label, text in dict(self.prose, **{"003": self.three_text}).items():
            for match in re.finditer(r"(?P<sha>[0-9a-f]{8}) is (?P<t>\d{2}:\d{2})", text):
                actual = self._commit_time(match.group("sha"))
                if not actual:
                    continue
                with self.subTest(record=label, phrase=match.group(0)):
                    self.assertEqual(match.group("t"), actual,
                                     f"git says {match.group('sha')} is {actual}")

    def test_a_campaign_named_with_a_revision_is_the_one_the_table_records(self) -> None:
        """F-2026-09-18-005 names each campaign's commit in prose. The table holds that
        mapping per workspace, so the prose is read against it rather than believed."""
        revisions = self.data["campaign_revisions"]
        for label, text in dict(self.prose, **{"003": self.three_text}).items():
            for match in re.finditer(r"(?P<ws>[a-z_]+-\d{8}-\d{6}) at "
                                     r"(?P<rev>open-cake-ir@[0-9a-f]{8})", text):
                with self.subTest(record=label, phrase=match.group(0)):
                    self.assertIn(match.group("ws"), revisions,
                                  "no such campaign on this host")
                    self.assertEqual(revisions[match.group("ws")], match.group("rev"),
                                     "that campaign ran at a different commit")

    def test_every_row_names_the_run_receipt_and_revision_it_came_from(self) -> None:
        for task, row in self.rows.items():
            with self.subTest(task=task):
                self.assertTrue(row["workspace"].startswith(f"{task}-"), row["workspace"])
                self.assertRegex(row["receipt"], r"^objects/sha256/[0-9a-f]{2}/[0-9a-f]{64}$")
                self.assertRegex(row["compiler_revision"], r"^open-cake-ir@[0-9a-f]{8}$")
                self.assertIn(row["workspace"], self.data["campaign_revisions"])
                self.assertEqual(self.data["campaign_revisions"][row["workspace"]],
                                 row["compiler_revision"])
        workspaces = [r["workspace"] for r in self.rows.values()]
        self.assertEqual(len(set(workspaces)), len(workspaces))

    def test_the_table_says_how_many_runs_and_how_many_tasks(self) -> None:
        self.assertEqual(self.data["qualified_tasks"], len(self.rows))
        self.assertEqual(self.data["qualified_runs"],
                         len(self.rows) + sum(len(r) for r in self.reruns.values()))
        self.assertGreater(self.data["qualified_runs"], self.data["qualified_tasks"])

    def test_the_table_names_the_instrument_and_says_which_fields_it_owns(self) -> None:
        """`collected` claimed the tool 'regenerates every field below' while it emitted
        ten keys of fifteen, so running it with --out would have deleted five."""
        tool = self.data["generated_by"]
        self.assertTrue((ROOT / tool).is_file(), f"{tool} is named but not in the checkout")
        self.assertIn(tool, self.data["collected"])
        owned = set(self.data["generated_fields"])
        self.assertIn("generated_fields", owned, "the list must include itself")
        prose = sorted(set(self.data) - owned)
        self.assertEqual(prose, ["collected", "footprint_sweep", "receipt_paths",
                                 "row_selection", "what"])
        for field in prose:
            self.assertIn(field, self.data["collected"],
                          "collected must name every field the tool does not regenerate")
        self.assertNotIn("regenerates every field", self.data["collected"])

    def test_every_campaign_revision_entry_is_shaped_like_what_it_claims(self) -> None:
        """Only entries that were also a row's workspace were checked, so the rest -- the
        majority -- could be deleted or given a made-up commit and nothing noticed. They
        are what a record citing an unqualified run is read against."""
        campaigns = self.data["campaign_revisions"]
        self.assertGreater(len(campaigns), len(self.rows),
                           "this table is meant to be wider than the rows")
        for workspace, revision in campaigns.items():
            with self.subTest(workspace=workspace):
                self.assertRegex(workspace, r"^[a-z_]+-\d{8}-\d{6}$")
                self.assertRegex(revision, r"^open-cake-ir@[0-9a-f]{8}$")
        for row in self.rows.values():
            with self.subTest(workspace=row["workspace"]):
                self.assertEqual(campaigns.get(row["workspace"]), row["compiler_revision"],
                                 "a row and the campaign table disagree about its commit")
        for runs in self.data["excluded_reruns"].values():
            for run in runs:
                with self.subTest(workspace=run["workspace"]):
                    self.assertEqual(campaigns.get(run["workspace"]),
                                     run["compiler_revision"])

    def test_every_campaign_ran_at_a_commit_that_exists_and_predates_it(self) -> None:
        """Shape alone admits any eight hex characters -- `deadbeef` passed. A commit a
        campaign ran at has to exist in this repository and has to be older than the run,
        which is the whole point of ADR 0065's stamp: retained evidence replays against the
        tools that produced it, and tools from the future did not produce anything."""
        import subprocess
        campaigns = self.data["campaign_revisions"]
        committed = {}
        for revision in sorted(set(campaigns.values())):
            sha = revision.split("@")[1]
            result = subprocess.run(["git", "log", "-1", "--format=%ct", sha],
                                    cwd=ROOT, capture_output=True, text=True)
            with self.subTest(revision=revision):
                self.assertEqual(result.returncode, 0,
                                 f"{revision} is not a commit in this repository")
                committed[revision] = int(result.stdout.strip())
        import datetime
        for workspace, revision in sorted(campaigns.items()):
            stamp = datetime.datetime.strptime(workspace.split("-", 1)[1],
                                               "%Y%m%d-%H%M%S").timestamp()
            with self.subTest(workspace=workspace):
                self.assertGreaterEqual(
                    stamp, committed[revision],
                    f"{workspace} started before {revision} existed")

    def test_the_sweep_days_are_the_days_these_campaigns_ran(self) -> None:
        self.assertEqual(sorted(self.data["sweep_days"]),
                         sorted({w.split("-")[-2] for w in self.data["campaign_revisions"]}))

    def test_the_prose_describing_the_row_rule_is_the_rule_enforced(self) -> None:
        """`row_selection` can be inverted to say LAST while the behaviour stays FIRST, and
        the two live in different files, so nothing made them agree."""
        prose = self.data["row_selection"]
        self.assertIn("FIRST qualified run", prose)
        self.assertNotIn("LAST qualified run", prose)
        for task, runs in self.data["excluded_reruns"].items():
            kept = self.rows[task]["workspace"]
            for run in runs:
                with self.subTest(task=task):
                    self.assertLess(kept, run["workspace"],
                                    "the kept row is not the earliest qualified run, so "
                                    "the prose and the table disagree")

    def test_the_two_groups_partition_the_collection(self) -> None:
        self.assertEqual(len(self._at_floor()) + len(self._separated()), len(self.rows))
        self.assertEqual(self._at_floor() & self._separated(), set())

    def test_one_arm_on_the_floor_still_separates(self) -> None:
        for task in ("prelu", "swiglu"):
            with self.subTest(task=task):
                row = self.rows[task]
                self.assertTrue(row["candidate"] == self.floor
                                or row["baseline"] == self.floor)
                self.assertIn(task, self._separated())

    def test_no_retracted_phrasing_stands_as_a_live_claim(self) -> None:
        for retracted in ("all seven passed", "eleven other tasks", "wrong three times",
                          "twenty-six qualified runs", "twenty-seven qualified runs",
                          "softsign the one qualified run absent", "here names 2e901f53",
                          "fourth time this number has moved",
                          "regenerates every field below"):
            for label, text in self.prose.items():
                self.assertNotIn(retracted, text, f"{label} still carries {retracted!r}")


if __name__ == "__main__":
    unittest.main()
