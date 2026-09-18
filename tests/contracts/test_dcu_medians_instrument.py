"""The instrument behind the DCU medians table does what the table says it does.

`findings/data/2026-09-18-dcu-confirmatory-medians.json` describes its own provenance, and
that description has been wrong on three successive attempts: two sweeps that cover eight
of its twenty-seven tasks, then an extractor in no commit, then a tool that emitted ten
keys of fifteen while the file said it rewrote the whole document. The third would have
been caught by any test that ran the tool once -- running it with `--out` would have
deleted five fields -- and there was no such test, so this is it.

Nothing here reaches bw1100. The evidence is a fixture: sealed events and receipts in the
shape the real workspaces have, built so the answers are known in advance. What that
establishes is the tool's rules -- which run each row comes from, what it rounds, which
fields it owns, what it refuses. Whether the numbers on the host are these numbers is not
something any test in this checkout can say, and the table's per-row `workspace` and
`receipt` exist so a reader with the machine can check that themselves.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/read_dcu_confirmatory_medians.py"
DATA = ROOT / "findings/data/2026-09-18-dcu-confirmatory-medians.json"
REVISION = "open-cake-ir@abcd1234"


def _workspace(root: Path, name: str, *, candidate=None, baseline=None,
               adherence="adhered", observation="qualified", revision=REVISION):
    """One campaign workspace, in the shape the launcher seals."""
    evidence = root / name / "campaign-evidence"
    events = evidence / "runs" / "run-1" / "events"
    events.mkdir(parents=True)
    objects = []
    if candidate is not None:
        digest = f"{abs(hash(name)):064x}"[:64]
        path = evidence / "objects" / "sha256" / digest[:2] / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "purpose": "confirmatory",
            "timing": {"pooled_medians_ms": {"candidate": candidate, "baseline": baseline},
                       "classification": "close_null"}}))
        objects.append({"role": "evaluation_receipt",
                        "relative_path": f"objects/sha256/{digest[:2]}/{digest}"})
    (events / "0001.json").write_text(json.dumps({
        "kind": "candidate_evaluated",
        "payload": {"objects": objects, "compiler_revision_id": revision}}))
    (events / "0002.json").write_text(json.dumps({
        "kind": "run_terminal",
        "payload": {"protocol_adherence": adherence, "endpoint_observation": observation}}))


def _run(runs: Path) -> dict:
    completed = subprocess.run([sys.executable, str(TOOL), "--runs", str(runs)],
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return json.loads(completed.stdout)


class TheInstrument(unittest.TestCase):
    def _fixture(self, directory: str) -> dict:
        runs = Path(directory)
        # Two runs of one task: the first qualified run is the row, the later one is not.
        _workspace(runs, "alpha-20260917-101010", candidate=0.005439, baseline=0.005439)
        _workspace(runs, "alpha-20260918-202020", candidate=0.001000, baseline=0.009000)
        # A task with one run, separated from the floor.
        _workspace(runs, "beta-20260918-030303", candidate=0.002000, baseline=0.004000)
        # A run that adhered but qualified nothing: no row, but its revision is recorded.
        _workspace(runs, "gamma-20260918-040404", observation="no_qualified_candidate",
                   revision="open-cake-ir@99998888")
        return _run(runs)

    def test_each_row_is_the_first_qualified_run_and_later_ones_are_kept_aside(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = self._fixture(directory)
        self.assertEqual(sorted(document["medians_ms"]), ["alpha", "beta"])
        self.assertEqual(document["medians_ms"]["alpha"]["workspace"],
                         "alpha-20260917-101010")
        self.assertEqual([run["workspace"] for run in document["excluded_reruns"]["alpha"]],
                         ["alpha-20260918-202020"])
        self.assertNotIn("beta", document["excluded_reruns"])

    def test_a_run_that_qualified_nothing_has_no_row_but_is_still_counted_once(self) -> None:
        """F-2026-09-18-003 cites exactly these runs for its stamp, so dropping them would
        leave a record's revisions resting on prose again."""
        with tempfile.TemporaryDirectory() as directory:
            document = self._fixture(directory)
        self.assertNotIn("gamma", document["medians_ms"])
        self.assertIn("gamma-20260918-040404", document["campaign_revisions"])
        self.assertEqual(document["campaign_revisions"]["gamma-20260918-040404"],
                         "open-cake-ir@99998888")
        self.assertEqual(document["qualified_runs"], 3)
        self.assertEqual(document["qualified_tasks"], 2)

    def test_it_lists_the_fields_it_owns_by_deriving_them_from_its_own_output(self) -> None:
        """The check that would have caught the third wrong provenance statement."""
        with tempfile.TemporaryDirectory() as directory:
            document = self._fixture(directory)
        self.assertEqual(document["generated_fields"], sorted(document))
        self.assertIn("generated_fields", document["generated_fields"])

    def test_the_committed_table_is_this_tools_shape(self) -> None:
        """Every field the tool owns is present in the committed file, and every field it
        does not own is prose the file names. Running --out must not silently drop one."""
        table = json.loads(DATA.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            document = self._fixture(directory)
        self.assertEqual(sorted(table["generated_fields"]), sorted(document))
        self.assertTrue(set(document) <= set(table))
        for field in sorted(set(table) - set(document)):
            # Prose, or a hand-kept record of evidence the tool cannot reach: the footprint
            # sweep needs the device, and the file says so where it states its provenance.
            self.assertIsInstance(table[field], (str, dict))
            self.assertIn(field, table["collected"])
            if isinstance(table[field], dict):
                self.assertIn("TRANSCRIBED", table[field]["source"])
                self.assertFalse(table[field]["is_campaign_evidence"])

    def test_it_refuses_evidence_that_names_two_compiler_revisions(self) -> None:
        """A run has one. Reporting the first would put a wrong stamp in a record, which is
        the defect three records on this branch already carry."""
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            _workspace(runs, "delta-20260918-050505", candidate=0.001, baseline=0.001)
            stray = (runs / "delta-20260918-050505" / "campaign-evidence" / "stray.json")
            stray.write_text(json.dumps({"compiler_revision_id": "open-cake-ir@deadbeef"}))
            completed = subprocess.run([sys.executable, str(TOOL), "--runs", str(runs)],
                                       capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Compiler revisions", completed.stderr)

    def test_it_rounds_to_the_six_places_the_assay_reports(self) -> None:
        """The binary tail a float carries is not measurement and must not reach the
        ledger: one row arrived as 0.010398999999999 from the device."""
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory)
            _workspace(runs, "eps-20260918-060606",
                       candidate=0.010398999999999, baseline=0.0100790000001)
            document = _run(runs)
        self.assertEqual(document["medians_ms"]["eps"]["candidate"], 0.010399)
        self.assertEqual(document["medians_ms"]["eps"]["baseline"], 0.010079)


if __name__ == "__main__":
    unittest.main()
