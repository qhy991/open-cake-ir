"""The environment doctor reports; it never repairs, and it never crashes on absence."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

from tools import check_environment

ROOT = Path(__file__).resolve().parents[2]
STATUSES = {"ok", "failed", "unchecked", "unsupported"}


class EnvironmentReportContracts(unittest.TestCase):
    def test_every_kind_reports_rather_than_raising(self):
        for kind in ("metal", "cuda", "amd"):
            with self.subTest(kind=kind):
                checks = check_environment.run(kind, None)
                self.assertTrue(checks)
                for check in checks:
                    self.assertIn(check.status, STATUSES)
                    # A failing or unsupported check has to say what to do about it.
                    if check.status in {"failed", "unsupported"}:
                        self.assertTrue(check.remediation, check.name)

    def test_an_unobservable_precondition_is_unchecked_and_not_a_failure(self):
        # Which targets have an Executor here changes with the inventory, so ask it rather
        # than naming one: a declared target the inventory does not carry cannot have its
        # pinned closure read, and that has to report unchecked rather than pass or fail.
        inventory = json.loads((ROOT / "inventory/EXECUTOR_REVISIONS.json").read_text())
        declared = {path.stem for path in (ROOT / "compiler/targets").glob("*.json")}
        absent = sorted(declared - set(inventory.get("current_by_target", {})))
        if not absent:
            self.skipTest("every declared target currently has an Executor in this checkout")
        statuses = {check.name: check.status for check in check_environment.run("cuda", absent[0])}
        self.assertEqual(statuses.get("current Executor"), "unchecked")

    def test_amd_is_reported_unsupported_until_the_checkout_names_it(self):
        checks = {check.name: check for check in check_environment.run("amd", None)}
        for name in ("AMD Target document", "AMD backend module", "AMD Executor host kind"):
            self.assertIn(name, checks)
            self.assertIn(checks[name].status, {"unsupported", "ok"})

    def test_an_undeclared_target_is_refused_by_name(self):
        done = subprocess.run([sys.executable, str(ROOT / "tools/check_environment.py"),
                               "--kind", "cuda", "--target", "gfx942"],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(done.returncode, 2)
        self.assertIn("this checkout declares", done.stderr)

    def test_the_json_report_states_that_nothing_was_repaired(self):
        done = subprocess.run([sys.executable, str(ROOT / "tools/check_environment.py"),
                               "--kind", "amd", "--json"], capture_output=True, text=True, cwd=ROOT)
        # An unsupported kind exits 3; a report is still produced.
        self.assertEqual(done.returncode, 3)
        report = json.loads(done.stdout)
        self.assertEqual(report["repairs_performed"], 0)
        self.assertEqual(report["counts"]["failed"], 0)

    def test_the_doctor_runs_no_installer(self):
        source = (ROOT / "tools/check_environment.py").read_text()
        for forbidden in ("pip install", "brew install", "xcode-select --install", "apt-get", "sudo"):
            # Remediation strings may name these; no call may execute one.
            for line in source.splitlines():
                if forbidden in line and "subprocess" in line:
                    self.fail(f"the doctor must not execute {forbidden!r}: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
