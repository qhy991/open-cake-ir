"""The guard that stops two checkouts minting one Executor ordinal (F-2026-09-10-012).

The release cycle derives the next id from the largest one visible locally. That is right
only when the checkout has seen every released descriptor, and until this guard existed
nothing established it: two checkouts that had not exchanged descriptors both minted v95
over different bytes. A reserved identity cannot be un-reserved, so the failure has to be
loud before the mint rather than discovered in the ledger afterwards.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "tools/check_executor_ledger.py"
NAME = "open-cake-ir-b200-v{}+{}.json"


def git(root: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True)


class ExecutorLedgerCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "checkout"
        (self.root / "runtime/executors").mkdir(parents=True)
        git(self.root, "init", "-q")
        git(self.root, "config", "user.name", "Ledger Test")
        git(self.root, "config", "user.email", "ledger@example.invalid")
        self.release("90", "a" * 64)
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "released v90")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def release(self, ordinal: str, digest: str) -> Path:
        path = self.root / "runtime/executors" / NAME.format(ordinal, digest)
        path.write_text("{}\n")
        return path

    def check(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(CHECK), "--project-root", str(self.root),
                               *arguments], capture_output=True, text=True)

    def test_an_uncommitted_descriptor_stops_the_mint(self):
        """Until it is committed it exists only here, so a peer will reuse its ordinal."""
        self.release("91", "b" * 64)
        result = self.check("--acknowledge-isolated", "no remote in this fixture")
        self.assertEqual(result.returncode, 3)
        self.assertIn("not committed", result.stderr)
        self.assertIn("v91", result.stderr)
        # Committing it is what makes the checkout's view publishable, and admissible.
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "released v91")
        self.assertEqual(self.check("--acknowledge-isolated", "no remote").returncode, 0)

    def test_a_descriptor_only_the_remote_has_stops_the_mint(self):
        """The exact shape of the collision: a peer released and this checkout cannot see it."""
        remote = Path(self.temporary.name) / "peer"
        subprocess.run(["git", "clone", "-q", str(self.root), str(remote)], capture_output=True)
        git(remote, "config", "user.name", "Peer")
        git(remote, "config", "user.email", "peer@example.invalid")
        (remote / "runtime/executors" / NAME.format("91", "c" * 64)).write_text("{}\n")
        git(remote, "add", "-A")
        git(remote, "commit", "-q", "-m", "peer released v91")
        git(self.root, "remote", "add", "peer", str(remote))
        result = self.check("--remote", "peer", "--branch", git(remote, "rev-parse",
                                                                "--abbrev-ref", "HEAD").stdout.strip())
        self.assertEqual(result.returncode, 3)
        self.assertIn("never seen", result.stderr)
        self.assertIn("v91", result.stderr)

    def test_an_unreachable_remote_is_refused_unless_acknowledged(self):
        """An isolated host cannot establish completeness, and that is not the same as being complete."""
        git(self.root, "remote", "add", "origin", str(Path(self.temporary.name) / "absent"))
        refused = self.check()
        self.assertEqual(refused.returncode, 3)
        self.assertIn("completeness of the released set is unestablished", refused.stderr)
        accepted = self.check("--acknowledge-isolated", "GPU host has no egress")
        self.assertEqual(accepted.returncode, 0)
        # The acceptance says what is being taken on faith, in the release log.
        self.assertIn("UNVERIFIED", accepted.stdout)
        self.assertIn("GPU host has no egress", accepted.stdout)

    def test_a_synchronised_checkout_reports_what_it_can_see(self):
        remote = Path(self.temporary.name) / "peer2"
        subprocess.run(["git", "clone", "-q", str(self.root), str(remote)], capture_output=True)
        git(self.root, "remote", "add", "peer", str(remote))
        branch = git(remote, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        result = self.check("--remote", "peer", "--branch", branch)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("highest released ordinal visible here is v90", result.stdout)


if __name__ == "__main__":
    unittest.main()
