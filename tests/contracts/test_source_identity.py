"""A clean git commit is the whole source identity (ADR 0065)."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from open_cake_ir.source_identity import (
    SourceIdentityError,
    checkout_commit,
    checkout_commit_or_none,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=fixture",
         "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
         *arguments],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


class CheckoutCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve() / "checkout"
        self.root.mkdir()
        _git(self.root, "init", "-q")
        (self.root / ".gitignore").write_text("cache/\n")
        (self.root / "module.py").write_text("VALUE = 1\n")
        _git(self.root, "add", ".")
        _git(self.root, "commit", "-q", "--no-verify", "-m", "fixture")
        self.commit = _git(self.root, "rev-parse", "HEAD")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_a_clean_checkout_is_identified_by_its_head_commit(self) -> None:
        self.assertEqual(checkout_commit(self.root), self.commit)
        self.assertEqual(checkout_commit_or_none(self.root), self.commit)

    def test_a_modified_tracked_file_leaves_no_identity(self) -> None:
        (self.root / "module.py").write_text("VALUE = 2\n")
        with self.assertRaisesRegex(SourceIdentityError, "module.py"):
            checkout_commit(self.root)
        self.assertIsNone(checkout_commit_or_none(self.root))

    def test_an_untracked_file_leaves_no_identity(self) -> None:
        (self.root / "shadow.py").write_text("VALUE = 3\n")
        with self.assertRaisesRegex(SourceIdentityError, "shadow.py"):
            checkout_commit(self.root)

    def test_ignored_files_do_not_change_the_identity(self) -> None:
        (self.root / "cache").mkdir()
        (self.root / "cache" / "entry.bin").write_bytes(b"cache")
        self.assertEqual(checkout_commit(self.root), self.commit)

    def test_a_directory_outside_any_checkout_has_no_identity(self) -> None:
        outside = Path(self.temporary.name).resolve() / "plain"
        outside.mkdir()
        with self.assertRaises(SourceIdentityError):
            checkout_commit(outside)
        self.assertIsNone(checkout_commit_or_none(outside))

    def test_a_subdirectory_does_not_borrow_its_checkout_identity(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        with self.assertRaisesRegex(SourceIdentityError, "not the top"):
            checkout_commit(nested)


if __name__ == "__main__":
    unittest.main()
