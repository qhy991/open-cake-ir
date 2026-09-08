"""Tracked source scanning shares the Evidence rule without printing credentials."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools.check_tracked_secrets import finding_lines, main
from tools.verify_aka_qualified_review_export import _sensitive_findings


class TrackedSecretsTests(unittest.TestCase):
    def test_export_uses_shared_token_detection_and_retains_its_path_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "record.txt").write_bytes(b"github_pat_" + b"synthetic" * 5)
            self.assertEqual(_sensitive_findings(root), [("record.txt", 1, "credential_shape")])
            (root / "record.txt").write_text("reference to auth.json")
            self.assertEqual(_sensitive_findings(root), [("record.txt", 1, "auth_filename")])

    def test_policy_exception_is_an_exact_literal_and_never_a_file_exemption(self):
        name = "src/open_cake_ir/evidence/secret_detection.py"
        marker = b"OPENAI_" + b"API_KEY="
        declaration = b"    " + repr(marker).encode() + b","
        self.assertEqual(finding_lines(name, declaration), [])
        token = b"github_pat_" + b"synthetic" * 5
        self.assertEqual(finding_lines(name, declaration + b"\n" + token), [2])
        self.assertEqual(finding_lines("unreviewed.py", declaration), [1])

    def test_scan_detects_added_credentials_but_does_not_echo_their_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            source = root / "tracked.txt"
            source.write_text("ordinary source\n")
            subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["--project-root", str(root)]), 0)
            token = b"ghp_" + b"synthetic" * 5
            source.write_bytes(b"ordinary source\n" + token)
            with redirect_stderr(StringIO()) as errors:
                self.assertEqual(main(["--project-root", str(root)]), 1)
            self.assertIn("tracked.txt:2", errors.getvalue())
            self.assertNotIn(token.decode(), errors.getvalue())
