"""Executor identity: one clean commit over one committed host capture (ADR 0065)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.lab import ExecutorRevision  # noqa: E402
from open_cake_ir.lab.executor import HOSTS_DIRECTORY  # noqa: E402
from open_cake_ir.source_identity import checkout_commit_or_none  # noqa: E402

HOSTS = ROOT / HOSTS_DIRECTORY


def _synthetic_cuda_host() -> dict:
    """Schema-only CPU fixture, following the existing paired/Flash fixtures.

    These deliberately nonexistent runtime paths must never be admitted as a
    real host. The fixture tests reference admission and CUDA advisory contracts,
    independently of whichever hardware owns a captured host.
    """
    return {
        "python": {"invocation_path": "/SYNTHETIC/not-an-executable/python",
                   "version": "SYNTHETIC", "resolved_sha256": "0" * 64},
        "packages": {"triton": "SYNTHETIC"},
        "cupti_python": {"site_packages_path": "/SYNTHETIC/no-runtime", "distribution": "SYNTHETIC",
                         "version": "SYNTHETIC", "files": [
                             {"path": "not-a-runtime", "sha256": "0" * 64, "size_bytes": 1}]},
        "flashinfer_helper": {"path": "/SYNTHETIC/not-a-helper", "distribution": "SYNTHETIC",
                              "version": "SYNTHETIC", "sha256": "0" * 64, "size_bytes": 1},
    }


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=fixture",
         "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", *arguments],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


class CommittedHostCaptureTests(unittest.TestCase):
    """What this checkout publishes: one capture per exact target, and nothing else."""

    def test_every_capture_is_named_by_the_target_it_describes(self) -> None:
        captures = sorted(HOSTS.glob("*.json"))
        self.assertTrue(captures, "this checkout publishes no host capture")
        for path in captures:
            with self.subTest(target=path.stem):
                document = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(set(document), {"schema_version", "target", "host_environment"})
                self.assertEqual(document["schema_version"], 1)
                self.assertEqual(document["target"], path.stem)
                host = document["host_environment"]
                self.assertIn(host.get("kind"), {None, "metal", "hip", "maca"})
                if host.get("kind") == "metal":
                    self.assertEqual(host["host"]["target"], path.stem)

    def test_a_capture_carries_the_checkout_commit_as_its_identity(self) -> None:
        commit = checkout_commit_or_none(ROOT)
        if commit is None:
            self.skipTest("this checkout has uncommitted or untracked changes")
        for path in sorted(HOSTS.glob("*.json")):
            with self.subTest(target=path.stem):
                executor = ExecutorRevision.for_target(ROOT, path.stem)
                self.assertEqual(executor.executor_id, f"{path.stem}@{commit}")
                self.assertEqual(executor.relative_path, f"{HOSTS_DIRECTORY}/{path.stem}.json")
                self.assertEqual(executor.document["commit"], commit)
                with self.assertRaises(TypeError):
                    executor.document["host_environment"]["packages"]["torch"] = "changed"

    def test_a_target_without_a_capture_is_reported_as_having_none(self) -> None:
        with self.assertRaisesRegex(ValueError, "no host capture is published"):
            ExecutorRevision.for_target(ROOT, "synthetic-absent-target")

    @unittest.skipUnless(os.environ.get("OPEN_CAKE_RUN_BOUND_EXECUTOR_TESTS") == "1",
                         "requires the captured host's exact runtime/profiler; opt in with "
                         "OPEN_CAKE_RUN_BOUND_EXECUTOR_TESTS=1")
    def test_a_capture_pins_the_attribution_profiler(self) -> None:
        captures = sorted(HOSTS.glob("*.json"))
        executor = ExecutorRevision.for_target(ROOT, captures[0].stem)
        profiler = executor.admit_profiler()
        if executor.document["host_environment"].get("kind") == "metal":
            self.assertEqual(dict(profiler),
                             dict(executor.document["host_environment"]["observer_executable"]))
            return  # This observer has no --version command; do not dispatch it.
        completed = subprocess.run([str(profiler["path"]), "--version"], check=True,
                                   capture_output=True, text=True, timeout=30)
        match = re.search(r"^Version (\S+)", completed.stdout, re.MULTILINE)
        self.assertIsNotNone(match, completed.stdout)
        assert match is not None
        self.assertEqual(profiler["version"], match.group(1))


class ExecutorReferenceTests(unittest.TestCase):
    """Exact-reference admission over a tiny committed fixture checkout."""

    def setUp(self):
        from types import SimpleNamespace

        temporary = tempfile.TemporaryDirectory(prefix="cake-executor-reference-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / HOSTS_DIRECTORY / "fixture-target.json"
        self.path.parent.mkdir(parents=True)
        self.document = {"schema_version": 1, "target": "fixture-target",
                         "host_environment": _synthetic_cuda_host()}
        self.path.write_text(json.dumps(self.document, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8")
        _git(self.root, "init", "-q")
        _git(self.root, "add", ".")
        _git(self.root, "commit", "-q", "--no-verify", "-m", "fixture")
        self.commit = _git(self.root, "rev-parse", "HEAD")
        self.executor = ExecutorRevision.for_target(self.root, "fixture-target")
        self.reference = dict(self.executor.reference)
        self.lock = SimpleNamespace(document={"execution": {"executor_revision": self.reference}})

    def test_identity_is_the_target_and_the_commit(self):
        self.assertEqual(self.executor.executor_id, f"fixture-target@{self.commit}")
        self.assertEqual(self.reference["path"], f"{HOSTS_DIRECTORY}/fixture-target.json")
        loaded = ExecutorRevision.load_reference(self.root, self.reference, "fixture")
        self.assertEqual(loaded.executor_id, self.executor.executor_id)
        self.assertEqual(set(self.reference), {"path", "executor_id"})

    def test_current_and_frozen_binding_return_the_verified_object(self):
        from open_cake_ir.lab.bindings import resolve_executor

        current = resolve_executor(self.root, {"binding": "current_release"}, "study.execution",
                                   template=True, target="fixture-target")
        self.assertEqual(current.executor_id, self.executor.executor_id)
        frozen = resolve_executor(self.root, self.reference, "study.execution", template=False)
        self.assertEqual(frozen.executor_id, self.executor.executor_id)
        for value, template in ((self.reference, True), ({"binding": "current_release"}, False)):
            with self.subTest(template=template), self.assertRaises(ValueError):
                resolve_executor(self.root, value, "study.execution", template=template)

    def test_an_untracked_file_leaves_the_checkout_without_an_identity(self):
        (self.root / "shadow.py").write_text("VALUE = 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "clean committed checkout"):
            ExecutorRevision.load_reference(self.root, self.reference, "replay")

    def test_a_later_commit_does_not_satisfy_a_pinned_reference(self):
        (self.root / "note.md").write_text("second\n", encoding="utf-8")
        _git(self.root, "add", ".")
        _git(self.root, "commit", "-q", "--no-verify", "-m", "second")
        with self.assertRaisesRegex(ValueError, "Executor Revision differs"):
            ExecutorRevision.load_reference(self.root, self.reference, "replay")

    def test_reference_fields_identity_and_paths_are_checked(self):
        specimens = [None, {}, {**self.reference, "extra": True},
                     {**self.reference, "executor_id": "other"},
                     {**self.reference, "canonical_sha256": "0" * 64}]
        for path in (True, "../executor.json", str(self.path), "dir\\fixture-target.json"):
            specimens.append({**self.reference, "path": path})
        for reference in specimens:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                ExecutorRevision.load_reference(self.root, reference, "fixture")

    def test_a_capture_describing_another_target_is_refused(self):
        other = self.root / HOSTS_DIRECTORY / "other-target.json"
        other.write_text(json.dumps(self.document, sort_keys=True), encoding="utf-8")
        _git(self.root, "add", ".")
        _git(self.root, "commit", "-q", "--no-verify", "-m", "mislabelled capture")
        with self.assertRaisesRegex(ValueError, "fields, schema or target differ"):
            ExecutorRevision.for_target(self.root, "other-target")

    def test_a_capture_outside_the_hosts_directory_is_refused(self):
        stray = self.root / "fixture-target.json"
        stray.write_text(json.dumps(self.document, sort_keys=True), encoding="utf-8")
        _git(self.root, "add", ".")
        _git(self.root, "commit", "-q", "--no-verify", "-m", "stray capture")
        with self.assertRaisesRegex(ValueError, "host capture lives at"):
            ExecutorRevision.load(self.root, stray)

    def test_composition_and_worker_each_refuse_a_forged_exact_reference(self):
        from unittest.mock import patch
        from open_cake_ir.tasks.compose import _admit_executor
        from open_cake_ir.tasks import evaluate as worker

        self.reference["extra"] = True
        with patch.object(ExecutorRevision, "admit_host",
                          side_effect=AssertionError("must reject before live host")):
            with self.assertRaisesRegex(ValueError, "fields differ"):
                _admit_executor(self.root, self.lock)
        request = self.root / "request.json"
        request.write_text(json.dumps({"executor_revision": self.reference}), encoding="utf-8")
        with patch.object(worker, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "fields differ"):
                worker._load_authority(request)


if __name__ == "__main__":
    unittest.main()
