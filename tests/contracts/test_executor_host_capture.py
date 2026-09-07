from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tools import capture_executor_host as capture  # noqa: E402


class ExecutorHostCaptureContractTests(unittest.TestCase):
    def run_fixture_capture(self, root: Path, helper_prefix: str = "") -> subprocess.CompletedProcess[str]:
        """Use importable CPU fixtures to exercise the fresh-process software path."""

        site = root / "site"
        for name, package, sources in (
            ("cupti-python", "cupti", {"__init__.py": "cupti = object()\n"}),
            ("flashinfer-python", "flashinfer", {
                "testing/utils.py": helper_prefix + "\n" + "\n".join(
                    f"def bench_gpu_time_with_{suffix}(): pass"
                    for suffix in ("cupti", "cuda_event", "cudagraph")
                ),
            }),
        ):
            metadata_name = name.replace("-", "_") + "-1.0.dist-info"
            metadata = site / metadata_name
            metadata.mkdir(parents=True)
            records = []
            for relative, source in sources.items():
                path = site / package / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source)
                records.append(path.relative_to(site).as_posix())
            (metadata / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
            records += [f"{metadata_name}/METADATA", f"{metadata_name}/RECORD"]
            (metadata / "RECORD").write_text("".join(f"{path},,\n" for path in records))
        profiler = root / "ncu"
        profiler.write_text("#!/bin/sh\nprintf 'Version 1.0\\n'\n")
        profiler.chmod(0o755)
        return subprocess.run(
            [
                sys.executable, str(ROOT / "tools/capture_executor_host.py"),
                "--package", "cupti-python", "--package", "flashinfer-python",
                "--cupti-distribution", "cupti-python",
                "--flashinfer-distribution", "flashinfer-python",
                "--ncu", str(profiler), "--output", str(root / "host.json"),
            ],
            cwd=root, env={
                **os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join((str(site), str(ROOT / "src"))),
            },
            capture_output=True, text=True, timeout=30,
        )

    def test_fresh_process_capture_writes_only_an_admitted_host_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            completed = self.run_fixture_capture(root)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            output = root / "host.json"
            host = json.loads(output.read_text())
            self.assertEqual(set(host), {
                "python", "packages", "cupti_python", "flashinfer_helper", "nsight_compute",
            })
            self.assertEqual(host["python"]["invocation_path"], sys.executable)
            self.assertEqual(host["packages"], {"cupti-python": "1.0", "flashinfer-python": "1.0"})
            self.assertEqual(host["nsight_compute"]["version"], "1.0")
            self.assertEqual(json.loads(completed.stdout), {
                "output": str(output), "host_admitted": True, "profiler_admitted": True,
            })
            self.assertEqual(list(root.glob("*.json")), [output])

    def test_cold_helper_import_failure_produces_no_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            completed = self.run_fixture_capture(root, "raise RuntimeError('cold import failed')")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("cold import failed", completed.stderr)
            self.assertFalse((root / "host.json").exists())

    def test_profiler_change_during_host_admission_produces_no_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            completed = self.run_fixture_capture(
                root, f"from pathlib import Path\nPath({str(root / 'ncu')!r}).write_text('changed')",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Executor Nsight Compute bytes differ", completed.stderr)
            self.assertFalse((root / "host.json").exists())

    def arguments(self, output: Path) -> list[str]:
        return [
            "--package", "open-cake-ir-no-such-distribution-for-test",
            "--cupti-distribution", "cupti-python",
            "--flashinfer-distribution", "flashinfer-python",
            "--ncu", sys.executable, "--output", str(output),
        ]

    def distribution(
        self, root: Path, records: list[str],
    ) -> importlib.metadata.Distribution:
        metadata = root / "cupti_python-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Name: cupti-python\nVersion: 1.0\n")
        (metadata / "RECORD").write_text("".join(f"{path},,\n" for path in records))
        for relative in records:
            if relative.startswith("cupti/") and ".." not in Path(relative).parts:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test distribution file\n")
        return importlib.metadata.Distribution.at(metadata)

    def test_capture_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "host.json"
            output.write_text("existing capture\n")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                capture.main(self.arguments(output))
            self.assertEqual(output.read_text(), "existing capture\n")

    def test_capture_refuses_a_dangling_output_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "host.json"
            target = Path(directory) / "missing.json"
            output.symlink_to(target)
            with self.assertRaises(FileExistsError):
                capture.main(self.arguments(output))
            self.assertTrue(output.is_symlink())
            self.assertFalse(target.exists())

    def test_capture_refuses_output_in_a_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").write_text("gitdir: /test/worktree\n")
            output = root / "host.json"
            with self.assertRaisesRegex(ValueError, "outside project checkouts"):
                capture.main(self.arguments(output))
            self.assertFalse(output.exists())

    def test_missing_distribution_produces_no_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "host.json"
            with self.assertRaises(importlib.metadata.PackageNotFoundError):
                capture.main(self.arguments(output))
            self.assertFalse(output.exists())

    def test_cupti_file_capture_uses_the_distribution_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                "cupti/__init__.py", "cupti/cupti.cpython-test.so", "cupti/cupti.pyi",
                "cupti/__pycache__/generated.pyc",
                "cupti_python-1.0.dist-info/METADATA", "cupti_python-1.0.dist-info/RECORD",
            ]
            distribution = self.distribution(root, records)
            with patch.object(capture.importlib.metadata, "distribution", return_value=distribution):
                result = capture._capture_cupti("cupti-python")
            self.assertEqual(result["version"], "1.0")
            self.assertEqual(result["site_packages_path"], str(root.resolve()))
            self.assertEqual(
                [record["path"] for record in result["files"]],
                sorted(path for path in records if "__pycache__" not in path),
            )

    def test_cupti_incomplete_manifest_is_rejected(self) -> None:
        for omitted in (
            "cupti/__init__.py", "cupti_python-1.0.dist-info/METADATA",
            "cupti_python-1.0.dist-info/RECORD",
        ):
            with self.subTest(omitted=omitted), tempfile.TemporaryDirectory() as directory:
                records = [
                    "cupti/__init__.py", "cupti_python-1.0.dist-info/METADATA",
                    "cupti_python-1.0.dist-info/RECORD",
                ]
                records.remove(omitted)
                distribution = self.distribution(Path(directory), records)
                with patch.object(capture.importlib.metadata, "distribution", return_value=distribution):
                    with self.assertRaisesRegex(ValueError, "lacks its package"):
                        capture._capture_cupti("cupti-python")

    def test_cupti_manifest_cannot_escape_its_distribution_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            distribution = self.distribution(Path(directory), [
                "cupti/__init__.py", "cupti/../../outside.py",
                "cupti_python-1.0.dist-info/METADATA", "cupti_python-1.0.dist-info/RECORD",
            ])
            with patch.object(capture.importlib.metadata, "distribution", return_value=distribution):
                with self.assertRaisesRegex(ValueError, "path is unsafe"):
                    capture._capture_cupti("cupti-python")

    def test_cupti_missing_recorded_source_is_not_silently_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            distribution = self.distribution(root, [
                "cupti/__init__.py", "cupti/missing.py",
                "cupti_python-1.0.dist-info/METADATA", "cupti_python-1.0.dist-info/RECORD",
            ])
            (root / "cupti/missing.py").unlink()
            with patch.object(capture.importlib.metadata, "distribution", return_value=distribution):
                with self.assertRaises(FileNotFoundError):
                    capture._capture_cupti("cupti-python")

    def test_distribution_rejects_a_malformed_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            distribution = self.distribution(root, ["cupti/__init__.py"])
            (root / "cupti_python-1.0.dist-info/RECORD").write_text(
                "cupti/__init__.py\n"
            )
            with self.assertRaisesRegex(ValueError, "malformed file entry"):
                capture._distribution_files(distribution)

    def test_flashinfer_helper_must_be_owned_by_the_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            distribution = self.distribution(Path(directory), [
                "cupti_python-1.0.dist-info/METADATA", "cupti_python-1.0.dist-info/RECORD",
            ])
            with patch.object(capture.importlib.metadata, "distribution", return_value=distribution):
                with self.assertRaisesRegex(ValueError, "does not own its testing helper"):
                    capture._capture_flashinfer("flashinfer-python")

    def test_profiler_requires_an_absolute_regular_executable(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit absolute executable"):
            capture._capture_profiler(Path("ncu"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ncu"
            path.write_text("not executable\n")
            with self.assertRaisesRegex(ValueError, "explicit absolute executable"):
                capture._capture_profiler(path)

    def test_non_ncu_version_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "version output differs"):
            capture._capture_profiler(Path(sys.executable).resolve(strict=True))

    def test_canonical_schema_rejection_produces_no_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "host.json"
            with patch.object(capture, "_capture_host", return_value={}):
                with self.assertRaisesRegex(ValueError, "host environment fields differ"):
                    capture.main(self.arguments(output))
            self.assertFalse(output.exists())

    def test_canonical_host_rejection_produces_no_capture(self) -> None:
        inventory = json.loads((ROOT / "inventory/EXECUTOR_REVISIONS.json").read_text())
        host = json.loads((ROOT / inventory["current"]["path"]).read_text())["host_environment"]
        host["python"]["invocation_path"] = sys.executable
        host["python"]["version"] = "not-the-running-python-version"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "host.json"
            with patch.object(capture, "_capture_host", return_value=host):
                with self.assertRaisesRegex(ValueError, "Python runtime differs"):
                    capture.main(self.arguments(output))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
