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

# The host_environment of the released pre-kind CUDA descriptor
# `git show history:runtime/executors/open-cake-ir-b200-v6.json`, with the CUPTI file
# list cut to its first record. Retired descriptors resolve through
# docs/history/identities.json; this copy exists only to give the Python-runtime refusal
# a CUDA-shaped subject.
RELEASED_PRE_KIND_CUDA_HOST = """{
    "cupti_python": {
        "distribution": "cupti-python",
        "files": [
            {
                "path": "cupti/__init__.py",
                "sha256": "a0833b8fd41566149fdcdabd3e59461488709e1ecb39def32a4a0b86c02a38b7",
                "size_bytes": 100
            }
        ],
        "site_packages_path": "/home/qinhaiyan/sol-execbench-main/.venv/lib/python3.12/site-packages",
        "version": "13.0.1"
    },
    "flashinfer_helper": {
        "distribution": "flashinfer-python",
        "path": "/home/qinhaiyan/megakernel-exp/.venv-flashinfer/lib/python3.12/site-packages/flashinfer/testing/utils.py",
        "sha256": "3a7515a237a3c6531ffa15829a5d46e3d399b3c23f7d3dba853f9e6be46da490",
        "size_bytes": 67087,
        "version": "0.6.16.post2"
    },
    "packages": {
        "cuda-bindings": "13.3.1",
        "flashinfer-python": "0.6.16.post2",
        "numpy": "2.5.1",
        "torch": "2.13.0+cu130",
        "triton": "3.7.1"
    },
    "python": {
        "invocation_path": "/home/qinhaiyan/megakernel-exp/.venv-flashinfer/bin/python",
        "resolved_sha256": "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118",
        "version": "3.12.3"
    }
}"""


def declare_target(root: Path, target: str) -> None:
    """A capture belongs to a checkout that declares the target it describes."""
    directory = root / "compiler/targets"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{target}.json").write_bytes(
        (ROOT / "compiler/targets" / f"{target}.json").read_bytes()
    )


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
        declare_target(root, "sm_103a")
        return subprocess.run(
            [
                sys.executable, str(ROOT / "tools/capture_executor_host.py"),
                "--project-root", str(root), "--target", "sm_103a",
                "--package", "cupti-python", "--package", "flashinfer-python",
                "--cupti-distribution", "cupti-python",
                "--flashinfer-distribution", "flashinfer-python",
                "--ncu", str(profiler),
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
            output = root / "runtime/hosts/sm_103a.json"
            document = json.loads(output.read_text())
            self.assertEqual(set(document), {"schema_version", "target", "host_environment"})
            self.assertEqual(document["target"], "sm_103a")
            host = document["host_environment"]
            self.assertEqual(set(host), {
                "python", "packages", "cupti_python", "flashinfer_helper", "nsight_compute",
            })
            self.assertEqual(host["python"]["invocation_path"], sys.executable)
            self.assertEqual(host["packages"], {"cupti-python": "1.0", "flashinfer-python": "1.0"})
            self.assertEqual(host["nsight_compute"]["version"], "1.0")
            self.assertEqual(json.loads(completed.stdout), {
                "output": str(output), "target": "sm_103a",
                "host_admitted": True, "profiler_admitted": True,
            })
            self.assertEqual(sorted((root / "runtime/hosts").glob("*.json")), [output])

    def test_cold_helper_import_failure_produces_no_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            completed = self.run_fixture_capture(root, "raise RuntimeError('cold import failed')")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("cold import failed", completed.stderr)
            self.assertFalse((root / "runtime/hosts/sm_103a.json").exists())

    def test_profiler_change_during_host_admission_produces_no_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            completed = self.run_fixture_capture(
                root, f"from pathlib import Path\nPath({str(root / 'ncu')!r}).write_text('changed')",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Executor Nsight Compute bytes differ", completed.stderr)
            self.assertFalse((root / "runtime/hosts/sm_103a.json").exists())

    def arguments(self, root: Path, target: str = "sm_103a") -> list[str]:
        declare_target(root, target)
        return [
            "--project-root", str(root), "--target", target,
            "--package", "open-cake-ir-no-such-distribution-for-test",
            "--cupti-distribution", "cupti-python",
            "--flashinfer-distribution", "flashinfer-python",
            "--ncu", sys.executable,
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

    def test_capture_refuses_to_recapture_without_being_asked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            arguments = self.arguments(root)
            output = root / "runtime/hosts/sm_103a.json"
            output.parent.mkdir(parents=True)
            output.write_text("existing capture\n")
            with self.assertRaisesRegex(FileExistsError, "recapture"):
                capture.main(arguments)
            self.assertEqual(output.read_text(), "existing capture\n")

    def test_capture_refuses_a_target_this_checkout_does_not_declare(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            arguments = self.arguments(root)
            arguments[arguments.index("--target") + 1] = "synthetic-absent-target"
            with self.assertRaisesRegex(ValueError, "not a Target this checkout declares"):
                capture.main(arguments)
            self.assertFalse((root / "runtime/hosts").exists())

    def test_missing_distribution_produces_no_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(importlib.metadata.PackageNotFoundError):
                capture.main(self.arguments(root))
            self.assertFalse((root / "runtime/hosts/sm_103a.json").exists())

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
            root = Path(directory).resolve()
            with patch.object(capture, "_capture_host", return_value={}):
                with self.assertRaisesRegex(ValueError, "host environment fields differ"):
                    capture.main(self.arguments(root))
            self.assertFalse((root / "runtime/hosts/sm_103a.json").exists())

    def test_canonical_host_rejection_produces_no_capture(self) -> None:
        # A CUDA capture needs a CUDA-shaped host. Taking whichever descriptor happens to
        # be current stopped working the moment a HIP host was one, and refused for the
        # kind rather than for the Python mismatch this case is about. The released
        # pre-kind descriptors left the tree for the `history` branch, so the subject is
        # the released `open-cake-ir-b200-v6` host_environment inlined from
        # `git show history:runtime/executors/open-cake-ir-b200-v6.json`, with its CUPTI
        # file list cut to the one record the validator's non-empty check needs.
        host = json.loads(RELEASED_PRE_KIND_CUDA_HOST)
        host["python"]["invocation_path"] = sys.executable
        host["python"]["version"] = "not-the-running-python-version"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(capture, "_capture_host", return_value=host):
                with self.assertRaisesRegex(ValueError, "Python runtime differs"):
                    capture.main(self.arguments(root))
            self.assertFalse((root / "runtime/hosts/sm_103a.json").exists())


class HipExecutorHostCaptureContractTests(unittest.TestCase):
    def run_fixture_capture(
        self, root: Path, *, import_failure: bool = False, changing_profiler: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Model Linux and importable HIP packages; no device API is exposed."""
        site = root / "site"
        for name in capture.HIP_PACKAGES:
            metadata = site / f"{name}-1.0.dist-info"
            metadata.mkdir(parents=True)
            (metadata / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
        torch = site / "torch.py"
        torch.write_text(
            "raise RuntimeError('cold HIP import failed')\n" if import_failure else
            "from types import SimpleNamespace\n"
            "version = SimpleNamespace(hip='7.2.1', cuda=None)\n"
            "def __getattr__(name): raise AssertionError('device access: ' + name)\n"
        )
        paths = {}
        for kind in (*capture.HIP_BUILD_TOOLS, "amd-smi", "rocprofv3"):
            tool = root / kind
            mutation = f"printf '# changed\\n' >> '{tool}'\n" if changing_profiler and kind == "rocprofv3" else ""
            tool.write_text("#!/bin/sh\n" + mutation + "printf 'Version 1.0\\n'\n")
            tool.chmod(0o755)
            paths[kind] = tool
        library = root / "libxml2.so.2"
        library.write_bytes(b"modeled compatibility library")
        declare_target(root, "gfx938")
        arguments = ["--project-root", str(root), "--target", "gfx938",
                     "--runtime-kind", "hip",
                     "--device-monitor", "amd-smi", str(paths["amd-smi"])]
        for name in sorted(capture.HIP_PACKAGES):
            arguments += ["--package", name]
        for kind in sorted(capture.HIP_BUILD_TOOLS):
            arguments += ["--hip-build-tool", kind, str(paths[kind])]
        arguments += [
            "--hip-profiler", "rocprofv3", str(paths["rocprofv3"]),
            "--hip-runtime-library", "libxml2.so.2", str(library),
            # The jail clears the environment, so a HIP host states what its toolchain
            # needs inside it rather than inheriting an env.sh.
            "--hip-build-environment", "ROCM_PATH", str(root),
        ]
        program = (
            "import platform, runpy; platform.system = lambda: 'Linux'; "
            f"runpy.run_path({str(ROOT / 'tools/capture_executor_host.py')!r}, run_name='__main__')"
        )
        return subprocess.run(
            [sys.executable, "-c", program, *arguments], cwd=root,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                 "PYTHONPATH": os.pathsep.join((str(site), str(ROOT / "src")))},
            capture_output=True, text=True, timeout=30,
        )

    def test_fresh_hip_capture_uses_canonical_software_admission_without_devices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            completed = self.run_fixture_capture(root)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            document = json.loads((root / "runtime/hosts/gfx938.json").read_text())
            self.assertEqual(document["target"], "gfx938")
            host = document["host_environment"]
            capture.ExecutorRevision._validate_host_document(host)
            self.assertEqual(host["kind"], "hip")
            self.assertEqual(host["platform"]["system"], "Linux")
            self.assertEqual(host["runtime"]["torch_hip_version"], "7.2.1")
            # Declared, not inherited: the isolated build jail runs --clearenv.
            self.assertEqual(host["runtime"]["build_environment"], {"ROCM_PATH": str(root)})
            self.assertEqual(set(host["packages"]), capture.HIP_PACKAGES)
            self.assertEqual(host["tools"]["profilers"][0]["kind"], "rocprofv3")
            self.assertTrue(json.loads(completed.stdout)["host_admitted"])

    def test_cold_hip_import_failure_produces_no_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            completed = self.run_fixture_capture(root, import_failure=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("cold HIP import failed", completed.stderr)
            self.assertFalse((root / "runtime/hosts/sm_103a.json").exists())

    def test_hip_profiler_changed_during_version_probe_produces_no_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            completed = self.run_fixture_capture(root, changing_profiler=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("profilers[0] bytes differ", completed.stderr)
            self.assertFalse((root / "runtime/hosts/sm_103a.json").exists())

    def test_cuda_inputs_cannot_be_mixed_into_hip_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            declare_target(root, "gfx938")
            with self.assertRaisesRegex(ValueError, "HIP capture requires"):
                capture.main([
                    "--project-root", str(root), "--target", "gfx938",
                    "--runtime-kind", "hip", "--package", "torch",
                    "--cupti-distribution", "cupti-python",
                ])
            self.assertFalse((root / "runtime/hosts/gfx938.json").exists())


if __name__ == "__main__":
    unittest.main()
