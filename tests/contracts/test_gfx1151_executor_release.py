from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools import release_gfx1151_executor_cycle as release  # noqa: E402
from tools.release_executor import _validate_output_boundary  # noqa: E402


def _host() -> dict[str, object]:
    return {
        "runtime_kind": "hip",
        "platform": {
            "system": "Linux",
            "machine": "x86_64",
            "kernel_release": "fixture",
        },
        "python": {
            "invocation_path": "/qualified/python",
            "version": "3.12.0",
            "resolved_sha256": "a" * 64,
        },
        "packages": {
            "packaging": "25.0",
            "pybind11": "3.0.4",
            "psutil": "7.2.2",
            "setuptools": "80.0.0",
            "torch": "2.9.1",
            "triton": "3.5.1",
        },
        "runtime": {
            "backend": "hip",
            "torch_hip_version": "7.2.1",
            "visible_device_count": 1,
        },
        "tools": {
            "build_tools": [
                {
                    "kind": kind,
                    "path": f"/qualified/{kind}",
                    "version": "fixture",
                    "sha256": "cdef012"[index] * 64,
                    "size_bytes": 1,
                }
                for index, kind in enumerate(
                    (
                        "cxx",
                        "git",
                        "hipcc",
                        "hipconfig",
                        "ninja",
                        "rocminfo",
                        "sh",
                    )
                )
            ],
            "device_monitor": {
                "kind": "amd-smi",
                "path": "/qualified/amd-smi",
                "version": "fixture",
                "sha256": "b" * 64,
                "size_bytes": 1,
            },
            "profilers": [],
        },
    }


class Gfx1151HostCaptureTests(unittest.TestCase):
    def test_rocm_aliases_are_normalized_to_one_canonical_tool_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            rocm_bin = root / "rocm/bin"
            aliases = root / "aliases"
            rocm_bin.mkdir(parents=True)
            aliases.mkdir()
            for name in ("hipcc", "hipconfig", "rocminfo"):
                path = rocm_bin / name
                path.write_text(f"#!/bin/sh\n# {name}\n", encoding="utf-8")
                path.chmod(0o755)
            alias = aliases / "hipcc"
            alias.symlink_to(rocm_bin / "hipcc")

            with (
                patch.object(release.shutil, "which", return_value=str(alias)),
                patch.dict(
                    release.os.environ,
                    {"ROCM_HOME": "", "ROCM_PATH": ""},
                ),
            ):
                observed = release._rocm_build_tool_paths()

        self.assertEqual(observed["hipcc"], rocm_bin / "hipcc")
        self.assertEqual(observed["hipconfig"], rocm_bin / "hipconfig")
        self.assertEqual(observed["rocminfo"], rocm_bin / "rocminfo")

    def test_captures_only_consumed_hip_host_facts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            rocm_bin = root / "rocm/bin"
            rocm_bin.mkdir(parents=True)
            commands = {
                "amd-smi": root / "amd-smi",
                "c++": root / "c++",
                "git": root / "git",
                "hipcc": rocm_bin / "hipcc",
                "hipconfig": rocm_bin / "hipconfig",
                "ninja": root / "ninja",
                "rocminfo": rocm_bin / "rocminfo",
                "sh": root / "sh",
            }
            for name, path in commands.items():
                path.write_text(
                    f"#!/bin/sh\necho '{name} fixture'\n", encoding="utf-8"
                )
                path.chmod(0o755)
            torch = SimpleNamespace(version=SimpleNamespace(hip="7.2.1"))

            def which(name: str) -> str | None:
                key = "sh" if name == "/bin/sh" else name
                return str(commands[key]) if key in commands else None

            with (
                patch.object(release.platform, "system", return_value="Linux"),
                patch.object(release.platform, "machine", return_value="x86_64"),
                patch.object(release.platform, "release", return_value="6.10-fixture"),
                patch.object(release, "admit_exact_hip", return_value=(torch, object(), object())),
                patch.object(release.shutil, "which", side_effect=which),
                patch.object(
                    release.importlib.metadata,
                    "version",
                    side_effect=lambda name: {
                        "packaging": "25.0",
                        "pybind11": "3.0.4",
                        "psutil": "7.2.2",
                        "setuptools": "80.0.0",
                        "torch": "2.9.1",
                        "triton": "3.5.1",
                    }[name],
                ),
            ):
                observed = release.collect_gfx1151_host_environment()

        self.assertEqual(observed["runtime_kind"], "hip")
        self.assertEqual(observed["runtime"]["torch_hip_version"], "7.2.1")
        self.assertEqual(observed["runtime"]["visible_device_count"], 1)
        self.assertEqual(observed["tools"]["device_monitor"]["kind"], "amd-smi")
        self.assertEqual(observed["tools"]["profilers"], [])
        self.assertEqual(
            {value["kind"] for value in observed["tools"]["build_tools"]},
            {"cxx", "git", "hipcc", "hipconfig", "ninja", "rocminfo", "sh"},
        )
        self.assertEqual(
            set(observed["packages"]),
            {
                "packaging",
                "pybind11",
                "psutil",
                "setuptools",
                "torch",
                "triton",
            },
        )
        self.assertNotIn("gfx1151", json.dumps(observed))

    def test_refuses_to_capture_on_another_platform(self) -> None:
        with (
            patch.object(release.platform, "system", return_value="Darwin"),
            self.assertRaisesRegex(RuntimeError, "Linux x86_64"),
        ):
            release.collect_gfx1151_host_environment()


class Gfx1151ReleaseTransitionTests(unittest.TestCase):
    def test_low_level_writer_cannot_install_a_schema2_release_directly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runtime = root / "runtime/executors"
            runtime.mkdir(parents=True)
            identity = "open-cake-ir-gfx1151-v1"
            with self.assertRaisesRegex(ValueError, "hidden candidate"):
                _validate_output_boundary(
                    root,
                    runtime / f"{identity}.json",
                    schema_version=2,
                    executor_id=identity,
                )
            _validate_output_boundary(
                root,
                runtime / f".{identity}.candidate.123.json",
                schema_version=2,
                executor_id=identity,
            )

    @staticmethod
    def _working_descriptor(root: Path) -> Path:
        path = root / "runtime/executors/open-cake-ir-gfx1151-v1.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(
            json.dumps(
                {
                    "schema_version": 2,
                    "executor_id": "open-cake-ir-gfx1151-v1",
                    "state": "released",
                }
            ).encode()
        )
        return path

    def test_failed_candidate_build_preserves_the_working_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            working = self._working_descriptor(root)
            before = working.read_bytes()
            failure = subprocess.CalledProcessError(1, ["release_executor.py"])

            with (
                patch.object(release, "collect_gfx1151_host_environment", return_value=_host()),
                patch.object(release.subprocess, "run", side_effect=failure),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                release.release_gfx1151_executor(root)

            self.assertEqual(working.read_bytes(), before)
            self.assertEqual(
                list((root / "runtime/executors").glob(".*.candidate.*.json")),
                [],
            )

    def test_failed_candidate_validation_preserves_the_working_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            working = self._working_descriptor(root)
            before = working.read_bytes()

            def build(command: list[str], **_: object) -> object:
                output = Path(command[command.index("--output") + 1])
                output.write_text("{}", encoding="utf-8")
                return SimpleNamespace(returncode=0)

            with (
                patch.object(release, "collect_gfx1151_host_environment", return_value=_host()),
                patch.object(release.subprocess, "run", side_effect=build),
                patch.object(
                    release.ExecutorRevision,
                    "load",
                    side_effect=ValueError("candidate differs"),
                ),
                self.assertRaisesRegex(ValueError, "candidate differs"),
            ):
                release.release_gfx1151_executor(root)

            self.assertEqual(working.read_bytes(), before)
            self.assertEqual(
                list((root / "runtime/executors").glob(".*.candidate.*.json")),
                [],
            )

    def test_new_frozen_witness_before_commit_preserves_working_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            working = self._working_descriptor(root)
            before = working.read_bytes()

            def build(command: list[str], **_: object) -> object:
                output = Path(command[command.index("--output") + 1])
                output.write_text("{}", encoding="utf-8")
                return SimpleNamespace(returncode=0)

            def freeze() -> object:
                path = root / "contracts/calibrations/frozen.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "state": "frozen",
                            "executor": {
                                "executor_id": "open-cake-ir-gfx1151-v1",
                                "canonical_sha256": "c" * 64,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return SimpleNamespace(executor_id="open-cake-ir-gfx1151-v1")

            released = SimpleNamespace(
                executor_id="open-cake-ir-gfx1151-v1",
                canonical_sha256="d" * 64,
                admit_hip_host=Mock(side_effect=freeze),
            )
            with (
                patch.object(release, "collect_gfx1151_host_environment", return_value=_host()),
                patch.object(release.subprocess, "run", side_effect=build),
                patch.object(release.ExecutorRevision, "load", return_value=released),
                self.assertRaisesRegex(ValueError, "witness state changed"),
            ):
                release.release_gfx1151_executor(root)

            self.assertEqual(working.read_bytes(), before)

    def test_candidate_revalidation_failure_preserves_working_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            working = self._working_descriptor(root)
            before = working.read_bytes()

            def build(command: list[str], **_: object) -> object:
                output = Path(command[command.index("--output") + 1])
                output.write_text("{}", encoding="utf-8")
                return SimpleNamespace(returncode=0)

            released = SimpleNamespace(
                executor_id="open-cake-ir-gfx1151-v1",
                canonical_sha256="d" * 64,
                admit_hip_host=Mock(
                    return_value=SimpleNamespace(
                        executor_id="open-cake-ir-gfx1151-v1"
                    )
                ),
            )
            with (
                patch.object(release, "collect_gfx1151_host_environment", return_value=_host()),
                patch.object(release.subprocess, "run", side_effect=build),
                patch.object(
                    release.ExecutorRevision,
                    "load",
                    side_effect=[released, ValueError("candidate source changed")],
                ),
                self.assertRaisesRegex(ValueError, "candidate source changed"),
            ):
                release.release_gfx1151_executor(root)

            self.assertEqual(working.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
