"""Observable gfx1151 successor release behavior with modeled Linux/HIP facts.

The integration-owned cycle must be present before running this module. The host
admission and source loader remain real; only installed package/runtime observation
and the low-level descriptor build are modeled. This is no GPU qualification.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from open_cake_ir.evaluation import triton_hip as hip  # noqa: E402
from tests.contracts.test_amd_host_contract import HipExecutorFixture  # noqa: E402
from tools import release_gfx1151_executor_cycle as release  # noqa: E402


class Gfx1151ReleaseTransitionTests(unittest.TestCase):
    def _fixture(self) -> tuple[HipExecutorFixture, Path]:
        fixture = HipExecutorFixture(self)
        target = fixture.root / "compiler/targets/gfx1151.json"
        target.parent.mkdir(parents=True)
        target.write_bytes((ROOT / "compiler/targets/gfx1151.json").read_bytes())
        capture = fixture.root / "host.json"
        capture.write_text(json.dumps(fixture.host), encoding="utf-8")
        archived = fixture.root / "evidence/executors/reserved/runtime/executor.json"
        archived.parent.mkdir(parents=True)
        archived.write_text(json.dumps({
            "state": "released", "executor_id": "open-cake-ir-gfx1151-v3",
        }), encoding="utf-8")
        return fixture, capture

    @contextmanager
    def _runtime(self, fixture: HipExecutorFixture, *, arch: str = "gfx1151"):
        properties = SimpleNamespace(gcnArchName=arch, warp_size=32)
        torch = SimpleNamespace(
            version=SimpleNamespace(hip="7.2.1", cuda=None),
            cuda=SimpleNamespace(
                is_available=lambda: True, device_count=lambda: 1,
                get_device_properties=lambda _: properties,
            ),
        )
        modules = {
            "torch": torch, "triton": SimpleNamespace(),
            "triton.runtime.driver": SimpleNamespace(driver=SimpleNamespace(
                active=SimpleNamespace(get_current_target=lambda: SimpleNamespace(
                    backend="hip", arch=arch, warp_size=32,
                )),
            )),
        }
        with (
            patch("open_cake_ir.lab.executor.platform.system", return_value="Linux"),
            patch("open_cake_ir.lab.executor.importlib.metadata.version",
                  side_effect=fixture.host["packages"].__getitem__),
            patch.object(hip.importlib, "import_module", side_effect=modules.__getitem__),
        ):
            yield

    @staticmethod
    def _builder(fixture: HipExecutorFixture):
        def build(command: list[str], **_: object) -> object:
            proposal = json.loads(Path(command[command.index("--proposal") + 1]).read_text())
            proposal["state"] = "released"
            proposal["sources"] = copy.deepcopy(fixture.document["sources"])
            output = Path(command[command.index("--output") + 1])
            with output.open("x", encoding="utf-8") as stream:
                json.dump(proposal, stream)
            return SimpleNamespace(returncode=0)
        return build

    def _assert_preserved(self, fixture: HipExecutorFixture, original: bytes) -> None:
        self.assertEqual(fixture.path.read_bytes(), original)
        self.assertFalse((fixture.path.parent / "open-cake-ir-gfx1151-v4.json").exists())
        self.assertEqual(list(fixture.path.parent.glob(".*.candidate.*.json")), [])

    def test_public_cycle_reserves_current_and_archived_ids_and_admits_real_boundary(self) -> None:
        fixture, capture = self._fixture()
        original = fixture.path.read_bytes()
        # Keep the actual strict HIP admission callable. Missing artifact roles in
        # the cycle's requirements must fail before mocked runtime imports.
        self.assertIs(release.admit_exact_hip, hip.admit_exact_hip)
        with (
            self._runtime(fixture),
            patch.object(release.subprocess, "run", side_effect=self._builder(fixture)),
        ):
            result = release.release_gfx1151_executor(fixture.root, host_environment=capture)
        self.assertEqual(result["executor_id"], "open-cake-ir-gfx1151-v4")
        self.assertTrue(result["host_admitted"])
        self.assertEqual(result["target_admitted"], "gfx1151")
        installed = fixture.root / result["path"]
        self.assertEqual(json.loads(installed.read_text())["executor_id"], result["executor_id"])
        self.assertEqual(fixture.path.read_bytes(), original)
        self.assertEqual(list(fixture.path.parent.glob(".*.candidate.*.json")), [])

    def test_missing_explicit_capture_fails_before_build_or_install(self) -> None:
        fixture, _ = self._fixture()
        original = fixture.path.read_bytes()
        with (
            patch.object(release.subprocess, "run") as build,
            self.assertRaises(FileNotFoundError),
        ):
            release.release_gfx1151_executor(fixture.root, host_environment=fixture.root / "missing.json")
        build.assert_not_called()
        self._assert_preserved(fixture, original)

    def test_malformed_capture_fails_before_build_or_install(self) -> None:
        fixture, capture = self._fixture()
        original = fixture.path.read_bytes()
        host = copy.deepcopy(fixture.host)
        host["runtime"]["visible_device_count"] = True
        capture.write_text(json.dumps(host), encoding="utf-8")
        with (
            patch.object(release.subprocess, "run") as build,
            self.assertRaises(ValueError),
        ):
            release.release_gfx1151_executor(fixture.root, host_environment=capture)
        build.assert_not_called()
        self._assert_preserved(fixture, original)

    def test_wrong_observed_runtime_target_preserves_releases(self) -> None:
        fixture, capture = self._fixture()
        original = fixture.path.read_bytes()
        with (
            self._runtime(fixture, arch="gfx942"),
            patch.object(release.subprocess, "run", side_effect=self._builder(fixture)),
            self.assertRaisesRegex(RuntimeError, "runtime Target"),
        ):
            release.release_gfx1151_executor(fixture.root, host_environment=capture)
        self._assert_preserved(fixture, original)

    def test_failed_build_preserves_releases(self) -> None:
        fixture, capture = self._fixture()
        original = fixture.path.read_bytes()
        with (
            patch.object(release.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["builder"])),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            release.release_gfx1151_executor(fixture.root, host_environment=capture)
        self._assert_preserved(fixture, original)

    def test_changed_candidate_source_is_refused_before_install(self) -> None:
        fixture, capture = self._fixture()
        original = fixture.path.read_bytes()
        build = self._builder(fixture)

        def drift(command: list[str], **kwargs: object) -> object:
            result = build(command, **kwargs)
            (fixture.root / "runner.py").write_text("CHANGED = True\n", encoding="utf-8")
            return result

        with (
            patch.object(release.subprocess, "run", side_effect=drift),
            self.assertRaises(ValueError),
        ):
            release.release_gfx1151_executor(fixture.root, host_environment=capture)
        self._assert_preserved(fixture, original)

    def test_cli_requires_capture_without_creating_a_candidate(self) -> None:
        fixture, _ = self._fixture()
        original = fixture.path.read_bytes()
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools/release_gfx1151_executor_cycle.py"),
             "--project-root", str(fixture.root)],
            cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--host-environment", completed.stderr)
        self._assert_preserved(fixture, original)


if __name__ == "__main__":
    unittest.main()
