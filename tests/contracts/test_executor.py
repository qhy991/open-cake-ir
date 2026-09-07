from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The current Executor Revision advances whenever a runtime source changes. Read it from
# the inventory rather than naming a version, so a bump is not a test edit.
_INVENTORY = json.loads(
    (ROOT / "inventory/EXECUTOR_REVISIONS.json").read_text(encoding="utf-8")
)
CURRENT_EXECUTOR = ROOT / _INVENTORY["current"]["path"]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.lab import ExecutorRevision  # noqa: E402


class ExecutorRevisionContractTests(unittest.TestCase):
    def test_inventory_observation_plans_keep_bound_executor_bytes_resolvable(self) -> None:
        bindings: list[tuple[str, str]] = []

        def collect(value: object) -> None:
            if isinstance(value, dict):
                if {
                    "executor_descriptor",
                    "executor_descriptor_raw_sha256",
                } <= set(value):
                    bindings.append(
                        (
                            str(value["executor_descriptor"]),
                            str(value["executor_descriptor_raw_sha256"]),
                        )
                    )
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        for path in (ROOT / "inventory").glob("*.json"):
            collect(json.loads(path.read_text(encoding="utf-8")))

        self.assertTrue(bindings)
        for relative, expected_sha256 in bindings:
            with self.subTest(path=relative):
                source = ROOT / relative
                self.assertTrue(source.is_file())
                self.assertEqual(sha256(source.read_bytes()).hexdigest(), expected_sha256)

    def test_released_executor_covers_the_complete_runtime_source_closure(self) -> None:
        executor = ExecutorRevision.load(ROOT, CURRENT_EXECUTOR)
        self.assertEqual(executor.executor_id, _INVENTORY["current"]["executor_id"])
        self.assertEqual(
            CURRENT_EXECUTOR.stat().st_mode & 0o444,
            0o444,
        )
        observed = {
            str(record["path"])
            for record in executor.document["sources"]
        }
        expected = {
            path.relative_to(ROOT).as_posix()
            for directory in (
                ROOT / "src/open_cake_ir/lab",
                ROOT / "src/open_cake_ir/evaluation",
                ROOT / "src/open_cake_ir/evidence",
            )
            for path in directory.glob("*.py")
        } | {
            "docs/en/PAIRED_TRITON.md",
            "examples/gpu/flash_kmeans_quickstart.py",
            "src/open_cake_ir/__init__.py",
            "src/open_cake_ir/cli.py",
            "src/open_cake_ir/tasks/qsa/assets/qsa_direct_reference_v1.cu",
            "src/open_cake_ir/tasks/qsa/assets/qsa_direct_reference_v1.json",
            "src/open_cake_ir/tasks/flash_kmeans/calibrate.py",
            "tools/capture_executor_host.py",
            "src/open_cake_ir/tasks/evaluate.py",
            "src/open_cake_ir/tasks/qsa/evaluate.py",
            "tools/observe_target_peak.py",
            "src/open_cake_ir/tasks/qsa/project_feedback.py",
        }

        self.assertEqual(observed, expected)
        with self.assertRaises(TypeError):
            executor.document["host_environment"]["packages"]["torch"] = "changed"

    def test_current_executor_pins_the_attribution_profiler(self) -> None:
        profiler = ExecutorRevision.load(ROOT, CURRENT_EXECUTOR).admit_profiler()
        completed = subprocess.run(
            [str(profiler["path"]), "--version"], check=True, capture_output=True,
            text=True, timeout=30,
        )
        match = re.search(r"^Version (\S+)", completed.stdout, re.MULTILINE)
        self.assertIsNotNone(match, completed.stdout)
        assert match is not None
        self.assertEqual(profiler["version"], match.group(1))

    def test_g8_inventory_resolves_the_complete_historical_executor(self) -> None:
        inventory = json.loads(
            (ROOT / "inventory/G8_SYSTEM_QUALIFICATION_20260822.json").read_text()
        )
        reference = inventory["executor_revision"]
        archive_root = ROOT / reference["archive_root"]
        executor = ExecutorRevision.load(
            archive_root,
            archive_root / "runtime/executor.json",
        )

        self.assertEqual(executor.executor_id, reference["executor_id"])
        self.assertEqual(executor.canonical_sha256, reference["canonical_sha256"])
        self.assertEqual(
            sha256((archive_root / "runtime/executor.json").read_bytes()).hexdigest(),
            reference["descriptor_raw_sha256"],
        )
        self.assertEqual(len(executor.document["sources"]), reference["source_count"])
        self.assertNotEqual(
            executor.canonical_sha256,
            ExecutorRevision.load(ROOT, CURRENT_EXECUTOR).canonical_sha256,
        )

    def test_executor_revision_inventory_resolves_current_and_archived_closures(self) -> None:
        inventory = json.loads(
            (ROOT / "inventory/EXECUTOR_REVISIONS.json").read_text(encoding="utf-8")
        )
        current = inventory["current"]
        current_executor = ExecutorRevision.load(ROOT, ROOT / current["path"])
        self.assertEqual(current_executor.executor_id, current["executor_id"])
        self.assertEqual(current_executor.canonical_sha256, current["canonical_sha256"])
        self.assertEqual(len(current_executor.document["sources"]), current["source_count"])
        self.assertEqual(
            sha256((ROOT / current["path"]).read_bytes()).hexdigest(),
            current["descriptor_raw_sha256"],
        )
        for archived in inventory["archives"]:
            archive_root = ROOT / archived["archive_root"]
            executor = ExecutorRevision.load(
                archive_root,
                archive_root / "runtime/executor.json",
            )
            self.assertEqual(executor.executor_id, archived["executor_id"])
            self.assertEqual(executor.canonical_sha256, archived["canonical_sha256"])
            self.assertEqual(len(executor.document["sources"]), archived["source_count"])
            self.assertEqual(
                sha256((archive_root / "runtime/executor.json").read_bytes()).hexdigest(),
                archived["descriptor_raw_sha256"],
            )

    def test_executor_release_is_create_only_and_world_readable(self) -> None:
        document = json.loads(CURRENT_EXECUTOR.read_text(encoding="utf-8"))
        document["executor_id"] = "open-cake-ir-test-v1"
        document["state"] = "draft"
        document["sources"] = []
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            proposal = temporary / "proposal.json"
            output = temporary / "released.json"
            proposal.write_text(json.dumps(document), encoding="utf-8")
            command = (
                sys.executable,
                str(ROOT / "tools/release_executor.py"),
                "--project-root",
                str(ROOT),
                "--proposal",
                str(proposal),
                "--output",
                str(output),
            )

            first = subprocess.run(command, capture_output=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            released = output.read_bytes()
            self.assertEqual(output.stat().st_mode & 0o444, 0o444)

            second = subprocess.run(command, capture_output=True, check=False)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), released)

            document["executor_id"] = "open-cake-ir-b200-v2"
            collision_proposal = temporary / "collision-proposal.json"
            collision_output = temporary / "collision-release.json"
            collision_proposal.write_text(json.dumps(document), encoding="utf-8")
            collision = subprocess.run(
                (*command[:-3], str(collision_proposal), "--output", str(collision_output)),
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(collision.returncode, 0)
            self.assertFalse(collision_output.exists())

    def test_release_cycle_preserves_released_ids_without_local_run_witnesses(self) -> None:
        from tools.release_executor import _SOURCE_FILES

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            shutil.copytree(ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__"))
            for relative in (
                *_SOURCE_FILES, "tools/release_executor.py", "tools/release_executor_cycle.sh",
                "tools/release_runtime.sh",
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            runtime = root / "runtime/executors"
            runtime.mkdir(parents=True)
            preserved = {}
            for version in (41, 42):
                path = runtime / f"open-cake-ir-b200-v{version}.json"
                raw = (ROOT / path.relative_to(root)).read_bytes()
                path.write_bytes(raw)
                preserved[path] = raw
            archived = root / "evidence/executors/open-cake-ir-b200-v46-review/runtime/executor.json"
            archived.parent.mkdir(parents=True)
            document = json.loads(CURRENT_EXECUTOR.read_text())
            document["executor_id"] = "open-cake-ir-b200-v46"
            archived.write_text(json.dumps(document))
            preserved[archived] = archived.read_bytes()
            inventory_path = root / "inventory/EXECUTOR_REVISIONS.json"
            inventory_path.parent.mkdir()
            previous = next(
                entry for entry in (_INVENTORY["current"], *_INVENTORY["superseded"])
                if entry["executor_id"] == "open-cake-ir-b200-v42"
            )
            inventory_path.write_text(json.dumps({
                "current": previous, "archives": [], "superseded": [],
            }))
            host = document["host_environment"]
            host["python"]["invocation_path"] = "/verified/executor/python"
            host_path = root / "verified-host.json"
            host_path.write_text(json.dumps(host))
            runtime_log = root.parent / "runtime-invocations"
            selected_python = root.parent / "selected python"
            selected_python.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$OPEN_CAKE_PYTHON\" >> "
                + shlex.quote(str(runtime_log)) + "\nexec "
                + shlex.quote(sys.executable) + ' "$@"\n'
            )
            selected_python.chmod(0o755)
            commands = root.parent / "commands"
            commands.mkdir()
            fallback = commands / "python3"
            fallback.write_text("#!/bin/sh\nexit 97\n")
            fallback.chmod(0o755)
            environment = dict(os.environ)
            environment["OPEN_CAKE_PYTHON"] = "./selected python"
            environment["PATH"] = str(commands) + os.pathsep + environment["PATH"]
            environment.pop("OPEN_CAKE_REUSE_VERIFIED_HOST", None)
            command = [
                "bash", str(root / "tools/release_executor_cycle.sh"),
                "--host-environment", str(host_path),
            ]
            for version in (47, 48):
                completed = subprocess.run(
                    command, cwd=root.parent, env=environment, capture_output=True, text=True,
                    check=False, timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                released = runtime / f"open-cake-ir-b200-v{version}.json"
                executor = ExecutorRevision.load(root, released)
                self.assertEqual(executor.executor_id, f"open-cake-ir-b200-v{version}")
                self.assertEqual(json.loads(released.read_text())["host_environment"], host)
                for path, raw in preserved.items():
                    self.assertEqual(path.read_bytes(), raw)
                preserved[released] = released.read_bytes()
            self.assertGreater(len(runtime_log.read_text().splitlines()), 2)
            self.assertEqual(
                {Path(value).resolve() for value in runtime_log.read_text().splitlines()},
                {selected_python.resolve()},
            )
            inventory = json.loads(inventory_path.read_text())
            self.assertEqual(inventory["current"]["executor_id"], "open-cake-ir-b200-v48")
            self.assertEqual(
                {entry["executor_id"] for entry in inventory["superseded"]},
                {"open-cake-ir-b200-v42", "open-cake-ir-b200-v47"},
            )

if __name__ == "__main__":
    unittest.main()
