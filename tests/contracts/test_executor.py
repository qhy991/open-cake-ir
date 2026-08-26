from __future__ import annotations

import json
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
            "examples/gpu/aiter_rmsnorm_amd_baseline.py",
            "examples/gpu/amd_triton_quickstart.py",
            "examples/gpu/flash_kmeans_quickstart.py",
            "examples/gpu/llama_q8_1_amd_quickstart.py",
            "examples/gpu/rmsnorm_amd_quickstart.py",
            "examples/gpu/rmsnorm_amd_search.py",
            "examples/gpu/swiglu_amd_quickstart.py",
            "src/open_cake_ir/__init__.py",
            "src/open_cake_ir/cli.py",
            "tools/evaluate_flash_candidate.py",
        }

        self.assertEqual(observed, expected)
        with self.assertRaises(TypeError):
            executor.document["host_environment"]["packages"]["torch"] = "changed"

    def test_current_executor_pins_the_attribution_profiler(self) -> None:
        profiler = ExecutorRevision.load(ROOT, CURRENT_EXECUTOR).admit_profiler()
        self.assertEqual(profiler["version"], "2026.1.1.0")
        self.assertEqual(
            profiler["sha256"],
            sha256(Path(str(profiler["path"])).read_bytes()).hexdigest(),
        )

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

    def test_release_can_build_beside_but_not_overwrite_an_unwitnessed_working_id(self) -> None:
        working = ROOT / "runtime/executors/open-cake-ir-b200-v30.json"
        before = working.read_bytes()
        document = json.loads(working.read_text(encoding="utf-8"))
        document["state"] = "draft"
        document["sources"] = []
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            proposal = temporary / "proposal.json"
            output = temporary / "replacement.json"
            proposal.write_text(json.dumps(document), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/release_executor.py"),
                    "--project-root",
                    str(ROOT),
                    "--proposal",
                    str(proposal),
                    "--output",
                    str(output),
                    "--replace-unwitnessed",
                    str(working),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertTrue(output.is_file())
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["executor_id"],
                "open-cake-ir-b200-v30",
            )
            self.assertEqual(working.read_bytes(), before)

    def test_release_refuses_to_replace_a_witnessed_executor(self) -> None:
        witnessed = ROOT / "runtime/executors/open-cake-ir-b200-v29.json"
        before = witnessed.read_bytes()
        document = json.loads(witnessed.read_text(encoding="utf-8"))
        document["state"] = "draft"
        document["sources"] = []
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            proposal = temporary / "proposal.json"
            output = temporary / "replacement.json"
            proposal.write_text(json.dumps(document), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/release_executor.py"),
                    "--project-root",
                    str(ROOT),
                    "--proposal",
                    str(proposal),
                    "--output",
                    str(output),
                    "--replace-unwitnessed",
                    str(witnessed),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"not an unwitnessed working revision", completed.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(witnessed.read_bytes(), before)

if __name__ == "__main__":
    unittest.main()
