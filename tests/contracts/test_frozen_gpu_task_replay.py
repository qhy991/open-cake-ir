from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE_SOURCE = "c40bb399c7bebe026bcd30e27255dfc15ad23f5c"
FMA_SOURCE = "d9d56e835cf96eecf70e0259b65bc1b20c4f6f0d"


def _checkout(destination: Path, revision: str) -> None:
    # A local clone leaves the user's worktrees and refs alone. Its objects include
    # the complete frozen task and Compiler; an archive of current main cannot do this.
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", "--quiet", str(ROOT), str(destination)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--quiet", "--detach", revision],
        check=True, capture_output=True, text=True,
    )


def _prepare(source: Path, script: str, output: Path, *extra: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(source / script), "--output-root", str(output), *extra],
        cwd=source, env=environment, capture_output=True, text=True, timeout=30,
    )


class FrozenGpuTaskReplayTests(unittest.TestCase):
    def test_state_v5_prepares_from_its_complete_v41_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "bundle"
            _checkout(source, STATE_SOURCE)
            script = "examples/gpu/state_store_b200_correctness/prepare_candidate.py"
            completed = _prepare(source, script, output)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            preflight = json.loads(completed.stdout)
            provenance = json.loads((output / "candidate/provenance.json").read_text())
            kernel = (output / "candidate/kernel.py").read_text()
            self.assertEqual(preflight["compiler_revision_id"], "open-cake-ir-sm100a-v41")
            self.assertTrue(preflight["positive"]["accepted"])
            self.assertTrue(preflight["positive"]["lowering_eligible"])
            for name, code, path in (
                ("owner", "STATE_STORE_PROGRAM_OWNER", "access_maps[2].indices[0]"),
                ("axis", "STATE_STORE_PROGRAM_AXIS_COVERAGE", "program_map.axes[1]"),
            ):
                negative = preflight["negative"][f"state-store-b8-smoke-{name}-drift.json"]
                self.assertFalse(negative["accepted"])
                self.assertFalse(negative["lowering_eligible"])
                self.assertEqual(negative["decisive_findings"], [[code, path]])
            compile(kernel, "<frozen-state-store>", "exec")
            self.assertIn("tl.store(\n        state + batch * D_STATE_1 + state_d1_offsets,", kernel)
            self.assertNotIn("torch.empty((8, 128)", kernel)
            self.assertEqual(provenance["compiler_source_commit"], "ee32b76870c4166c8143e8774c37c1212a0b6e1c")
            self.assertEqual(provenance["shape"], [8, 128])
            self.assertIn("no arbitrary-n", provenance["claim_boundary"])
            repeated = _prepare(source, script, output)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual((output / "candidate/kernel.py").read_text(), kernel)

    def test_fma_task_prepares_from_its_complete_v41_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            output = Path(directory) / "bundle"
            _checkout(source, FMA_SOURCE)
            completed = _prepare(
                source, "examples/gpu/fma_b200_correctness/prepare.py", output,
                "--python", sys.executable, "--judge-cwd", str(source),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            contract = json.loads(completed.stdout)
            task = json.loads((output / "task.json").read_text())
            self.assertEqual(contract["compiler_revision_id"], "open-cake-ir-sm100a-v41")
            self.assertEqual(contract["shape"], [8, 128])
            self.assertEqual(len(contract["workloads"]), 12)
            self.assertFalse(contract["performance_measured"])
            self.assertEqual([stage["kind"] for stage in task["stages"]], ["correctness", "sanitize", "sanitize"])
            for name in ("ordinary", "nested"):
                generated = (output / f"candidate/{name}.py").read_text()
                compile(generated, f"<frozen-{name}>", "exec")
                self.assertIn("fma.rn.f32", generated)


if __name__ == "__main__":
    unittest.main()
