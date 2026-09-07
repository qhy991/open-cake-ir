from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CYCLES = ("release_compiler_cycle.sh", "release_executor_cycle.sh")


class ReleaseRuntimeContractTests(unittest.TestCase):
    def test_unsupported_python_stops_both_cycles_before_any_release_write(self) -> None:
        for cycle in CYCLES:
            for explicit in (False, True):
                with self.subTest(cycle=cycle, explicit=explicit), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    project = root / "project"
                    (project / "tools").mkdir(parents=True)
                    for name in (*CYCLES, "release_runtime.sh"):
                        shutil.copyfile(ROOT / "tools" / name, project / "tools" / name)
                    for relative in (
                        "compiler/revision.json", "compiler/revision.lock.json",
                        "compiler/corpus-gate-report.json", "compiler/release-approval.json",
                        "inventory/EXECUTOR_REVISIONS.json", "runtime/executors/existing.json",
                    ):
                        path = project / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("preserved release artifact\n")
                    before = {
                        path.relative_to(project): path.read_bytes()
                        for path in project.rglob("*") if path.is_file()
                    }
                    commands = root / "commands"
                    commands.mkdir()
                    # Exercise the actual version check with an emulated 3.9 runtime;
                    # this test must also work on hosts without an installed Python 3.9.
                    unsupported = commands / "python3"
                    unsupported.write_text(
                        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " -c "
                        + shlex.quote(
                            "import sys; sys.version_info = (3, 9, 0); "
                            "sys.version = '3.9.0 (test runtime)'; exec(sys.argv[1])"
                        ) + ' "$2"\n'
                    )
                    unsupported.chmod(0o755)
                    attempted_write = root / "attempted-mktemp"
                    mktemp = commands / "mktemp"
                    mktemp.write_text(
                        "#!/bin/sh\n: > " + shlex.quote(str(attempted_write)) + "\nexit 97\n"
                    )
                    mktemp.chmod(0o755)
                    environment = {
                        **os.environ,
                        "PATH": str(commands) + os.pathsep + os.environ["PATH"],
                    }
                    environment.pop("OPEN_CAKE_PYTHON", None)
                    if explicit:
                        environment["OPEN_CAKE_PYTHON"] = str(unsupported)
                    completed = subprocess.run(
                        ["bash", str(project / "tools" / cycle)],
                        cwd=root, env=environment, capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
                    self.assertIn("require Python >= 3.10", completed.stderr)
                    self.assertIn("3.9.0", completed.stderr)
                    self.assertFalse(attempted_write.exists())
                    self.assertEqual(
                        {path.relative_to(project): path.read_bytes()
                         for path in project.rglob("*") if path.is_file()},
                        before,
                    )

    def test_supported_relative_interpreter_survives_the_cycle_directory_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "selected python"
            runtime.write_text(
                "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n'
            )
            runtime.chmod(0o755)
            project = root / "project"
            project.mkdir()
            completed = subprocess.run(
                [
                    "bash", "-euc",
                    'source "$1"; cd "$2"; "$OPEN_CAKE_PYTHON" -c '
                    "'import os, sys; print(os.environ[\"OPEN_CAKE_PYTHON\"]); print(sys.executable)'",
                    "release-runtime-test", str(ROOT / "tools/release_runtime.sh"), str(project),
                ],
                cwd=root, env={**os.environ, "OPEN_CAKE_PYTHON": "./selected python"},
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            selected, actual = completed.stdout.splitlines()
            self.assertEqual(Path(selected).resolve(), runtime.resolve())
            self.assertEqual(actual, sys.executable)


if __name__ == "__main__":
    unittest.main()
