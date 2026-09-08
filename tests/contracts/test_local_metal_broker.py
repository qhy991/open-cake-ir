"""Local job admission does not borrow CUDA exclusive-card recovery semantics."""
import os
import fcntl
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.attempts import BrokerAttempt, is_resubmittable_admission_failure, valid_job_mode
from open_cake_ir.evaluation.local_broker import _acquire, observe_local_metal_job
from open_cake_ir.lab.runtime import _BROKER_JOB_OBSERVATION
from open_cake_ir.lab.process import SupervisedProcessTimeout, run_supervised


class LocalMetalBrokerTests(unittest.TestCase):
    def test_namespaces_preserve_real_allocation_domain(self):
        self.assertTrue(valid_job_mode("metal-123456789abc", "local_serialized"))
        self.assertTrue(valid_job_mode("gpuq-123456789abc", "exclusive"))
        for job, mode in (("metal-123456789abc", "exclusive"), ("gpuq-123456789abc", "local_serialized"), ("metal-000000000000", "local_serialized")):
            self.assertFalse(valid_job_mode(job, mode))
        self.assertEqual(_BROKER_JOB_OBSERVATION.findall(b"[metal-run] accepted job metal-123456789abc\n"), [b"metal-123456789abc"])
        self.assertEqual(_BROKER_JOB_OBSERVATION.findall(b"[gpu-run] accepted job metal-123456789abc\n"), [])

    def test_metal_never_borrows_the_cuda_zero_work_retry(self):
        common = dict(candidate_sha256="a" * 64, manifest_sha256="b" * 64, policy_sha256="c" * 64,
            evaluator_arguments_sha256="d" * 64, admitted=False, error="gpu_admission_differs",
            compiler_invocations=0, module_loads=0, preflight_calls=0, kernel_calls=0, timing_samples=0, fallback_calls=0, receipt=None)
        self.assertFalse(is_resubmittable_admission_failure(BrokerAttempt(job_id="metal-123456789abc", mode="local_serialized", **common)))
        self.assertTrue(is_resubmittable_admission_failure(BrokerAttempt(job_id="gpuq-123456789abc", mode="exclusive", **common)))

    def test_live_lock_descriptor_is_required_and_a_second_job_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metal.lock"
            fd = _acquire(path)
            try:
                with self.assertRaises(BlockingIOError):
                    _acquire(path)
                with patch("open_cake_ir.evaluation.local_broker._lock_path", return_value=path), \
                     patch.dict(os.environ, METAL_JOB_ID="metal-123456789abc", METAL_BROKER_LOCK_FD=str(fd)):
                    self.assertEqual(observe_local_metal_job(), "metal-123456789abc")
                with patch.dict(os.environ, METAL_JOB_ID="metal-123456789abc", METAL_BROKER_LOCK_FD="0"):
                    with self.assertRaises(ValueError):
                        observe_local_metal_job()
            finally:
                os.close(fd)
            next_fd = _acquire(path)
            os.close(next_fd)


_CPU_WORKER = r"""import argparse, fcntl, json, os, sys, time
from pathlib import Path
import open_cake_ir
from open_cake_ir.evaluation import local_broker
parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
job = local_broker.observe_local_metal_job()
try:
    fd = local_broker._acquire(local_broker._lock_path())
except BlockingIOError:
    contended = True
else:
    os.close(fd)
    contended = False
record = {"job_id": job, "pid": os.getpid(), "process_group": os.getpgrp(),
          "isolated": sys.flags.isolated, "lock_contended": contended,
          "lock_path": str(local_broker._lock_path()),
          "worker_file": __file__, "broker_file": local_broker.__file__,
          "package_file": open_cake_ir.__file__, "sys_path": sys.path,
          "python_environment": {name: os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME")}}
with Path(args.output).open("x") as stream:
    json.dump(record, stream)
print("CPU worker reached through actual broker exec", flush=True)
if request.get("sleep"):
    time.sleep(30)
"""


class SourceBoundBrokerProcessTests(unittest.TestCase):
    """Actual CPU broker→exec→worker processes; no provider, candidate or GPU."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="source-bound-broker-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        checkout = Path(__file__).resolve().parents[2]
        self.source = self.root / "bound/src"
        shutil.copytree(checkout / "src/open_cake_ir", self.source / "open_cake_ir",
                        ignore=shutil.ignore_patterns("__pycache__"))
        self.worker = self.source / "open_cake_ir/cpu_worker_fixture.py"
        self.worker.write_text(_CPU_WORKER)
        self.bootstrap = self.source / "open_cake_ir/evaluation/source_bootstrap.py"
        spec = importlib.util.spec_from_file_location("cpu_fixture_bootstrap", self.bootstrap)
        self.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.builder)
        self.unrelated = self.root / "unrelated-cwd"
        self.unrelated.mkdir()
        self.lock_root = self.root / "locks"
        self.lock_root.mkdir()
        self.request = self.root / "request.json"
        self.request.write_text("{}")
        self.output = self.root / "output.json"
        self.hostile = self.root / "other-checkout"
        (self.hostile / "open_cake_ir").mkdir(parents=True)
        (self.hostile / "open_cake_ir/__init__.py").write_text('raise RuntimeError("HOSTILE CHECKOUT IMPORTED")')

    def environment(self, poisoned):
        # Only fixture child environments change; real host/canary state is untouched.
        environment = dict(os.environ, TMPDIR=str(self.lock_root))
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        if poisoned:
            environment.update(PYTHONPATH=str(self.hostile), PYTHONHOME=str(self.hostile))
        return environment

    def command(self):
        return self.builder.module_command(sys.executable, "open_cake_ir.evaluation.local_broker",
            "--worker-module", "open_cake_ir.cpu_worker_fixture",
            "--request", str(self.request), "--output", str(self.output))

    def test_both_hops_bind_source_without_or_despite_python_environment(self):
        for poisoned in (False, True):
            with self.subTest(poisoned=poisoned):
                self.output = self.root / f"output-{poisoned}.json"
                environment = self.environment(poisoned)
                process = subprocess.Popen(self.command(), cwd=self.unrelated, env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                stdout, stderr = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, stderr.decode())
                record = json.loads(self.output.read_text())
                self.assertEqual(record["pid"], process.pid)  # Broker exec, not another child.
                self.assertEqual(record["process_group"], process.pid)
                self.assertEqual(record["isolated"], 1)
                self.assertEqual(record["worker_file"], str(self.worker))
                self.assertEqual(record["broker_file"], str(self.bootstrap.with_name("local_broker.py")))
                self.assertEqual(record["package_file"], str(self.source / "open_cake_ir/__init__.py"))
                self.assertEqual(record["sys_path"][0], str(self.source))
                self.assertNotIn(str(self.hostile), record["sys_path"])
                self.assertTrue(record["lock_contended"])
                self.assertEqual(record["python_environment"], {name: environment.get(name) for name in ("PYTHONPATH", "PYTHONHOME")})
                self.assertIn(record["job_id"].encode(), stderr)
                self.assertIn(b"actual broker exec", stdout)

    def test_generic_worker_package_entrypoint_keeps_the_bound_source(self):
        package = self.source / "open_cake_ir/cpu_package_fixture"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "__main__.py").write_text(_CPU_WORKER)
        command = self.command()
        command[command.index("--worker-module") + 1] = "open_cake_ir.cpu_package_fixture"
        result = subprocess.run(command, cwd=self.unrelated, env=self.environment(True),
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        record = json.loads(self.output.read_text())
        self.assertEqual(record["worker_file"], str(package / "__main__.py"))
        self.assertTrue(record["lock_contended"])

    def test_wrong_root_or_module_refuses_before_target_execution(self):
        relocated = self.root / "relocated.py"
        shutil.copyfile(self.bootstrap, relocated)
        outside = self.root / "escaped-worker.py"
        outside.write_text('raise AssertionError("OUTSIDE WORKER EXECUTED")')
        (self.source / "open_cake_ir/escaped.py").symlink_to(outside)
        cases = ((self.bootstrap, "json"), (relocated, "open_cake_ir.cpu_worker_fixture"),
                 (self.bootstrap, "open_cake_ir.escaped"))
        for script, module in cases:
            with self.subTest(script=script, module=module):
                process = subprocess.run([sys.executable, "-I", str(script), module],
                    cwd=self.unrelated, env=self.environment(True), capture_output=True, timeout=15)
                self.assertNotEqual(process.returncode, 0)
                self.assertNotIn(b"OUTSIDE WORKER EXECUTED", process.stderr)
                self.assertFalse(self.output.exists())
        process = subprocess.run([sys.executable, str(self.bootstrap), "open_cake_ir.cpu_worker_fixture"],
            cwd=self.unrelated, env=self.environment(False), capture_output=True, timeout=15)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(b"requires isolated Python", process.stderr)

    def test_existing_supervisor_timeout_releases_the_inherited_lock(self):
        self.request.write_text('{"sleep":true}')
        with self.assertRaises(SupervisedProcessTimeout) as captured:
            run_supervised(self.command(), cwd=self.unrelated, environment=self.environment(True), timeout_seconds=2)
        self.assertIn(b"CPU worker reached", captured.exception.stdout)
        record = json.loads(self.output.read_text())
        self.assertTrue(record["lock_contended"])
        fd = os.open(record["lock_path"], os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


if __name__ == "__main__":
    unittest.main()
