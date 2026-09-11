"""CPU safety probes only: no sudo, NCU, CUDA context or GPU evidence."""
from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(Path(__file__).resolve().parent)]
import native_cuda_evaluate as e
import test_qualification as qualification_tests


class ProfileLaunchBoundary(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="TEST_ONLY_profile_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.row = {"id": "gemm-tiny", "source_sha256": "a" * 64,
                    "metadata": {"kernel_entry_point": "fixture_kernel"}}
        self.directory = self.root / self.row["id"] / "run"
        self.directory.mkdir(parents=True)
        env = {name: None for name in e.PROFILE_ENVIRONMENT}
        env.update(CUDA_VISIBLE_DEVICES="7", GPUQ_JOB_ID="gpuq-fixture", PYTHONDONTWRITEBYTECODE="1")
        for name in ("TMPDIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
            path = self.root / name.lower(); path.mkdir(); env[name] = str(path)
        self.admission = {"uid": 1010, "euid": 1010, "pid": 100, "visible_device": "7",
            "broker_job_id": "gpuq-fixture", "gpu_uuid": "fixture-uuid", "device_name": "NVIDIA B300 SXM6 AC",
            "compute_capability": [10, 3], "profile_environment": env, "profile_python": sys.executable,
            "profile_script": str(Path(e.__file__).resolve()), "evidence_root": str(self.root)}
        e.write_json(self.root / "admission.json", self.admission)
        self.context = {"uid": 1010, "euid": 1010, "pid": 100, "parent_pid": 99, "environment": env}
        self.ncu = self.root / "ncu"
        self.ncu.write_bytes(b"\x7fELFTEST_ONLY_not_a_GPU_executable")
        self.output_access = e.profile_output_access

    def request(self, sudo=False):
        return {"case_id": self.row["id"], "arm": "candidate", "source_sha256": self.row["source_sha256"],
            "evaluation_pid": 100, "sudo": sudo, "expected": e.profile_expected(self.admission, sudo),
            "ncu": str(self.ncu), "ncu_executable_sha256": "b" * 64,
            "version_probe": {"argv": [str(self.ncu), "--version"], "returncode": 0, "stdout": "fixture", "stderr": ""},
            "python": sys.executable, "script": self.admission["profile_script"], "evidence_root": str(self.root)}

    def successful_capture(self, argv, *, cwd, receipt):
        request = e.read_json(cwd / "profile-candidate-request.json")
        e.write_bytes(cwd / "profile-candidate-output.bin", b"TEST_ONLY")
        e.write_json(cwd / "profile-candidate-output.json", {"passed": True, "output": "profile-candidate-output.bin"})
        execution = {**request["expected"], "pid": 200, "parent_pid": 199, "status": "observed",
                     "output_access": self.output_access(cwd, "profile-candidate")}
        e.write_json(cwd / "profile-candidate-execution.json", execution)
        raw = io.StringIO(); writer = csv.writer(raw)
        writer.writerow(["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
        for metric in e.NCU_ATTRIBUTION_METRICS:
            writer.writerow(["fixture_kernel", metric, "unit", "1"])
        result = {"argv": argv, "returncode": 0, "stdout": raw.getvalue(), "stderr": ""}
        e.write_json(receipt, result)
        return result

    def launch(self, sudo):
        with patch.object(e, "profile_context", return_value=self.context), \
             patch.object(e, "capture", side_effect=lambda argv, **kw: {
                 "argv": argv, "returncode": 0, "stdout": "fixture-version", "stderr": ""}), \
             patch.object(e, "capture_profile", side_effect=self.successful_capture):
            e.profile(self.root, self.row, self.directory, str(self.ncu), sudo=sudo)

    def test_sudo_prefix_keeps_real_tool_identity_and_fixed_environment(self):
        self.launch(True)
        request = e.read_json(self.directory / "profile-candidate-request.json")
        raw = e.read_json(self.directory / "profile-candidate-command.json")
        profile = e.read_json(self.directory / "profile-candidate.json")
        self.assertEqual(raw["argv"][:6], ["/usr/bin/sudo", "-n",
            "--preserve-env=CUDA_VISIBLE_DEVICES,GPUQ_JOB_ID,TRITON_CACHE_DIR,CUDA_CACHE_PATH,PYTHONDONTWRITEBYTECODE", "--",
            "/usr/bin/env", "TMPDIR=" + self.admission["profile_environment"]["TMPDIR"]])
        self.assertEqual(raw["argv"][6:10], ["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", "900s"])
        self.assertEqual(raw["argv"][10], str(self.ncu.resolve()))
        self.assertIn("--forward-signals", raw["argv"])
        for option, value in (("--page", "details"), ("--print-details", "all"),
                ("--print-metric-name", "name"), ("--print-units", "base"),
                ("--metrics", ",".join(e.NCU_ATTRIBUTION_METRICS)), ("--launch-count", "1"),
                ("--kernel-name", "regex:^fixture_kernel$")):
            self.assertEqual(raw["argv"].count(option), 1)
            self.assertEqual(raw["argv"][raw["argv"].index(option) + 1], value)
        self.assertEqual(request["expected"]["uid"], 0)
        self.assertEqual(request["version_probe"]["argv"], [str(self.ncu.resolve()), "--version"])
        self.assertEqual(profile["tool"], {"version": "fixture-version",
                         "executable_sha256": sha256(self.ncu.read_bytes()).hexdigest()})
        self.assertEqual(self.context["euid"], 1010)

    def test_direct_mode_preserves_ordinary_child(self):
        self.launch(False)
        request = e.read_json(self.directory / "profile-candidate-request.json")
        raw = e.read_json(self.directory / "profile-candidate-command.json")
        self.assertEqual(request["expected"]["uid"], 1010)
        self.assertEqual(raw["argv"][0], "/usr/bin/timeout")
        self.assertNotIn("/usr/bin/sudo", raw["argv"])

    def test_cli_passes_explicit_privilege_scope_only_to_run(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
             patch.object(e, "run_all", return_value={"TEST_ONLY": True}) as run, \
             patch.object(e, "prepare") as prepare:
            for sudo in (False, True):
                argv = ["run", "--evidence-root", str(self.root)] + (["--ncu-sudo"] if sudo else [])
                self.assertEqual(e.main(argv), 0)
                run.assert_called_with(self.root.resolve(), e.DEFAULT_NCU, ncu_sudo=sudo)
            self.assertEqual(e.main(["prepare", "--evidence-root", str(self.root), "--ncu-sudo"]), 3)
            prepare.assert_not_called()

    def test_shell_launcher_is_refused_before_version_or_execution(self):
        self.ncu.write_bytes(b"#!/bin/sh\nexec sudo ncu \"$@\"\n")
        with patch.object(e, "profile_context", return_value=self.context), patch.object(e, "capture") as capture:
            with self.assertRaisesRegex(e.QualificationError, "real NCU ELF"):
                e.profile(self.root, self.row, self.directory, str(self.ncu))
        capture.assert_not_called()

    def test_old_wide_raw_page_is_not_rewritten_into_a_canonical_profile(self):
        # TEST_ONLY complete numeric metrics in the observed raw-page shape.
        # The real 40ac wide stdout stays outside source, unchanged and failed.
        wide = io.StringIO(); writer = csv.writer(wide)
        writer.writerow(["Kernel Name", *e.NCU_ATTRIBUTION_METRICS])
        writer.writerow(["", *["unit"] * len(e.NCU_ATTRIBUTION_METRICS)])
        writer.writerow(["fixture_kernel", *["1"] * len(e.NCU_ATTRIBUTION_METRICS)])
        with self.assertRaisesRegex(ValueError, "CSV header is missing"):
            e.build_ncu_attribution_profile(candidate_sha256=self.row["source_sha256"],
                case_id=self.row["id"], kernel_name="fixture_kernel", ncu_version="TEST_ONLY",
                ncu_executable_sha256="b" * 64, stdout=wide.getvalue().encode(), stderr=b"")

    def test_sudo_requires_existing_private_cache_environment(self):
        for name, value in (("TMPDIR", "/tmp"), ("PYTHONDONTWRITEBYTECODE", None)):
            with self.subTest(name=name):
                admission = copy.deepcopy(self.admission); admission["profile_environment"][name] = value
                context = {**self.context, "environment": admission["profile_environment"]}
                with patch.object(e, "read_json", return_value=admission), patch.object(e, "profile_context", return_value=context):
                    with self.assertRaisesRegex(e.QualificationError, "task-private"):
                        e.profile(self.root, self.row, self.directory, str(self.ncu), sudo=True)

    def test_output_permission_failure_preserves_raw_and_does_not_promote(self):
        with patch.object(e, "profile_output_access", side_effect=PermissionError("TEST_ONLY unreadable root output")):
            with self.assertRaises(PermissionError):
                self.launch(True)
        self.assertEqual(e.read_json(self.directory / "profile-candidate-command.json")["returncode"], 0)
        self.assertTrue((self.directory / "profile-candidate-output.bin").exists())
        self.assertFalse((self.directory / "profile-candidate.json").exists())

    def test_child_uid_job_cvd_and_gpu_mismatches_stop_before_launch(self):
        request = self.request(True)
        e.write_json(self.directory / "profile-candidate-request.json", request)
        correct = {**request["expected"], "pid": 200, "parent_pid": 199}
        for mutate in (lambda d: d.update(uid=1010), lambda d: d.update(euid=1010),
                       lambda d: d.update(uid=False), lambda d: d.update(euid=False),
                       lambda d: d["environment"].update(GPUQ_JOB_ID="gpuq-other"),
                       lambda d: d["environment"].update(CUDA_VISIBLE_DEVICES="6"),
                       lambda d: d["environment"].update(TMPDIR="/tmp")):
            observed = copy.deepcopy(correct); mutate(observed)
            with self.assertRaises(e.QualificationError): e.validate_profile_context(request, observed)
        observed = copy.deepcopy(correct); observed.update(gpu={**correct["gpu"], "gpu_uuid": "wrong"}, status="observed")
        with self.assertRaises(e.QualificationError): e.validate_profile_execution(request, observed)
        wrong = {**correct, "euid": 1010}
        with patch.object(e, "load_manifest", return_value={"rows": [self.row]}), \
             patch.object(e, "verify_compile"), patch.object(e, "profile_context", return_value=wrong), \
             patch.object(e, "load_inputs") as inputs, patch.object(e, "one_launch") as launch:
            with self.assertRaises(e.QualificationError): e.profile_child(self.root, self.row["id"], "candidate")
        inputs.assert_not_called(); launch.assert_not_called()
        execution = e.read_json(self.directory / "profile-candidate-execution.json")
        self.assertEqual(execution["euid"], 1010)
        self.assertEqual(execution["status"], "failed")

    def test_missing_tmpdir_is_not_repaired_before_child_observation(self):
        request = self.request(True)
        e.write_json(self.directory / "profile-candidate-request.json", request)
        observed = {**copy.deepcopy(request["expected"]), "pid": 200, "parent_pid": 199}
        observed["environment"]["TMPDIR"] = None
        with patch.object(e, "load_manifest", return_value={"rows": [self.row]}), \
             patch.object(e, "verify_compile"), patch.object(e, "profile_context", return_value=observed), \
             patch.object(e, "load_inputs") as inputs, patch.object(e, "one_launch") as launch:
            with self.assertRaises(e.QualificationError): e.profile_child(self.root, self.row["id"], "candidate")
        inputs.assert_not_called(); launch.assert_not_called()
        execution = e.read_json(self.directory / "profile-candidate-execution.json")
        self.assertIsNone(execution["environment"]["TMPDIR"])
        self.assertEqual(execution["status"], "failed")
        self.assertEqual(execution["failure_class"], "admission")


class ProfileEvidenceReplay(unittest.TestCase):
    def setUp(self):
        self.fixture = qualification_tests.ReceiptReplay(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_identity_tool_and_exact_launcher_tampering_are_refused(self):
        directory = self.fixture.directory
        cases = [("request", lambda r: r.update(ncu_executable_sha256="d" * 64)),
                 ("request", lambda r: r.update(sudo=True)),
                 ("request", lambda r: r["expected"].update(uid=0)),
                 ("execution", lambda r: r.update(euid=0)),
                 ("execution", lambda r: r["gpu"].update(gpu_uuid="other")),
                 ("execution", lambda r: r["environment"].update(GPUQ_JOB_ID="gpuq-other")),
                 ("command", lambda r: r["argv"].insert(0, "/tmp/sudo-wrapper")),
                 ("command", lambda r: r["argv"].__setitem__(r["argv"].index("--page") + 1, "raw")),
                 ("command", lambda r: r["argv"].__setitem__(r["argv"].index("--print-metric-name") + 1, "label")),
                 ("command", lambda r: r.update(error="timeout", cleanup="unknown")),
                 ("version", lambda r: r["argv"].__setitem__(0, "/tmp/wrapper"))]
        for kind, mutate in cases:
            with self.subTest(kind=kind):
                path = directory / ("profile-candidate-" + kind + ".json")
                original = path.read_bytes(); value = json.loads(original); mutate(value)
                path.write_text(json.dumps(value))
                with self.assertRaises(e.QualificationError): self.fixture.verify()
                path.write_bytes(original)


class ProfileCaptureSafety(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="TEST_ONLY_profile_capture_")
        self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)

    def test_actual_cpu_timeout_retains_output_and_reaps_synchronous_process(self):
        argv = [sys.executable, "-c", "import time;print('retained-before-timeout',flush=True);time.sleep(30)"]
        receipt = self.root / "command.json"
        with patch.object(e, "PROFILE_CAPTURE_TIMEOUT", 0.2), patch.object(e, "PROFILE_CLEANUP_TIMEOUT", 2):
            with self.assertRaises(subprocess.TimeoutExpired): e.capture_profile(argv, cwd=self.root, receipt=receipt)
        raw = e.read_json(receipt)
        self.assertEqual(raw["stdout"], "retained-before-timeout\n")
        self.assertEqual(raw["returncode"], -signal.SIGTERM)
        self.assertEqual(raw["cleanup"], "launcher_reaped_and_pipes_closed")

    def test_parent_group_sigterm_and_sigint_retain_receipt_and_stop_child(self):
        def state(pid):
            result = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "pid=,pgid=,stat=,command="],
                                    capture_output=True, text=True)
            fields = result.stdout.split(None, 3)
            return fields if fields and not fields[2].startswith("Z") else None
        for signum in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=signum):
                directory = self.root / str(signum); directory.mkdir()
                marker = directory / "child.json"; receipt = directory / "capture.json"
                child_code = (
                    "import os,json,time;from pathlib import Path;"
                    "p=Path(" + repr(str(marker)) + ");"
                    "p.write_text(json.dumps({'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgrp()}));"
                    "print('TEST_ONLY_profile_signal_child_ready',flush=True);time.sleep(30)")
                parent_code = (
                    "import sys;from pathlib import Path;sys.path[:0]=" + repr([str(ROOT / "tools"), str(ROOT / "src")]) + ";"
                    "import native_cuda_evaluate as e;e.PROFILE_CAPTURE_TIMEOUT=10;e.PROFILE_CLEANUP_TIMEOUT=2;"
                    "e.capture_profile(" + repr([sys.executable, "-c", child_code]) + ",cwd=Path(" + repr(str(directory))
                    + "),receipt=Path(" + repr(str(receipt)) + "))")
                parent = subprocess.Popen([sys.executable, "-B", "-c", parent_code],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                child_pid = None
                try:
                    deadline = time.monotonic() + 5
                    while not marker.exists() and parent.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(marker.exists(), "controlled CPU child did not start")
                    observed = json.loads(marker.read_text()); child_pid = observed["pid"]
                    self.assertEqual(observed["ppid"], parent.pid)
                    self.assertEqual(observed["pgid"], child_pid)
                    self.assertEqual(os.getpgid(parent.pid), parent.pid)
                    os.killpg(parent.pid, signum)
                    parent.communicate(timeout=5)
                    self.assertIn(parent.returncode, (128 + signum, -signum))
                    self.assertIsNone(state(child_pid), "controlled child outlived cancelled parent")
                    raw = e.read_json(receipt)
                    self.assertEqual(raw["cleanup_signal"], "TERM_to_direct_launcher")
                    self.assertEqual(raw["cleanup"], "launcher_reaped_and_pipes_closed")
                    self.assertEqual(raw["stdout"], "TEST_ONLY_profile_signal_child_ready\n")
                    if signum == signal.SIGTERM:
                        self.assertEqual(parent.returncode, 143)
                        self.assertEqual(raw["termination_signal"], signal.SIGTERM)
                        self.assertEqual(raw["error"], "SystemExit")
                finally:
                    if parent.poll() is None:
                        parent.kill(); parent.wait(timeout=3)
                    if child_pid is not None:
                        live = state(child_pid)
                        if live is not None:
                            self.assertEqual(int(live[1]), child_pid)
                            self.assertIn("TEST_ONLY_profile_signal_child_ready", live[3])
                            os.kill(child_pid, signal.SIGTERM)
                            deadline = time.monotonic() + 3
                            while state(child_pid) is not None and time.monotonic() < deadline: time.sleep(0.01)
                            self.assertIsNone(state(child_pid), "owned CPU fixture cleanup failed")

    def test_sigint_and_sigterm_handlers_are_restored_after_success_and_failure(self):
        originals = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
        custom = lambda signum, frame: None
        for signum in originals: signal.signal(signum, custom)
        try:
            for failure in (False, True):
                process = Mock(returncode=0); process.poll.return_value = None
                process.communicate.side_effect = ([RuntimeError("TEST_ONLY"), (b"", b"")]
                                                   if failure else [(b"", b"")])
                with patch.object(e.subprocess, "Popen", return_value=process):
                    if failure:
                        with self.assertRaises(RuntimeError):
                            e.capture_profile(["fixture"], cwd=self.root, receipt=self.root / "failure.json")
                    else:
                        e.capture_profile(["fixture"], cwd=self.root, receipt=self.root / "success.json")
                for signum in originals: self.assertIs(signal.getsignal(signum), custom)
        finally:
            for signum, handler in originals.items(): signal.signal(signum, handler)

    def test_first_cancellation_is_deferred_at_real_spawn_and_receipt_boundaries(self):
        for edge, signum in (("spawn", signal.SIGTERM), ("spawn", signal.SIGINT),
                             ("receipt", signal.SIGTERM), ("receipt", signal.SIGINT)):
            with self.subTest(edge=edge, signal=signum):
                directory = self.root / (edge + str(signum)); directory.mkdir()
                marker = directory / "child.json"; receipt = directory / "capture.json"
                outcome = directory / "outcome.json"
                child_code = ("import os,json,time;from pathlib import Path;Path(" + repr(str(marker))
                    + ").write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp()}));"
                    + ("print('TEST_ONLY_spawn_edge',flush=True);time.sleep(30)" if edge == "spawn"
                       else "print('completed-child',flush=True)"))
                injection = (f'''
OriginalPopen=e.subprocess.Popen
class AtRealSpawnReturn(OriginalPopen):
    def _execute_child(self,*args,**kwargs):
        super()._execute_child(*args,**kwargs)
        deadline=time.monotonic()+5
        while not Path({str(marker)!r}).exists():
            assert time.monotonic()<deadline
            time.sleep(.01)
        os.kill(os.getpid(),{int(signum)})
e.subprocess.Popen=AtRealSpawnReturn
''' if edge == "spawn" else f'''
original_write=e.write_json
def at_receipt(path,value):
    info['signal_disposition_is_ignore']=signal.getsignal({int(signum)})==signal.SIG_IGN
    os.kill(os.getpid(),{int(signum)})
    info['writer_continued_after_signal']=True
    original_write(path,value)
e.write_json=at_receipt
''')
                parent_code = f'''
import os,signal,sys,json,time
from pathlib import Path
sys.path[:0]={[str(ROOT / "tools"), str(ROOT / "src")]!r}
import native_cuda_evaluate as e
e.PROFILE_CAPTURE_TIMEOUT=10;e.PROFILE_CLEANUP_TIMEOUT=2
info={{'capture_returned_success':False}}
{injection}
try:
    e.capture_profile({[sys.executable, "-c", child_code]!r},cwd=Path({str(directory)!r}),receipt=Path({str(receipt)!r}))
    info['capture_returned_success']=True
except (SystemExit,KeyboardInterrupt) as error:
    info['exit_code']=error.code if isinstance(error,SystemExit) else 130
    raise
finally:
    info['handler_is_default']=signal.getsignal(signal.SIGTERM)==signal.SIG_DFL
    info['sigint_handler_restored']=signal.getsignal(signal.SIGINT)==signal.default_int_handler
    Path({str(outcome)!r}).write_text(json.dumps(info))
'''
                parent = subprocess.Popen([sys.executable, "-B", "-c", parent_code],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                child_pid = None
                try:
                    stdout, stderr = parent.communicate(timeout=10)
                    self.assertIn(parent.returncode, (128 + signum, -signum), stderr.decode())
                    info = json.loads(outcome.read_text())
                    self.assertFalse(info["capture_returned_success"])
                    self.assertTrue(info["handler_is_default"])
                    self.assertTrue(info["sigint_handler_restored"])
                    self.assertEqual(info["exit_code"], 128 + signum)
                    raw = e.read_json(receipt)
                    if edge == "spawn":
                        observed = json.loads(marker.read_text()); child_pid = observed["pid"]
                        state = subprocess.run(["/bin/ps", "-p", str(child_pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
                        self.assertTrue(not state or state.startswith("Z"), "child survived constructor-edge cancellation")
                        self.assertEqual(raw["termination_signal"], signum)
                        self.assertEqual(raw["cleanup_signal"], "TERM_to_direct_launcher")
                    else:
                        self.assertFalse(info["signal_disposition_is_ignore"])
                        self.assertTrue(info["writer_continued_after_signal"])
                        # The child completed before cancellation during its raw
                        # receipt write. Its bytes remain true; the call cancels.
                        self.assertEqual(raw["returncode"], 0)
                        self.assertEqual(raw["stdout"], "completed-child\n")
                finally:
                    if parent.poll() is None:
                        parent.kill(); parent.wait(timeout=3)
                    if edge == "spawn" and marker.exists():
                        observed = json.loads(marker.read_text()); child_pid = observed["pid"]
                        status = subprocess.run(["/bin/ps", "-p", str(child_pid), "-o", "pgid=,stat=,command="], capture_output=True, text=True).stdout.split(None, 2)
                        if status and not status[1].startswith("Z"):
                            self.assertEqual(int(status[0]), observed["pgid"])
                            self.assertIn("TEST_ONLY_spawn_edge", status[2])
                            os.kill(child_pid, signal.SIGTERM)
                            deadline = time.monotonic() + 3
                            while time.monotonic() < deadline:
                                remaining = subprocess.run(["/bin/ps", "-p", str(child_pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
                                if not remaining or remaining.startswith("Z"): break
                                time.sleep(0.01)
                            self.assertTrue(not remaining or remaining.startswith("Z"), "owned constructor-edge fixture cleanup failed")

    def test_interrupt_and_exception_preserve_original_cause_and_accumulated_output(self):
        for index, error in enumerate((KeyboardInterrupt("TEST_ONLY interrupt"), RuntimeError("TEST_ONLY failure"),
                subprocess.TimeoutExpired(["fixture"], 1, output=b"first\n"))):
            process = Mock(returncode=143)
            process.poll.return_value = None
            process.communicate.side_effect = [error, (b"first\nlast\n", b"raw-stderr")]
            path = self.root / f"command-{index}.json"
            with patch.object(e.subprocess, "Popen", return_value=process) as launch:
                with self.assertRaises(type(error)) as caught: e.capture_profile(["fixture"], cwd=self.root, receipt=path)
            self.assertIs(caught.exception, error)
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            process.send_signal.assert_called_once_with(signal.SIGTERM)
            process.kill.assert_not_called()
            self.assertEqual(e.read_json(path)["stdout"], "first\nlast\n")

    def test_descendant_pipe_timeout_is_unknown_without_killing_sudo_frontend(self):
        original = KeyboardInterrupt("TEST_ONLY interrupt")
        process = Mock(returncode=None); process.poll.return_value = None
        process.communicate.side_effect = [original, subprocess.TimeoutExpired(["fixture"], 15, output=b"retained\n")]
        path = self.root / "unknown.json"
        with patch.object(e.subprocess, "Popen", return_value=process):
            with self.assertRaises(KeyboardInterrupt) as caught: e.capture_profile(["fixture"], cwd=self.root, receipt=path)
        self.assertIs(caught.exception, original)
        raw = e.read_json(path)
        self.assertEqual(raw["cleanup"], "unknown")
        self.assertEqual(raw["stdout"], "retained\n")
        self.assertIsNone(raw["returncode"])
        process.kill.assert_not_called()
        self.assertGreater(e.PROFILE_CAPTURE_TIMEOUT, e.PROFILE_TIMEOUT + e.PROFILE_KILL_AFTER)


if __name__ == "__main__": unittest.main()
