"""Actual CPU helper processes exercise Metal lease lifetime, without GPU APIs."""
import json
import os
from pathlib import Path
import sys
import signal
import subprocess
import time
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation import local_broker
from open_cake_ir.lab.metal_build import MetalArchiveHost
from open_cake_ir.lab.faults import RunProtocolFault


HELPER = '''import fcntl, json, os, sys, time
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
lock = Path(os.environ['TMPDIR']) / ('open-cake-ir-metal-' + str(os.geteuid()) + '.lock')
fd = os.open(lock, os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locked = False
except BlockingIOError:
    locked = True
finally:
    os.close(fd)
report = {'job': os.environ.get('METAL_JOB_ID'), 'locked': locked,
          'pid': os.getpid(), 'group': os.getpgrp(),
          'descriptor': os.fstat(int(os.environ['METAL_BROKER_LOCK_FD'])).st_ino}
if request.get('marker'):
    Path(request['marker']).write_text(json.dumps(report))
print(json.dumps(report), flush=True)
if request.get('sleep'):
    time.sleep(30)
sys.exit(request.get('exit', 0))
'''


class MetalDevicePhaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='metal-device-phase-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.helper = self.root / 'helper'
        self.helper.write_text('#!' + sys.executable + '\n' + HELPER)
        self.helper.chmod(0o700)
        self.lock = self.root / f'open-cake-ir-metal-{os.geteuid()}.lock'
        self.host = MetalArchiveHost(self.helper, {}, timeout_seconds=5)

    def environment(self):
        return patch.dict(os.environ, {'PATH': os.defpath, 'TMPDIR': str(self.root)}, clear=True)

    def unlocked(self):
        descriptor = local_broker._acquire(self.lock)
        os.close(descriptor)

    def test_fresh_helper_holds_lock_and_releases_before_parent_report(self):
        with self.environment():
            self.unlocked()
            report = self.host.invoke({'target': 'apple_gpu_family9'}, self.root / 'first')
            self.assertTrue(report['locked'])
            self.assertRegex(report['job'], r'^metal-[0-9a-f]{12}$')
            self.assertEqual(report['group'], os.getpgrp())
            self.assertNotEqual(report['pid'], os.getpid())
            self.assertNotIn('METAL_BROKER_LOCK_FD', os.environ)
            self.unlocked()
            second = self.host.invoke({'target': 'apple_gpu_family9'}, self.root / 'second')
            self.assertNotEqual(second['job'], report['job'])
            self.unlocked()

    def test_busy_refuses_before_helper_without_releasing_other_owner(self):
        descriptor = local_broker._acquire(self.lock)
        try:
            with self.environment(), self.assertRaises(RunProtocolFault):
                self.host.invoke({'target': 'apple_gpu_family9'}, self.root / 'busy')
            self.assertEqual((self.root / 'busy/stdout.json').read_bytes(), b'')
            self.assertIn(b'metal_broker_busy', (self.root / 'busy/stderr.log').read_bytes())
            with self.assertRaises(BlockingIOError):
                local_broker._acquire(self.lock)
        finally:
            os.close(descriptor)

    def test_failure_and_timeout_keep_streams_and_release_by_process_exit(self):
        popen = subprocess.Popen
        for name, extra, timeout, reason in [
            ('failure', {'exit': 7}, 10, 'helper process failed'),
            ('timeout', {'sleep': True}, 1, 'helper did not finish'),
            ('startup_timeout', {'sleep': True}, 0, 'helper did not finish'),
        ]:
            children = []
            def start(*args, **kwargs):
                child = popen(*args, **kwargs)
                children.append(child)
                return child
            with (self.subTest(name=name), self.environment(),
                  patch.object(subprocess, 'Popen', side_effect=start)):
                host = MetalArchiveHost(self.helper, {}, timeout_seconds=timeout)
                with self.assertRaisesRegex(RunProtocolFault, reason) as fault:
                    host.invoke({'target': 'apple_gpu_family9', **extra}, self.root / name)
                self.assertEqual(len(children), 1)
                child = children[0]
                stdout = (self.root / name / 'stdout.json').read_bytes()
                stderr = (self.root / name / 'stderr.log').read_bytes()
                if name == 'failure':
                    self.assertEqual(child.returncode, 7)
                    report = json.loads(stdout)
                    self.assertEqual(report['pid'], child.pid)
                    self.assertTrue(report['locked'])
                else:
                    timeout_error = fault.exception.__cause__
                    self.assertIsInstance(timeout_error, subprocess.TimeoutExpired)
                    self.assertEqual(child.returncode, -signal.SIGKILL)
                    # Timeout can precede a complete report, including interpreter startup.
                    self.assertEqual(stdout, timeout_error.stdout or b'')
                    self.assertEqual(stderr, timeout_error.stderr or b'')
                    if name == 'startup_timeout':
                        self.assertEqual(stdout, b'')
                with self.assertRaises(ProcessLookupError):
                    os.kill(child.pid, 0)
                self.unlocked()

    def test_borrowed_allocation_remains_owned_after_helper_exit(self):
        with self.environment(), patch.object(local_broker, '_lock_path', return_value=self.lock):
            job = local_broker.admit_local_job('metal')
            descriptor = int(os.environ['METAL_BROKER_LOCK_FD'])
            try:
                report = self.host.invoke({'target': 'apple_gpu_family9'}, self.root / 'borrowed')
                self.assertEqual(report['job'], job)
                self.assertTrue(report['locked'])
                with self.assertRaises(BlockingIOError):
                    local_broker._acquire(self.lock)
                self.assertEqual(local_broker.observe_local_metal_job(), job)
            finally:
                os.close(descriptor)
        self.unlocked()

    def test_stale_or_wrong_family_identity_never_starts_native_helper(self):
        for values in ({'METAL_JOB_ID': 'metal-123456789abc'},
                       {'METAL_JOB_ID': 'maca-123456789abc', 'METAL_BROKER_LOCK_FD': '0'}):
            with self.subTest(values=values), self.environment(), patch.dict(os.environ, values):
                destination = self.root / ('stale-' + str(len(values)))
                with self.assertRaises(ValueError):
                    self.host.invoke({'target': 'apple_gpu_family9'}, destination)
                self.assertFalse((destination / 'stdout.json').exists())
                self.unlocked()

    def test_outer_supervisor_cancellation_includes_the_device_helper(self):
        marker = self.root / 'started.json'
        code = (
            'from pathlib import Path; from open_cake_ir.lab.metal_build import MetalArchiveHost; '
            f'MetalArchiveHost(Path({str(self.helper)!r}), {{}}, timeout_seconds=20).invoke('
            f'{{"target":"apple_gpu_family9","sleep":True,"marker":{str(marker)!r}}}, '
            f'Path({str(self.root / "cancelled")!r}))'
        )
        source = Path(__file__).resolve().parents[2] / 'src'
        environment = dict(PATH=os.defpath, TMPDIR=str(self.root), PYTHONPATH=str(source))
        process = subprocess.Popen([sys.executable, '-c', code], env=environment,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(marker.exists(), 'native helper did not start')
            report = json.loads(marker.read_text())
            self.assertTrue(report['locked'])
            self.assertEqual(report['group'], process.pid)
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            deadline = time.monotonic() + 5
            while True:
                try:
                    self.unlocked()
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
