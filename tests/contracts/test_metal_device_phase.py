"""Actual CPU helper processes exercise Metal lease lifetime, without GPU APIs."""
import json
import os
from pathlib import Path
import sys
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
            self.assertEqual(report['pid'], report['group'])
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
        for name, extra in [('failure', {'exit': 7}), ('timeout', {'sleep': True})]:
            with self.subTest(name=name), self.environment():
                host = MetalArchiveHost(self.helper, {}, timeout_seconds=1)
                with self.assertRaises(RunProtocolFault):
                    host.invoke({'target': 'apple_gpu_family9', **extra}, self.root / name)
                report = json.loads((self.root / name / 'stdout.json').read_bytes())
                self.assertTrue(report['locked'])
                with self.assertRaises(ProcessLookupError):
                    os.kill(report['pid'], 0)
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
