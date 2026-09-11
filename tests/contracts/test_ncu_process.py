import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.lab import ncu_process as ncu


class NcuProcessTests(unittest.TestCase):
    def test_only_restricted_linux_nonroot_needs_sudo(self):
        for platform, uid, params, expected in (
            ('linux', 1010, 'RmProfilingAdminOnly: 1\n', True),
            ('linux', 1010, 'RmProfilingAdminOnly: 0\n', False),
            ('linux', 0, 'RmProfilingAdminOnly: 1\n', False),
            ('darwin', 501, '', False),
        ):
            with self.subTest(platform=platform, uid=uid, params=params), \
                 patch.object(ncu.sys, 'platform', platform), \
                 patch.object(ncu.os, 'geteuid', return_value=uid), \
                 patch.object(Path, 'read_text', return_value=params):
                self.assertEqual(ncu.requires_sudo(), expected)

    def test_direct_execution_keeps_existing_supervisor(self):
        with patch.object(ncu, 'requires_sudo', return_value=False), \
             patch.object(ncu, 'run_supervised', return_value='direct') as run:
            self.assertEqual(ncu.run_ncu(['/ncu'], cwd=Path('/tmp'), environment={}), 'direct')
        self.assertEqual(run.call_args.args[0], ['/ncu'])

    def test_privilege_requires_broker_assignment_before_spawn(self):
        with patch.object(ncu, 'requires_sudo', return_value=True), \
             patch.object(ncu.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'CUDA_VISIBLE_DEVICES'):
                ncu.run_ncu(['/ncu'], cwd=Path('/tmp'), environment={})
        spawn.assert_not_called()

    def test_output_creation_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result'
            ncu.write_new(path, b'first', (os.geteuid(), os.getegid()))
            with self.assertRaises(FileExistsError):
                ncu.write_new(path, b'second')
            self.assertEqual(path.read_bytes(), b'first')

    def test_sudo_environment_is_narrow_and_noninteractive(self):
        environment = {'CUDA_VISIBLE_DEVICES': 'cpu-fixture', 'GPUQ_JOB_ID': 'cpu-fixture',
                       'PATH': '/untrusted', 'LD_LIBRARY_PATH': '/untrusted', 'AUTH_TOKEN': 'excluded'}
        with patch.object(ncu, 'requires_sudo', return_value=True), \
             patch.object(ncu.subprocess, 'Popen') as spawn:
            spawn.return_value.poll.return_value = 0
            spawn.return_value.returncode = 0
            ncu.run_ncu(['/ncu'], cwd=Path('/tmp'), environment=environment)
        command = spawn.call_args.args[0]
        self.assertEqual(command[:4], ['/usr/bin/sudo', '-n', '--', '/usr/bin/timeout'])
        self.assertIn('PATH=/usr/bin:/bin', command)
        self.assertIn('CUDA_VISIBLE_DEVICES=cpu-fixture', command)
        self.assertFalse(any('/untrusted' in arg or 'excluded' in arg for arg in command))

    def test_cancellation_retains_diagnostics(self):
        error = ncu.NcuProcessCancelled(15, b'output', b'diagnostic')
        self.assertEqual((error.code, error.stdout, error.stderr), (143, b'output', b'diagnostic'))

    def test_output_owner_distinguishes_direct_root_and_privileged_child(self):
        with patch.dict(os.environ, {'SUDO_UID': '1010', 'SUDO_GID': '1010'}), \
             patch.object(ncu.os, 'geteuid', return_value=0), \
             patch.object(ncu.os, 'getegid', return_value=0):
            self.assertEqual(ncu.profile_output_owner({'uid': 0, 'gid': 0}), (0, 0))
            self.assertEqual(ncu.profile_output_owner({'uid': 1010, 'gid': 1010}), (1010, 1010))
            with self.assertRaises(ValueError):
                ncu.profile_output_owner({'uid': 2000, 'gid': 2000})
        with patch.object(ncu.os, 'geteuid', return_value=1010), \
             patch.object(ncu.os, 'getegid', return_value=1010):
            with self.assertRaises(ValueError):
                ncu.profile_output_owner({'uid': 0, 'gid': 0})

    def test_output_privileges_restore_on_creation_failure(self):
        calls = []
        with patch.object(ncu.os, 'geteuid', return_value=0), \
             patch.object(ncu.os, 'getegid', return_value=0), \
             patch.object(ncu.os, 'seteuid', side_effect=lambda x: calls.append(('uid', x))), \
             patch.object(ncu.os, 'setegid', side_effect=lambda x: calls.append(('gid', x))), \
             patch.object(Path, 'open', side_effect=FileExistsError):
            with self.assertRaises(FileExistsError):
                ncu.write_new(Path('/unused'), b'content', (1010, 1010))
        self.assertEqual(calls, [('gid', 1010), ('uid', 1010), ('uid', 0), ('gid', 0)])


if __name__ == '__main__':
    unittest.main()
