"""Resource refusal isolates one failed task only with complete failed-worker evidence."""
from copy import deepcopy
import json
import unittest
from types import SimpleNamespace

from open_cake_ir.evaluation.metax_failures import is_local_launch_resource_failure
from tools.launch_task_matrix import dispatch_must_stop


class LaunchResourceFailure(unittest.TestCase):
    def setUp(self):
        self.result = {'schema_version': 2, 'failure_class': 'MetaxLaunchResourceError',
            'error': 'evaluator_failed', 'receipt': None, 'admitted': True,
            'mode': 'local_serialized', 'job_id': 'maca-123456789abc',
            'failure_artifacts': {'launch_resource': 'failure-launch_resource.bin'},
            'counters': {'kernel_calls': 0, 'timing_samples': 0, 'compiler_invocations': 0,
                         'fallback_calls': 0, 'preflight_calls': 0, 'module_loads': 5}}
        self.diagnostic = {'kind': 'maca_launch_resource_failure_v1',
            'operation': 'mcModuleLaunchKernel', 'status': 32,
            'status_name': 'mcErrorMemoryValueTooLarge', 'completed_target_calls': 0,
            'purpose': 'search', 'arm': 'candidate', 'phase': 'preflight',
            'teardown_completed': True, 'job_id': self.result['job_id'],
            'coverage': 'failed_dispatch_no_correctness_or_timing_receipt',
            'resources': {'local_bytes': 7680}, 'candidate_sha256': 'c' * 64}
        self.stdout = ('Error in allocating private memory because the private memory size required in the kernel '
            'is greater than the maximum value set by the system. '
            'The system is set to :4 KB/Thread,kernel request:7 KB/thread\n'
            'mcModuleLaunchKernel: Returned mcErrorMemoryValueTooLarge\n')
        self.stderr = ('[maca-run] accepted job maca-123456789abc\n'
            'MetaxLaunchResourceError: MACA mcModuleLaunchKernel failed with status 32 (mcErrorMemoryValueTooLarge)\n')

    def accepts(self, result=None, diagnostic=None, stdout=None, stderr=None):
        return is_local_launch_resource_failure(self.result if result is None else result,
            self.diagnostic if diagnostic is None else diagnostic,
            self.stdout if stdout is None else stdout, self.stderr if stderr is None else stderr)

    def test_dynamic_sdk_limit_is_evidence_not_a_target_constant(self):
        self.assertTrue(self.accepts())
        other = deepcopy(self.diagnostic); other['resources']['local_bytes'] = 13 * 1024
        self.assertTrue(self.accepts(diagnostic=other,
            stdout=self.stdout.replace(':4 KB/Thread', ':8 KB/Thread').replace(':7 KB/thread', ':13 KB/thread')))

    def test_unknown_baseline_cleanup_and_partial_launch_failures_still_stop(self):
        for key, value in [('status', 1009), ('phase', 'postflight'), ('arm', 'baseline'),
                           ('purpose', 'confirmatory'), ('teardown_completed', False),
                           ('completed_target_calls', 1), ('job_id', 'maca-other')]:
            with self.subTest(field=key):
                self.assertFalse(self.accepts(diagnostic={**self.diagnostic, key: value}))
        for key, value in [('failure_class', 'RuntimeError'), ('receipt', {}), ('admitted', False)]:
            self.assertFalse(self.accepts(result={**self.result, key: value}))
        for name in ['kernel_calls', 'timing_samples', 'compiler_invocations', 'fallback_calls']:
            changed = deepcopy(self.result); changed['counters'][name] = 1
            self.assertFalse(self.accepts(result=changed))
        self.assertFalse(self.accepts(stdout='mcErrorMemoryValueTooLarge'))
        self.assertFalse(self.accepts(stdout=self.stdout.replace('request:7', 'request:4')))
        self.assertFalse(self.accepts(stderr=self.stderr + 'RuntimeError: device lost\n'))
        wrong = deepcopy(self.diagnostic); wrong['resources']['local_bytes'] = 0
        self.assertFalse(self.accepts(diagnostic=wrong))

    def test_dispatch_requires_custody_and_retained_attempt_not_message(self):
        report = {'run_id': 'run', 'audit': {'archive_integrity': True,
            'filesystem_custody_verified': True, 'protocol_adherence': 'broker_fault'},
            'replay': {'refusals': []}}
        fault = {'stage': 'evaluation', 'turn': 2, 'fault': 'broker_fault'}
        objects = {'attempt_1_evaluator_result': json.dumps(self.result).encode(),
            'attempt_1_failure_launch_resource': json.dumps(self.diagnostic).encode(),
            'attempt_1_stdout': self.stdout.encode(), 'attempt_1_stderr': self.stderr.encode()}
        attempt = {'kind': 'evaluation_attempt_completed', 'payload': {'turn': 2,
            'purpose': 'search', 'candidate_sha256': 'c' * 64,
            'objects': [{'role': role} for role in objects]}}
        evidence = SimpleNamespace(replay_events=lambda run: [attempt, {'kind':'run_fault','payload':fault}],
            read_object=lambda ref: objects[ref['role']])
        self.assertTrue(dispatch_must_stop(report, 0, fault))
        self.assertFalse(dispatch_must_stop(report, 0, fault, evidence=evidence))
        self.assertEqual(report['audit']['protocol_adherence'], 'broker_fault')
        bad = deepcopy(report); bad['audit']['filesystem_custody_verified'] = False
        self.assertTrue(dispatch_must_stop(bad, 0, fault, evidence=evidence))
        bad = deepcopy(report); bad['replay']['refusals'] = ['bad receipt']
        self.assertTrue(dispatch_must_stop(bad, 0, fault, evidence=evidence))
        attempt['payload']['candidate_sha256'] = 'wrong'
        self.assertTrue(dispatch_must_stop(report, 0, fault, evidence=evidence))
