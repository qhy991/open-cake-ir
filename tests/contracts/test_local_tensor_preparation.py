"""Original CPU oracles precede the real local lock; device checks retain them."""
from copy import deepcopy
from hashlib import sha256
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation import local_broker
from open_cake_ir.evaluation.core import EvaluationProtocol, LaunchableCandidate, TensorLaunchManifest
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks import evaluate as worker, workloads
from open_cake_ir.tasks.normalization.study import evaluation_policy
from open_cake_ir.tasks.tiles.evaluation import PreparedTensorCase, evaluate_tile_validation_case


class LocalTensorPreparation(unittest.TestCase):
    def setUp(self):
        document, _ = workloads.create_task('rmsnorm', backend='triton-metax', rows=2, columns=16)
        self.workload = WorkloadContract(document)
        self.manifest = TensorLaunchManifest.for_workload(self.workload, 'primary', target='xcore1002',
            kernel_name='fixture', grid=[2, 1, 1], block=[64, 1, 1],
            dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=0)
        payloads = {'mcfatbin': b'CPU contract fixture',
                    'launch_manifest': canonical_json_bytes(self.manifest.as_dict())}
        self.candidate = LaunchableCandidate('a' * 64, 'xcore1002', 'fixture',
            {k: sha256(v).hexdigest() for k, v in payloads.items()},
            self.manifest.canonical_sha256, payloads)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.lock = self.root / 'device.lock'
        self.request = self.root / 'request.json'
        self.request.write_text('{}')
        self.output = self.root / 'result.json'
        self.authority = worker._Authority(
            {'purpose': 'confirmatory', 'evaluation_protocol': evaluation_policy(self.workload)},
            self.root, None, self.workload, self.manifest, self.candidate,
            self.candidate.artifact_payloads, 'primary', self.candidate,
            allocation_mode='local_broker')

    def unlocked(self):
        fd = local_broker._acquire(self.lock)
        os.close(fd)

    def test_prepared_reference_is_immutable_and_does_not_weaken_output_or_input_checks(self):
        prepared = PreparedTensorCase(self.workload, 'primary')
        with self.assertRaises(TypeError):
            prepared.inputs['x'][0] = 1
        with self.assertRaises(TypeError):
            prepared.expected['out'] = ()
        with self.assertRaises(ValueError):
            prepared.check(self.workload, 'zeros')
        protocol = EvaluationProtocol('fixture', 'confirmatory', self.workload.canonical_sha256,
                                      'primary', 'none')
        class Launcher:
            wrong = False
            mutate = False
            def launch_tensors(inner, candidate, manifest, inputs):
                out = {k: list(v) for k, v in prepared.expected.items()}
                after = deepcopy(inputs)
                if inner.wrong:
                    out['out'][-1] += 1
                if inner.mutate:
                    after['x'][0] += 1
                return out, after, {'candidate_sha256': candidate.candidate_sha256,
                                    'kernel_calls': 1, 'fallback_calls': 0}
        launcher = Launcher()
        with patch.object(workloads, 'materialize_case', side_effect=AssertionError('late inputs')), \
             patch.object(workloads, 'reference_outputs', side_effect=AssertionError('late oracle')):
            for wrong, mutate, expected in ((False, False, True), (True, False, False),
                                             (False, True, False), (False, False, True)):
                launcher.wrong, launcher.mutate = wrong, mutate
                result = evaluate_tile_validation_case(self.candidate, self.workload, protocol,
                                                       launcher, prepared=prepared)
                self.assertEqual(result.correctness_passed, expected)

    def run_worker(self, device, *, oracle=None):
        original_inputs = workloads.materialize_case
        original_reference = workloads.reference_outputs
        def materialize(*args):
            self.unlocked()
            return original_inputs(*args)
        def reference(*args):
            self.unlocked()
            return original_reference(*args) if oracle is None else oracle(*args)
        original_write = worker._write_new
        def write(path, value):
            if path == self.output:
                if os.environ.get('METAL_BROKER_LOCK_FD'):
                    with self.assertRaises(BlockingIOError):
                        local_broker._acquire(self.lock)
                else:
                    self.unlocked()
            original_write(path, value)
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(local_broker, '_lock_path', return_value=self.lock), \
             patch.object(worker, '_load_authority', return_value=self.authority), \
             patch.object(worker, '_platform', return_value=SimpleNamespace(evaluate=device)), \
             patch.object(workloads, 'materialize_case', side_effect=materialize) as inputs, \
             patch.object(workloads, 'reference_outputs', side_effect=reference) as gold, \
             patch.object(worker, '_write_new', side_effect=write), \
             patch('sys.argv', ['evaluate', '--request', str(self.request), '--output', str(self.output),
                                '--local-kind', 'maca']):
            try:
                self.assertEqual(worker.main(), 0)
            finally:
                # This test invokes main in-process. Production releases this
                # process-owned descriptor at worker exit, not at main's return.
                fd = os.environ.get('METAL_BROKER_LOCK_FD')
                if fd is not None:
                    os.close(int(fd))
        return json.loads(self.output.read_text()), inputs.call_count, gold.call_count

    def test_all_cases_are_prepared_once_before_allocation_and_reused_while_owned(self):
        seen = []
        def device(authority, result):
            seen.append(local_broker.observe_local_job('maca'))
            with self.assertRaises(BlockingIOError):
                local_broker._acquire(self.lock)
            for case_id in self.workload.case_ids:
                for _ in range(4):
                    inputs = worker._inputs_for(authority, case_id)
                    self.assertIs(worker._reference_for(authority, case_id, inputs),
                                  authority.prepared_cases[case_id].expected)
            result['admitted'] = True
        result, inputs, gold = self.run_worker(device)
        self.assertEqual((inputs, gold), (5, 5))
        self.assertEqual(result['job_id'], seen[0])
        self.assertTrue(result['admitted'])
        self.assertIsNone(result['error'])
        self.unlocked()

    def test_device_exception_keeps_ownership_until_worker_exit_and_refuses_a_receipt(self):
        def device(authority, result):
            local_broker.observe_local_job('maca')
            raise RuntimeError('device fixture failure')
        result, inputs, gold = self.run_worker(device)
        self.assertEqual((inputs, gold), (5, 5))
        self.assertEqual(result['failure_class'], 'RuntimeError')
        self.assertIsNone(result['receipt'])
        self.unlocked()

    def test_oracle_failure_does_not_enter_device_phase(self):
        def oracle(*args):
            raise ValueError('CPU oracle fixture failure')
        def device(*args):
            self.fail('device phase followed failed CPU preparation')
        result, inputs, gold = self.run_worker(device, oracle=oracle)
        self.assertEqual((inputs, gold), (1, 1))
        self.assertFalse(result['admitted'])
        self.assertEqual(result['counters']['module_loads'], 0)
        self.assertEqual(result['failure_class'], 'ValueError')

    def test_nested_allocation_is_refused_before_bulk_cpu_preparation(self):
        for key, value in (('METAL_BROKER_LOCK_FD', '7'), ('GPUQ_JOB_ID', 'gpuq-123456789abc')):
            with self.subTest(key=key), patch.dict(os.environ, {key: value}, clear=True), \
                 patch.object(worker, 'PreparedTensorCase') as prepare:
                with self.assertRaisesRegex(ValueError, 'existing allocation'):
                    worker._prepare_local_tensor_work(self.authority, 'maca')
                prepare.assert_not_called()

    def test_busy_broker_preserves_other_ownership_and_names_its_own_job(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(local_broker, '_lock_path', return_value=self.lock):
            fd = local_broker._acquire(self.lock)
            try:
                with self.assertRaises(local_broker.LocalBrokerBusy) as caught:
                    local_broker.admit_local_job('maca')
                self.assertTrue(caught.exception.job_id.startswith('maca-'))
                with self.assertRaises(BlockingIOError):
                    local_broker._acquire(self.lock)
                self.assertNotIn('METAL_BROKER_LOCK_FD', os.environ)
            finally:
                os.close(fd)
        self.unlocked()

    def test_process_failure_keeps_the_lock_through_reporting_and_releases_at_exit(self):
        source = """
from pathlib import Path
import sys
from open_cake_ir.evaluation import local_broker
local_broker._lock_path = lambda kind: Path(sys.argv[1])
local_broker.admit_local_job('maca')
try:
    raise RuntimeError('partial device constructor')
except RuntimeError:
    print('reporting', flush=True)
    sys.stdin.readline()
raise SystemExit(7)
"""
        env = dict(os.environ)
        for key in ('METAL_JOB_ID', 'METAL_BROKER_LOCK_FD', 'GPUQ_JOB_ID'):
            env.pop(key, None)
        process = subprocess.Popen([sys.executable, '-c', source, str(self.lock)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env)
        try:
            self.assertEqual(process.stdout.readline().strip(), 'reporting')
            with self.assertRaises(BlockingIOError):
                local_broker._acquire(self.lock)
            process.communicate('finish\n', timeout=10)
            self.assertEqual(process.returncode, 7)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        self.unlocked()
