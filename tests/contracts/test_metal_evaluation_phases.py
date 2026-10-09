"""CPU executable double checks the actual Metal process/worker phase boundary."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation import local_broker, metal_runtime
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks import evaluate, workloads
from tests.contracts import test_metal_evaluation as fixtures


HELPER = r'''import fcntl, json, math, os, struct, sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
directory = Path(request['output_directory'])
descriptor = os.open(Path(os.environ['TMPDIR']) / ('open-cake-ir-metal-' + str(os.geteuid()) + '.lock'), os.O_RDWR)
try:
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raise RuntimeError('observer started without a device lease')
except BlockingIOError:
    pass
finally:
    os.close(descriptor)
(directory / 'device-process.json').write_text(json.dumps({'pid': os.getpid(), 'job_id': os.environ['METAL_JOB_ID']}))
if os.environ.get('METAL_FIXTURE_FAIL'):
    print('retained native fixture failure', file=sys.stderr)
    raise SystemExit(7)
participants = {p['role']: p for p in request['participants']}
cases = {c['id']: c for c in request['cases']}
launches = []
for row in request['launches']:
    case = cases[row['input_case_id']]
    paths = {}
    for name, source in case['inputs'].items():
        dest = directory / (str(row['index']) + '-' + name + '.bin')
        dest.write_bytes(Path(source).read_bytes())
        paths[name] = dest.name
    for name, oracle in case['oracles'].items():
        raw = Path(oracle['expected']).read_bytes()
        values = struct.unpack('<' + str(len(raw)//8) + 'd', raw)
        dest = directory / (str(row['index']) + '-' + name + '.bin')
        dest.write_bytes(struct.pack('<' + str(len(values)) + 'f', *values))
        paths[name] = dest.name
    command = {'launch_index': row['index'], 'completed': True, 'timed': row['timed'],
        'dispatches': row['dispatches'], 'gpu_start_seconds': 100.0 + row['index'],
        'gpu_end_seconds': 100.001 + row['index']}
    observed = {k: row[k] for k in ('index', 'role', 'phase', 'input_case_id')}
    observed.update(buffer_paths=paths, preflight_guard_passed=True, command_buffer=command)
    if row['profile']:
        observed['profile'] = {'counter_set':'Timestamp', 'counter':'GPUTimestamp',
            'sampling_boundary':'compute_stage', 'units':'raw_device_timestamp_units',
            'sample_indices':[0,1], 'resolved_bytes':16, 'timestamps':[100,200], 'command_buffer':command}
    launches.append(observed)
print(json.dumps({'status':'completed','host':request['expected_host'], 'source_library_rebuilt':False,
    'archive_miss_policy':'failOnBinaryArchiveMiss', 'module_loads':len(participants), 'launches':launches}))
'''


class MetalEvaluationPhases(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='metal-evaluation-phases-')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.lock = self.root / f'open-cake-ir-metal-{os.geteuid()}.lock'
        self.helper = self.root / 'observer'
        self.helper.write_text('#!' + sys.executable + '\n' + HELPER)
        self.helper.chmod(0o700)
        document, _ = workloads.create_task('rmsnorm', backend='metal-m1-pro', rows=1, columns=2)
        self.workload = WorkloadContract(document)
        self.candidate, self.manifest = fixtures.candidate_fixture(self.workload, 'candidate')
        self.baseline, _ = fixtures.candidate_fixture(self.workload, 'baseline')
        self.request = self.root / 'request.json'
        self.request.write_text('{}')
        self.output = self.root / 'result.json'

    def unlocked(self):
        fd = local_broker._acquire(self.lock)
        os.close(fd)
        self.assertNotIn('METAL_BROKER_LOCK_FD', os.environ)

    def run_worker(self, purpose='search', fail=False, oracle_fail=False):
        def admit():
            self.unlocked()
            return {'kind':'metal', 'host':fixtures.HOST, 'observer_executable':str(self.helper)}
        authority = evaluate._Authority({'purpose':purpose, 'evaluation_protocol':fixtures.policy_fixture()},
            self.root, SimpleNamespace(admit_host=admit), self.workload, self.manifest, self.candidate,
            self.candidate.artifact_payloads, 'primary', self.baseline, allocation_mode='local_serialized')
        original_inputs, original_oracle = workloads.materialize_case, workloads.reference_outputs
        original_compare, original_write = metal_runtime.compare_tile_outputs, evaluate._write_new
        def inputs(*args):
            self.unlocked()
            return original_inputs(*args)
        def oracle(*args):
            self.unlocked()
            if oracle_fail:
                raise ValueError('fixture oracle failed before allocation')
            return original_oracle(*args)
        def compare(*args):
            self.unlocked()
            return original_compare(*args)
        def write(*args):
            self.unlocked()
            return original_write(*args)
        env = dict(PATH=os.defpath, TMPDIR=str(self.root))
        if fail:
            env['METAL_FIXTURE_FAIL'] = '1'
        with patch.dict(os.environ, env, clear=True), \
             patch.object(evaluate, '_load_authority', return_value=authority), \
             patch.object(workloads, 'materialize_case', side_effect=inputs) as materialize, \
             patch.object(workloads, 'reference_outputs', side_effect=oracle) as reference, \
             patch.object(metal_runtime, 'compare_tile_outputs', side_effect=compare) as checks, \
             patch.object(evaluate, '_write_new', side_effect=write), \
             patch('sys.argv', ['evaluate', '--request', str(self.request), '--output', str(self.output), '--local-kind', 'metal']):
            self.assertEqual(evaluate.main(), 0)
            self.unlocked()
        return json.loads(self.output.read_text()), materialize.call_count, reference.call_count, checks.call_count

    def test_paired_plan_prepares_all_cases_before_one_device_process_and_checks_after_exit(self):
        result, inputs, oracles, checks = self.run_worker()
        self.assertIsNone(result['error'])
        self.assertTrue(result['receipt']['correctness_passed'])
        self.assertEqual((inputs, oracles, checks), (5, 5, 32))
        directory = self.root / 'metal-observation'
        device = json.loads((directory / 'device-process.json').read_text())
        self.assertEqual(result['job_id'], device['job_id'])
        with self.assertRaises(ProcessLookupError):
            os.kill(device['pid'], 0)
        launches = json.loads((directory / 'request.json').read_text())['launches']
        self.assertEqual(sum(row['phase'] == 'preflight' for row in launches), 10)
        self.assertEqual(sum(row['phase'] == 'postflight' for row in launches), 10)
        self.assertEqual(result['counters']['timing_samples'], 8)

    def test_profile_still_prepares_and_checks_every_distribution(self):
        result, inputs, oracles, checks = self.run_worker('attribution')
        self.assertIsNone(result['error'])
        self.assertEqual((inputs, oracles, checks), (5, 5, 6))
        self.assertIsNone(result['receipt']['timing'])

    def test_native_failure_retains_job_and_stderr_but_never_receipt(self):
        result, inputs, oracles, checks = self.run_worker(fail=True)
        self.assertEqual((inputs, oracles, checks), (5, 5, 0))
        self.assertTrue(result['admitted'])
        self.assertIsNone(result['receipt'])
        directory = self.root / 'metal-observation'
        self.assertEqual(json.loads((directory / 'allocation.json').read_text())['job_id'], result['job_id'])
        self.assertIn('retained native fixture failure', (directory / 'observer.stderr.log').read_text())

    def test_failed_cpu_oracle_never_reaches_admission(self):
        result, inputs, oracles, checks = self.run_worker(oracle_fail=True)
        self.assertEqual((inputs, oracles, checks), (1, 1, 0))
        self.assertFalse(result['admitted'])
        self.assertIsNone(result['receipt'])
        self.assertFalse((self.root / 'metal-observation').exists())

    def test_busy_observer_keeps_the_other_owner_and_preserves_refusal_identity(self):
        from open_cake_ir.evaluation.metal_device_process import run_metal_process, read_metal_admission
        request = self.root / 'device-request.json'
        request.write_text('{}')
        record = self.root / 'allocation.json'
        descriptor = local_broker._acquire(self.lock)
        try:
            with patch.dict(os.environ, dict(PATH=os.defpath, TMPDIR=str(self.root)), clear=True):
                result = run_metal_process(self.helper, request, target=self.candidate.target,
                    timeout_seconds=5, allocation_output=record)
            self.assertEqual(result.returncode, 3)
            self.assertEqual(result.stdout, b'')
            admission = read_metal_admission(record)
            self.assertFalse(admission['admitted'])
            self.assertRegex(admission['job_id'], r'^metal-[0-9a-f]{12}$')
            with self.assertRaises(BlockingIOError):
                local_broker._acquire(self.lock)
        finally:
            os.close(descriptor)

    def submit_through_temporary_transport(self, *, fail=False):
        import grp
        import pwd
        from hashlib import sha256
        from open_cake_ir.lab.executor import ExecutorRevision
        from open_cake_ir.lab.runtime import CommandBrokerSubmitter
        from open_cake_ir.serialization import canonical_json_bytes
        from tests.contracts._executor_fixture import compiler_reference
        source = Path(__file__).resolve().parents[2]
        workload_path = self.root / 'workload.json'
        workload_path.write_bytes(canonical_json_bytes(self.workload.document))
        wrapper = self.root / 'worker.py'
        wrapper.write_text(
            'import sys; sys.path.insert(0, ' + repr(str(source / 'src')) + ')\n'
            'from unittest.mock import patch\n'
            'from open_cake_ir.lab.executor import ExecutorRevision\n'
            'from open_cake_ir.tasks import evaluate\n'
            'with patch.object(ExecutorRevision, "admit_host", return_value=' + repr({
                'kind': 'metal', 'host': fixtures.HOST, 'observer_executable': str(self.helper)}) + '):\n'
            '    raise SystemExit(evaluate.main())\n')
        policy = fixtures.policy_fixture()
        submitter = CommandBrokerSubmitter(command=(sys.executable, str(wrapper), '--local-kind', 'metal'),
            workload_path=workload_path, workload_sha256=self.workload.canonical_sha256,
            protocol_sha256=sha256(canonical_json_bytes(policy)).hexdigest(), cwd=source,
            executor=ExecutorRevision.for_target(source, self.candidate.target),
            compiler_reference=compiler_reference(source),
            service_user=pwd.getpwuid(os.geteuid()).pw_name,
            service_group=grp.getgrgid(os.getegid()).gr_name,
            evaluation_protocol=policy, baseline=self.baseline, workload_loader=workloads.load_workload)
        env = dict(PATH=os.defpath, TMPDIR=str(self.root))
        if fail:
            env['METAL_FIXTURE_FAIL'] = '1'
        with patch.dict(os.environ, env, clear=True):
            return submitter.submit(self.candidate, case_id='primary', purpose='search', attempt=1)

    def test_actual_submitter_accepts_job_and_receipt_after_temporary_transport_closes(self):
        attempt = self.submit_through_temporary_transport()
        self.assertTrue(attempt.admitted)
        self.assertIsNone(attempt.error)
        self.assertTrue(attempt.receipt.correctness_passed)
        request = json.loads(attempt.artifact_payloads['evaluator_request'])
        # TemporaryDirectory has exited; retained bytes must suffice.
        self.assertFalse(Path(request['artifact_paths']['launch_manifest']).is_absolute())
        self.assertIn(attempt.job_id.encode(), attempt.artifact_payloads['stderr'])

    def test_actual_submitter_keeps_failed_native_streams_after_temporary_transport_closes(self):
        attempt = self.submit_through_temporary_transport(fail=True)
        self.assertTrue(attempt.admitted)
        self.assertIsNone(attempt.receipt)
        self.assertIn(b'retained native fixture failure', attempt.artifact_payloads['failure_metal_observer_stderr'])
        self.assertEqual(json.loads(attempt.artifact_payloads['failure_metal_allocation'])['job_id'], attempt.job_id)
