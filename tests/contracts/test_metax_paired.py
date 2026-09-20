"""Native-record custody through the real paired worker and receipt reader (CPU doubles)."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.core import EvaluationReceipt, LaunchableCandidate, TensorLaunchManifest
from open_cake_ir.evaluation.metax_benchmark import RESET, TIMER
from open_cake_ir.evaluation.paired import validate_paired_broker
from open_cake_ir.evaluation.triton_metax import MetaxDeviceAdmission
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.serialization import canonical_json_bytes as encoded
from open_cake_ir.tasks import evaluate as worker
from open_cake_ir.tasks.normalization.study import evaluation_policy
from open_cake_ir.tasks.workloads import create_task, reference_outputs
from tests.contracts.test_metax_measurement import capture, kernel


class MacaPairedReceipts(unittest.TestCase):
    def setUp(self):
        document, _ = create_task('rmsnorm', backend='triton-metax', rows=2, columns=128)
        self.workload = WorkloadContract(document)
        self.policy = evaluation_policy(self.workload)
        self.admission = MetaxDeviceAdmission('maca-123456789abc', 'xcore1002', 'xcore1002',
            'MetaX C550', 64, '0000:0f:00', '/opt/maca/lib/libmcruntime.so')
        self.candidates = {}
        self.manifests = {}
        for role in ('candidate', 'baseline'):
            manifest = TensorLaunchManifest.for_workload(self.workload, 'primary', target='xcore1002',
                kernel_name=role, grid=[2, 1, 1], block=[64, 1, 1],
                dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=0)
            payloads = {'mcfatbin': b'CPU fixture ' + role.encode(), 'launch_manifest': encoded(manifest.as_dict())}
            self.candidates[role] = LaunchableCandidate(sha256(role.encode()).hexdigest(), 'xcore1002', role,
                {k: sha256(v).hexdigest() for k, v in payloads.items()}, manifest.canonical_sha256, payloads)
            self.manifests[role] = manifest
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def execute(self):
        owner = self
        class Loaded:
            module_count = 1
            def __init__(self, candidate, manifest, inputs, admission):
                self.candidate, self.manifest, self.inputs = candidate, manifest, inputs
                self.loaded = SimpleNamespace(launch_calls=0, resources={
                    'registers_per_thread':16, 'local_bytes':0, 'dynamic_shared_bytes':0})
            def fresh_argument_sets(self, count):
                return [{} for _ in range(count)]
            def launch(self, values):
                self.loaded.launch_calls += 1
                values.update(reference_outputs(owner.workload, 'primary', self.inputs))
            def snapshot(self, values):
                return values, self.inputs
            def launch_tensors(self, candidate, manifest, inputs):
                values = {}; self.launch(values)
                return values, inputs, {'candidate_sha256':candidate.candidate_sha256,
                    'kernel_calls':1, 'fallback_calls':0, 'manifest_sha256':manifest.canonical_sha256,
                    'device_admission':asdict(owner.admission), 'resources':self.loaded.resources}
            def close(self):
                pass
        sequence = 0
        class Assay:
            def __init__(self, manifest, **kwargs):
                self.manifest = manifest
                owner.assertEqual(kwargs['l2_cache_bytes'], 8388608)
            def __call__(self, function, **kwargs):
                nonlocal sequence
                for _ in range(kwargs['dry_run_iters'] + kwargs['repeat_iters']):
                    function()
                reset = kernel('fill', 1, 1000)
                reset_activity = capture(reset); reset_activity['records'][-1]['cbid'] = 56
                records = []
                duration = 2048 if self.manifest.kernel_name == 'candidate' else 4096
                for index in range(kwargs['repeat_iters']):
                    start = 10000 + sequence * 10000
                    records += [kernel('fill', 2 + sequence*2, start),
                        {**kernel(self.manifest.kernel_name, 3 + sequence*2, start+3000,
                                  grid=self.manifest.grid, block=self.manifest.block),
                         'end_ns':start+3000+duration}]
                    sequence += 1
                activity = capture(*records)
                for row in activity['records']:
                    if row['kind'] == 5 and row['correlation'] % 2 == 0:
                        row['cbid'] = 56
                self.last_activity = dict(timer=TIMER, cache_policy=RESET, l2_cache_bytes=8388608,
                    reset_bytes=4*8388608, reset_record=reset, reset_activity=reset_activity, activity=activity)
                self.non_target_dispatches = 0
                return [duration/1e6] * kwargs['repeat_iters']
        candidate = self.candidates['candidate']
        host = SimpleNamespace(admit_host=lambda:{'runtime_library':self.admission.runtime_library,
                                                 'activity_library':'/opt/maca/lib/libmcpti.so'})
        authority = worker._Authority({'purpose':'search', 'evaluation_protocol':self.policy},
            self.root, host, self.workload, self.manifests['candidate'], candidate,
            candidate.artifact_payloads, 'primary', self.candidates['baseline'])
        self.result = worker._base_result(self.admission.broker_job_id)
        with patch.object(worker, 'LoadedTorchTensorCandidate', Loaded), \
             patch('open_cake_ir.evaluation.metax_benchmark.McptiDispatchBenchmark', Assay):
            worker._evaluate_metax_candidate(authority, self.result, collect_timing=True, admission=self.admission)
        self.payloads = {role:(self.root/name).read_bytes()
                         for role,name in self.result['receipt']['artifacts'].items()}
        return self.receipt()

    def receipt(self, payloads=None):
        payloads = self.payloads if payloads is None else payloads
        value = self.result['receipt']
        return EvaluationReceipt(self.candidates['candidate'].candidate_sha256, self.workload.canonical_sha256,
            sha256(encoded(self.policy)).hexdigest(), 'search', 'primary', value['correctness_passed'],
            value['correctness'], 1, 0, sha256(payloads['launch_receipt']).hexdigest(),
            value['timing'], artifact_payloads=payloads)

    def test_common_worker_uses_native_paired_records_and_complete_oracle_cases(self):
        receipt = self.execute()
        self.assertTrue(receipt.correctness_passed)
        self.assertTrue(receipt.timing['measurement_quality_passed'])
        self.assertEqual(receipt.timing['speedup'], 2.0)
        self.assertEqual(self.result['counters']['timing_samples'], 500)
        self.assertEqual(self.result['counters']['kernel_calls'], 740)
        validate_paired_broker(receipt, self.admission.broker_job_id, self.result['counters'])
        launch = json.loads(self.payloads['launch_receipt'])
        self.assertEqual(launch['allocation_mode'], 'local_serialized')
        self.assertEqual(launch['external_gpu_activity'], 'not_excluded')
        self.assertEqual(launch['device_admission'], asdict(self.admission))

    def test_native_record_reset_manifest_device_and_samples_are_not_replaceable(self):
        self.execute()
        raw = json.loads(self.payloads['timing_samples'])
        mutations = []
        value = deepcopy(raw);value['measurements'][0]['arms']['candidate']['samples_ms'][0] *= 2;mutations.append(value)
        value = deepcopy(raw);value['measurements'][0]['arms']['candidate']['native_activity']['reset_bytes'] = 4;mutations.append(value)
        value = deepcopy(raw);value['measurements'][0]['arms']['candidate']['native_activity']['activity']['records'].pop();mutations.append(value)
        value = deepcopy(raw);value['launch_manifests']['candidate']['workload_sha256'] = 'f'*64;mutations.append(value)
        value = deepcopy(raw);value['device_admission']['pci_bus_id'] = '0000:10:00';mutations.append(value)
        value = deepcopy(raw);value['measurements'][1]['arms'] = deepcopy(value['measurements'][0]['arms']);mutations.append(value)
        value = deepcopy(raw)
        native = value['measurements'][2]['arms']['candidate']['native_activity']
        for activity in (native['activity'], native['reset_activity']):
            for row in activity['records']:
                if row['kind'] == 10:
                    row.update(context=7, stream=9)
        native['reset_record'].update(context=7, stream=9)
        mutations.append(value)
        for index,value in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                self.receipt({**self.payloads, 'timing_samples':encoded(value)})
        launch = json.loads(self.payloads['launch_receipt'])
        launch['resources']['candidate']['registers_per_thread'] = 99
        with self.assertRaisesRegex(ValueError, 'resources'):
            self.receipt({**self.payloads, 'launch_receipt':encoded(launch)})
