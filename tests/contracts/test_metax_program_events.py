"""Complete Program event intervals using the real ordered loader and CPU devices.

These contracts do not qualify a physical MetaX target for performance Runs.
"""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.evaluation import metax_event_benchmark as events
from open_cake_ir.evaluation.program import admit_program_execution
from open_cake_ir.evaluation import program as program_runtime
from open_cake_ir.evaluation.core import EvaluationReceipt
from open_cake_ir.evaluation.paired import validate_paired_broker, validate_receipt_policy
from open_cake_ir.evaluation.triton_metax import MetaxDeviceAdmission
from open_cake_ir.serialization import canonical_json_bytes as encoded
from open_cake_ir.tasks import evaluate as worker
from open_cake_ir.tasks.normalization.study import evaluation_policy
from tests.contracts.test_metax_measurement import capture, kernel
from tests.contracts.test_native_program_tensors import build
from tests.contracts.test_ordered_launch_plan import document
from tests.contracts import test_program_evaluation as program_fixtures

ROOT = Path(__file__).resolve().parents[2]


class ProgramEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = document()
        value['target'] = 'xcore1002'
        for stage in value['stages']:
            stage['schedule']['target'] = 'xcore1002'
        cls.program = Program.from_dict(value)
        cls.workload = program_fixtures.workload_for(cls.program)
        cls.workload.case_ids = ('primary',)
        cls.workload.document['validation']['primary_case'] = 'primary'
        cls.candidate = build(cls.program, cls.workload, Compiler.load(ROOT))

    def fixture(self, *, omit=False):
        native, manifest, tensor, calls, children = program_fixtures.ProgramEvaluationTests().loaded(self.candidate)
        native._stream = 0  # The fixture kernel has no real device stream.
        def fresh(count):
            result = []
            for _ in range(count):
                args = [tensor([1, 2, 3, 4], (4,), 'fp32'),
                        tensor([float('nan')] * 4, (4,), 'fp32')]
                native.prepare_arguments(args)
                result.append(args)
            return result
        sets = fresh(16)
        for child in children.values():
            child.resources = {'registers_per_thread': 16, 'local_bytes': 0, 'dynamic_shared_bytes': 0}
        order = []
        stream = SimpleNamespace(cuda_stream=0, synchronize=lambda: order.append('sync'))
        class Event:
            def __init__(self, **kwargs): pass
            def record(self, actual):
                self.assert_stream = actual is stream
                if not self.assert_stream: raise ValueError('wrong stream')
                order.append('event')
            def synchronize(self): order.append('end_sync')
            def elapsed_time(self, other): return .25
        torch = SimpleNamespace(version=SimpleNamespace(maca='CPU fixture'), float32=object(),
            empty=lambda *args, **kwargs: SimpleNamespace(fill_=lambda value: order.append('reset')),
            cuda=SimpleNamespace(default_stream=lambda device: stream,
                get_device_properties=lambda device: SimpleNamespace(L2_cache_size=8388608),
                Event=Event, stream=lambda value: nullcontext(), synchronize=lambda: None))
        def launch(args):
            order.append('program')
            if not omit:
                native.launch(args, tensor_contract=manifest, stream=0)
        wrapper = SimpleNamespace(manifest=manifest, candidate=self.candidate,
                                  loaded=native, launch=launch, fresh_argument_sets=fresh)
        self.addCleanup(lambda: native.close(synchronize=lambda: None))
        return wrapper, sets, torch, calls, order

    def test_full_program_has_fresh_outputs_and_ordered_reset_event_interval(self):
        loaded, sets, torch, calls, order = self.fixture()
        timer = events.MacaProgramEventBenchmark(loaded.manifest, l2_cache_bytes=8388608)
        with patch.dict(sys.modules, torch=torch):
            values = timer.capture_loaded_cohort(loaded, sets, dry_run_iters=11, repeat_iters=5)
        self.assertEqual(values, [.25] * 5)
        self.assertEqual(calls, ['first', 'second'] * 16)
        self.assertTrue(all(args[-1].data == [3, 4, 5, 6] for args in sets))
        self.assertEqual(order[-25:], ['reset', 'event', 'program', 'event', 'end_sync'] * 5)
        record = {'native_activity': timer.last_activity, 'samples_ms': values}
        events.validate_cohort(record, loaded.manifest, sample_count=5)
        for name, value in [('completed_stage_calls', 31), ('profiler_enabled', True),
                            ('stage_names', ['second', 'first']), ('interval', events.INTERVAL)]:
            bad = deepcopy(record)
            bad['native_activity'][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                events.validate_cohort(bad, loaded.manifest, sample_count=5)

    def test_missing_program_dispatch_cannot_become_a_latency(self):
        loaded, sets, torch, _, _ = self.fixture(omit=True)
        timer = events.MacaProgramEventBenchmark(loaded.manifest, l2_cache_bytes=8388608)
        with patch.dict(sys.modules, torch=torch), self.assertRaisesRegex(ValueError, 'stage count'):
            timer.capture_loaded_cohort(loaded, sets, dry_run_iters=11, repeat_iters=5)

    def test_implementation_does_not_imply_device_qualification(self):
        with self.assertRaisesRegex(ValueError, 'not implemented|not qualified'):
            admit_program_execution('xcore1002', timing=True, attribution=True)

    def execute_worker(self, *, profile=False, failed_profile=False):
        owner = self
        instances = []
        class Loaded:
            def __init__(self, candidate, manifest, inputs, admission):
                wrapper, _, _, _, _ = owner.fixture()
                self.__dict__.update(wrapper.__dict__)
                self.module_count = 2
                self.inputs = inputs
                instances.append(self)
            def snapshot(self, arguments):
                return {'y': arguments[1].data[:]}, {'x': arguments[0].data[:]}
            def release_argument_sets(self, arguments):
                for args in arguments:
                    self.loaded.release_arguments(args)
            def launch_tensors(self, candidate, manifest, inputs):
                args = self.fresh_argument_sets(1)[0]
                before = self.loaded.launch_calls
                self.launch(args)
                observed, after = self.snapshot(args)
                self.release_argument_sets([args])
                return observed, after, {'candidate_sha256': candidate.candidate_sha256,
                    'kernel_calls': self.loaded.launch_calls - before, 'fallback_calls': 0,
                    'manifest_sha256': manifest.canonical_sha256, 'device_admission': asdict(admission),
                    'module_unloaded': self.loaded.closed, 'resources': self.loaded.resources}
            def close(self): self.loaded.close(synchronize=lambda: None)
        _, _, torch, _, _ = self.fixture()
        manifest, _, manifests = program_runtime.program_components(self.candidate)
        policy = evaluation_policy(self.workload, metax_mean10=True)
        policy.pop('validation_case_ids')
        admission = MetaxDeviceAdmission('maca-123456789abc', 'xcore1002', 'xcore1002',
            'MetaX C550', 64, '0000:0f:00', '/opt/maca/lib/libmcruntime.so')
        host = SimpleNamespace(admit_host=lambda: {'runtime_library': admission.runtime_library,
                                                   'activity_library': 'CPU fixture'})
        native_records = capture(*(kernel(spec.kernel_name, i + 1, 1000 + i * 4000,
                                          grid=spec.grid, block=spec.block)
                                   for i, spec in enumerate(manifests.values())))
        if failed_profile:
            native_records['records'][0]['name'] = 'unrelated_kernel'
        collector = SimpleNamespace(begin=lambda: None, finish=lambda: native_records)
        purpose = 'attribution' if profile else 'search'
        inputs, expected = {'x': [1., 2., 3., 4.]}, {'y': [3., 4., 5., 6.]}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, torch=torch), \
             patch.object(program_runtime, '_MACA_PROGRAM_MEASUREMENT_EVIDENCE', frozenset({'xcore1002'})), \
             patch.object(worker, 'LoadedTorchTensorCandidate', Loaded), \
             patch.object(worker, 'materialize_evaluation_inputs', return_value=inputs), \
             patch.object(worker, 'reference_evaluation_outputs', return_value=expected), \
             patch('open_cake_ir.tasks.workloads.materialize_evaluation_case', return_value=(inputs, expected)), \
             patch('open_cake_ir.evaluation.metax_program_profile.activity_collector', return_value=collector) as profiler:
            authority = worker._Authority({'purpose': purpose, 'evaluation_protocol': policy},
                Path(directory), host, self.workload, manifest, self.candidate,
                self.candidate.artifact_payloads, 'primary', None if profile else self.candidate)
            result = worker._base_result(admission.broker_job_id)
            worker._evaluate_metax_candidate(authority, result, collect_timing=not profile, admission=admission)
            if profile:
                self.assertEqual(profiler.call_count, 1)
            else:
                profiler.assert_not_called()
            value = result['receipt']
            payloads = {role: (Path(directory) / path).read_bytes()
                        for role, path in value['artifacts'].items()}
            receipt = EvaluationReceipt(self.candidate.candidate_sha256, self.workload.canonical_sha256,
                sha256(encoded(policy)).hexdigest(), purpose, 'primary', value['correctness_passed'],
                value['correctness'], 2, 0, sha256(payloads['launch_receipt']).hexdigest(),
                value['timing'], payloads)
            validate_receipt_policy(receipt, policy, None if profile else worker.candidate_identity(self.candidate),
                                    self.candidate)
            if not profile:
                validate_paired_broker(receipt, admission.broker_job_id, result['counters'])
            self.assertTrue(all(item.loaded.closed for item in instances))
            return receipt, result

    def test_real_paired_worker_checks_all_outputs_and_replays_ten_program_samples(self):
        receipt, result = self.execute_worker()
        self.assertTrue(receipt.correctness_passed)
        self.assertEqual(receipt.timing['pooled_sample_counts'], {'candidate': 10, 'baseline': 10})
        self.assertEqual(receipt.timing['pooled_mean_ms'], .25)
        self.assertEqual(result['counters']['kernel_calls'], 136)

    def test_real_worker_profiles_every_stage_separately_and_unloads_before_receipt(self):
        receipt, result = self.execute_worker(profile=True)
        self.assertIsNone(receipt.timing)
        self.assertEqual(result['counters']['timing_samples'], 0)
        self.assertEqual(result['counters']['kernel_calls'], 4)
        self.assertEqual([row['stage'] for row in receipt.attribution_feedback['stages']], ['first', 'second'])
        with self.assertRaisesRegex(ValueError, 'sealed native launch') as caught:
            self.execute_worker(profile=True, failed_profile=True)
        self.assertIn('program_activity', caught.exception.artifact_payloads)


if __name__ == '__main__':
    unittest.main()
