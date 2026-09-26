"""Complete reset/Program/synchronization sequences, CPU activity fixtures."""
from copy import deepcopy
from types import SimpleNamespace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.metax_benchmark import RESET
from open_cake_ir.evaluation.metax_program_benchmark import (
    McptiProgramBenchmark, PROGRAM_TIMER, SYNCHRONIZATION, program_samples,
)
from open_cake_ir.evaluation.paired import candidate_identity
from tests.contracts import test_metax_program_profile as profile_fixtures
from tests.contracts.test_metax_measurement import capture, kernel


def sync(correlation, start, end):
    return dict(kind=5, cbid=18, correlation=correlation, start_ns=start, end_ns=end, return_value=0)


class ProgramTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        profile_fixtures.ProgramProfile.setUpClass()

    def setUp(self):
        fixture = profile_fixtures.ProgramProfile
        calibration = kernel('fill', 90, 1000, grid=(32768, 1, 1), block=(256, 1, 1))
        reset_capture = capture(calibration)
        reset_capture['records'][-1]['cbid'] = 56
        records = []
        for index in range(2):
            offset = 10000 + 30000 * index
            reset = {**calibration, 'start_ns': offset, 'end_ns': offset + 2048,
                     'correlation': 100 + index * 10}
            rows = capture(reset)['records']; rows[-1]['cbid'] = 56
            records.extend(rows)
            records.append(sync(200 + 2 * index, offset - 50, offset + 2100))
            for stage_index, spec in enumerate(fixture.manifests.values()):
                item = {**kernel(spec.kernel_name, 101 + index * 10 + stage_index,
                                offset + 4000 + stage_index * 4000, grid=spec.grid, block=spec.block),
                        'dynamic_shared_bytes': spec.dynamic_shared_memory_bytes}
                records.extend(capture(item)['records'])
            records.append(sync(201 + 2 * index, offset + 17000, offset + 19000))
        records.append(sync(999, 60000, 60100))
        self.raw = {**deepcopy(fixture.raw), 'candidate': candidate_identity(fixture.candidate),
                    'timer': PROGRAM_TIMER, 'cache_policy': RESET, 'synchronization': SYNCHRONIZATION,
                    'l2_cache_bytes': 8388608, 'reset_bytes': 33554432,
                    'reset_record': calibration, 'reset_activity': reset_capture,
                    'activity': {'source': 'mcpti_activity', 'api_version': 18, 'dropped_records': 0,
                                 'pending_buffers': 0, 'records': records}}

    def test_interval_includes_stage_gaps_but_excludes_reset(self):
        self.assertEqual(program_samples(self.raw, repeats=2), [0.014048, 0.014048])
        self.assertNotEqual(program_samples(self.raw, repeats=2)[0], 4 * 0.002048)

    def test_missing_extra_wrong_stage_and_synchronization_are_refused(self):
        mutations = []
        for kind, field, value in [(10, 'name', 'unknown'), (10, 'stream', 1),
                                  (10, 'dynamic_shared_bytes', 4096), (5, 'cbid', 46)]:
            raw = deepcopy(self.raw)
            item = next(row for row in raw['activity']['records']
                        if row['kind'] == kind and (row.get('correlation') == 101 if kind == 10 else row.get('cbid') == 18))
            item[field] = value; mutations.append(raw)
        raw = deepcopy(self.raw); raw['activity']['records'].pop(); mutations.append(raw)
        raw = deepcopy(self.raw); raw['activity']['records'] += capture(kernel('extra', 888, 70000))['records']; mutations.append(raw)
        raw = deepcopy(self.raw); raw['reset_record']['name'] = 'other'; mutations.append(raw)
        raw = deepcopy(self.raw); raw['activity']['dropped_records'] = 1; mutations.append(raw)
        raw = deepcopy(self.raw); raw['synchronization'] = 'none'; mutations.append(raw)
        raw = deepcopy(self.raw); raw['activity']['records'].append(deepcopy(raw['activity']['records'][-1])); mutations.append(raw)
        for raw in mutations:
            with self.subTest(raw=raw), self.assertRaises(ValueError): program_samples(raw, repeats=2)

    def test_sampling_reuses_collector_reset_and_exact_fresh_call_budget(self):
        fixture = profile_fixtures.ProgramProfile
        with patch('open_cake_ir.evaluation.metax_benchmark.activity_collector'):
            benchmark = McptiProgramBenchmark(fixture.candidate, activity_library='CPU fixture', l2_cache_bytes=8388608)
        events = []
        benchmark._reset = SimpleNamespace(fill_=lambda value: events.append('reset'))
        benchmark._reset_record = self.raw['reset_record']
        benchmark._reset_activity = self.raw['reset_activity']
        benchmark._prepare_reset = lambda: None
        def collect(function):
            function()
            return self.raw['activity']
        benchmark._collect = collect
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: events.append('sync')))
        with patch.dict('sys.modules', {'torch': torch}):
            samples = benchmark(lambda: events.append('program'), dry_run_iters=1, repeat_iters=2,
                                cold_l2_cache=True, use_cuda_graph=False)
        self.assertEqual(samples, [0.014048, 0.014048])
        self.assertEqual(events, ['program', 'sync', 'reset', 'sync', 'program', 'sync',
                                  'reset', 'sync', 'program', 'sync'])
        self.assertEqual(benchmark.non_target_dispatches, 0)
        error = ValueError('partial capture')
        error.activity_snapshot = {**self.raw['activity'], 'collection_errors': ['failed drain']}
        def fail(function): raise error
        benchmark._collect = fail
        with patch.dict('sys.modules', {'torch': torch}), self.assertRaisesRegex(ValueError, 'partial capture'):
            benchmark(lambda: None, dry_run_iters=1, repeat_iters=2, cold_l2_cache=True, use_cuda_graph=False)
        self.assertEqual(benchmark.last_activity['activity'], error.activity_snapshot)
        self.assertIsNone(benchmark.non_target_dispatches)

    def test_invalid_reset_capture_is_retained_before_validation(self):
        fixture = profile_fixtures.ProgramProfile
        with patch('open_cake_ir.evaluation.metax_benchmark.activity_collector'):
            benchmark = McptiProgramBenchmark(fixture.candidate, activity_library='fixture', l2_cache_bytes=8388608)
        bad = capture(kernel('fill', 1, 1000), kernel('unexpected', 2, 5000))
        benchmark._collect = lambda function: bad
        tensor = SimpleNamespace(fill_=lambda value: None)
        torch = SimpleNamespace(float32='fixture', empty=lambda *args, **kwargs: tensor,
            cuda=SimpleNamespace(get_device_properties=lambda index: SimpleNamespace(L2_cache_size=8388608)))
        with patch.dict('sys.modules', {'torch': torch}), self.assertRaisesRegex(ValueError, 'reset'):
            benchmark(lambda: None, dry_run_iters=1, repeat_iters=1, cold_l2_cache=True, use_cuda_graph=False)
        self.assertEqual(benchmark.last_activity['reset_activity'], bad)

    def test_measurement_composer_preserves_cohorts_on_final_check_failure(self):
        from open_cake_ir.lab.faults import RunProtocolFault
        fixture = profile_fixtures.ProgramProfile
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('measure_fixture', root / 'tools/qualify_tensor_program.py')
        tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
        preflight = SimpleNamespace(correctness_passed=True, artifact_payloads={'correctness_output': b'{"fixture":true}'})
        resources = {'kind': 'ordered_program', 'stages': {name: {'registers_per_thread': 16,
            'dynamic_shared_bytes': item.dynamic_shared_memory_bytes, 'local_bytes': 0}
            for name, item in fixture.manifests.items()}}
        native = SimpleNamespace(launch_calls=0, closed=False, resources=resources)
        def close(): native.closed = True
        loaded = SimpleNamespace(loaded=native, close=close)
        benchmark = SimpleNamespace(last_activity={'activity': self.raw['activity']})
        def cohort(*args, **kwargs):
            native.launch_calls += 36 * fixture.manifest.kernels_per_call
            return [0.1] * 25, {'passed': True, 'checked_launches': 36, 'output_mismatches': 0,
                                'max_abs_error': 0.0, 'inputs_unchanged': True}
        fault = RunProtocolFault('harness_fault', 'postflight failed', artifact_payloads={'postflight_raw': b'kept'})
        with tempfile.TemporaryDirectory() as directory, patch.object(tool, 'evaluate_program_case', side_effect=[preflight, fault]), patch(
                'open_cake_ir.evaluation.core.LoadedTorchTensorCandidate', return_value=loaded), patch(
                'open_cake_ir.evaluation.metax_program_benchmark.McptiProgramBenchmark', return_value=benchmark), patch(
                'open_cake_ir.tasks.evaluate._fresh_tile_cohort', side_effect=cohort):
            output = Path(directory)
            with self.assertRaisesRegex(RunProtocolFault, 'postflight failed') as caught:
                tool.measure_program(SimpleNamespace(output=output), fixture.candidate, fixture.workload,
                    SimpleNamespace(case_id='primary'), fixture.admission, SimpleNamespace(inputs={}), {},
                    {'activity_library': 'fixture'}, {})
            diagnostic = json.loads((output / 'measurement-diagnostic.json').read_text())
            self.assertEqual(len(diagnostic['cohorts']), 5)
            self.assertEqual(diagnostic['kernel_calls'], 720)
            self.assertTrue(diagnostic['module_unloaded'])
            self.assertFalse(diagnostic['passed'])
            self.assertEqual(caught.exception.artifact_payloads['postflight_raw'], b'kept')
            self.assertEqual(len(list(output.glob('cohort-*.json'))), 5)
