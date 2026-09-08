"""CPU doubles for common Metal Evaluation receipts; no GPU observer invocation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
from subprocess import CompletedProcess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.evaluation.core import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evaluation.metal_manifest import MetalTensorLaunchManifest, compile_options
from open_cake_ir.evaluation.metal_observations import METAL_PROFILE_KIND
from open_cake_ir.evaluation.paired import PAIRED_METAL_KIND, paired_protocol, validate_paired_broker
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.evaluation import metal_runtime
from open_cake_ir.tasks import evaluate, workloads

HOST = {'device_name': 'Apple M1 Pro', 'device_registry_id': '1234',
        'operating_system': 'explicit unit-test host', 'target': 'apple_gpu_family7'}
CASES = ['primary', 'zeros', 'near_zero', 'alternating', 'mixed_magnitude']
JOB = 'metal-123456789abc'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def workload_fixture():
    return WorkloadContract({'workload_id': 'metal-evaluation-unit-fixture',
        'semantics': {'target': HOST['target'], 'candidate_abi': {'inputs': ['x'], 'outputs': ['out']}},
        'cases': [{'case_id': name, 'shape': {'R': 1, 'C': 2}, 'seed': i, 'mode': name} for i, name in enumerate(CASES)],
        'tensors': {name: {'shape': ['R', 'C'], 'dtype': 'fp32', 'layout': 'contiguous_row_major'} for name in ('x', 'out')},
        'validation': {'all_cases_required': True, 'primary_case': 'primary', 'comparison': 'elementwise_atol_rtol', 'atol': 1e-5, 'rtol': 1e-5}})


def policy_fixture():
    return {'case_id': 'primary', 'validation_case_ids': CASES.copy(),
        'attribution_evaluation': 'correctness_then_profile_each_search_survivor',
        'search_evaluation': 'correctness_then_paired_metal',
        'confirmatory_evaluation': 'fresh_fixed_candidate_correctness_then_paired_metal',
        'paired_timing': {'kind': PAIRED_METAL_KIND, 'arms': ['candidate', 'baseline'],
            'pair_order': [['candidate', 'baseline'], ['baseline', 'candidate']],
            'samples_per_cohort': 2, 'route_calls_per_cohort': 3,
            'maximum_cv': 0.2, 'materiality_ratio': 1.01, 'required_pair_wins': 2}}


def candidate_fixture(workload, label):
    manifest = MetalTensorLaunchManifest.for_workload(workload, 'primary', target=HOST['target'],
        kernel_name='fixture_kernel', grid=[1, 1, 1], block=[32, 1, 1])
    payloads = {'metal_binary_archive': b'not executed: explicit CPU artifact double ' + label.encode(),
        'launch_manifest': canonical(manifest.as_dict()),
        'metal_build_report': canonical({'build': {'host': HOST}, 'archive_only_reload': {'host': HOST, 'source_library_rebuilt': False}, 'compile_options': compile_options()})}
    candidate = LaunchableCandidate(sha256(label.encode()).hexdigest(), HOST['target'], 'fixture_kernel',
        {role: sha256(data).hexdigest() for role, data in payloads.items()}, manifest.canonical_sha256, payloads)
    return candidate, manifest


def fake_observe(**kwargs):
    """Counter observations are explicit test doubles, never called native evidence."""
    launches = []
    for row in kwargs['launch_plan']:
        start = 100.0 + row['index']
        command = {'launch_index': row['index'], 'completed': True, 'timed': row['timed'],
                   'gpu_start_seconds': start, 'gpu_end_seconds': start + (0.001 if row['role'] == 'candidate' else 0.002)}
        observed = {**row, 'passed': True, 'metrics': {'output_mismatches': 0, 'max_abs_error': 0.0, 'inputs_unchanged': True}, 'command_buffer': command}
        if row['profile']:
            observed['profile_raw'] = {'counter_set': 'Timestamp', 'counter': 'GPUTimestamp', 'sampling_boundary': 'compute_stage',
                'units': 'raw_device_timestamp_units', 'sample_indices': [0, 1], 'resolved_bytes': 16,
                'timestamps': [100, 200], 'command_buffer': command}
        launches.append(observed)
    return {'host': HOST, 'module_loads': len(kwargs['candidates']), 'launches': launches}


class MetalEvaluationContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cake-metal-evaluation-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workload = workload_fixture()
        self.candidate, self.manifest = candidate_fixture(self.workload, 'candidate')
        self.baseline, _ = candidate_fixture(self.workload, 'baseline')
        self.policy = policy_fixture()

    def authority(self, purpose='search'):
        return evaluate._Authority({'purpose': purpose, 'evaluation_protocol': self.policy}, self.root,
            SimpleNamespace(admit_host=lambda: {'kind': 'metal', 'host': HOST, 'observer_executable': '/unit-test/not-executed'}),
            self.workload, self.manifest, self.candidate, self.candidate.artifact_payloads, 'primary', self.baseline)

    def run_worker(self, purpose='search', observer=fake_observe):
        result = evaluate._base_result('unassigned')
        def inputs(workload, case_id):
            return {'x': [float(CASES.index(case_id)), 2.0]}
        with patch.dict(os.environ, {'METAL_JOB_ID': JOB}), \
             patch.object(workloads, 'materialize_case', side_effect=inputs, create=True) as materialize, \
             patch.object(workloads, 'reference_outputs', side_effect=lambda w,c,i: {'out': list(i['x'])}, create=True), \
             patch.object(metal_runtime, 'observe', side_effect=observer) as observe:
            evaluate._evaluate_metal_candidate(self.authority(purpose), result)
        self.assertEqual([call.args[1] for call in materialize.call_args_list], CASES)
        self.assertEqual(result['mode'], 'local_serialized')
        self.assertEqual(observe.call_count, 1)
        return result

    def receipt(self, result, *, payloads=None):
        value = result['receipt']
        data = payloads or {role: (self.root / name).read_bytes() for role, name in value['artifacts'].items()}
        return EvaluationReceipt(self.candidate.candidate_sha256, self.workload.canonical_sha256,
            sha256(canonical(self.policy)).hexdigest(), 'attribution' if 'profile' in data else 'search', 'primary',
            value['correctness_passed'], value['correctness'], 1, 0, sha256(data['launch_receipt']).hexdigest(), value['timing'], data)

    def test_real_worker_path_reuses_common_receipt_derivation_and_all_input_cases(self):
        result = self.run_worker()
        receipt = self.receipt(result)
        self.assertTrue(receipt.correctness_passed)
        self.assertTrue(receipt.timing['measurement_quality_passed'])
        self.assertGreater(receipt.timing['speedup'], 1.9)
        validate_paired_broker(receipt, JOB, result['counters'])
        self.assertEqual(result['counters']['compiler_invocations'], 0)
        self.assertEqual(result['counters']['module_loads'], 2)
        self.assertEqual(result['counters']['preflight_calls'], 10)
        self.assertEqual(result['counters']['kernel_calls'], 32)
        self.assertEqual(result['counters']['timing_samples'], 8)

    def test_raw_timestamp_and_validation_coverage_mutations_are_rejected(self):
        result = self.run_worker()
        base = {role: (self.root / name).read_bytes() for role, name in result['receipt']['artifacts'].items()}
        for role, mutate in (
            ('timing_samples', lambda d: d['measurements'][0]['arms']['candidate']['samples_ms'].__setitem__(0, 77.0)),
            ('timing_samples', lambda d: d.update(timer='CUPTI')),
            ('timing_samples', lambda d: d.update(allocation_mode='exclusive')),
            ('correctness_output', lambda d: d['participants']['candidate']['preflight']['launches'].pop()),
        ):
            changed = deepcopy(base)
            document = json.loads(changed[role]); mutate(document); changed[role] = canonical(document)
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.receipt(result, payloads=changed)

    def test_a_failed_nonprimary_case_prevents_timing_and_retains_all_preflight_checks(self):
        def rejected(**kwargs):
            observation = fake_observe(**kwargs)
            observation['launches'] = [row for row in observation['launches'] if row['phase'] == 'preflight']
            row = next(row for row in observation['launches'] if row['role'] == 'candidate' and row['input_case_id'] == 'zeros')
            row['passed'] = False
            row['metrics']['output_mismatches'] = 1
            row['metrics']['max_abs_error'] = 1.0
            return observation
        result = self.run_worker(observer=rejected)
        receipt = self.receipt(result)
        self.assertFalse(receipt.correctness_passed)
        self.assertIsNone(receipt.timing)
        self.assertEqual(result['counters']['kernel_calls'], 10)
        self.assertEqual(result['counters']['timing_samples'], 0)
        validate_paired_broker(receipt, JOB, result['counters'])

    def test_profile_is_separate_raw_uint64_stage_timestamps_with_no_timing_claim(self):
        result = self.run_worker('attribution')
        receipt = self.receipt(result)
        self.assertIsNone(receipt.timing)
        self.assertEqual(receipt.attribution_feedback['kind'], METAL_PROFILE_KIND)
        self.assertEqual(receipt.attribution_feedback['timestamp_delta'], 100)
        self.assertEqual(receipt.attribution_feedback['units'], 'raw_device_timestamp_units')
        payloads = dict(receipt.artifact_payloads)
        profile = json.loads(payloads['profile']); profile['summary']['timestamp_delta'] = 999
        payloads['profile'] = canonical(profile)
        with self.assertRaises(ValueError):
            self.receipt(result, payloads=payloads)

    def test_missing_broker_or_case_projection_refuses_before_observer(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(metal_runtime, 'observe') as observe:
            with self.assertRaisesRegex(ValueError, 'broker job'):
                evaluate._evaluate_metal_candidate(self.authority(), evaluate._base_result('unassigned'))
            observe.assert_not_called()
        self.policy['validation_case_ids'] = ['primary']
        with patch.dict(os.environ, {'METAL_JOB_ID': JOB}), patch.object(metal_runtime, 'observe') as observe:
            with self.assertRaisesRegex(ValueError, 'case projection'):
                evaluate._evaluate_metal_candidate(self.authority(), evaluate._base_result('unassigned'))
            observe.assert_not_called()

    def test_metal_policy_does_not_borrow_cupti_callback_counts_or_allow_unwarmed_samples(self):
        self.assertIsNotNone(paired_protocol(self.policy))
        wrong = deepcopy(self.policy); wrong['paired_timing']['route_calls_per_cohort'] = 2
        with self.assertRaisesRegex(ValueError, 'warmup'):
            paired_protocol(wrong)

    def test_missing_native_observer_never_compiles_or_dispatches(self):
        with patch.object(metal_runtime.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'observer executable'):
                metal_runtime.observe(workload=self.workload, candidates={'candidate': self.candidate},
                    manifests={'candidate': self.manifest}, input_cases={}, launch_plan=[],
                    observer_executable=self.root / 'missing', expected_host=HOST, directory=self.root / 'observation')
            run.assert_not_called()

    def test_observer_preparation_rejects_an_enclosing_parent_checkout(self):
        parent = self.root / 'parent-project'
        parent.mkdir()
        (parent / '.git').mkdir()
        output = parent / 'nested' / 'artifacts'
        with patch.object(metal_runtime.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'enclosing checkout'):
                metal_runtime.compile_observer(output)
            run.assert_not_called()
        self.assertFalse(output.exists())

    def test_observer_binary_snapshots_are_checked_by_independent_common_oracle(self):
        executable = self.root / 'observer-double'; executable.write_bytes(b'explicit test double; never executed')
        inputs = {'primary': {'inputs': {'x': [1.0, 2.0]}, 'expected': {'out': [1.0, 2.0]}}}
        plan = [{'index': 0, 'role': 'candidate', 'phase': 'preflight', 'input_case_id': 'primary', 'timed': False, 'profile': False}]
        def process(command, **kwargs):
            request = json.loads(Path(command[1]).read_text()); directory = Path(request['output_directory'])
            (directory / 'x.bin').write_bytes(struct.pack('<2f', 1.0, 2.0))
            (directory / 'out.bin').write_bytes(struct.pack('<2f', 1.0, 2.0))
            row = {'index': 0, 'role': 'candidate', 'phase': 'preflight', 'input_case_id': 'primary',
                'buffer_paths': {'x': 'x.bin', 'out': 'out.bin'}, 'preflight_guard_passed': True,
                'command_buffer': {'launch_index': 0, 'completed': True, 'timed': False, 'gpu_start_seconds': 1.0, 'gpu_end_seconds': 1.001}}
            report = {'status': 'completed', 'host': HOST, 'source_library_rebuilt': False,
                'archive_miss_policy': 'failOnBinaryArchiveMiss', 'module_loads': 1, 'launches': [row]}
            return CompletedProcess(command, 0, canonical(report), b'')
        with patch.object(metal_runtime.subprocess, 'run', side_effect=process):
            result = metal_runtime.observe(workload=self.workload, candidates={'candidate': self.candidate},
                manifests={'candidate': self.manifest}, input_cases=inputs, launch_plan=plan,
                observer_executable=executable, expected_host=HOST, directory=self.root / 'observation')
        self.assertTrue(result['launches'][0]['passed'])
        request = json.loads((self.root / 'observation' / 'request.json').read_text())
        self.assertNotIn('source_path', request['participants'][0])
        self.assertEqual(request['participants'][0]['archive_path'].split('.')[-1], 'metallib')


if __name__ == '__main__':
    unittest.main()
