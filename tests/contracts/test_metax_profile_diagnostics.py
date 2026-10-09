"""Preflight diagnostics stay intact while common profile metrics aggregate."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.core import EvaluationProtocol, EvaluationReceipt, _plain_json
from open_cake_ir.evaluation.metax_observations import validate_profile_correctness
from open_cake_ir.serialization import canonical_json_bytes
from tests.contracts import test_metax_program_profile as fixtures

ROOT = Path(__file__).resolve().parents[2]


def observations():
    preflight = {'output_mismatches': 0, 'inputs_unchanged': True, 'max_abs_error': .125,
                 'task_diagnostic': {'phase': 'preflight', 'rows': [1, '诊断', None]}}
    instrumented = {'output_mismatches': 0, 'inputs_unchanged': True, 'max_abs_error': .25,
                    'task_diagnostic': {'phase': 'instrumented', 'rows': [2, '不同记录', False]}}
    metrics = {**deepcopy(preflight), 'max_abs_error': .25}
    return {'passed': True, 'correctness_launches': 2, 'preflight': preflight, 'metrics': metrics,
            'instrumented': {'passed': True, 'metrics': instrumented}}


class ProfileDiagnosticContract(unittest.TestCase):
    def test_complete_preflight_diagnostics_are_the_aggregate_basis(self):
        record = observations()
        before = deepcopy(record)
        validate_profile_correctness({'correctness_launches': 2}, record)
        self.assertEqual(record, before)
        self.assertNotEqual(record['metrics']['task_diagnostic'], record['instrumented']['metrics']['task_diagnostic'])

    def test_aggregate_extra_fields_cannot_be_added_removed_changed_or_wrongly_sourced(self):
        record = observations()
        for change in (
            lambda d: d['metrics'].pop('task_diagnostic'),
            lambda d: d['metrics'].update(unowned='extra'),
            lambda d: d['metrics']['task_diagnostic']['rows'].append('changed'),
            lambda d: d['metrics'].update(task_diagnostic=deepcopy(d['instrumented']['metrics']['task_diagnostic'])),
            lambda d: d['metrics'].update(max_abs_error=.125),
        ):
            value = deepcopy(record); change(value)
            with self.assertRaisesRegex(ValueError, 'correctness aggregate'):
                validate_profile_correctness({'correctness_launches': 2}, value)

    def test_plain_old_metrics_and_strict_oracle_guards_are_unchanged(self):
        record = observations()
        for key in ('preflight', 'metrics'): record[key].pop('task_diagnostic')
        record['instrumented']['metrics'].pop('task_diagnostic')
        original = canonical_json_bytes(record)
        validate_profile_correctness({'correctness_launches': 2}, record)
        self.assertEqual(canonical_json_bytes(record), original)
        for field, value in (('output_mismatches', 1), ('inputs_unchanged', False),
                             ('max_abs_error', float('nan')), ('max_abs_error', -1.)):
            changed = deepcopy(record); changed['preflight'][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, 'oracle metrics'):
                validate_profile_correctness({'correctness_launches': 2}, changed)

    def test_standalone_profile_composer_retains_both_distinct_real_receipts(self):
        fixtures.ProgramProfile.setUpClass()
        fixture = fixtures.ProgramProfile('runTest')
        spec = importlib.util.spec_from_file_location('diagnostic_profile_tool', ROOT/'tools/qualify_tensor_program.py')
        tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
        protocol = EvaluationProtocol('profile-diagnostic-fixture', 'confirmatory',
                                     fixture.workload.canonical_sha256, 'primary', 'none')
        record = observations()
        receipts = []
        for phase, metrics in (('preflight', record['preflight']), ('instrumented', record['instrumented']['metrics'])):
            launch = {'candidate_sha256': fixture.candidate.candidate_sha256,
                'manifest_sha256': fixture.manifest.canonical_sha256,
                'kernel_calls': fixture.manifest.kernels_per_call, 'fallback_calls': 0,
                'device_admission': asdict(fixture.admission), 'module_unloaded': True,
                'resources': {'kind': 'ordered_program', 'stages': {name: {
                    'registers_per_thread': 16, 'local_bytes': 492,
                    'dynamic_shared_bytes': item.dynamic_shared_memory_bytes}
                    for name, item in fixture.manifests.items()}}}
            check = {'passed': True, 'metrics': metrics}
            if phase == 'instrumented': check['native_activity'] = fixture.raw
            payloads = {'correctness_output': canonical_json_bytes(check),
                        'launch_receipt': canonical_json_bytes(launch), 'timing_samples': b'null'}
            receipts.append(EvaluationReceipt(fixture.candidate.candidate_sha256, fixture.workload.canonical_sha256,
                protocol.canonical_sha256, 'confirmatory', 'primary', True, metrics,
                fixture.manifest.kernels_per_call, 0, sha256(payloads['launch_receipt']).hexdigest(), None, payloads))
        prepared = object()
        with patch.object(tool, 'evaluate_program_case', side_effect=receipts) as evaluate:
            result = tool.profile_program(fixture.candidate, fixture.workload, protocol, fixture.admission,
                                          prepared, {'activity_library': 'CPU fixture'})
        self.assertEqual(evaluate.call_count, 2)
        self.assertNotIn('observe', evaluate.call_args_list[0].kwargs)
        self.assertIn('observe', evaluate.call_args_list[1].kwargs)
        correctness = json.loads(result.artifact_payloads['correctness_output'])
        self.assertEqual(correctness['metrics'], record['metrics'])
        self.assertEqual(correctness['preflight'], record['preflight'])
        self.assertEqual(correctness['instrumented']['metrics'], record['instrumented']['metrics'])
        self.assertEqual(correctness['observations']['preflight']['metrics'], record['preflight'])
        self.assertEqual(correctness['observations']['instrumented']['metrics'], record['instrumented']['metrics'])
        self.assertEqual(_plain_json(result.correctness), record['metrics'])
        self.assertIsNone(result.timing)
        self.assertEqual(len(result.attribution_feedback['stages']), fixture.manifest.kernels_per_call)


if __name__ == '__main__':
    unittest.main()
