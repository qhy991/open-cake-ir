"""Run common worker output paths with real immutable receipts and CPU devices."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.core import _plain_json, _freeze_json
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks import evaluate as worker
from tests.contracts.test_metax_paired import MacaPairedReceipts
from tests.contracts.test_metax_program_events import ProgramEvents


DETAIL = {'passed': True, 'outputs': {'out': {'passed': True, 'reason': 'upstream_numeric',
    'max_absolute_error': .0625, 'max_relative_error': .013888889,
    'extra': {'label': '原始比较器', 'coverage': ['finite', None, False]}}}}


def detailed(function):
    """Keep the real verdict/launch, adding the nested diagnostic shape Bench owns."""
    def invoke(*args, **kwargs):
        receipt = function(*args, **kwargs)
        metrics = _plain_json(receipt.correctness)
        metrics['original_bench_check'] = {**DETAIL, 'passed': receipt.correctness_passed}
        payloads = dict(receipt.artifact_payloads)
        payloads['correctness_output'] = canonical_json_bytes(
            {'passed': receipt.correctness_passed, 'metrics': metrics})
        return replace(receipt, correctness=metrics, artifact_payloads=payloads)
    return invoke


class ImmutableEvaluationEvidence(unittest.TestCase):
    def maca(self):
        fixture = MacaPairedReceipts('runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    @contextmanager
    def captured_writes(self):
        records = {}
        original = worker._write_new
        def write(path, value):
            expected = canonical_json_bytes(_plain_json(value))
            original(path, value)
            payload = path.read_bytes()
            self.assertEqual(payload, expected)
            records[path.name] = json.loads(payload)
        with patch.object(worker, '_write_new', side_effect=write):
            yield records

    def test_paired_all_case_output_keeps_nested_diagnostics_and_raw_measurements(self):
        fixture = self.maca()
        with patch.object(worker, 'evaluate_tile_validation_case',
                          side_effect=detailed(worker.evaluate_tile_validation_case)), self.captured_writes() as records:
            receipt = fixture.execute()
            worker._write_new(fixture.root/'worker-result.json', fixture.result)
        self.assertTrue(receipt.correctness_passed)
        document = records['correctness-output.json']
        for role in ('candidate', 'baseline'):
            for phase in ('preflight', 'postflight'):
                for row in document['participants'][role][phase]['launches']:
                    self.assertEqual(row['metrics']['original_bench_check'], DETAIL)
        self.assertEqual(sum(len(arm['samples_ms']) for row in records['timing-samples.json']['measurements']
                             for arm in row['arms'].values()), fixture.result['counters']['timing_samples'])
        self.assertEqual(records['worker-result.json']['receipt']['timing'], _plain_json(receipt.timing))

    def test_untimed_all_case_output_keeps_the_real_verdict_and_nested_diagnostics(self):
        fixture = self.maca()
        def untimed(authority, result, *, collect_timing, admission):
            worker._evaluate_untimed_validation_cases(authority, result, admission)
        with patch.object(worker, '_evaluate_metax_candidate', side_effect=untimed), \
             patch.object(worker, 'evaluate_tile_validation_case',
                          side_effect=detailed(worker.evaluate_tile_validation_case)), self.captured_writes() as records:
            receipt = fixture.execute()
            worker._write_new(fixture.root/'worker-result.json', fixture.result)
        self.assertTrue(receipt.correctness_passed)
        self.assertIsNone(records['timing-samples.json'])
        self.assertEqual(len(records['correctness-output.json']['validation_cases']), len(fixture.workload.case_ids))
        for row in records['correctness-output.json']['validation_cases']:
            self.assertEqual(row['metrics']['original_bench_check'], DETAIL)

    def test_attribution_output_keeps_frozen_preflight_and_separate_profile(self):
        ProgramEvents.setUpClass()
        fixture = ProgramEvents('runTest')
        self.addCleanup(fixture.doCleanups)
        with patch.object(worker, 'evaluate_tile_workload', side_effect=detailed(worker.evaluate_tile_workload)), \
             self.captured_writes() as records:
            receipt, result = fixture.execute_worker(profile=True)
        self.assertTrue(receipt.correctness_passed)
        self.assertEqual(records['correctness-output.json']['preflight']['original_bench_check'], DETAIL)
        self.assertEqual(records['correctness-output.json']['metrics']['original_bench_check'], DETAIL)
        self.assertIn('instrumented', records['correctness-output.json'])
        self.assertEqual(len(records['profile.json']['summary']['stages']), 2)
        self.assertEqual(result['counters']['timing_samples'], 0)

    def test_failed_paired_activity_retains_original_error_and_nested_observations(self):
        fixture = self.maca()
        with patch.object(worker, 'evaluate_tile_validation_case',
                          side_effect=detailed(worker.evaluate_tile_validation_case)), \
             self.assertRaisesRegex(ValueError, 'serialized samples overlap') as caught:
            fixture.execute(reject_timing=True)
        raw = json.loads(caught.exception.artifact_payloads['paired_activity'])
        self.assertEqual(raw['error_class'], 'ValueError')
        self.assertIn('serialized samples overlap', raw['error'])
        self.assertEqual(raw['correctness_observations']['candidate']['preflight']['launches'][0]
                         ['metrics']['original_bench_check'], DETAIL)
        self.assertEqual(len(raw['native_activity']['candidate']['activity']['records']), 100)
        self.assertIsNone(fixture.result['receipt'])

    def test_failed_program_profile_retains_frozen_raw_activity_and_original_error(self):
        from open_cake_ir.evaluation import metax_program_profile
        ProgramEvents.setUpClass()
        fixture = ProgramEvents('runTest')
        self.addCleanup(fixture.doCleanups)
        original = metax_program_profile.capture_program_activity
        def capture(*args, **kwargs):
            return MappingProxyType(original(*args, **kwargs))
        with patch.object(metax_program_profile, 'capture_program_activity', side_effect=capture), \
             self.assertRaisesRegex(ValueError, 'sealed native launch') as caught:
            fixture.execute_worker(profile=True, failed_profile=True)
        raw = json.loads(caught.exception.artifact_payloads['program_activity'])
        self.assertIn('manifest', raw)
        self.assertIn('stage_manifests', raw)
        self.assertEqual(raw['activity']['records'][0]['name'], 'unrelated_kernel')

    def test_writer_keeps_canonical_bytes_and_does_not_mutate_frozen_records(self):
        document = {'diagnostic': DETAIL, 'values': [1, False, None, '中文']}
        frozen = _freeze_json(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'record.json'
            worker._write_new(path, frozen)
            self.assertEqual(path.read_bytes(), canonical_json_bytes(document))
            with self.assertRaises(FileExistsError): worker._write_new(path, frozen)
        with self.assertRaises(TypeError): frozen['diagnostic']['passed'] = False
        self.assertEqual(_plain_json(frozen), document)

    def test_unknown_leaves_nonstring_keys_and_nonfinite_numbers_are_refused_before_write(self):
        for invalid in ({'nested': object()}, {'nested': {1: 'not a JSON key'}},
                        {'nested': float('nan')}, {'nested': float('inf')}):
            with self.subTest(value=repr(invalid)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'record.json'
                with self.assertRaises((ValueError, TypeError)): worker._write_new(path, invalid)
                self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
