"""Successor event receipts through the real paired worker; legacy MCPTI stays intact."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation import metax_event_benchmark as event
from open_cake_ir.evaluation.paired import paired_protocol, validate_paired_broker
from open_cake_ir.tasks.normalization.study import evaluation_policy
from open_cake_ir.serialization import canonical_json_bytes
from tests.contracts import test_metax_paired as fixtures


class MetaXMean10Tests(unittest.TestCase):
    def fixture(self):
        f = fixtures.MacaPairedReceipts(); f.setUp()
        self.addCleanup(f.doCleanups)
        return f

    def test_declared_count_and_legacy_default(self):
        f = self.fixture()
        old = paired_protocol(f.policy)
        self.assertEqual(old.statistic, 'median')
        self.assertEqual(len(old.pair_order) * old.samples_per_cohort, 250)
        policy = evaluation_policy(f.workload, metax_mean10=True)
        new = paired_protocol(policy)
        self.assertEqual(new.statistic, 'mean')
        self.assertEqual(len(new.pair_order) * new.samples_per_cohort, 10)
        self.assertIsNone(new.maximum_cv)
        for key, value in [('samples_per_cohort', 10), ('statistic', 'median'), ('maximum_cv', .05)]:
            malformed = deepcopy(policy); malformed['paired_timing'][key] = value
            with self.assertRaises(ValueError): paired_protocol(malformed)

    def execute(self, *, zero=False, native=False):
        f = self.fixture(); f.policy = evaluation_policy(f.workload,
            metax_mean10=not native, metax_native_mean10=native)
        class Assay:
            def __init__(self, manifest, **kwargs): self.manifest = manifest
            def __call__(self, function, **kwargs):
                for _ in range(kwargs['dry_run_iters'] + kwargs['repeat_iters']): function()
                # A distinct mean/median exercises real selection and receipt derivation.
                values = [1., 1., 1., 1., 6.] if self.manifest.kernel_name == 'candidate' else [4.] * 5
                if zero: values[0] = 0.
                self.last_activity = {
                    'kind':'maca_native_event_samples_v1' if native else 'maca_event_samples_v1',
                    'timer':event.NATIVE_TIMER if native else event.TIMER,
                    'cache_policy':event.NATIVE_RESET if native else event.RESET,
                    'interval':event.NATIVE_INTERVAL if native else event.INTERVAL,
                    'coverage':event.COVERAGE,'profiler_enabled':False,
                    'target':self.manifest.target,'device':0,'stream':0,
                    'l2_cache_bytes':8388608,'reset_bytes':33554432,
                    'warmup_calls':11,'event_pair_primed':True,
                    'launch':{'kernel_name':self.manifest.kernel_name,'grid':list(self.manifest.grid),
                              'block':list(self.manifest.block),'dynamic_shared_memory_bytes':0},
                    'samples':[{'index':i,'elapsed_ms':v,'reset_enqueued_before_start':True,
                                'end_synchronized':True} for i,v in enumerate(values)]}
                return values
        if native:
            def capture_loaded_cohort(assay, loaded, arguments, *, dry_run_iters, repeat_iters):
                used = iter(arguments)
                return assay(lambda: loaded.launch(next(used)), dry_run_iters=dry_run_iters,
                             repeat_iters=repeat_iters)
            Assay.capture_loaded_cohort = capture_loaded_cohort
        with patch.object(event, 'MacaNativeEventBenchmark' if native else 'MacaEventBenchmark', Assay), \
             patch('open_cake_ir.evaluation.metax_observations.collect_maca_activity',
                   side_effect=AssertionError('profiler called during primary timing')) as profiler:
            receipt = f.execute()
            profiler.assert_not_called()
        return f, receipt

    def test_native_worker_keeps_full_oracle_and_broker_receipt_validation(self):
        f, receipt = self.execute(native=True)
        self.assertTrue(receipt.correctness_passed)
        self.assertEqual(receipt.timing['kind'], 'fixed_baseline_paired_maca_native_event_v1')
        self.assertEqual(receipt.timing['pooled_sample_counts'], {'candidate':10,'baseline':10})
        self.assertEqual(f.result['counters']['kernel_calls'],84)
        validate_paired_broker(receipt, f.admission.broker_job_id, f.result['counters'])
        raw = json.loads(f.payloads['timing_samples'])
        raw['measurements'][0]['arms']['candidate']['native_activity']['interval'] = event.INTERVAL
        with self.assertRaises(ValueError):
            f.receipt({**f.payloads, 'timing_samples':canonical_json_bytes(raw)})

    def test_real_worker_mean_counts_and_broker_replay(self):
        f, receipt = self.execute()
        self.assertTrue(receipt.correctness_passed)
        self.assertEqual(receipt.timing['pooled_mean_ms'], 2.)
        self.assertEqual(receipt.timing['pooled_median_ms'], 1.)
        self.assertEqual(receipt.timing['speedup'], 2.)
        self.assertEqual(receipt.timing['dispersion_gate'], 'diagnostic_only')
        self.assertEqual(f.result['counters']['timing_samples'], 20)
        self.assertEqual(f.result['counters']['kernel_calls'], 84)
        validate_paired_broker(receipt, f.admission.broker_job_id, f.result['counters'])

    def test_bad_event_observations_do_not_become_valid_means(self):
        f, _ = self.execute(); original = json.loads(f.payloads['timing_samples'])
        for label in ['zero', 'missing', 'wrong_stream', 'reset_order', 'unsynchronized', 'profiler', 'foreign_coverage', 'changed_sample']:
            raw = deepcopy(original); arm = raw['measurements'][0]['arms']['candidate']; data = arm['native_activity']
            if label == 'zero': data['samples'][0]['elapsed_ms'] = 0
            elif label == 'missing': data['samples'].pop()
            elif label == 'wrong_stream': data['stream'] = 1
            elif label == 'reset_order': data['samples'][0]['reset_enqueued_before_start'] = False
            elif label == 'unsynchronized': data['samples'][0]['end_synchronized'] = False
            elif label == 'profiler': data['profiler_enabled'] = True
            elif label == 'foreign_coverage': arm['non_target_dispatches'] = 0
            else: arm['samples_ms'][0] = 2
            with self.subTest(label=label), self.assertRaises(ValueError):
                f.receipt({**f.payloads,'timing_samples':canonical_json_bytes(raw)})

    def test_zero_from_producer_is_retained_without_receipt(self):
        with self.assertRaises(ValueError): self.execute(zero=True)

    def test_legacy_measurement_source_and_new_source_share_one_platform_owner(self):
        from open_cake_ir.evaluation.platforms import platform_for
        row = platform_for('xcore1002')
        self.assertEqual(row.measurement_source, 'mcpti_dispatch')
        self.assertEqual(row.measurement_source_for('fixed_baseline_paired_mcpti_dispatch_v1'), 'mcpti_dispatch')
        self.assertEqual(row.measurement_source_for('fixed_baseline_paired_maca_event_v1'), 'maca_event')
        with self.assertRaises(ValueError): row.measurement_source_for('unknown')
