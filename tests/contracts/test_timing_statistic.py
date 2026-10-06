"""Mean and median can disagree; selection must follow the declared estimand."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

from open_cake_ir.evaluation.timing import (
    PairedTimingProtocol, derive_paired_timing, summarize_cohort,
    timing_latency_ms, timing_latencies,
)
from open_cake_ir.evaluation.paired import paired_protocol, paired_summary
from open_cake_ir.lab.selection import _receipt_latency_ms
from open_cake_ir.lab.feedback import baseline_comparison_feedback
from tests.contracts.test_paired_execution import protocol as legacy_policy


def measurements(protocol):
    # Two cohorts: each candidate has a 1 ms median but a 3 ms mean.
    # Baseline is uniformly 2 ms. No sample may be discarded as an outlier.
    rows = []
    for i, order in enumerate(protocol.pair_order):
        arms = {}
        for position, arm in enumerate(order):
            values = [1.] * 14 + [31.] if arm == 'candidate' else [2.] * 15
            arms[arm] = dict(position=position, samples_ms=values,
                route_calls=protocol.route_calls_per_cohort, summary=summarize_cohort(values))
        rows.append(dict(pair_index=i, order=list(order), arms=arms))
    return rows


class TimingStatisticTest(unittest.TestCase):
    def setUp(self):
        self.protocol = PairedTimingProtocol(('candidate', 'baseline'),
            (('candidate', 'baseline'), ('baseline', 'candidate')), 15, 32,
            None, 1.05, 0, statistic='mean')

    def test_mean_keeps_all_thirty_samples_and_reverses_median_winner(self):
        observed = derive_paired_timing(measurements(self.protocol), self.protocol)
        self.assertEqual(dict(observed.pooled_sample_counts), dict(candidate=30, baseline=30))
        self.assertEqual(dict(observed.pooled_means_ms), dict(candidate=3., baseline=2.))
        self.assertEqual(dict(observed.pooled_medians_ms), dict(candidate=1., baseline=2.))
        self.assertTrue(observed.measurement_quality_passed)
        self.assertEqual(observed.classification, 'second_arm_faster')
        median = replace(self.protocol, statistic='median', maximum_cv=.05, required_pair_wins=1)
        self.assertFalse(derive_paired_timing(measurements(median), median).measurement_quality_passed)

    def test_mean_summary_and_author_feedback_use_mean_without_relabeling_median(self):
        policy = legacy_policy()
        policy['paired_timing'].update(statistic='mean', maximum_cv=None, required_pair_wins=0,
            samples_per_cohort=15, route_calls_per_cohort=32,
            pair_order=[list(p) for p in self.protocol.pair_order])
        admitted = paired_protocol(policy)
        summary = paired_summary({'evaluation_protocol':policy, 'measurements':measurements(admitted)})
        self.assertEqual(summary['pooled_mean_ms'], 3.)
        self.assertEqual(summary['pooled_median_ms'], 1.)
        self.assertEqual(timing_latency_ms(summary), 3.)
        self.assertEqual(timing_latencies(summary)['baseline'], 2.)
        self.assertEqual(_receipt_latency_ms(SimpleNamespace(timing=summary)), 3.)
        feedback = baseline_comparison_feedback(SimpleNamespace(document={'execution':{}}), summary)
        self.assertEqual(feedback['statistic'], 'mean')
        self.assertEqual(feedback['baseline_latency_ms'], 2.)

    def test_legacy_summary_has_no_new_fields_and_keeps_median_selection(self):
        protocol = replace(self.protocol, statistic='median', maximum_cv=.05, required_pair_wins=1)
        policy = legacy_policy()
        policy['paired_timing'].update(maximum_cv=.05, required_pair_wins=1,
            samples_per_cohort=15, route_calls_per_cohort=32,
            pair_order=[list(p) for p in protocol.pair_order])
        summary = paired_summary({'evaluation_protocol':policy, 'measurements':measurements(protocol)})
        self.assertNotIn('statistic', summary)
        self.assertNotIn('pooled_mean_ms', summary)
        self.assertEqual(timing_latency_ms(summary), 1.)

    def test_diagnostic_only_does_not_admit_missing_or_invalid_samples(self):
        for value in (float('nan'), float('inf'), 0., -1., True):
            with self.subTest(value=value):
                rows = measurements(self.protocol)
                rows[0]['arms']['candidate']['samples_ms'][0] = value
                with self.assertRaises(ValueError):
                    derive_paired_timing(rows, self.protocol)
        rows = measurements(self.protocol)
        rows[0]['arms']['candidate']['samples_ms'].pop()
        with self.assertRaises(ValueError):
            derive_paired_timing(rows, self.protocol)

    def test_no_implicit_mean_or_unknown_estimator(self):
        with self.assertRaises(ValueError):
            replace(self.protocol, statistic='median')
        with self.assertRaises(ValueError):
            replace(self.protocol, statistic='trimmed_mean')
        self.assertIsNone(timing_latency_ms({'statistic':'mean','pooled_median_ms':1.}))
