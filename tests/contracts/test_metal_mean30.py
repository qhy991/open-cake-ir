"""CPU policy checks: counts and diagnostic-only dispersion are explicit."""
import unittest
from open_cake_ir.tasks.workloads import create_task
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.evaluation.paired import paired_protocol
from open_cake_ir.tasks.normalization.study import evaluation_policy

class MetalMean30Test(unittest.TestCase):
    def test_thirty_normalized_samples_per_arm_not_per_cohort(self):
        document, _ = create_task('rmsnorm', backend='metal-m4', rows=128, columns=1024)
        workload = WorkloadContract(document)
        policy = evaluation_policy(workload, metal_mean30=True)
        protocol = paired_protocol(policy)
        self.assertEqual(protocol.statistic, 'mean')
        self.assertEqual(len(protocol.pair_order) * protocol.samples_per_cohort, 30)
        self.assertEqual(protocol.route_calls_per_cohort - protocol.samples_per_cohort, 3)
        self.assertEqual(protocol.dispatches_per_sample, 64)
        self.assertIsNone(protocol.maximum_cv)
        self.assertIsNone(protocol.maximum_relative_iqr)
        self.assertEqual(protocol.required_pair_wins, 0)
        with self.assertRaisesRegex(ValueError, '64 dispatches'):
            evaluation_policy(workload, metal_mean30=True, dispatches_per_sample=1)

    def test_absent_option_preserves_python_legacy_policy(self):
        document, _ = create_task('rmsnorm', backend='metal-m4', rows=128, columns=1024)
        workload = WorkloadContract(document)
        protocol = paired_protocol(evaluation_policy(workload))
        self.assertEqual(protocol.statistic, 'median')
        self.assertEqual(len(protocol.pair_order) * protocol.samples_per_cohort, 250)
