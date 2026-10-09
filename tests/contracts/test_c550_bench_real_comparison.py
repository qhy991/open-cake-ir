"""Exercise the pinned external comparison on CPU, including rejected statistics."""
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.core import compare_tile_output_values
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks.c550_bench.binding import BenchProblem
from open_cake_ir.tasks.c550_bench.workload import BenchWorkload


@unittest.skipUnless(os.environ.get('CAKE_C550_BENCH_ROOT'), 'requires the pinned private Bench checkout and CPU Torch')
class OriginalBenchComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = BenchProblem.open(Path(os.environ['CAKE_C550_BENCH_ROOT']), 'L1/069_rms_norm')

    def compare(self, actual, expected, tolerance=None, *, reference_mapping=dict, candidate_mapping=dict,
                reference_name='out', candidate_name='out'):
        from sol_execbench.core.data import Definition, ToleranceSpec
        definition = Definition.model_validate({
            'name': 'comparison_control', 'axes': {'n': {'type': 'var'}},
            'inputs': {'x': {'shape': ['n'], 'dtype': 'float32'}},
            'outputs': {'out': {'shape': ['n'], 'dtype': 'float32'}},
            'reference': 'def run(x):\n    return x\n',
        })
        row = SimpleNamespace(uuid='synthetic', axes={'n': actual.numel()},
                              tolerance=tolerance or ToleranceSpec())
        problem = BenchProblem(self.original.root, 'synthetic', self.original.api, definition,
                               (row,), ({},), self.original.task_root)
        expected_outputs = reference_mapping({reference_name: expected})
        actual_outputs = candidate_mapping({candidate_name: actual})
        raw = problem.compare('synthetic', expected_outputs, actual_outputs)
        workload = BenchWorkload({'workload_id': 'comparison-control',
            'cases': [{'case_id': 'primary', 'shape': {'N': actual.numel()}, 'seed': 200, 'mode': 'fixture'}]})
        with patch('open_cake_ir.tasks.c550_bench.workload.problem_for', return_value=(problem, 'synthetic')):
            passed, metrics = compare_tile_output_values(workload, expected_outputs, actual_outputs)
        self.assertEqual(passed, raw['passed'])
        return passed, json.loads(canonical_json_bytes(metrics))

    def test_immutable_named_outputs_preserve_pass_rejection_and_original_ratio(self):
        import torch
        from sol_execbench.core.data import ToleranceSpec
        expected = torch.ones(100)
        one_wrong = expected.clone(); one_wrong[0] = 2.
        for actual, tolerance in ((expected.clone(), None), (torch.zeros(100), None),
                                  (one_wrong, ToleranceSpec(required_matched_ratio=.99))):
            baseline = self.compare(actual, expected, tolerance)
            for reference_mapping, candidate_mapping in (
                    (MappingProxyType, dict), (dict, MappingProxyType), (MappingProxyType, MappingProxyType)):
                with self.subTest(reference=reference_mapping.__name__, candidate=candidate_mapping.__name__):
                    self.assertEqual(self.compare(actual, expected, tolerance,
                        reference_mapping=reference_mapping, candidate_mapping=candidate_mapping), baseline)

    def test_immutable_wrong_output_names_remain_original_rejections(self):
        import torch
        expected = torch.ones(4)
        for names in ({'reference_name': 'wrong'}, {'candidate_name': 'wrong'}):
            baseline = self.compare(expected.clone(), expected, **names)
            self.assertFalse(baseline[0])
            self.assertEqual(baseline[1]['original_bench_check']['reason'], 'output_names')
            self.assertEqual(self.compare(expected.clone(), expected,
                reference_mapping=MappingProxyType, candidate_mapping=MappingProxyType, **names), baseline)

    def test_finite_fp32_subtraction_overflow_is_a_retained_numeric_rejection(self):
        import torch
        largest = torch.finfo(torch.float32).max
        passed, record = self.compare(torch.tensor([largest]), torch.tensor([-largest]))
        self.assertFalse(passed)
        self.assertEqual(record['original_bench_check']['outputs']['out']['max_absolute_error'], 'Infinity')
        self.assertEqual(record['output_mismatches'], 1)

    def test_nan_and_infinity_reject_without_serialization_fault(self):
        import torch
        for value in (float('nan'), float('inf')):
            passed, record = self.compare(torch.tensor([value]), torch.ones(1))
            self.assertFalse(passed)
            self.assertEqual(record['output_mismatches'], 1)

    def test_finite_match_and_original_matched_ratio_are_preserved(self):
        import torch
        from sol_execbench.core.data import ToleranceSpec
        expected = torch.ones(100)
        self.assertTrue(self.compare(expected.clone(), expected)[0])
        actual = expected.clone()
        actual[0] = 2.
        passed, record = self.compare(actual, expected, ToleranceSpec(required_matched_ratio=.99))
        self.assertTrue(passed)
        self.assertEqual(record['original_bench_check']['outputs']['out']['max_absolute_error'], 1.)
        self.assertEqual(record['comparison_unit'], 'original_bench_output_contract')

    def test_dense_transpose_view_preserves_pointer_and_inverse_values_without_copy(self):
        import torch
        from open_cake_ir.tasks.c550_bench.binding import physical_input_view
        original = torch.arange(24).reshape(2, 3, 4).transpose(0, 1)
        physical = physical_input_view(original, [1, 0, 2])
        self.assertTrue(physical.is_contiguous())
        self.assertEqual(physical.data_ptr(), original.data_ptr())
        self.assertTrue(torch.equal(physical.permute(1, 0, 2), original))
        with self.assertRaisesRegex(ValueError, 'zero-copy dense'):
            physical_input_view(original, [0, 1, 2])
        sliced = torch.arange(40).reshape(5, 8)[:, ::2]
        with self.assertRaisesRegex(ValueError, 'zero-copy dense'):
            physical_input_view(sliced, [0, 1])


if __name__ == '__main__':
    unittest.main()
