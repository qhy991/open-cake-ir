"""Explicit launch-width transformation; GPU numerics and speed require evaluation."""
import ast
import copy
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.tasks.workloads import create_task

ROOT = Path(__file__).resolve().parents[2]


class WarpSpecializationTests(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def document(self, task='silu', backend='triton-b300'):
        _, source = create_task(task, backend=backend, rows=128, columns=1024)
        return frontend.parse(source).document

    def apply(self, document, width=16):
        return self.compiler.specialize_triton_warps(document, num_warps=width,
            schedule_id=document['schedule_id'] + '-width', entry_point='width_candidate')

    def test_two_evidence_families_and_held_out_task_keep_math_and_abi(self):
        for task in ('rmsnorm_input_gradient', 'swiglu', 'softmax_backward'):
            for backend in ('triton-b200', 'triton-b300'):
                with self.subTest(task=task, backend=backend):
                    document = self.document(task, backend)
                    original = copy.deepcopy(document)
                    result = self.apply(document)
                    self.assertTrue(result.applied, (result.reason, result.message))
                    self.assertEqual(document, original)
                    candidate = result.schedule
                    for field in original.keys() - {'roles', 'schedule_id', 'lowering'}:
                        self.assertEqual(candidate[field], original[field], field)
                    self.assertEqual(candidate['roles'][0]['warps'], list(range(16)))
                    before = self.compiler.lower(self.compiler.assess(document))
                    after = self.compiler.lower(result.assessment)
                    self.assertEqual(after.toolchain_requirements['compile_options']['num_warps'], 16)
                    def kernel(lowering):
                        node = next(n for n in ast.parse(lowering.source).body
                                    if isinstance(n, ast.FunctionDef)
                                    and n.name == lowering.toolchain_requirements['kernel_entry_point'])
                        node.name = 'same_kernel'
                        return ast.dump(node)
                    self.assertEqual(kernel(before), kernel(after))
                    candidate['roles'][0]['warps'].clear()
                    self.assertEqual(result.schedule['roles'][0]['warps'], list(range(16)))

    def test_invalid_and_noop_choices_do_not_generate_candidates(self):
        document = self.document()
        for value in (0, -1, 3, 6, True, 4.0, '8'):
            with self.subTest(value=value):
                result = self.apply(document, value)
                self.assertFalse(result.applied)
                self.assertEqual(result.reason, 'warp_count')
        result = self.apply(document, len(document['roles'][0]['warps']))
        self.assertEqual(result.reason, 'unchanged')
        for value in (64, 2**80):
            self.assertEqual(self.apply(document, value).reason, 'warp_count')

    def test_bad_input_rejected_by_backend_without_silent_repair(self):
        document = self.document()
        document['roles'][0]['warps'] = list(range(6))
        Schedule.from_dict(document)
        assessment = self.compiler.assess(document)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn('TRITON_NUM_WARPS_UNSUPPORTED', [f.code for f in assessment.findings])
        self.assertEqual(self.apply(document).reason, 'input_refused')
        self.assertEqual(document['roles'][0]['warps'], list(range(6)))

    def test_valid_explicit_resource_commitment_is_ineligible(self):
        document = self.document()
        document['residency'] = {'ctas_per_multiprocessor': 1}
        self.assertTrue(self.compiler.assess(document).lowering_eligible)
        self.assertEqual(self.apply(document).reason, 'execution_commitments')

    def test_non_triton_target_remains_outside_pass(self):
        document = self.document(backend='metal-m1-pro')
        self.assertTrue(self.compiler.assess(document).lowering_eligible)
        self.assertEqual(self.apply(document).reason, 'target_route')

    def test_invalid_candidate_identity_refused(self):
        document = self.document()
        result = self.compiler.specialize_triton_warps(document, num_warps=16,
            schedule_id=document['schedule_id'], entry_point='candidate')
        self.assertEqual(result.reason, 'result_identity')


if __name__ == '__main__':
    unittest.main()
