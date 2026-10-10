"""Exact C550 pass outputs match device-qualified explicit candidates."""
from copy import deepcopy
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.toolchain import project_triton_kernel
from open_cake_ir.tasks.workloads import create_task
from tools.benchmarks.c550.width_fixtures import extension_cases, output_loop_cases
from tools.qualify_c550_widths import CASES, width_source
from tests.contracts.test_dcu_pure_width_specialization import document as pure_document

ROOT = Path(__file__).resolve().parents[2]


class C550WidthAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def apply(self, doc, width):
        return self.compiler.specialize_triton_warps(doc, num_warps=width,
            schedule_id=doc['schedule_id']+'-qualified-width',
            entry_point=doc['lowering']['entry_point'])

    def test_actual_pass_preserves_all_qualified_kernel_compile_inputs(self):
        cases = extension_cases() + output_loop_cases()
        for task, rows, columns, widths in CASES:
            _, source = create_task(task, backend='triton-metax', rows=rows, columns=columns)
            cases.append(dict(name=task, widths=widths, source=source))
        rewritten = 0
        for row in cases:
            original = frontend.parse(row['source']).document
            saved = deepcopy(original)
            for width in row['widths']:
                with self.subTest(case=row['name'], width=width):
                    result = self.apply(original, width)
                    self.assertEqual(original, saved)
                    if row['name'] == 'ieee_mma_output_loop':
                        self.assertEqual(result.reason, 'loop_domain')
                        continue
                    if width == 1:
                        self.assertEqual(result.reason, 'unchanged')
                        continue
                    if width == 16:
                        self.assertEqual(result.reason, 'result_refused')
                        self.assertIn('MACA_WARP_COUNT_UNQUALIFIED', result.message)
                        continue
                    self.assertTrue(result.applied, (result.reason, result.message))
                    direct = frontend.parse(width_source(self.compiler, row['source'], width)).document
                    candidate = result.schedule
                    candidate['schedule_id'] = direct['schedule_id']
                    self.assertEqual(candidate, direct)
                    by_pass = self.compiler.lower(result.assessment)
                    explicit = self.compiler.lower(self.compiler.assess(direct))
                    self.assertEqual(by_pass.toolchain_requirements, explicit.toolchain_requirements)
                    self.assertEqual(project_triton_kernel(by_pass.source.encode(), by_pass.toolchain_requirements),
                                     project_triton_kernel(explicit.source.encode(), explicit.toolchain_requirements))
                    rewritten += 1
        self.assertEqual(rewritten, 29)

    def test_accepted_commitments_and_loop_cast_are_refused_by_their_own_guards(self):
        for kind in ('residency', 'integer_loop_cast'):
            with self.subTest(kind=kind):
                doc = pure_document(target='xcore1002', input_dtype='int32' if kind=='integer_loop_cast' else 'fp32')
                if kind == 'residency':
                    doc['residency'] = {'ctas_per_multiprocessor': 1}
                self.assertTrue(self.compiler.assess(doc).lowering_eligible)
                result = self.apply(doc, 2)
                self.assertEqual(result.reason, 'execution_commitments' if kind=='residency' else 'loop_domain')

    def test_width16_source_cannot_be_silently_repaired_by_width_pass(self):
        doc = pure_document(target='xcore1002', loop=False)
        doc['roles'][0]['execution_groups'] = list(range(16))
        self.assertFalse(self.compiler.assess(doc).lowering_eligible)
        result = self.apply(doc, 4)
        self.assertEqual(result.reason, 'input_refused')
        self.assertEqual(doc['roles'][0]['execution_groups'], list(range(16)))


if __name__ == '__main__':
    unittest.main()
