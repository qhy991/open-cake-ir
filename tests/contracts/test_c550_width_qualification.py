"""The bounded preparation changes exactly one hardware choice, not the oracle."""
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.tasks.workloads import create_task
from tools.qualify_c550_widths import CASES, width_source

ROOT = Path(__file__).resolve().parents[2]


class C550WidthPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def test_all_eight_candidates_keep_all_other_schedule_fields(self):
        count = 0
        for task, batch, hidden, widths in CASES:
            _, source = create_task(task, backend='triton-metax', rows=batch, columns=hidden)
            original = frontend.parse(source).document
            for width in widths:
                with self.subTest(task=task, width=width):
                    candidate = frontend.parse(width_source(self.compiler, source, width)).document
                    for field in original.keys() - {'roles'}:
                        self.assertEqual(original[field], candidate[field], field)
                    expected = dict(original['roles'][0], execution_groups=list(range(width)))
                    self.assertEqual(candidate['roles'], [expected])
                    self.assertTrue(self.compiler.assess(candidate).lowering_eligible)
                    count += 1
        self.assertEqual(count, 8)

    def test_qualified_action_and_invalid_widths(self):
        _, source = create_task('fib_rmsnorm_h4096', backend='triton-metax', rows=64, columns=4096)
        document = frontend.parse(source).document
        result = self.compiler.specialize_triton_warps(document, num_warps=4,
            schedule_id='qualification_not_promotion', entry_point='candidate')
        self.assertTrue(result.applied, (result.reason, result.message))
        for bad in (True, 0, 3, 32):
            with self.subTest(width=bad), self.assertRaises(ValueError):
                width_source(self.compiler, source, bad)

    def test_another_target_cannot_supply_qualification(self):
        _, source = create_task('rmsnorm', backend='triton-dcu', rows=64, columns=4096)
        with self.assertRaisesRegex(ValueError, 'exact C550'):
            width_source(self.compiler, source, 4)


if __name__ == '__main__':
    unittest.main()
