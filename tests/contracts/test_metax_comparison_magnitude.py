"""Only comparison projections may lose an intermediate magnitude's sign bits."""
import struct
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends.metax import comparison_magnitude_input
from open_cake_ir.compiler.ir import OperationKind, Schedule
from open_cake_ir.tasks.metax_fp8_gemm import bucketed_source
from open_cake_ir.tasks.workloads import load_workload

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name='compare-magnitude', target='xcore1002', backend='triton', entry_point='compare_magnitude')
def kernel(lm, x: cake.Tensor((8,128), 'fp32'), out: cake.Tensor((8,128), 'int32', mode='output')):
    compute = lm.role(execution_groups=[0,1,2,3])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[row,:])
        sign = lm.compare(values, 0.0, op='ge')
        negative = lm.mul(values, -1.0)
        magnitude = lm.select(sign, values, negative)
        answer = lm.compare(magnitude, 0.0, op='gt')
        lm.store(out[row,:], answer)
'''


class MetaxComparisonMagnitude(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def lower(self, source):
        assessment = self.compiler.assess(frontend.parse(source).document)
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        return assessment, self.compiler.lower(assessment).source

    def test_numeric_comparison_projection_uses_abs_on_maca(self):
        for dtype in ('fp32', 'fp16'):
            with self.subTest(dtype=dtype):
                _, emitted = self.lower(SOURCE.replace("'fp32'", repr(dtype)))
                self.assertIn(f'magnitude = tl.abs(values).to(tl.float{32 if dtype == "fp32" else 16})', emitted)

    def test_qualified_recipe_recognizes_all_compare_only_magnitudes(self):
        workload = load_workload(ROOT / 'contracts/workloads/metax-fp8-e4m3-gemm-fp32-xcore1002-m64-n64-k64-v2.json')
        assessment, emitted = self.lower(bucketed_source(workload))
        schedule = Schedule.from_dict(frontend.parse(bucketed_source(workload)).document)
        selected = [op for op in schedule.operations if op.kind is OperationKind.SELECT]
        self.assertEqual(sum(comparison_magnitude_input(schedule, op) is not None for op in selected), 18)
        self.assertEqual(emitted.count(' = tl.abs('), 18)

    def test_stored_zero_and_arithmetic_consumers_keep_the_original_select(self):
        stored = SOURCE.replace("'int32', mode='output'", "'fp32', mode='output'")
        stored = stored.replace("        answer = lm.compare(magnitude, 0.0, op='gt')\n", '')
        stored = stored.replace('out[row,:], answer', 'out[row,:], magnitude')
        arithmetic = SOURCE.replace('        answer = lm.compare(magnitude, 0.0, op=\'gt\')',
            "        shifted = lm.add(magnitude, 1.0)\n        answer = lm.compare(shifted, 0.0, op='gt')")
        for source in (stored, arithmetic):
            _, emitted = self.lower(source)
            self.assertIn('magnitude = tl.where(', emitted)
            self.assertNotIn('magnitude = tl.abs(', emitted)
        value = -0.0
        selected = value if value >= 0 else -value
        self.assertEqual(struct.pack('<f', selected), bytes.fromhex('00000080'))
        self.assertEqual(struct.pack('<f', abs(value)), bytes(4))

    def test_nonmatching_legal_patterns_and_other_vendor_are_not_folded(self):
        cases = [SOURCE.replace("values, 0.0, op='ge'", "values, 1.0, op='ge'"),
                 SOURCE.replace('values, -1.0', 'values, -2.0'),
                 SOURCE.replace("'fp32'", "'bf16'"),
                 SOURCE.replace("target='xcore1002'", "target='sm_103a'")]
        for source in cases:
            with self.subTest(source=source):
                _, emitted = self.lower(source)
                self.assertIn('magnitude = tl.where(', emitted)
                self.assertNotIn('magnitude = tl.abs(', emitted)
