"""A BF16 expert projection is assembled from ordinary Cake operations."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda_row_dot import preflight


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / 'examples/schedules/native/bf16-row-dot-h16-i32.json'


def document():
    return json.loads(DOCUMENT.read_text())


class NativeBf16RowDot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.target = Target.load(ROOT / 'compiler/targets/sm_103a.json')

    def test_complete_projection_lowers_with_explicit_operations(self):
        value = document()
        assessment = self.compiler.assess(value)
        self.assertTrue(assessment.lowering_eligible,
                        [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
        lowered = self.compiler.lower(assessment)
        source = lowered.source
        self.assertIn('__bfloat162float', source)
        self.assertIn('__shfl_down_sync(0x0000ffffu', source)
        self.assertEqual(lowered.toolchain_requirements['grid'], [32, 1, 1])
        self.assertEqual(lowered.toolchain_requirements['block'], [32, 1, 1])
        self.assertEqual(set(lowered.source_map),
                         {operation['id'] for operation in value['operations']})
        self.assertNotIn('triton', source.lower())

    def test_reduction_access_and_precision_counterexamples(self):
        base = document()
        value = deepcopy(base)
        value['operations'][5]['parameters']['op'] = 'max'
        self.assertIn('NATIVE_ROW_DOT_REDUCE',
                      {f.code for f in preflight(Schedule.from_dict(value), self.target)})
        value = deepcopy(base)
        value['access_maps'][1]['indices'][0]['name'] = 'other'
        self.assertIn('NATIVE_ROW_DOT_ACCESS',
                      {f.code for f in preflight(Schedule.from_dict(value), self.target)})
        value = deepcopy(base)
        value['buffers'][1]['dtype'] = 'fp16'
        self.assertIn('NATIVE_ROW_DOT_GLOBALS',
                      {f.code for f in preflight(Schedule.from_dict(value), self.target)})
        value = deepcopy(base)
        value['buffers'][1]['shape'][1] = 8
        self.assertIn('NATIVE_ROW_DOT_SHAPE',
                      {f.code for f in preflight(Schedule.from_dict(value), self.target)})


if __name__ == '__main__':
    unittest.main()
