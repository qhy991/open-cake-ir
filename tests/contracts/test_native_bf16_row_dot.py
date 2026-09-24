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
TRITON_DOCUMENT = ROOT / 'examples/schedules/triton/bf16-row-dot-h16-i32.json'
SELECTED_DOCUMENT = ROOT / 'examples/schedules/native/bf16-selected-expert-row-dot-h16-i32.json'
SELECTED_TRITON_DOCUMENT = ROOT / 'examples/schedules/triton/bf16-selected-expert-row-dot-h16-i32.json'


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
        value = deepcopy(base)
        value['operations'][4].pop('depends_on')
        self.assertIn('NATIVE_ROW_DOT_DEPENDENCIES',
                      {f.code for f in preflight(Schedule.from_dict(value), self.target)})
        value = deepcopy(base)
        value['buffers'][0]['byte_offset'] = 4
        self.assertIn('NATIVE_ROW_DOT_REFINEMENT',
                      {f.code for f in preflight(Schedule.from_dict(value), self.target)})

    def test_triton_comparator_keeps_identical_math_and_access(self):
        native = document()
        triton = json.loads(TRITON_DOCUMENT.read_text())
        for value in (native, triton):
            value.pop('schedule_id')
            value.pop('lowering')
        self.assertEqual(native, triton)
        assessment = self.compiler.assess(json.loads(TRITON_DOCUMENT.read_text()))
        self.assertTrue(assessment.lowering_eligible,
                        [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
        lowered = self.compiler.lower(assessment)
        self.assertIn('tl.sum(products.to(tl.float32), axis=0)', lowered.source)
        self.assertEqual(lowered.toolchain_requirements['grid'], [32, 1, 1])
        self.assertEqual(set(lowered.source_map),
                         {operation['id'] for operation in triton['operations']})

    def test_selected_expert_is_a_masked_runtime_index(self):
        value = json.loads(SELECTED_DOCUMENT.read_text())
        assessment = self.compiler.assess(value)
        self.assertTrue(assessment.lowering_eligible,
                        [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
        lowered = self.compiler.lower(assessment)
        self.assertIn('>= 0 &&', lowered.source)
        self.assertIn('< 2) ? ', lowered.source)
        self.assertIn(': __float2bfloat16(0.0f)', lowered.source)
        self.assertIn('bfloat162float', lowered.source)
        self.assertEqual(lowered.toolchain_requirements['argument_order'],
                         ['x', 'weight', 'expert_id', 'y'])
        self.assertEqual(set(lowered.source_map),
                         {operation['id'] for operation in value['operations']})

    def test_selected_expert_pair_and_invalid_index_contract(self):
        native = json.loads(SELECTED_DOCUMENT.read_text())
        triton = json.loads(SELECTED_TRITON_DOCUMENT.read_text())
        for value in (native, triton):
            value.pop('schedule_id')
            value.pop('lowering')
        self.assertEqual(native, triton)
        assessment = self.compiler.assess(json.loads(SELECTED_TRITON_DOCUMENT.read_text()))
        self.assertTrue(assessment.lowering_eligible,
                        [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
        self.assertIn('other=0.0', self.compiler.lower(assessment).source)

        base = json.loads(SELECTED_DOCUMENT.read_text())
        value = deepcopy(base)
        next(access for access in value['access_maps']
             if access['operation'] == 'load_w')['indices'][0]['name'] = 'wrong'
        self.assertIn('NATIVE_ROW_DOT_ACCESS',
                      {f.code for f in preflight(Schedule.from_dict(value), self.target)})
        value = deepcopy(base)
        next(buffer for buffer in value['buffers']
             if buffer['name'] == 'expert_id')['dtype'] = 'fp32'
        self.assertIn('NATIVE_ROW_DOT_EXPERT_INPUT',
                      {f.code for f in preflight(Schedule.from_dict(value), self.target)})
        value = deepcopy(base)
        next(operation for operation in value['operations']
             if operation['id'] == 'load_w').pop('depends_on')
        self.assertIn('NATIVE_ROW_DOT_DEPENDENCIES',
                      {f.code for f in preflight(Schedule.from_dict(value), self.target)})


if __name__ == '__main__':
    unittest.main()
