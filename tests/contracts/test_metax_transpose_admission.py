"""Exact C550 admission reuses register-transpose typing and coordinate semantics."""
from pathlib import Path
import struct
import unittest

from open_cake_ir.compiler import Compiler, Target, frontend
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.performance.work import work_bound
from tests.contracts.test_register_transpose import source
from tests.contracts.test_triton_loop_scopes import _execute

ROOT = Path(__file__).resolve().parents[2]
SHAPES = ((16, 32), (35, 67), (1, 3))
DTYPES = ('fp32', 'fp16', 'bf16', 'int32', 'int64', 'bool')


def document(rows, columns, dtype):
    text = source(rows, columns, dtype).replace('target="gfx938"', 'target="xcore1002"')
    text = text.replace('execution_groups=[0]', 'execution_groups=[0, 1, 2, 3]')
    return frontend.parse(text).document


def values_for(dtype, count):
    if dtype == 'bool':
        return [[bool((i >> bit) & 1) for i in range(count)]
                for bit in range(max(1, (count - 1).bit_length()))]
    if dtype == 'int32':
        return [[2**24 + i for i in range(count)]]
    if dtype == 'int64':
        return [[2**54 + i for i in range(count)]]
    if dtype == 'fp16':
        return [[struct.unpack('<e', struct.pack('<H', 0x3000 + i))[0] for i in range(count)]]
    prefix, shift = (0x3000, 16) if dtype == 'bf16' else (0x3f000000, 0)
    return [[struct.unpack('<f', struct.pack('<I', (prefix + i) << shift))[0] for i in range(count)]]


class MetaxTransposeAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.target = Target.load(ROOT / 'compiler/targets/xcore1002.json')

    def test_real_target_emission_preserves_each_coordinate_dtype_and_single_store(self):
        for dtype in DTYPES:
            for rows, columns in SHAPES:
                with self.subTest(dtype=dtype, shape=(rows, columns)):
                    raw = document(rows, columns, dtype)
                    assessment = self.compiler.assess(raw)
                    self.assertTrue(assessment.lowering_eligible, assessment.findings)
                    lowering = self.compiler.lower(assessment)
                    self.assertIn('t = tl.trans(v)', lowering.source)
                    self.assertNotIn('t = tl.trans(v.to', lowering.source)
                    self.assertEqual(work_bound(assessment.typed_schedule).flops, 0)
                    schedule = Schedule.from_dict(raw)
                    for values in values_for(dtype, rows * columns):
                        memory = {'x': values[:], 'out': [None] * len(values)}
                        trace = _execute(emit(schedule, self.target), memory)
                        self.assertEqual(memory['out'], [values[r * columns + c]
                            for c in range(columns) for r in range(rows)])
                        self.assertEqual(memory['x'], values)
                        self.assertEqual(len(trace.stores), rows * columns)
                        self.assertEqual(set(trace.stores.values()), {1})

    def test_existing_value_typing_owns_invalid_shape_dtype_rank_storage_and_arity(self):
        for mutation in ('shape', 'dtype', 'rank', 'storage', 'arity'):
            raw = document(35, 67, 'fp32')
            buffers = {b['name']: b for b in raw['buffers']}
            op = next(o for o in raw['operations'] if o['id'] == 'transpose')
            if mutation == 'shape': buffers['t']['shape'] = [16, 32]
            elif mutation == 'dtype': buffers['t']['dtype'] = 'bf16'
            elif mutation == 'rank': buffers['v']['shape'] = [512]
            elif mutation == 'storage': buffers['t']['space'] = 'global'
            else: op['reads'].append('v')
            assessment = self.compiler.assess(raw)
            codes = {f.code for f in assessment.findings if f.blocks_lowering}
            with self.subTest(mutation=mutation):
                self.assertFalse(assessment.lowering_eligible)
                self.assertIn('VALUE_OPERATION_TYPE', codes)
                self.assertNotIn('TARGET_OPERATION_UNSUPPORTED', codes)

    def test_fp8_keeps_both_its_type_and_maca_refusal(self):
        for rows, columns in SHAPES:
            raw = document(rows, columns, 'fp8_e4m3')
            assessment = self.compiler.assess(raw)
            codes = {f.code for f in assessment.findings if f.blocks_lowering}
            self.assertFalse(assessment.lowering_eligible)
            self.assertIn('VALUE_OPERATION_TYPE', codes)
            self.assertNotIn('TARGET_OPERATION_UNSUPPORTED', codes)
            # Invalid value typing stops Compiler.assess before backend preflight.
            # Exercise the existing backend refusal separately, at its own boundary.
            backend_codes = {f.code for f in preflight(Schedule.from_dict(raw), self.target)
                             if f.blocks_lowering}
            self.assertIn('MACA_FP8_OPERATION_UNQUALIFIED', backend_codes)

    def test_admission_does_not_open_another_target(self):
        raw = document(16, 32, 'fp32')
        raw['target'] = 'gfx1151'
        assessment = self.compiler.assess(raw)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn('TARGET_OPERATION_UNSUPPORTED', {f.code for f in assessment.findings})


if __name__ == '__main__':
    unittest.main()
