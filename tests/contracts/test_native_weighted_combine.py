"""The origin's weighted FP32-to-BF16 combine is visible in native CUDA."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda_combine import preflight


ROOT = Path(__file__).resolve().parents[2]


def document(tokens, route):
    return json.loads((ROOT / f'examples/schedules/{route}/'
                       f'weave-weighted-combine-t{tokens}-h16.json').read_text())


class NativeWeightedCombine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.target = Target.load(ROOT / 'compiler/targets/sm_103a.json')

    def test_matched_native_and_triton_combine_lower_for_tail_and_full(self):
        for tokens in (7, 8):
            with self.subTest(tokens=tokens):
                native = document(tokens, 'native')
                triton = document(tokens, 'triton')
                for value in (native, triton):
                    value.pop('schedule_id')
                    value.pop('lowering')
                self.assertEqual(native, triton)
                for route in ('native', 'triton'):
                    value = document(tokens, route)
                    assessment = self.compiler.assess(value)
                    self.assertTrue(assessment.lowering_eligible,
                                    [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
                    lowered = self.compiler.lower(assessment)
                    self.assertEqual(lowered.toolchain_requirements['grid'],
                                     [tokens, 1, 1])
                    self.assertEqual(set(lowered.source_map),
                                     {op['id'] for op in value['operations']})
                    if route == 'native':
                        self.assertIn('__float2bfloat16_rn', lowered.source)
                        self.assertIn('cake_token * 32 + route * 16 + cake_lane',
                                      lowered.source)
                        self.assertNotIn('triton', lowered.source.lower())
                    else:
                        self.assertIn('tl.sum(', lowered.source)

    def test_wrong_axis_broadcast_rounding_and_address_refuse(self):
        base = document(7, 'native')
        for operation_id, field, value, code in (
                ('weight', 'broadcast_axis', 1, 'NATIVE_COMBINE_MULTIPLY'),
                ('reduce_routes', 'axis', 1, 'NATIVE_COMBINE_REDUCE'),
                ('round_bf16', 'to', 'fp32', 'NATIVE_COMBINE_CAST')):
            changed = deepcopy(base)
            next(op for op in changed['operations']
                 if op['id'] == operation_id)['parameters'][field] = value
            self.assertIn(code, {finding.code for finding in preflight(
                Schedule.from_dict(changed), self.target)})
        changed = deepcopy(base)
        next(access for access in changed['access_maps']
             if access['operation'] == 'load_contributions')['indices'][2]['dimension'] = 1
        self.assertIn('NATIVE_COMBINE_ACCESS',
                      {finding.code for finding in preflight(
                          Schedule.from_dict(changed), self.target)})


if __name__ == '__main__':
    unittest.main()
