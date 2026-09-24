"""The EP origin's weighted BF16 combine is an ordinary Cake Schedule."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'contracts/workloads/weave-ep4-bf16-moe-b300-v1.json'


class WeaveWeightedCombine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.cases = json.loads(CONTRACT.read_text())['cases']

    def document(self, tokens):
        return json.loads((ROOT / f'examples/schedules/triton/'
                           f'weave-weighted-combine-t{tokens}-h16.json').read_text())

    def test_t7_and_t8_complete_combine_lower(self):
        shapes = {case['shape']['T']: case['shape'] for case in self.cases}
        self.assertEqual(set(shapes), {7, 8})
        for tokens, shape in shapes.items():
            with self.subTest(tokens=tokens):
                document = self.document(tokens)
                schedule = Schedule.from_dict(document)
                self.assertEqual(schedule.buffer('contributions').shape,
                                 (tokens, shape['K'], shape['H']))
                self.assertEqual(schedule.buffer('weights').shape,
                                 (tokens, shape['K']))
                self.assertEqual(schedule.buffer('output').shape,
                                 (tokens, shape['H']))
                self.assertEqual([operation.kind.value for operation in schedule.operations],
                                 ['load', 'load', 'elementwise', 'reduce', 'cast', 'store'])
                assessment = self.compiler.assess(document)
                self.assertTrue(assessment.lowering_eligible,
                                [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
                lowered = self.compiler.lower(assessment)
                self.assertEqual(set(lowered.source_map),
                                 {operation.op_id for operation in schedule.operations})
                self.assertIn('tl.sum(', lowered.source)
                self.assertEqual(lowered.toolchain_requirements['grid'],
                                 [tokens, 1, 1])

    def test_wrong_route_axis_and_rounding_refuse(self):
        base = self.document(7)
        changed = deepcopy(base)
        next(operation for operation in changed['operations']
             if operation['id'] == 'reduce_routes')['parameters']['axis'] = 1
        self.assertFalse(self.compiler.assess(changed).lowering_eligible)
        changed = deepcopy(base)
        next(buffer for buffer in changed['buffers']
             if buffer['name'] == 'rounded')['dtype'] = 'fp32'
        self.assertFalse(self.compiler.assess(changed).lowering_eligible)


if __name__ == '__main__':
    unittest.main()
