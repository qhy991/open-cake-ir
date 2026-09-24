"""Native worker lowering keeps leaf mathematics and scheduling effects distinct."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.program import LoweredWorkerProgram


ROOT = Path(__file__).resolve().parents[2]
LEAF = ROOT / 'examples/schedules/native/fma-row-b512.json'


def document() -> dict:
    leaf = json.loads(LEAF.read_text())
    shape = [512, 16]
    tensors = {name: {'shape': shape, 'dtype': 'fp32'}
               for name in ('a', 'b', 'c', 'middle0', 'middle1', 'out')}
    tensors.update({name: {'shape': [1], 'dtype': 'int32'}
                    for name in ('first_ctas', 'chunks', 'steal_budget')})
    stages = [{'name': name, 'schedule': deepcopy(leaf),
               'bindings': {'a': source, 'b': 'b', 'c': 'c', 'y': destination}}
              for name, source, destination in (
                  ('send', 'a', 'middle0'), ('calculate', 'middle0', 'middle1'),
                  ('finish', 'middle1', 'out'))]
    return {
        'schema_version': 2, 'program_id': 'native-fma-worker-b512',
        'target': 'sm_103a', 'tensors': tensors,
        'inputs': ['a', 'b', 'c', 'first_ctas', 'chunks', 'steal_budget'],
        'outputs': ['out'], 'stages': stages,
        'execution': {
            'kind': 'cooperative_workers',
            'lowering': {'backend': 'native_cuda', 'entry_point': 'cake_fma_workers'},
            'controls': {'first_class_ctas': 'first_ctas', 'chunk_count': 'chunks',
                         'steal_budget': 'steal_budget'},
            'workers': [
                {'name': 'first', 'phases': ['send', 'finish']},
                {'name': 'second', 'phases': ['calculate', 'finish']},
            ],
            'queues': [
                {'stage': 'send', 'workers': ['first']},
                {'stage': 'calculate', 'workers': ['first', 'second']},
                {'stage': 'finish', 'workers': ['first', 'second']},
            ],
            'handoffs': [
                {'payload': 'middle0', 'producer': 'send', 'consumer': 'calculate',
                 'order': 'release_acquire', 'scope': 'device'},
                {'payload': 'middle1', 'producer': 'calculate', 'consumer': 'finish',
                 'order': 'release_acquire', 'scope': 'device'},
            ],
            'steal': {'borrower': 'first', 'stage': 'calculate',
                      'after': 'send', 'before': 'finish'},
        },
    }


class NativeWorkerProgram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def test_complete_fma_program_emits_one_cooperative_kernel(self):
        lowered = self.compiler.lower_program(Program.from_dict(document()))
        self.assertIsInstance(lowered, LoweredWorkerProgram)
        lowered.validate_binding()
        self.assertEqual(lowered.toolchain_requirements['grid'], [148, 1, 1])
        self.assertEqual(lowered.toolchain_requirements['state_bytes'], 6176)
        self.assertEqual(lowered.toolchain_requirements['argument_order'],
                         list(document()['tensors']))
        source = lowered.source
        self.assertEqual(source.count('{'), source.count('}'))
        self.assertEqual(source.count('fma.rn.f32 %0, %1, %2, %3;'), 3)
        self.assertIn('atom.relaxed.gpu.global.add.s32', source)
        self.assertIn('st.release.gpu.global.s32', source)
        self.assertIn('ld.acquire.gpu.global.s32', source)
        self.assertIn('cudaMemsetAsync(h->state, 0, 6176', source)
        self.assertIn('cudaLaunchCooperativeKernel', source)
        self.assertIn('// CAKE_OP: send.fma', source)
        self.assertIn('// CAKE_OP: calculate.fma', source)
        self.assertIn('// CAKE_OP: finish.fma', source)
        self.assertNotIn('triton', source.lower())

    def test_refuses_unproved_routes_and_handoffs(self):
        value = document()
        value['stages'][1]['schedule']['lowering']['backend'] = 'triton'
        with self.assertRaisesRegex(ValueError, 'stage .*native_cuda leaf route'):
            self.compiler.lower_program(Program.from_dict(value))
        value = document()
        value['execution']['handoffs'][0]['scope'] = 'system'
        with self.assertRaisesRegex(ValueError, 'system-scope handoff'):
            self.compiler.lower_program(Program.from_dict(value))
        value = document()
        value['stages'][1]['schedule']['program_map']['cooperative'] = False
        with self.assertRaisesRegex(ValueError, 'cooperative one-CTA-per-SM'):
            self.compiler.lower_program(Program.from_dict(value))
        value = document()
        value['execution']['lowering']['entry_point'] = 'bad();'
        with self.assertRaisesRegex(ValueError, 'entry_point must be an identifier'):
            self.compiler.lower_program(Program.from_dict(value))


if __name__ == '__main__':
    unittest.main()
