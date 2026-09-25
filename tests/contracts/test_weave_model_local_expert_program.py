"""Model-width expert FFN stages meet at explicit Cake tensor dtypes."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / 'examples/programs/weave-model-local-expert-ffn-native-b300.json'


class WeaveModelLocalExpertProgram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def test_three_complete_native_stages_bind_without_implicit_cast(self):
        program = Program.from_dict(json.loads(DOCUMENT.read_text()))
        self.assertEqual([stage.name for stage in program.stages],
                         ['up_gate', 'activation', 'down'])
        self.assertEqual(program.tensors['up_gate'].shape, (128, 1536))
        self.assertEqual(program.tensors['activated'].shape, (128, 768))
        self.assertEqual(program.tensors['activated'].dtype.value, 'bf16')
        self.assertFalse(any(binding.singleton_view for stage in program.stages
                             for binding in stage.bindings.values()))
        lowered = self.compiler.lower_program(program)
        self.assertEqual(len(lowered.lowerings), 3)
        self.assertTrue(all(stage.generated for stage in lowered.lowerings))
        self.assertTrue(all('triton' not in stage.source.lower()
                            for stage in lowered.lowerings))
        self.assertEqual(Program.from_dict(program.document), program)

    def test_down_input_cannot_silently_reinterpret_bf16_as_fp32(self):
        changed = deepcopy(json.loads(DOCUMENT.read_text()))
        down = changed['stages'][2]['schedule']
        next(row for row in down['buffers'] if row['name'] == 'a')['dtype'] = 'fp32'
        with self.assertRaisesRegex(ValueError, 'down.*shape/dtype'):
            Program.from_dict(changed)


if __name__ == '__main__':
    unittest.main()
