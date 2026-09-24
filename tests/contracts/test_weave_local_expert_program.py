"""A complete local BF16 expert FFN remains visible as three Cake stages."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / 'examples/programs/weave-local-expert-ffn-b300.json'


class WeaveLocalExpertProgram(unittest.TestCase):
    def test_complete_local_math_lowers_without_an_opaque_expert_opcode(self):
        document = json.loads(DOCUMENT.read_text())
        program = Program.from_dict(document)
        self.assertEqual([stage.name for stage in program.stages],
                         ['up_gate', 'activation', 'down'])
        self.assertEqual(program.tensors['up_gate'].shape, (64,))
        self.assertEqual(program.tensors['activated'].shape, (32,))
        self.assertEqual(program.tensors['out'].shape, (16,))
        self.assertTrue(program.stages[1].bindings['up_gate'].singleton_view)
        self.assertTrue(program.stages[1].bindings['activated'].singleton_view)
        lowered = Compiler.load(ROOT).lower_program(program)
        self.assertEqual(len(lowered.lowerings), 3)
        self.assertTrue(all(stage.generated for stage in lowered.lowerings))
        self.assertEqual(Program.from_dict(program.document), program)

    def test_shape_views_and_fp32_activation_are_required(self):
        base = json.loads(DOCUMENT.read_text())
        value = deepcopy(base)
        value['stages'][1]['bindings']['up_gate'] = 'up_gate'
        with self.assertRaisesRegex(ValueError, 'shape/dtype differs'):
            Program.from_dict(value)
        value = deepcopy(base)
        value['stages'][2]['bindings']['x'] = 'x'
        with self.assertRaisesRegex(ValueError, 'shape/dtype differs'):
            Program.from_dict(value)


if __name__ == '__main__':
    unittest.main()
