"""The local BF16 expert FFN composes native CUDA leaves with visible math."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda_row_dot import preflight as row_dot_preflight


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / 'examples/programs/weave-local-expert-ffn-native-b300.json'


class NativeLocalExpertProgram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.target = Target.load(ROOT / 'compiler/targets/sm_103a.json')

    def test_three_complete_native_stages_compose(self):
        document = json.loads(DOCUMENT.read_text())
        program = Program.from_dict(document)
        self.assertEqual([stage.name for stage in program.stages],
                         ['up_gate', 'activation', 'down'])
        self.assertEqual(program.tensors['up_gate'].shape, (64,))
        self.assertEqual(program.tensors['activated'].shape, (32,))
        self.assertTrue(program.stages[1].bindings['up_gate'].singleton_view)
        self.assertTrue(program.stages[1].bindings['activated'].singleton_view)
        lowered = self.compiler.lower_program(program)
        self.assertEqual(len(lowered.lowerings), 3)
        self.assertTrue(all(stage.generated for stage in lowered.lowerings))
        self.assertTrue(all('triton' not in stage.source.lower()
                            for stage in lowered.lowerings))
        self.assertEqual(Program.from_dict(program.document), program)

    def test_native_leaves_keep_matched_triton_math(self):
        names = ('bf16-expert-up-gate-h16-i32', 'bf16-gated-activation-i32',
                 'bf16-expert-down-i32-h16')
        for name in names:
            with self.subTest(stage=name):
                native = json.loads((ROOT / f'examples/schedules/native/{name}.json').read_text())
                triton = json.loads((ROOT / f'examples/schedules/triton/{name}.json').read_text())
                for document in (native, triton):
                    document.pop('schedule_id')
                    document.pop('lowering')
                self.assertEqual(native, triton)
                assessment = self.compiler.assess(json.loads(
                    (ROOT / f'examples/schedules/triton/{name}.json').read_text()))
                self.assertTrue(assessment.lowering_eligible,
                                [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])

    def test_down_projection_needs_fp32_activation_and_exact_index(self):
        path = ROOT / 'examples/schedules/native/bf16-expert-down-i32-h16.json'
        base = json.loads(path.read_text())
        value = deepcopy(base)
        next(buffer for buffer in value['buffers'] if buffer['name'] == 'x')['dtype'] = 'bf16'
        self.assertIn('NATIVE_ROW_DOT_GLOBALS',
                      {finding.code for finding in row_dot_preflight(
                          Schedule.from_dict(value), self.target)})
        value = deepcopy(base)
        next(access for access in value['access_maps']
             if access['operation'] == 'load_w')['indices'][0]['name'] = 'wrong'
        self.assertIn('NATIVE_ROW_DOT_ACCESS',
                      {finding.code for finding in row_dot_preflight(
                          Schedule.from_dict(value), self.target)})


if __name__ == '__main__':
    unittest.main()
