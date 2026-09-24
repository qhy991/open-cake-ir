"""The local expert's FP32 gate remains an explicit Cake arithmetic graph."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda_activation import preflight


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / 'examples/schedules/native/bf16-gated-activation-i32.json'
TRITON = ROOT / 'examples/schedules/triton/bf16-gated-activation-i32.json'


class NativeGatedActivation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.target = Target.load(ROOT / 'compiler/targets/sm_103a.json')

    def test_matched_native_and_triton_math_lower(self):
        native = json.loads(NATIVE.read_text())
        triton = json.loads(TRITON.read_text())
        for document in (native, triton):
            document.pop('schedule_id')
            document.pop('lowering')
        self.assertEqual(native, triton)
        for path, expected in ((NATIVE, 'expf('), (TRITON, 'tl.exp(')):
            document = json.loads(path.read_text())
            assessment = self.compiler.assess(document)
            self.assertTrue(assessment.lowering_eligible,
                            [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
            lowered = self.compiler.lower(assessment)
            self.assertIn(expected, lowered.source)
            self.assertEqual(lowered.toolchain_requirements['grid'], [1, 1, 1])
            self.assertEqual(set(lowered.source_map),
                             {operation['id'] for operation in document['operations']})
            if path == NATIVE:
                self.assertIn('32 + cake_lane', lowered.source)

    def test_wrong_subrange_arithmetic_and_dependency_refuse(self):
        base = json.loads(NATIVE.read_text())
        value = deepcopy(base)
        next(access for access in value['access_maps']
             if access['operation'] == 'load_gate')['indices'][1]['offset'] = 31
        self.assertIn('NATIVE_ACTIVATION_ACCESS',
                      {finding.code for finding in preflight(Schedule.from_dict(value), self.target)})
        value = deepcopy(base)
        next(operation for operation in value['operations']
             if operation['id'] == 'exp_gate')['parameters']['op'] = 'rsqrt'
        self.assertIn('NATIVE_ACTIVATION_ARITHMETIC',
                      {finding.code for finding in preflight(Schedule.from_dict(value), self.target)})
        value = deepcopy(base)
        next(operation for operation in value['operations']
             if operation['id'] == 'multiply').pop('depends_on')
        self.assertIn('NATIVE_ACTIVATION_DEPENDENCIES',
                      {finding.code for finding in preflight(Schedule.from_dict(value), self.target)})


if __name__ == '__main__':
    unittest.main()
