"""CPU admission follows separate bounded device evidence; this file does not run a HCU."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler

ROOT=Path(__file__).resolve().parents[2]


class Gfx938RegisterValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')

    def document(self,name): return json.loads((ROOT/f'corpus/schedules/{name}.json').read_text())

    def test_qualified_scan_and_broadcast_regions_lower_at_the_exact_target(self):
        for name in ['gfx938-int32-scan-129-reverse','gfx938-fp32-scan-33-forward','gfx938-register-outer-broadcast',
                     'gfx938-effective-row-validity-broadcast','gfx938-fp8-register-broadcast']:
            with self.subTest(name=name):
                assessment=self.compiler.assess(self.document(name))
                self.assertTrue(assessment.lowering_eligible,[f.to_dict() for f in assessment.findings])
                lowered=self.compiler.lower(assessment)
                self.assertEqual(lowered.target,'gfx938')
                self.assertEqual(lowered.toolchain_requirements['warp_size'],64)
        scan=self.compiler.lower(self.compiler.assess(self.document('gfx938-int32-scan-129-reverse'))).source
        self.assertIn('tl.cumsum(values.to(tl.int32)',scan)

    def test_declared_operation_does_not_override_shape_or_dtype_rules(self):
        d=self.document('gfx938-int32-scan-129-reverse')
        next(b for b in d['buffers'] if b['name']=='prefix')['dtype']='fp32'
        self.assertIn('SCAN_DTYPE_MISMATCH',{f.code for f in self.compiler.assess(d).findings})
        d=self.document('gfx938-register-outer-broadcast')
        next(op for op in d['operations'] if op['kind']=='broadcast_in_dim')['parameters']['dimensions']=[1]
        self.assertIn('VALUE_OPERATION_TYPE',{f.code for f in self.compiler.assess(d).findings})

    def test_this_target_decision_does_not_widen_other_targets(self):
        d=self.document('gfx938-register-outer-broadcast');d['target']='gfx1151'
        self.assertIn('TARGET_OPERATION_UNSUPPORTED',{f.code for f in self.compiler.assess(d).findings})
