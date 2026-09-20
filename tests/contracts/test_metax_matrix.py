"""Measured complete MACA dot contracts do not admit NVIDIA-only refinements."""
import copy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.backends import triton
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import declared_target

ROOT = Path(__file__).resolve().parents[2]


class MetaxMatrixAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def test_each_measured_mode_lowers_at_the_targets_own_width(self):
        for dtype in ('fp16', 'bf16', 'fp32'):
            with self.subTest(dtype=dtype):
                result = self.compiler.assess_file(ROOT / f'corpus/schedules/xcore1002-dot-{dtype}-64.py')
                self.assertTrue(result.accepted)
                self.assertTrue(result.lowering_eligible)
                lowering = self.compiler.lower(result)
                self.assertEqual(lowering.toolchain_requirements['code_object'], 'mcfatbin')
                self.assertEqual(lowering.toolchain_requirements['warp_size'], 64)
                self.assertEqual(lowering.toolchain_requirements['compile_options']['num_warps'], 4)
                self.assertIn('tl.dot(', lowering.source)
                self.assertNotIn('inline_asm_elementwise', lowering.source)

    def test_failed_precision_hypothesis_and_mismatched_operand_dtype_are_owned_refusals(self):
        for name, code in (('tf32-unqualified', 'TARGET_INSTRUCTION_UNSUPPORTED'),
                           ('operand-dtype-drift', 'MMA_OPERAND_DTYPE_DIFFERS')):
            with self.subTest(name=name):
                result = self.compiler.assess_file(ROOT / f'corpus/schedules/xcore1002-dot-{name}-64.py')
                self.assertFalse(result.lowering_eligible)
                self.assertIn(code, [f.code for f in result.findings])

    def test_partial_k_cannot_reach_the_nvidia_inline_assembly_route(self):
        document = json.loads((ROOT / 'examples/schedules/triton/kmeans-partials.json').read_text())
        original = copy.deepcopy(document)
        document['target'] = 'xcore1002'
        schedule = Schedule.from_dict(document)
        target = declared_target('xcore1002')
        findings = triton.preflight(schedule, target)
        self.assertIn('TRITON_MMA_K_RANGES_UNSUPPORTED', [f.code for f in findings])
        with self.assertRaises(EmitError):
            triton.emit(schedule, target)
        # This negative is the otherwise supported original refinement, with only
        # its target changed; an unrelated dtype/interval failure is not the guard.
        self.assertFalse(any(f.blocks_lowering for f in triton.preflight(
            Schedule.from_dict(original), declared_target(original['target']))))
