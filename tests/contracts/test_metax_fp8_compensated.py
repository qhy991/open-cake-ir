"""The C550 FP8 matrix route is a bounded SIMT program, not native FP8 dot."""
import copy
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends import triton
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import declared_target
from open_cake_ir.compiler.toolchain import project_triton_kernel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'examples/python/xcore1002_fp8_compensated.py'


class MetaxCompensatedFP8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        cls.document = frontend.read_schedule(SOURCE).document

    def test_exact_tile_lowers_to_compensated_simt_without_native_dot(self):
        assessment = self.compiler.assess(self.document)
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        lowering = self.compiler.lower(assessment)
        self.assertEqual(lowering.toolchain_requirements['code_object'], 'mcfatbin')
        self.assertEqual(lowering.toolchain_requirements['compile_options']['num_warps'], 4)
        self.assertIn('_maca_fp8_correction = _maca_fp8_correction + _maca_fp8_error', lowering.source)
        self.assertNotIn('tl.dot(', lowering.source)
        self.assertNotIn('inline_asm_elementwise', lowering.source)
        kernel = project_triton_kernel(lowering.source.encode(), lowering.toolchain_requirements)
        self.assertIn(b'for _maca_fp8_k in tl.range(0, 64):', kernel)

    def test_unmeasured_domain_is_refused_by_maca(self):
        for mutation in ('tile_shape', 'warps', 'global_shape', 'program_tile', 'persistent'):
            with self.subTest(mutation=mutation):
                document = copy.deepcopy(self.document)
                if mutation == 'tile_shape':
                    next(op for op in document['operations'] if op['kind'] == 'mma')['parameters']['tile_shape'] = [4, 64, 64]
                elif mutation == 'warps':
                    document['roles'][0]['execution_groups'] = list(range(8))
                elif mutation == 'global_shape':
                    next(buffer for buffer in document['buffers'] if buffer['name'] == 'out')['shape'] = [128, 64]
                elif mutation == 'program_tile':
                    document['program_map']['axes'][0]['tile'] = 4
                else:
                    document['program_map']['persistent'] = True
                findings = triton.preflight(Schedule.from_dict(document), declared_target('xcore1002'))
                self.assertIn('MACA_FP8_COMPENSATED_DOMAIN_UNQUALIFIED', [f.code for f in findings])

    def test_direct_fp8_dot_remains_unadmitted(self):
        document = copy.deepcopy(self.document)
        next(op for op in document['operations'] if op['kind'] == 'mma')['parameters']['instruction']['contract'] = 'triton.dot.fp8e4m3_fp32'
        assessment = self.compiler.assess(document)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn('TARGET_INSTRUCTION_UNSUPPORTED', [f.code for f in assessment.findings])
