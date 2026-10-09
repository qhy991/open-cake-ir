"""C550 FMA uses its own admitted instruction and no PTX source escape."""
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.backends import triton
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import declared_target
from open_cake_ir.compiler.toolchain import project_triton_kernel

ROOT = Path(__file__).resolve().parents[2]


def fma_document():
    document = json.loads((ROOT / 'corpus/schedules/fma-b8-smoke.json').read_text())
    document['target'] = 'xcore1002'
    document.pop('residency')  # MACA has no verified maxnreg option.
    next(op for op in document['operations'] if op['id'] == 'fma')['parameters']['instruction']['contract'] = 'maca.fma.f32'
    return document


class MetaxFmaLowering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def test_metax_contract_emits_admitted_fused_operation(self):
        assessment = self.compiler.assess(fma_document())
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        lowering = self.compiler.lower(assessment)
        self.assertIn('tl.fma(a_tile, b_tile, c_tile)', lowering.source)
        self.assertNotIn('inline_asm_elementwise', lowering.source)
        self.assertEqual(lowering.toolchain_requirements['code_object'], 'mcfatbin')
        kernel = project_triton_kernel(lowering.source.encode(), lowering.toolchain_requirements)
        self.assertIn(b'tl.fma(a_tile, b_tile, c_tile)', kernel)

    def test_directed_contracts_keep_rounding_visible_through_source_admission(self):
        for mode in ('rz', 'rd', 'ru'):
            document = fma_document()
            instruction = next(op for op in document['operations'] if op['id'] == 'fma')['parameters']['instruction']
            instruction['contract'] = f'maca.fma.{mode}.f32'
            assessment = self.compiler.assess(document)
            self.assertTrue(assessment.lowering_eligible, assessment.findings)
            lowered = self.compiler.lower(assessment)
            projected = project_triton_kernel(lowered.source.encode(), lowered.toolchain_requirements)
            self.assertIn(f'libdevice.fma_{mode}(a_tile, b_tile, c_tile)'.encode(), projected)
            self.assertNotIn(b'tl.fma(a_tile, b_tile, c_tile)', projected)

    def test_directed_contract_typing_refuses_fp16_operands(self):
        document = fma_document()
        for buffer in document['buffers']:
            if buffer['dtype'] == 'fp32':
                buffer['dtype'] = 'fp16'
        instruction = next(op for op in document['operations'] if op['id'] == 'fma')['parameters']['instruction']
        instruction['contract'] = 'maca.fma.rz.f32'
        assessment = self.compiler.assess(document)
        self.assertFalse(assessment.accepted)
        self.assertIn('ELEMENTWISE_INSTRUCTION_DTYPE_DIFFERS', [f.code for f in assessment.findings])

    def test_vendor_instruction_names_do_not_cross_routes(self):
        document = fma_document()
        op = next(op for op in document['operations'] if op['id'] == 'fma')
        op['parameters']['instruction']['contract'] = 'ptx.fma.rn.f32'
        metax = self.compiler.assess(document)
        self.assertFalse(metax.lowering_eligible)
        self.assertIn('TARGET_INSTRUCTION_UNSUPPORTED', [f.code for f in metax.findings])
        with self.assertRaises(EmitError):
            triton.emit(Schedule.from_dict(document), declared_target('xcore1002'))

        document['target'] = 'sm_100a'
        op['parameters']['instruction']['contract'] = 'maca.fma.f32'
        nvidia = self.compiler.assess(document)
        self.assertFalse(nvidia.lowering_eligible)
        self.assertIn('TARGET_INSTRUCTION_UNSUPPORTED', [f.code for f in nvidia.findings])
        with self.assertRaises(EmitError):
            triton.emit(Schedule.from_dict(document), declared_target('sm_100a'))
