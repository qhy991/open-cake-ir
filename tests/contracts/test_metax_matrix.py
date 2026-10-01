"""Measured complete MACA dot contracts do not admit NVIDIA-only refinements."""
import copy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
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

    def test_maca_warp_count_above_canary_qualification_is_refused_before_device(self):
        source = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="warp-canary", target="xcore1002", backend="triton", entry_point="kernel")
def candidate(lm, x: cake.Tensor((72, 128), "fp32"), out: cake.Tensor((72, 128), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        value = lm.load(x[row, :], id="load")
        lm.store(out[row, :], value, coalesced=False, id="store")
'''
        result = self.compiler.assess(frontend.parse(source).document)
        self.assertIn('MACA_WARP_COUNT_UNQUALIFIED', [f.code for f in result.findings])
        self.assertFalse(result.lowering_eligible)

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
        # Isolate selected-K from the separately refused MACA loop keyword.
        # Keeping the NVIDIA-only hint would stop at that earlier owner.
        for loop in document['tile_loops']:
            loop['range_options']['disallow_acc_multi_buffer'] = False
        schedule = Schedule.from_dict(document)
        target = declared_target('xcore1002')
        findings = triton.preflight(schedule, target)
        self.assertIn('TRITON_MMA_K_RANGES_UNSUPPORTED', [f.code for f in findings])
        with self.assertRaises(EmitError):
            triton.emit(schedule, target)
        # The original NVIDIA refinement is supported, and no unrelated
        # dtype/interval failure is being used as the selected-K guard.
        self.assertFalse(any(f.blocks_lowering for f in triton.preflight(
            Schedule.from_dict(original), declared_target(original['target']))))

    def test_maca_range_refuses_unsupported_keywords_before_compilation(self):
        document = frontend.read_schedule(ROOT / 'examples/python/b300_gemm_bias.py').document
        document['target'] = 'xcore1002'
        # The register cap has its own existing MACA refusal. Remove it so the
        # loop keyword under investigation owns each negative result.
        document.pop('residency')
        options = document['tile_loops'][0]['range_options']
        options['disallow_acc_multi_buffer'] = False
        self.assertEqual(options['num_stages'], 2)
        self.assertTrue(self.compiler.assess(document).lowering_eligible)
        source = triton.emit(Schedule.from_dict(document), declared_target('xcore1002')).source
        self.assertIn('num_stages=2', source)
        for name, value in (('loop_unroll_factor', 2), ('flatten', True),
                            ('disallow_acc_multi_buffer', True), ('disable_licm', True)):
            with self.subTest(option=name):
                candidate = copy.deepcopy(document)
                candidate['tile_loops'][0]['range_options'][name] = value
                assessment = self.compiler.assess(candidate)
                self.assertFalse(assessment.lowering_eligible)
                findings = [f for f in assessment.findings
                            if f.code == 'MACA_LOOP_OPTION_UNSUPPORTED']
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0].path, f'tile_loops[0].range_options.{name}')
                with self.assertRaises(EmitError):
                    triton.emit(Schedule.from_dict(candidate), declared_target('xcore1002'))
                candidate['target'] = 'sm_103a'
                self.assertFalse(any(f.blocks_lowering for f in triton.preflight(
                    Schedule.from_dict(candidate), declared_target('sm_103a'))))
