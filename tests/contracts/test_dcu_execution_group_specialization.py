"""A launch-width rewrite retains the complete graph and bounded MMA loop."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.compiler.reduction_tiling import tile_squared_difference
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.contraction.authoring import starter_source
from open_cake_ir.tasks.contraction.workload import workload_document

ROOT = Path(__file__).resolve().parents[2]


class ExecutionGroupSpecialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def gemm(self, target='gfx938'):
        document = json.loads((ROOT / 'corpus/schedules/gemm-bias-b1-smoke.json').read_text())
        document['target'] = target
        document.pop('residency', None)
        for role in document['roles']:
            role.pop('registers_per_thread', None)
        if target == 'gfx938':
            for buffer in document['buffers']:
                if buffer['name'] in {'a', 'b', 'a_tile', 'b_tile'}:
                    buffer['dtype'] = 'fp16'
            document['operations'][2]['parameters']['instruction']['contract'] = 'triton.dot.fp16_fp32'
        return document

    def apply(self, document, width=8):
        return self.compiler.specialize_triton_warps(document, num_warps=width,
            schedule_id='group_candidate', entry_point='group_candidate')

    def test_loop_math_accesses_and_abi_survive_width_change(self):
        for target in ('gfx938', 'sm_100a'):
            original = self.gemm(target)
            saved = deepcopy(original)
            result = self.apply(original)
            self.assertTrue(result.applied, (result.reason, result.message))
            self.assertEqual(original, saved)
            for key in saved.keys() - {'roles', 'schedule_id', 'lowering'}:
                self.assertEqual(result.schedule[key], saved[key], key)
            before = self.compiler.lower(self.compiler.assess(original))
            after = self.compiler.lower(result.assessment)
            def kernel(lowering):
                node = next(n for n in ast.parse(lowering.source).body
                            if isinstance(n, ast.FunctionDef)
                            and n.name == lowering.toolchain_requirements['kernel_entry_point'])
                node.name = 'same_kernel'
                return ast.dump(node)
            self.assertEqual(kernel(before), kernel(after))
            self.assertEqual(after.toolchain_requirements['compile_options']['num_warps'], 8)

    def test_complete_program_binding_survives_explicit_specialization(self):
        program = Program.from_schedule(self.gemm())
        result = self.compiler.rewrite_program(program, 'specialize_triton_warps',
            dict(stage=program.stages[0].name, num_warps=2,
                 schedule_id='group_candidate', entry_point='group_candidate'))
        self.assertTrue(result.applied, result.message)
        self.assertEqual(result.program.inputs, program.inputs)
        self.assertEqual(result.program.outputs, program.outputs)
        self.assertEqual(result.program.tensors, program.tensors)

    def test_valid_non_mma_loop_is_refused_by_the_loop_domain(self):
        workload = WorkloadContract(workload_document('pairwise_sqdist', rows=64,
            depth=256, columns=32, backend='triton-b300'))
        source = parse(starter_source(workload)).document
        tiled = tile_squared_difference(self.compiler, source, k_tile=64,
            schedule_id='tiled', entry_point='tiled')
        self.assertTrue(tiled.applied, tiled.message)
        self.assertTrue(self.compiler.assess(tiled.schedule).lowering_eligible)
        self.assertEqual(self.apply(tiled.schedule).reason, 'loop_domain')

    def test_valid_residency_commitment_is_not_silently_changed(self):
        document = self.gemm()
        document['residency'] = {'ctas_per_multiprocessor': 1}
        self.assertTrue(self.compiler.assess(document).lowering_eligible)
        self.assertEqual(self.apply(document).reason, 'execution_commitments')

    def test_backend_register_refusal_is_not_relaxed_by_the_pass(self):
        document = self.gemm('sm_100a')
        document['roles'][0]['registers_per_thread'] = 128
        document['residency'] = {'registers_per_thread': 128}
        self.assertFalse(self.compiler.assess(document).lowering_eligible)
        self.assertEqual(self.apply(document).reason, 'input_refused')
