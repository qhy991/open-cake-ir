"""Compiler rewrite guard and emitted loop semantics, without device claims."""
import copy
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.compiler.ir import Program
from open_cake_ir.compiler.reduction_tiling import tile_squared_difference
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.contraction.authoring import starter_source
from open_cake_ir.tasks.contraction.workload import workload_document
from open_cake_ir.tasks.contraction.arithmetic import validate_candidate_arithmetic

ROOT = Path(__file__).resolve().parents[2]


class SquaredDifferenceTiling(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler.load(ROOT,ROOT/'compiler/revision.json')
        self.workload = WorkloadContract(workload_document('pairwise_sqdist',rows=64,depth=256,columns=32,backend='triton-metax'))
        self.schedule = parse(starter_source(self.workload)).document

    def test_explicit_rewrite_changes_residency_not_operand_arithmetic(self):
        original = copy.deepcopy(self.schedule)
        result = tile_squared_difference(self.compiler,self.schedule,k_tile=64,schedule_id='tiled',
                                         entry_point='cake_contraction_pairwise_sqdist_fp32')
        self.assertTrue(result.applied, result.message)
        self.assertEqual(self.schedule,original)
        candidate = result.schedule
        validate_candidate_arithmetic(candidate,self.workload)
        self.assertEqual(candidate['tile_loops'][0]['tile'],64)
        self.assertTrue(candidate['operations'][4]['parameters']['across_loop'])
        source = self.compiler.lower(result.assessment).source
        self.assertIn('range(',source)
        self.assertNotIn('tl.dot(',source)
        program = Program.from_schedule(self.schedule)
        rewritten = self.compiler.rewrite_program(program,'tile_squared_difference',
            {'stage':program.stages[0].name,'k_tile':64,'schedule_id':'tiled','entry_point':'cake_contraction_pairwise_sqdist_fp32'})
        self.assertTrue(rewritten.applied,rewritten.message)

    def test_counterexamples_are_owned_by_the_intended_guard(self):
        cases=[]
        d=copy.deepcopy(self.schedule);d['operations'][2]['parameters']['op']='add';cases.append((d,'arithmetic_domain'))
        d=copy.deepcopy(self.schedule);d['buffers'][3]['dtype']='bf16';cases.append((d,'storage_domain'))
        d=copy.deepcopy(self.schedule);d['access_maps'][1]['indices'][1]['offset']=1;cases.append((d,'access_domain'))
        for d,reason in cases:
            result=tile_squared_difference(self.compiler,d,k_tile=64,schedule_id='tiled',entry_point='tiled')
            self.assertFalse(result.applied);self.assertEqual(result.reason,reason,result.message)
        for tile in (True,0,3,256,512):
            result=tile_squared_difference(self.compiler,self.schedule,k_tile=tile,schedule_id='tiled',entry_point='tiled')
            self.assertEqual(result.reason,'tile_extent')

    def test_workload_refuses_algebraic_expansion_and_low_precision_before_build(self):
        d=copy.deepcopy(self.schedule);d['operations'][3]['parameters']['op']='mul'
        with self.assertRaisesRegex(ValueError,'form the FP32 difference'):validate_candidate_arithmetic(d,self.workload)
        d=copy.deepcopy(self.schedule);d['buffers'][3]['dtype']='bf16'
        with self.assertRaisesRegex(ValueError,'remain FP32'):validate_candidate_arithmetic(d,self.workload)
        validate_candidate_arithmetic(self.schedule,self.workload)
