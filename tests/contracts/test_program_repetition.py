"""Report repeated whole-row work without rejecting valid tiling tradeoffs."""
from pathlib import Path
import unittest
from open_cake_ir.compiler import Compiler,frontend
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.performance.program_repetition import program_repetition_guidance

ROOT=Path(__file__).resolve().parents[2]
SOURCE='''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="repeated-row",target="xcore1002",backend="triton",entry_point="repeated_row")
def kernel(lm,x:cake.Tensor((8,1024),"fp32"),out:cake.Tensor((8,1024),"fp32",mode="output")):
    compute=lm.role(execution_groups=[0])
    row=lm.program(x,axis=0,dimension=0,tile=1)
    col=lm.program(out,axis=1,dimension=1,tile=256)
    with compute:
        full=lm.load(x[row,:],id="full_load")
        total=lm.reduce(full,op="sum",axis=0,scope="cta",across_loop=False,id="row_sum")
        part=lm.load(x[row,col],id="part_load")
        result=part+total
        lm.store(out[row,col],result,coalesced=True,id="store")
'''

class ProgramRepetition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')

    def test_full_row_work_is_localized_nonblocking_and_target_neutral(self):
        for target in ['xcore1002','sm_100a','gfx938']:
            assessment=self.compiler.assess(frontend.parse(SOURCE.replace('xcore1002',target)).document)
            self.assertTrue(assessment.accepted,assessment.findings)
            self.assertTrue(assessment.lowering_eligible,assessment.findings)
            hints=[f for f in assessment.guidance if f.code=='PROGRAM_INVARIANT_REDUCTION']
            self.assertEqual(len(hints),1)
            self.assertIn('4 times',hints[0].message)
            self.assertFalse(hints[0].blocks_lowering)
            self.assertEqual(hints[0].path,'operations[1]')

    def test_tile_dependent_reduction_and_single_program_axis_have_no_hint(self):
        for source in [SOURCE.replace('x[row,:]','x[row,col]'), SOURCE.replace('tile=256','tile=1024')]:
            assessment=self.compiler.assess(frontend.parse(source).document)
            self.assertTrue(assessment.lowering_eligible,assessment.findings)
            self.assertNotIn('PROGRAM_INVARIANT_REDUCTION',[f.code for f in assessment.guidance])

    def test_mutable_input_and_persistence_abstain(self):
        doc=frontend.parse(SOURCE).document
        doc['buffers'][0]['mode']='state'
        self.assertEqual(program_repetition_guidance(Schedule.from_dict(doc)),())
        doc=frontend.parse(SOURCE).document
        doc['program_map']['persistent']=True
        self.assertEqual(program_repetition_guidance(Schedule.from_dict(doc)),())
