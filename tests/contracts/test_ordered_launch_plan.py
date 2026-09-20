"""Typed ordered launches retain dependencies and keep task mathematics in kernels."""
from copy import deepcopy
from pathlib import Path
from dataclasses import replace
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.compiler import Program
from open_cake_ir.compiler.program import LoweredProgram
from open_cake_ir.evaluation.launch_plan import prepare_program

ROOT = Path(__file__).resolve().parents[2]


def stage(name):
    return parse(f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="{name}", target="sm_103a", backend="triton", entry_point="{name}")
def candidate(lm, x: cake.Tensor((4,), "fp32"), y: cake.Tensor((4,), "fp32", mode="output")):
    block = lm.program(y, axis=0, dimension=0, tile=4)
    compute = lm.role(execution_groups=[0])
    with compute:
        values = lm.load(x[block])
        result = values + 1.0
        lm.store(y[block], result, coalesced=False)
''').document


def document():
    return {'schema_version':1,'program_id':'two-stages','target':'sm_103a',
            'inputs':['x'],'outputs':['y'],
            'tensors':{name:{'shape':[4],'dtype':'fp32'} for name in ['x','middle','y']},
            'stages':[{'name':'first','schedule':stage('first'),'bindings':{'x':'x','y':'middle'}},
                      {'name':'second','schedule':stage('second'),'bindings':{'x':'middle','y':'y'}}]}


class OrderedLaunchPlanTests(unittest.TestCase):
    def test_plan_derives_storage_and_runs_every_stage_in_order(self):
        plan = Program.from_dict(document())
        self.assertEqual(plan.allocated_bytes, 32)
        order = []
        def load(name, lowering):
            def kernel(x, out):
                order.append(name)
                out[:] = [v+1 for v in x]
            return kernel
        compiled = Compiler.load(ROOT, ROOT/'compiler/revision.json').lower_program(plan)
        prepared = prepare_program(compiled, {'x':[1,2,3,4]}, allocate=lambda n,s:[None]*4,
                                    load_kernel=load, check_tensor=lambda tensor,spec:None,
                                    storage_span=lambda tensor:('fixture',id(tensor)*1000,id(tensor)*1000+16),
                                    execution_context=lambda:('fixture','stream'))
        self.assertEqual(prepared.run()['y'],[3,4,5,6])
        self.assertEqual(order,['first','second'])
        self.assertEqual(prepared.launch_calls,2)

    def test_distinct_views_with_overlapping_storage_are_refused(self):
        plan=Program.from_dict(document())
        compiled = Compiler.load(ROOT, ROOT/'compiler/revision.json').lower_program(plan)
        with self.assertRaisesRegex(ValueError,'overlaps'):
            prepare_program(compiled, {'x':[1,2,3,4]}, allocate=lambda n,s:[None]*4,
                check_tensor=lambda t,s:None, storage_span=lambda t:('same-device',1000,1016),
                execution_context=lambda:0, load_kernel=lambda n,l:None)

    def test_read_before_producer_and_double_writer_are_refused(self):
        bad = document();bad['stages'].reverse()
        with self.assertRaisesRegex(ValueError,'before its producer'):Program.from_dict(bad)
        bad = document();bad['stages'][1]['bindings']['y'] = 'middle'
        with self.assertRaises(ValueError):Program.from_dict(bad)

    def test_binding_shape_dtype_and_target_must_match(self):
        for field,value in [('shape',[8]),('dtype','bf16')]:
            bad=document();bad['tensors']['middle'][field]=value
            with self.assertRaisesRegex(ValueError,'shape/dtype differs'):Program.from_dict(bad)
        bad=document();bad['stages'][1]['schedule']['target']='sm_100a'
        with self.assertRaisesRegex(ValueError,'target differs'):Program.from_dict(bad)

    def test_inputs_cannot_be_written_or_ignored(self):
        bad=document();bad['stages'][0]['bindings']['y']='x'
        with self.assertRaises(ValueError):Program.from_dict(bad)
        bad=document();bad['tensors']['unused']={'shape':[4],'dtype':'fp32'};bad['inputs'].append('unused')
        with self.assertRaisesRegex(ValueError,'ignores a public input'):Program.from_dict(bad)

    def test_compiler_assesses_every_stage(self):
        compiler = Compiler.load(ROOT,ROOT/'compiler/revision.json')
        if compiler.commit is None:
            self.skipTest('compile identity requires committed test worktree')
        compiled = compiler.lower_program(Program.from_dict(document()))
        self.assertEqual(len(compiled.lowerings),2)
        self.assertTrue(all(lowering.generated for lowering in compiled.lowerings))
        bad=document();bad['stages'][1]['schedule']['operations'][0]['reads']=['missing']
        with self.assertRaisesRegex(ValueError,'refused'):compiler.lower_program(Program.from_dict(bad))
