"""Guarded complete pointwise candidates, address/tail effects and adversarial domains."""
from copy import deepcopy
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule
from tests.contracts.test_triton_loop_scopes import _execute

ROOT=Path(__file__).resolve().parents[2]


def pointwise_document(columns=32, *, target='xcore1002', extra=False):
    suffix=', extra: cake.Tensor((3, %d), "fp32", mode="output")' % columns if extra else ''
    more='        lm.store(extra[row, :], values, coalesced=False, id="store_extra")\n' if extra else ''
    return frontend.parse(f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="whole-row-pointwise", target="{target}", backend="triton", entry_point="whole_row_pointwise")
def candidate(lm, x: cake.Tensor((3, {columns}), "fp32"), bias: cake.Tensor(({columns},), "fp32"), out: cake.Tensor((3, {columns}), "fp32", mode="output"){suffix}):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[row, :], id="load_x")
        biases = lm.load(bias[:], id="load_bias")
        shifted = values + biases
        negative = shifted * -1.0
        decayed = lm.exp(negative, id="exp")
        gate = lm.reciprocal(decayed + 1.0, id="reciprocal")
        result = shifted * gate
        lm.store(out[row, :], result, coalesced=False, id="store_out")
{more}''').document


class PointwiseOutputTiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')

    def apply(self,document,tile=16):
        return self.compiler.tile_pointwise_outputs(document,output_tile=tile,
            schedule_id='pointwise-columns',entry_point='pointwise_columns')

    def test_registered_vectors_and_public_abi_preserve_arithmetic_and_masked_addresses(self):
        for target in ('xcore1002','sm_100a','gfx938','gfx1151'):
            for columns in (32,35):
                with self.subTest(target=target,columns=columns):
                    source=pointwise_document(columns,target=target,extra=True);before=deepcopy(source)
                    result=self.apply(source)
                    self.assertTrue(result.applied,(result.reason,result.message))
                    self.assertEqual(source,before)
                    self.assertEqual(source['operations'],result.schedule['operations'])
                    publics=lambda d:[b for b in d['buffers'] if b['space']=='global']
                    self.assertEqual(publics(source),publics(result.schedule))
                    target_doc=self.compiler._revision.targets[target]
                    emission=emit(Schedule.from_dict(result.schedule),target_doc)
                    self.assertEqual(emission.toolchain['grid'],[3,(columns+15)//16,1])
                    values=[(i%13-6)*.25 for i in range(3*columns)];bias=[(i%5)*.1 for i in range(columns)]
                    memory={'x':values[:],'bias':bias[:],'out':[None]*(3*columns),'extra':[None]*(3*columns)}
                    observer=_execute(emission,memory)
                    expected=[]
                    for row in range(3):
                        for column in range(columns):
                            x=values[row*columns+column]+bias[column];expected.append(x/(1+math.exp(-x)))
                    for actual,reference in zip(memory['out'],expected,strict=True):self.assertAlmostEqual(actual,reference,places=12)
                    self.assertEqual(memory['extra'],values)
                    self.assertEqual(memory['x'],values);self.assertEqual(memory['bias'],bias)
                    self.assertEqual(set(observer.stores.values()),{1})

    def test_cross_column_reduction_is_refused_by_the_rewrite_guard(self):
        # A resident softmax is a valid input, but its outputs are not independent.
        source='''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="resident-row-softmax",target="xcore1002",backend="triton",entry_point="softmax_row")
def candidate(lm,x:cake.Tensor((3,32),"fp32"),out:cake.Tensor((3,32),"fp32",mode="output")):
    compute=lm.role(execution_groups=[0])
    row=lm.program(x,axis=0,dimension=0,tile=1)
    with compute:
        values=lm.load(x[row,:],id="load_x")
        exponent=lm.exp(values,id="exp")
        total=lm.reduce(exponent,op="sum",axis=0,scope="cta",across_loop=False,id="sum")
        inverse=lm.reciprocal(total,id="inverse")
        result=exponent*inverse
        lm.store(out[row,:],result,coalesced=False,id="store_out")
'''
        d=frontend.parse(source).document
        self.assertTrue(self.compiler.assess(d).lowering_eligible)
        result=self.apply(d);self.assertFalse(result.applied);self.assertEqual(result.reason,'coupled_output_axis')

    def test_result_identity_bad_tiles_and_already_column_programmed_candidates_are_refused(self):
        source=pointwise_document()
        for tile in (True,0,-1,3,32,64):
            result=self.apply(source,tile);self.assertFalse(result.applied);self.assertEqual(result.reason,'tile_extent')
        self.assertFalse(self.compiler.tile_pointwise_outputs(source,output_tile=16,
            schedule_id=source['schedule_id'],entry_point='valid').applied)
        candidate=self.apply(source).schedule
        result=self.compiler.tile_pointwise_outputs(candidate,output_tile=8,
            schedule_id='columns-again',entry_point='columns_again')
        self.assertFalse(result.applied);self.assertEqual(result.reason,'program_shape')

    def test_sliced_inputs_and_explicit_residency_cannot_be_silently_reinterpreted(self):
        source=pointwise_document()
        source['residency']={'ctas_per_multiprocessor':1}
        self.assertTrue(self.compiler.assess(source).lowering_eligible)
        result=self.apply(source);self.assertFalse(result.applied);self.assertEqual(result.reason,'execution_commitments')

    def test_complete_program_rebinds_the_same_public_tensors(self):
        from open_cake_ir.compiler.ir import Program
        original=Program.from_schedule(pointwise_document())
        stage=original.stages[0].name
        result=self.compiler.rewrite_program(original,'tile_pointwise_outputs',
            {'stage':stage,'output_tile':16,'schedule_id':'program-columns','entry_point':'program_columns'})
        self.assertTrue(result.applied,(result.reason,result.message))
        self.assertEqual(result.program.tensors,original.tensors)
        self.assertEqual(result.program.outputs,original.outputs)
        self.assertEqual(result.program.stages[0].bindings,original.stages[0].bindings)
        self.assertEqual(len(result.program.stages),len(original.stages))

    def test_author_declared_metadata_survives_and_no_automatic_rewrite_occurs(self):
        source=pointwise_document();source['metadata']={'workload_contract_sha256':'1'*64}
        result=self.apply(source);self.assertTrue(result.applied,(result.reason,result.message))
        self.assertEqual(result.schedule['metadata'],source['metadata'])
        original=self.compiler.lower(self.compiler.assess(source))
        self.assertEqual(original.toolchain_requirements['grid'],[3,1,1])
