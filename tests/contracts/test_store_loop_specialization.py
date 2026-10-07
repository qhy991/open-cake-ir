"""Stage selection changes one existing store region, not reduction semantics."""
from copy import deepcopy
from pathlib import Path
import unittest
from open_cake_ir.compiler import Compiler, Program, frontend
from open_cake_ir.compiler.program_passes import TRANSFORMATIONS

ROOT=Path(__file__).resolve().parents[2]


def source(columns=128):
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="two-pass-rms", target="gfx938", backend="triton", entry_point="two_pass_rms")
def candidate(lm, x: cake.Tensor((3, {columns}), "fp32"),
              out: cake.Tensor((3, {columns}), "fp32", mode="output")):
    compute=lm.role(execution_groups=[0])
    row=lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        for c in lm.range(x, dimension=1, tile=32, name="sum_region"):
            v=lm.load(x[row,c], id="load_sum")
            q=lm.square(v, id="square")
            total=lm.reduce(q, op="sum", axis=0, scope="cta", across_loop=True, id="sum")
        inv=lm.rsqrt(total / {float(columns)} + 0.000001, id="inv")
        for c2 in lm.range(x, dimension=1, tile=32, name="store_region"):
            v2=lm.load(x[row,c2], id="load_apply")
            lm.store(out[row,c2], v2 * inv, id="store_out")
'''


class StoreLoopSpecializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler=Compiler.load(ROOT)

    def document(self, columns=128):
        return frontend.parse(source(columns)).document

    def apply(self, doc, **kw):
        args=dict(loop_name='store_region',num_stages=4,schedule_id='two-pass-staged',entry_point='two_pass_staged')
        args.update(kw)
        return self.compiler.specialize_triton_store_loop(doc,**args)

    def test_only_selected_range_depth_changes_and_tail_remains_masked(self):
        for columns in (128,117):
            with self.subTest(columns=columns):
                doc=self.document(columns);original=deepcopy(doc)
                result=self.apply(doc)
                self.assertTrue(result.applied,result.message)
                expected=deepcopy(doc);expected['schedule_id']='two-pass-staged';expected['lowering']['entry_point']='two_pass_staged'
                expected['tile_loops'][1]['range_options']['num_stages']=4
                self.assertEqual(result.schedule,expected)
                self.assertEqual(doc,original)
                text=self.compiler.lower(result.assessment).source
                self.assertIn('num_stages=4',text)
                self.assertIn('num_stages=1',text)
                self.assertIn('tl.sum(',text)
                self.assertIn('mask=',text)

    def test_complete_program_action_keeps_public_tensors_and_bindings(self):
        doc=self.document();program=Program.from_schedule(doc)
        stage=program.stages[0].name
        result=self.compiler.rewrite_program(program,'specialize_triton_store_loop',dict(
            stage=stage,loop_name='store_region',num_stages=4,schedule_id='new-rms',entry_point='new_rms'))
        self.assertTrue(result.applied,result.message)
        for key in ('tensors','inputs','outputs'):
            self.assertEqual(result.program.document[key],program.document[key])
        self.assertEqual(result.program.document['stages'][0]['bindings'],program.document['stages'][0]['bindings'])
        action=next(x for x in TRANSFORMATIONS if x.name=='specialize_triton_store_loop')
        self.assertIn('loop_name',action.parameters)

    def test_missing_or_reduction_region_has_a_local_refusal(self):
        self.assertEqual(self.apply(self.document(),loop_name='absent').reason,'loop_selection')
        self.assertEqual(self.apply(self.document(),loop_name='sum_region').reason,'loop_domain')

    def test_store_region_with_its_own_resident_reduction_is_outside_domain(self):
        text=source().replace('lm.store(out[row,c2], v2 * inv, id="store_out")',
            'local=lm.reduce(v2, op="sum", axis=0, scope="cta", id="local_sum")\n'
            '            lm.store(out[row,c2], v2 * local, id="store_out")')
        doc=frontend.parse(text).document
        self.assertTrue(self.compiler.assess(doc).lowering_eligible)
        self.assertEqual(self.apply(doc).reason,'loop_domain')

    def test_depth_is_bounded_by_real_trip_count_and_never_silently_clamped(self):
        for value in (True,0,-1,1.5,5):
            with self.subTest(value=value):
                self.assertEqual(self.apply(self.document(),num_stages=value).reason,'stage_count')
        self.assertEqual(self.apply(self.document(),num_stages=1).reason,'unchanged')

    def test_result_identity_and_unqualified_route_are_not_changed(self):
        self.assertEqual(self.apply(self.document(),entry_point='not-valid').reason,'result_identity')
        doc=self.document();doc['target']='sm_103a'
        self.assertTrue(self.compiler.assess(doc).lowering_eligible)
        self.assertEqual(self.apply(doc).reason,'target_route')

    def test_residency_and_persistent_commitments_are_refused(self):
        for persistent in (False,True):
            with self.subTest(persistent=persistent):
                doc=self.document();doc['residency']={'ctas_per_multiprocessor':1}
                doc['program_map']['persistent']=persistent
                result=self.apply(doc)
                self.assertEqual(result.reason,'execution_commitments')

    def test_output_loop_cannot_hide_a_cross_iteration_store_hazard(self):
        doc=self.document()
        dest=next(b for b in doc['buffers'] if b['name']=='out');dest['shape'][1]=32
        access=next(x for x in doc['access_maps'] if x['operation']=='store_out')
        access['indices'][1]={'source':'dimension','dimension':1}
        a=self.compiler.assess(doc)
        self.assertIn('TRITON_LOOP_STORE_OWNERSHIP',[f.code for f in a.findings])
        self.assertEqual(self.apply(doc).reason,'input_refused')


if __name__=='__main__':unittest.main()
