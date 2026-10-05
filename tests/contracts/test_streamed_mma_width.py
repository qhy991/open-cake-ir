"""Width API for a fixed output-tile loop; preserve rounding and store domains."""
import ast
import copy
from pathlib import Path
import unittest
from open_cake_ir.compiler import Compiler

ROOT=Path(__file__).resolve().parents[2]


def document(dtype='bf16', target='sm_103a'):
    buffers=[]
    for name,space,kind,shape,mode in [
        ('a','global','bf16',[17,32],'input'),('b','global','bf16',[70,32],'input'),
        ('c','global',dtype,[17,70],'output'),('at','register','bf16',[16,32],'scratch'),
        ('bt','register','bf16',[16,32],'scratch'),('acc','register','fp32',[16,16],'scratch'),
        ('rounded','register',dtype,[16,16],'scratch')]:
        buffers.append(dict(name=name,space=space,dtype=kind,shape=shape,mode=mode))
    def index(source,**kw):return dict(source=source,**kw)
    return {'schema_version':2,'schedule_id':'streamed-output','target':target,
        'roles':[{'name':'compute','execution_groups':list(range(8))}],
        'allocations':[],'pipelines':[],'barriers':[],'buffers':buffers,'outputs':['c'],
        'operations':[
            {'id':'load_a','kind':'load','role':'compute','reads':['a'],'writes':['at'],'parameters':{'movement':'global','reuse':'streamed'}},
            {'id':'load_b','kind':'load','role':'compute','reads':['b'],'writes':['bt'],'parameters':{'movement':'global','reuse':'streamed'}},
            {'id':'dot','kind':'mma','role':'compute','reads':['at','bt'],'writes':['acc'],
             'parameters':{'accumulator':'fp32','instruction':{'contract':'triton.dot.bf16_fp32'},'tile_shape':[16,16,32]},'depends_on':['load_a','load_b']},
            {'id':'round','kind':'cast','role':'compute','reads':['acc'],'writes':['rounded'],'parameters':{'to':dtype},'depends_on':['dot']},
            {'id':'store','kind':'store','role':'compute','reads':['rounded'],'writes':['c'],'parameters':{'coalesced':True},'depends_on':['round']}],
        'program_map':{'axes':[{'name':'m','axis':0,'buffer':'a','dimension':0,'tile':16}]},
        'tile_loops':[{'name':'output','iterator':'nn','buffer':'b','dimension':0,'tile':16,
            'body':['load_b','dot','round','store'],
            'range_options':{'num_stages':1,'loop_unroll_factor':1,'flatten':False,'warp_specialize':False,'disallow_acc_multi_buffer':True,'disable_licm':False}}],
        'access_maps':[
            {'operation':'load_a','buffer':'a','indices':[index('program_tile',name='m'),index('dimension',dimension=1)],'boundary':'mask_tiled_axes'},
            {'operation':'load_b','buffer':'b','indices':[index('loop_tile',name='nn'),index('dimension',dimension=1)],'boundary':'mask_tiled_axes'},
            {'operation':'store','buffer':'c','indices':[index('program_tile',name='m'),index('loop_tile',name='nn')],'boundary':'mask_tiled_axes'}],
        'metadata':{},'lowering':{'backend':'triton','entry_point':'streamed_output'}}


class StreamedMmaWidthTests(unittest.TestCase):
    def setUp(self):self.compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')
    def apply(self,d,width=2):
        return self.compiler.specialize_triton_warps(d,num_warps=width,schedule_id='streamed-width',entry_point='streamed_width')
    def kernel(self,lowering):
        n=next(n for n in ast.parse(lowering.source).body if isinstance(n,ast.FunctionDef) and n.name==lowering.toolchain_requirements['kernel_entry_point'])
        n.name='same_kernel';return ast.dump(n,include_attributes=False)
    def test_streamed_float_rounding_and_tail_store_body_unchanged(self):
        for target in ('sm_100a','sm_103a'):
            for dtype in ('fp16','bf16'):
                with self.subTest(target=target,dtype=dtype):
                    d=document(dtype,target);before=copy.deepcopy(d);a=self.compiler.assess(d)
                    self.assertTrue(a.lowering_eligible,[f.code for f in a.findings])
                    result=self.apply(d);self.assertTrue(result.applied,(result.reason,result.message))
                    self.assertEqual(d,before)
                    for key in d.keys()-{'roles','lowering','schedule_id'}:self.assertEqual(result.schedule[key],d[key])
                    original=self.compiler.lower(a);lower=self.compiler.lower(result.assessment)
                    self.assertEqual(self.kernel(original),self.kernel(lower))
                    self.assertEqual(lower.toolchain_requirements['compile_options']['num_warps'],2)
                    self.assertIn('.to(tl.'+('float16' if dtype=='fp16' else 'bfloat16')+')',lower.source)
    def test_non_streamed_repeated_output_remains_outside_domain(self):
        d=document();next(b for b in d['buffers'] if b['name']=='c')['shape']=[17,16]
        d['access_maps'][-1]['indices'][1]={'source':'dimension','dimension':1}
        self.assertTrue(self.compiler.assess(d).lowering_eligible)
        self.assertEqual(self.apply(d).reason,'loop_domain')
    def test_narrow_intermediate_mma_consumer_remains_outside_domain(self):
        d=document()
        for b in d['buffers']:
            if b['name'] in ('a','b','at','bt'):b['shape'][-1]=16
        d['operations'][2]['parameters']['tile_shape'][-1]=16
        d['buffers'].append({'name':'discard','space':'register','dtype':'fp32','shape':[16,16],'mode':'scratch'})
        d['operations'].insert(4,{'id':'reuse_round','kind':'mma','role':'compute','reads':['rounded','bt'],'writes':['discard'],
            'parameters':{'accumulator':'fp32','instruction':{'contract':'triton.dot.bf16_fp32'},'tile_shape':[16,16,16]},'depends_on':['round']})
        d['tile_loops'][0]['body'].insert(-1,'reuse_round')
        self.assertTrue(self.compiler.assess(d).lowering_eligible)
        self.assertEqual(self.apply(d).reason,'loop_domain')
    def test_fixed_execution_commitment_still_refused(self):
        d=document();d['residency']={'ctas_per_multiprocessor':1}
        self.assertTrue(self.compiler.assess(d).lowering_eligible)
        self.assertEqual(self.apply(d).reason,'execution_commitments')

if __name__=='__main__':unittest.main()
