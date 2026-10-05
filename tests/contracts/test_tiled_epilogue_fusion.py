"""Rounded dual-MMA K-loop composition; device/original Task acceptance is separate."""
from copy import deepcopy
import json
from pathlib import Path
import random
import unittest

from open_cake_ir.compiler import Compiler, Program
from tests.contracts.test_epilogue_fusion import execute, rounded

ROOT=Path(__file__).resolve().parents[2]


def tiled_program(*, m=19,n=23,k=33,tile=16,dtype='bf16',target='sm_100a'):
    p=json.loads((ROOT/'corpus/schedules/gemm-bias-b1-smoke.json').read_text())
    p['target']=target;p['schedule_id']='dual_producer';p['lowering']['entry_point']='dual_producer';p['metadata']={}
    p['buffers']=[b for b in p['buffers'] if b['name'] not in {'bias','bias_tile','c_tile'}]
    p['operations']=[o for o in p['operations'] if o['id'] not in {'load_bias','add_bias','store_c'}]
    p['access_maps']=[a for a in p['access_maps'] if a['operation'] in {'load_a','load_b'}]
    for b in p['buffers']:
        b['shape']={'a':[m,k],'b':[n,k],'c':[m,n],'a_tile':[tile,tile],'b_tile':[tile,tile],'acc':[tile,tile]}[b['name']]
        if b['name'] in {'a','b','a_tile','b_tile'}:b['dtype']='bf16'
        if b['name']=='c':b.update(name='gate',dtype=dtype)
    p['program_map']['axes'][0]['tile']=tile;p['program_map']['axes'][1]['tile']=tile
    p['tile_loops'][0]['tile']=tile;p['tile_loops'][0]['range_options']['num_stages']=1
    p['operations'][2]['parameters']['tile_shape']=[tile,tile,tile]
    p['buffers'] += [dict(name=name,space=space,dtype=dt,shape=shape,mode=mode) for name,space,dt,shape,mode in
        [('u','global','bf16',[n,k],'input'),('u_tile','register','bf16',[tile,tile],'scratch'),
         ('acc_u','register','fp32',[tile,tile],'scratch'),('up','global',dtype,[m,n],'output')]]
    load=deepcopy(p['operations'][1]);load.update(id='load_u',reads=['u'],writes=['u_tile'])
    dot=deepcopy(p['operations'][2]);dot.update(id='dot_u',reads=['a_tile','u_tile'],writes=['acc_u'],depends_on=['load_a','load_u'])
    p['operations'] += [load,dot];p['tile_loops'][0]['body'] += ['load_u','dot_u']
    access=deepcopy(p['access_maps'][1]);access.update(operation='load_u',buffer='u');p['access_maps'].append(access)
    if dtype!='fp32':
        for name,source,opid in [('g16','acc','round_g'),('u16','acc_u','round_u')]:
            p['buffers'].append(dict(name=name,space='register',dtype=dtype,shape=[tile,tile],mode='scratch'))
            p['operations'].append(dict(id=opid,kind='cast',role='compute',reads=[source],writes=[name],depends_on=['dot' if source=='acc' else 'dot_u'],parameters=dict(to=dtype)))
    for name,source,dep in [('gate','g16' if dtype!='fp32' else 'acc','round_g' if dtype!='fp32' else 'dot'),
                            ('up','u16' if dtype!='fp32' else 'acc_u','round_u' if dtype!='fp32' else 'dot_u')]:
        p['operations'].append(dict(id='store_'+name,kind='store',role='compute',reads=[source],writes=[name],depends_on=[dep],parameters=dict(coalesced=True)))
        p['access_maps'].append(dict(operation='store_'+name,buffer=name,boundary='mask_tiled_axes',indices=[dict(source='program_tile',name='m_block'),dict(source='program_tile',name='n_block')]))
    p['outputs']=['gate','up']
    e=deepcopy(p);e['schedule_id']='consumer';e['lowering']['entry_point']='consumer';e['tile_loops']=[]
    e['program_map']['axes']=[dict(name='row',axis=0,buffer='gate',dimension=0,tile=tile),dict(name='col',axis=1,buffer='gate',dimension=1,tile=tile)]
    e['buffers']=[dict(name=name,space=space,dtype=dt,shape=shape,mode=mode) for name,space,dt,shape,mode in
        [('gate','global',dtype,[m,n],'input'),('up','global',dtype,[m,n],'input'),('y','global',dtype,[m,n],'output'),
         ('g','register',dtype,[tile,tile],'scratch'),('u','register',dtype,[tile,tile],'scratch'),
         ('product','register','fp32',[tile,tile],'scratch')]]
    e['operations']=[];e['access_maps']=[]
    for name,result in [('gate','g'),('up','u')]:
        e['operations'].append(dict(id='load_'+name,kind='load',role='compute',reads=[name],writes=[result],parameters=dict(movement='global')))
        e['access_maps'].append(dict(operation='load_'+name,buffer=name,boundary='mask_tiled_axes',indices=[dict(source='program_tile',name='row'),dict(source='program_tile',name='col')]))
        if dtype!='fp32':
            e['buffers'].append(dict(name=result+'32',space='register',dtype='fp32',shape=[tile,tile],mode='scratch'))
            e['operations'].append(dict(id='wide_'+result,kind='cast',role='compute',reads=[result],writes=[result+'32'],depends_on=['load_'+name],parameters=dict(to='fp32')))
    e['operations'].append(dict(id='multiply',kind='elementwise',role='compute',reads=['g32','u32'] if dtype!='fp32' else ['g','u'],writes=['product'],depends_on=['wide_g','wide_u'] if dtype!='fp32' else ['load_gate','load_up'],parameters=dict(op='mul')))
    if dtype!='fp32':
        e['buffers'].append(dict(name='yb',space='register',dtype=dtype,shape=[tile,tile],mode='scratch'))
        e['operations'].append(dict(id='round_y',kind='cast',role='compute',reads=['product'],writes=['yb'],depends_on=['multiply'],parameters=dict(to=dtype)))
    e['operations'].append(dict(id='store_y',kind='store',role='compute',reads=['yb' if dtype!='fp32' else 'product'],writes=['y'],depends_on=['round_y' if dtype!='fp32' else 'multiply'],parameters=dict(coalesced=True)))
    e['access_maps'].append(dict(operation='store_y',buffer='y',boundary='mask_tiled_axes',indices=[dict(source='program_tile',name='row'),dict(source='program_tile',name='col')]))
    e['outputs']=['y']
    tensors={b['name']:{'shape':b['shape'],'dtype':b['dtype']} for s in [p,e] for b in s['buffers'] if b['space']=='global'}
    return dict(schema_version=1,program_id='tiled_dual_composition',target=target,tensors=tensors,inputs=['a','b','u'],outputs=['y'],
        stages=[dict(name=name,schedule=s,bindings={b['name']:b['name'] for b in s['buffers'] if b['space']=='global'}) for name,s in [('producer',p),('epilogue',e)]])


class TiledEpilogueFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')
    def fuse(self,document):
        return self.compiler.rewrite_program(Program.from_dict(document),'fuse_tiled_epilogue',dict(producer='producer',epilogue='epilogue',schedule_id='fused_tiles',entry_point='fused_tiles'))
    def test_two_rounded_outputs_and_original_k_loop_survive_full_program_rewrite(self):
        d=tiled_program();before=deepcopy(d);r=self.fuse(d);self.assertTrue(r.applied,r.message)
        self.assertEqual(d,before);self.assertEqual(r.program.inputs,('a','b','u'));self.assertEqual(r.program.outputs,('y',))
        s=r.program.document['stages'][0]['schedule'];self.assertEqual(s['tile_loops'],d['stages'][0]['schedule']['tile_loops'])
        self.assertEqual([o['id'] for o in s['operations'] if o['kind']=='mma'],['dot','dot_u'])
        self.assertTrue({'round_g','round_u'}<={o['id'] for o in s['operations']})
        self.assertEqual(sum(o['kind']=='store' for o in s['operations']),1)
        self.assertNotIn('gate',r.program.tensors);self.assertNotIn('up',r.program.tensors)
        self.compiler.lower_program(r.program)
    def test_emitted_before_after_match_with_m_n_k_tails_and_rounding(self):
        for dtype in ['bf16','fp16']:
            d=tiled_program(m=17,n=19,k=17,dtype=dtype);r=self.fuse(d);self.assertTrue(r.applied,r.message)
            rng=random.Random(991);inputs={'a':[rounded(rng.uniform(-.5,.5),'bf16') for _ in range(17*17)],
             'b':[rounded(rng.uniform(-.5,.5),'bf16') for _ in range(19*17)],'u':[rounded(rng.uniform(-.5,.5),'bf16') for _ in range(19*17)]}
            middle,_=execute(d['stages'][0]['schedule'],inputs);expected,_=execute(d['stages'][1]['schedule'],middle)
            actual,_=execute(r.program.document['stages'][0]['schedule'],inputs)
            self.assertEqual(list(actual.values()),list(expected.values()))
    def test_public_output_other_consumer_and_views_refuse_at_program_owner(self):
        d=tiled_program();d['outputs'].append('gate');self.assertEqual(self.fuse(d).reason,'public_intermediate')
        d=tiled_program();extra=deepcopy(d['stages'][1]);extra['name']='other';extra['bindings']['y']='other_y';d['tensors']['other_y']=deepcopy(d['tensors']['y']);d['outputs'].append('other_y');d['stages'].append(extra)
        self.assertEqual(self.fuse(d).reason,'intermediate_consumers')
    def test_fp32_arithmetic_materialization_remains_a_valid_input_but_refused_seam(self):
        d=tiled_program(dtype='fp32')
        for s in d['stages']:self.assertTrue(self.compiler.assess(s['schedule']).lowering_eligible)
        self.assertEqual(self.fuse(d).reason,'rounding_boundary')
    def test_mismatched_tiles_refuse_without_invalidating_either_stage(self):
        d=tiled_program();e=d['stages'][1]['schedule']
        for axis in e['program_map']['axes']:axis['tile']=32
        for b in e['buffers']:
            if b['space']=='register':b['shape']=[32,32]
        self.assertTrue(self.compiler.assess(e).lowering_eligible)
        self.assertEqual(self.fuse(d).reason,'tile_ownership')
