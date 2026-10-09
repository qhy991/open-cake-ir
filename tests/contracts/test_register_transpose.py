"""Register transpose keeps dtype, masks and contraction-axis provenance."""
from copy import deepcopy
from pathlib import Path
import unittest
from open_cake_ir.compiler import Compiler,frontend
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule,ScheduleParseError
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.performance.work import work_bound
from tests.contracts.test_triton_loop_scopes import _execute
from tests.contracts.test_cast_contraction_provenance import cast_gemm

ROOT=Path(__file__).resolve().parents[2]


def source(rows=17,columns=33,dtype='fp32',double=False):
    tail=('        restored=lm.transpose(t,id="restore")\n'
          '        lm.store(out[m,n],restored,id="store_out")') if double else '        lm.store(out[n,m],t,id="store_out")'
    out_shape=(rows,columns) if double else (columns,rows)
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="register-transpose",target="gfx938",backend="triton",entry_point="register_transpose")
def candidate(lm,x:cake.Tensor(({rows},{columns}),"{dtype}"),out:cake.Tensor({out_shape},"{dtype}",mode="output")):
    compute=lm.role(execution_groups=[0])
    m=lm.program(x,axis=0,dimension=0,tile=16)
    n=lm.program(x,axis=1,dimension=1,tile=32)
    with compute:
        v=lm.load(x[m,n],id="load_x")
        t=lm.transpose(v,id="transpose")
{tail}
'''


def kn_gemm_document(casted=False):
    d=cast_gemm();d['target']='gfx938'
    d.pop('residency',None)  # CUDA fixture register caps have no HCU enforcement.
    for buffer in d['buffers']:
        if not casted and buffer['name'] in ('a','b','a_tile','b_tile'):buffer['dtype']='fp32'
        if buffer['name'] in ('b_tile','b_tile_cast_0'):buffer['shape']=[32,16]
        if buffer['name'] in ('acc','c_tile'):buffer['shape']=[16,32]
        if buffer['name']=='bias_tile':buffer['shape']=[32]
    next(x for x in d['operations'] if x['id']=='dot')['parameters']['tile_shape']=[16,32,16]
    next(x for x in d['program_map']['axes'] if x['name']=='n_block')['tile']=32
    b=next(x for x in d['buffers'] if x['name']=='b');b['shape'].reverse()
    axis=next(x for x in d['program_map']['axes'] if x['name']=='n_block');axis['dimension']=1
    access=next(x for x in d['access_maps'] if x['operation']=='load_b');access['indices'].reverse()
    for name in ['b_tile','b_tile_cast_0']:
        next(x for x in d['buffers'] if x['name']==name)['shape'].reverse()
    dot=next(x for x in d['operations'] if x['id']=='dot');register=deepcopy(next(x for x in d['buffers'] if x['name']=='b_tile_cast_0'));register['name']='b_transposed';register['shape'].reverse();d['buffers'].append(register)
    trans=dict(id='transpose_b',kind='transpose',role=dot['role'],reads=['b_tile_cast_0'],writes=['b_transposed'],parameters={})
    d['operations'].insert(d['operations'].index(dot),trans);body=d['tile_loops'][0]['body'];body.insert(body.index('dot'),'transpose_b');dot['reads'][1]='b_transposed'
    if not casted:
        aliases={'a_tile_cast_0':'a_tile','b_tile_cast_0':'b_tile'}
        d['buffers']=[b for b in d['buffers'] if b['name'] not in aliases]
        d['operations']=[op for op in d['operations'] if op['id'] not in aliases]
        for op in d['operations']:
            op['reads']=[aliases.get(n,n) for n in op['reads']]
            if 'depends_on' in op:op['depends_on']=[n for n in op['depends_on'] if n not in aliases]
        d['tile_loops'][0]['body']=[n for n in d['tile_loops'][0]['body'] if n not in aliases]
    return d


class RegisterTransposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler=Compiler.load(ROOT)
        cls.target=Target.load(ROOT/'compiler/targets/gfx938.json')

    def test_non_square_tail_copy_and_double_transpose_store_each_element_once(self):
        for rows,columns in [(17,33),(1,65),(35,19)]:
            for double in [False,True]:
                with self.subTest(shape=(rows,columns),double=double):
                    doc=frontend.parse(source(rows,columns,double=double)).document
                    assessment=self.compiler.assess(doc);self.assertTrue(assessment.lowering_eligible,assessment.findings)
                    data=list(range(rows*columns));mem={'x':data,'out':[None]*len(data)}
                    execution=_execute(emit(Schedule.from_dict(doc),self.target),mem)
                    want=data if double else [data[i*columns+j] for j in range(columns) for i in range(rows)]
                    self.assertEqual(mem['out'],want)
                    self.assertEqual(mem['x'],data)
                    self.assertEqual(set(execution.stores.values()),{1})
                    self.assertEqual(work_bound(assessment.typed_schedule).flops,0)

    def test_declared_numeric_dtype_is_preserved_without_casts(self):
        for dtype in ['fp32','fp16','bf16','int32']:
            with self.subTest(dtype=dtype):
                doc=frontend.parse(source(dtype=dtype)).document;assessment=self.compiler.assess(doc)
                self.assertTrue(assessment.lowering_eligible,assessment.findings)
                self.assertEqual(next(b for b in doc['buffers'] if b['name']=='t')['dtype'],dtype)
                text=self.compiler.lower(assessment).source
                self.assertIn('t = tl.trans(v)',text)
                self.assertNotIn('t = tl.trans(v.to',text)

    def test_shape_dtype_rank_storage_arity_are_checked_by_value_typing(self):
        for change in ['shape','dtype','storage','arity','rank']:
            doc=frontend.parse(source()).document;op=next(x for x in doc['operations'] if x['id']=='transpose');dest=next(x for x in doc['buffers'] if x['name']=='t')
            if change=='shape':dest['shape']=[16,32]
            elif change=='dtype':dest['dtype']='bf16'
            elif change=='storage':dest['space']='global'
            elif change=='arity':op['reads'].append('v')
            else:next(x for x in doc['buffers'] if x['name']=='v')['shape']=[512]
            with self.subTest(change=change):
                self.assertIn('VALUE_OPERATION_TYPE',[f.code for f in self.compiler.assess(doc).findings])

    def test_no_implicit_other_target_or_arbitrary_permutation(self):
        doc=frontend.parse(source()).document;doc['target']='gfx1151'
        self.assertIn('TARGET_OPERATION_UNSUPPORTED',[f.code for f in self.compiler.assess(doc).findings])
        doc=frontend.parse(source()).document
        next(x for x in doc['operations'] if x['id']=='transpose')['parameters']={'dimensions':[1,0]}
        with self.assertRaises(ScheduleParseError):Schedule.from_dict(doc)

    def test_non_square_kn_operand_retains_every_k_tile_and_bias(self):
        doc=kn_gemm_document();schedule=Schedule.from_dict(doc);dot=schedule.operation('dot');loop=schedule.tile_loop('k_loop')
        self.assertTrue(schedule.mma_accumulates_over(dot,loop))
        a=self.compiler.assess(doc);self.assertTrue(a.lowering_eligible,a.findings)
        shapes={b['name']:b['shape'] for b in doc['buffers']};m,k=shapes['a'];n=shapes['b'][1]
        self.assertEqual(len({m,n,k}),3)
        x=[(i*3+t)%7-3 for i in range(m) for t in range(k)];b=[(t+2*j)%5-2 for t in range(k) for j in range(n)];bias=[j-2 for j in range(n)]
        mem={'a':x,'b':b,'bias':bias,'c':[None]*(m*n)};execution=_execute(emit(schedule,self.target),mem)
        self.assertEqual(mem['c'],[sum(x[i*k+t]*b[t*n+j] for t in range(k))+bias[j] for i in range(m) for j in range(n)])
        self.assertEqual(set(execution.stores.values()),{1})

    def test_cast_then_transpose_tracks_contraction_and_reversing_twice_restores_axis(self):
        doc=kn_gemm_document(casted=True);s=Schedule.from_dict(doc)
        self.assertEqual(s._staged_axis_filled_by('b_transposed',s.tile_loop('k_loop')),1)
        self.assertTrue(self.compiler.assess(doc).lowering_eligible)
        op=next(x for x in doc['operations'] if x['id']=='transpose_b')
        buf=deepcopy(next(x for x in doc['buffers'] if x['name']=='b_transposed'))
        buf['name']='b_restored';buf['shape'].reverse();doc['buffers'].append(buf)
        again=dict(id='transpose_again',kind='transpose',role=op['role'],reads=['b_transposed'],writes=['b_restored'],parameters={})
        doc['operations'].insert(doc['operations'].index(op)+1,again)
        body=doc['tile_loops'][0]['body'];body.insert(body.index('transpose_b')+1,'transpose_again')
        s=Schedule.from_dict(doc)
        self.assertEqual(s._staged_axis_filled_by('b_restored',s.tile_loop('k_loop')),0)

    def test_argmin_domain_tracks_transposed_candidate_axis(self):
        from dataclasses import replace
        from open_cake_ir.compiler.ir import OperationKind
        from tests.contracts.test_emit_triton import ArgminDomainTest, TARGET
        for k,carried in [(64,False),(130,True)]:
            with self.subTest(k=k,carried=carried):
                doc=ArgminDomainTest.scores(k,carried=carried)
                next(b for b in doc['buffers'] if b['name']=='scores')['shape']=[k,3]
                next(b for b in doc['buffers'] if b['name']=='tile')['shape']=[64,2]
                doc['access_maps'][0]['indices'].reverse()
                for axis in doc['program_map']['axes']:axis['dimension']=1-axis['dimension']
                if carried:doc['tile_loops'][0]['dimension']=0
                doc['buffers'].append(dict(name='transposed',space='register',dtype='fp32',shape=[2,64],mode='scratch'))
                trans=dict(id='transpose',kind='transpose',role='compute',reads=['tile'],writes=['transposed'],parameters={})
                doc['operations'].insert(1,trans);doc['operations'][2]['reads']=['transposed']
                doc['operations'][2]['depends_on']=['transpose']
                if carried:doc['tile_loops'][0]['body'].insert(1,'transpose')
                schedule=Schedule.from_dict(doc);self.assertIsNotNone(schedule.argmin_domain(schedule.operation('select')))
                # CPU semantic fixture only; no new sm_100a Target declaration.
                target=replace(TARGET,operation_kinds=TARGET.operation_kinds|{OperationKind.TRANSPOSE})
                values=[-1.0 if j==i+2 else 10.0 for j in range(k) for i in range(3)]
                mem={'scores':values,'indices':[None]*3}
                execution=_execute(emit(schedule,target),mem)
                self.assertEqual(mem['indices'],[2,3,4])
                self.assertEqual(set(execution.stores.values()),{1})

    def test_invalid_transpose_does_not_supply_contraction_proof(self):
        for change in ['shape','dtype','cycle','multiple_writers','storage']:
            doc=kn_gemm_document();op=next(x for x in doc['operations'] if x['id']=='transpose_b');buffer=next(b for b in doc['buffers'] if b['name']=='b_transposed')
            if change=='shape':buffer['shape']=[8,32]
            elif change=='dtype':buffer['dtype']='bf16'
            elif change=='cycle':op['reads']=['b_transposed']
            elif change=='storage':buffer['space']='global'
            else:extra=deepcopy(op);extra['id']='ambiguous';doc['operations'].append(extra)
            s=Schedule.from_dict(doc)
            with self.subTest(change=change):
                # The other valid operand alone still votes K; specifically the
                # malformed B operand must provide no axis proof of its own.
                self.assertIsNone(s._staged_axis_filled_by('b_transposed',s.tile_loop('k_loop')))


if __name__=='__main__':unittest.main()
