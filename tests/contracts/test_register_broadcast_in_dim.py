"""Explicit register broadcast for outer products and reduction validity predicates."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import Schedule, OperationKind, ScheduleParseError
from open_cake_ir.compiler.ir.value_ops import result_type
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

ROOT=Path(__file__).resolve().parents[2]


def outer_document():
    def buf(name,shape,mode,space='register',dtype='fp32'):
        return dict(name=name,shape=shape,mode=mode,space=space,dtype=dtype)
    return dict(schema_version=2,schedule_id='outer-broadcast-prototype',target='sm_103a',
        roles=[dict(name='compute',execution_groups=[0,1,2,3])],allocations=[],pipelines=[],barriers=[],
        buffers=[buf('a',[1,4],'input','global'),buf('b',[1,8],'input','global'),buf('out',[1,4,8],'output','global'),
                 buf('at',[4],'scratch'),buf('bt',[8],'scratch'),buf('aw',[4,8],'scratch'),buf('bw',[4,8],'scratch'),buf('product',[4,8],'scratch')],
        operations=[
            dict(id='la',kind='load',role='compute',reads=['a'],writes=['at'],parameters=dict(movement='global')),
            dict(id='lb',kind='load',role='compute',reads=['b'],writes=['bt'],parameters=dict(movement='global')),
            dict(id='ba',kind='broadcast_in_dim',role='compute',reads=['at'],writes=['aw'],depends_on=['la'],parameters=dict(dimensions=[0])),
            dict(id='bb',kind='broadcast_in_dim',role='compute',reads=['bt'],writes=['bw'],depends_on=['lb'],parameters=dict(dimensions=[1])),
            dict(id='mul',kind='elementwise',role='compute',reads=['aw','bw'],writes=['product'],depends_on=['ba','bb'],parameters=dict(op='mul')),
            dict(id='store',kind='store',role='compute',reads=['product'],writes=['out'],depends_on=['mul'],parameters=dict(coalesced=True))],
        program_map=dict(axes=[dict(name='row',axis=0,buffer='a',dimension=0,tile=1)]),tile_loops=[],
        access_maps=[
            dict(operation='la',buffer='a',boundary='mask_tiled_axes',indices=[dict(source='program',name='row'),dict(source='dimension',dimension=1)]),
            dict(operation='lb',buffer='b',boundary='mask_tiled_axes',indices=[dict(source='program',name='row'),dict(source='dimension',dimension=1)]),
            dict(operation='store',buffer='out',boundary='mask_tiled_axes',indices=[dict(source='program',name='row'),dict(source='dimension',dimension=1),dict(source='dimension',dimension=2)])],
        outputs=['out'],metadata={},lowering=dict(backend='triton',entry_point='outer_prototype'))


def prototype_target():
    doc=json.loads((ROOT/'compiler/targets/sm_103a.json').read_text())
    doc['operation_kinds'].append('broadcast_in_dim')
    # A test-local declaration exercises semantic/lowering contracts; it is not
    # new device qualification or a mutation of the committed Target.
    return Target.from_dict(doc)


class RegisterBroadcast(unittest.TestCase):
    def test_outer_product_keeps_explicit_axis_ownership_and_integer_work_accounting(self):
        schedule=Schedule.from_dict(outer_document());target=prototype_target()
        errors=[f for f in verify(schedule,target) if f.blocks_acceptance or f.blocks_lowering]
        self.assertEqual(errors,[])
        self.assertFalse(preflight(schedule,target))
        source=emit(schedule,target).source
        self.assertIn('aw = tl.broadcast_to(at[:, None], (4, 8))',source)
        self.assertIn('bw = tl.broadcast_to(bt[None, :], (4, 8))',source)
        self.assertNotIn('float32',source.split('# CAKE_OP:ba')[1].split('# CAKE_OP:bb')[0])
        from tests.contracts.test_triton_loop_scopes import _execute
        memory=dict(a=[1.,-2.,0.,3.],b=list(range(8)),out=[None]*32)
        _execute(emit(schedule,target),memory)
        self.assertEqual(memory['out'],[a*b for a in memory['a'] for b in memory['b']])
        self.assertTrue(work_bound(schedule).flops_exact)

    def test_bias_after_masked_load_does_not_pollute_tail_reduction(self):
        from tests.contracts.test_triton_loop_scopes import _execute
        for rows in [1, 3, 5, 7]:
            d=outer_document();d['program_map']['axes']=[
                dict(name='batch',axis=1,buffer='a',dimension=0,tile=1),
                dict(name='rows',axis=0,buffer='a',dimension=1,tile=8)]
            for b in d['buffers']:
                b['shape']={'a':[1,rows,8],'b':[1,8],'out':[1,8],
                            'at':[8,8],'bt':[8],'aw':[8,8],'bw':[8,8],'product':[8,8]}[b['name']]
            d['buffers'] += [dict(name=n,space='register',dtype=t,shape=sh,mode='scratch')
                            for n,t,sh in [('index','int32',[8]),('valid','int32',[8]),
                                           ('mask','int32',[8,8]),('clean','fp32',[8,8]),('total','fp32',[8])]]
            load_a=d['operations'][0];load_b=d['operations'][1];bias=d['operations'][3]
            add=d['operations'][4];add['parameters']['op']='add';add['reads']=['at','bw'];add['depends_on']=['la','bb']
            store=d['operations'][5];store['reads']=['total'];store['depends_on']=['fold']
            d['operations']=[load_a,load_b,bias,add,
                dict(id='coord',kind='coordinate',role='compute',reads=[],writes=['index'],parameters=dict(source='program_tile',name='rows')),
                dict(id='cmp',kind='compare',role='compute',reads=['index'],writes=['valid'],depends_on=['coord'],parameters=dict(op='lt',scalar=rows)),
                dict(id='mask_wide',kind='broadcast_in_dim',role='compute',reads=['valid'],writes=['mask'],depends_on=['cmp'],parameters=dict(dimensions=[0])),
                dict(id='neutral',kind='select',role='compute',reads=['mask','product'],writes=['clean'],depends_on=['mask_wide','mul'],parameters=dict(false_value=0)),
                dict(id='fold',kind='reduce',role='compute',reads=['clean'],writes=['total'],depends_on=['neutral'],parameters=dict(op='sum',axis=0,scope='cta',across_loop=False)),store]
            d['buffers']=[b for b in d['buffers'] if b['name']!='aw']
            d['access_maps']=[
                dict(operation='la',buffer='a',boundary='mask_tiled_axes',indices=[dict(source='program',name='batch'),dict(source='program_tile',name='rows'),dict(source='dimension',dimension=2)]),
                dict(operation='lb',buffer='b',boundary='mask_tiled_axes',indices=[dict(source='program',name='batch'),dict(source='dimension',dimension=1)]),
                dict(operation='store',buffer='out',boundary='mask_tiled_axes',indices=[dict(source='program',name='batch'),dict(source='dimension',dimension=1)])]
            schedule=Schedule.from_dict(d);errors=[f for f in verify(schedule,prototype_target()) if f.blocks_acceptance]
            self.assertEqual(errors,[])
            memories=dict(a=[0.]*(rows*8),b=[2.]*8,out=[None]*8)
            _execute(emit(schedule,prototype_target()),memories)
            self.assertEqual(memories['out'],[2.*rows]*8)

    def test_validity_predicate_replication_preserves_int32_and_select_shape(self):
        d=outer_document()
        d['buffers'][3]['dtype']='int32';d['buffers'][5]['dtype']='int32'
        d['buffers'][0]['dtype']='int32'
        schedule=Schedule.from_dict(d)
        kind,shape=result_type(schedule,schedule.operation('ba'))
        self.assertEqual(kind.value,'int32');self.assertEqual(shape,(4,8))

    def test_invalid_axes_extents_dtype_and_effects_refuse_by_value_typing(self):
        variants=[]
        d=outer_document();d['operations'][2]['parameters']['dimensions']=[1];variants.append(d)
        d=outer_document();d['operations'][2]['parameters']['dimensions']=[2];variants.append(d)
        d=outer_document();d['operations'][2]['parameters']['dimensions']=[0,1];variants.append(d)
        d=outer_document();d['buffers'][5]['dtype']='bf16';variants.append(d)
        d=outer_document();d['buffers'][5]['space']='global';variants.append(d)
        for d in variants:
            with self.subTest(document=d):
                codes={f.code for f in verify(Schedule.from_dict(d),prototype_target())}
                self.assertIn('VALUE_OPERATION_TYPE',codes)
        d=outer_document();d['operations'][2]['parameters']['dimensions']=[True]
        with self.assertRaises(ScheduleParseError):Schedule.from_dict(d)

    def test_target_admission_is_explicit_and_old_target_is_not_silently_widened(self):
        target=Target.load(ROOT/'compiler/targets/sm_103a.json')
        codes={f.code for f in verify(Schedule.from_dict(outer_document()),target)}
        self.assertIn('TARGET_OPERATION_UNSUPPORTED',codes)
