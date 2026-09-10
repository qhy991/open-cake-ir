"""Composition semantics and adversarial pass boundaries; no GPU qualification."""
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import re
import struct
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Target, frontend
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule
from tests.contracts.test_triton_loop_scopes import _TL, _Tile, _execute

ROOT = Path(__file__).resolve().parents[2]


def stage(name, dtype='bf16', target='sm_100a'):
    text = (ROOT/f'examples/python/epilogue_{name}.py').read_text()
    text = text.replace('bf16', dtype).replace('sm_100a', target)
    if dtype == 'fp32':
        pattern = r'^\s+(\w+) = lm.cast\((\w+), to="fp32", id="[^"]+"\)\n'
        aliases = re.findall(pattern, text, re.MULTILINE)
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
        for destination, source in aliases:
            text = re.sub(r'\b' + destination + r'\b', source, text)
    return frontend.parse(text).document


def rounded(value, dtype):
    if dtype == 'fp16':
        return struct.unpack('<e', struct.pack('<e', value))[0]
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    if dtype == 'bf16':
        bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def execute(document, inputs):
    """Run the emitted kernel AST using existing bounds-checked CPU memory helpers.

    Add only cast/exp/div for this example. It models the explicit rounding seam,
    not device transcendental approximations or compiler instruction contraction.
    """
    memory = deepcopy(inputs)
    for b in document['buffers']:
        if b['space'] == 'global' and b['mode'] == 'output':
            memory[b['name']] = [float('nan')]*math.prod(b['shape'])
    def cast(tile, dtype):
        return _Tile(tile.shape, [rounded(v, 'fp32' if dtype is _TL.float32 else dtype) for v in tile.values])
    with ExitStack() as stack:
        stack.enter_context(patch.object(_Tile, 'to', cast))
        stack.enter_context(patch.object(_Tile, '__truediv__', lambda a,b:a.binary(b,lambda x,y:x/y), create=True))
        stack.enter_context(patch.object(_Tile, '__rtruediv__', lambda a,b:a.binary(b,lambda x,y:y/x), create=True))
        stack.enter_context(patch.object(_TL, 'bfloat16', 'bf16', create=True))
        stack.enter_context(patch.object(_TL, 'float16', 'fp16', create=True))
        stack.enter_context(patch.object(_TL, 'exp', staticmethod(lambda a:_Tile(a.shape,[math.exp(v) for v in a.values])),create=True))
        trace = _execute(emit(Schedule.from_dict(document), Target.load(ROOT/f"compiler/targets/{document['target']}.json")),memory)
    return {name:memory[name] for name in document['outputs']}, trace


class EpilogueFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Author checks run against the explicit working draft, never a stale release.
        cls.compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')

    def fuse(self,p=None,e=None,**kwargs):
        return self.compiler.fuse_pointwise_epilogue(stage('producer') if p is None else p,
            stage('consumer') if e is None else e,private_intermediate=kwargs.get('private_intermediate','mid'),
            schedule_id=kwargs.get('schedule_id','fused_test'),entry_point='fused_test')

    def test_real_composition_removes_materialization_and_preserves_rounded_value(self):
        p,e=stage('producer'),stage('consumer')
        before=deepcopy((p,e));r=self.fuse(p,e)
        self.assertTrue(r.applied,r.message)
        self.assertEqual((p,e),before)
        d=r.schedule
        self.assertEqual(d['metadata'],{})
        self.assertNotIn('mid',[b['name'] for b in d['buffers']])
        self.assertNotIn('store_mid',[op['id'] for op in d['operations']])
        self.assertEqual(sum(op['kind']=='load' for op in d['operations']),3)
        self.assertEqual(sum(op['kind']=='store' for op in d['operations']),1)
        cast=next(op for op in d['operations'] if op['id']=='round_intermediate')
        self.assertEqual(cast['parameters']['to'],'bf16')
        source=self.compiler.lower(r.assessment).source
        self.assertIn('rounded = shifted.to(tl.bfloat16)',source)
        self.assertIn('= rounded.to(tl.float32)',source)
        self.assertNotIn('# CAKE_OP:store_mid',source)
        # Returned JSON is a projection; modifying it cannot mutate the assessed result.
        d['outputs'].clear()
        self.assertTrue(r.schedule['outputs'])

    def test_emitted_before_after_agree_and_skipping_the_round_is_detected(self):
        for dtype in ('bf16','fp16'):
            p,e=stage('producer',dtype),stage('consumer',dtype)
            r=self.fuse(p,e);self.assertTrue(r.applied,r.message)
            rng=random.Random(9941)
            cases=[{'a':[rounded(rng.uniform(-.5,.5),dtype) for _ in range(16)],
                    'b':[rounded(rng.uniform(-.5,.5),dtype) for _ in range(64)],
                    'bias':[rounded(rng.uniform(-.5,.5),'fp32') for _ in range(8)]},
                   {'a':[0.0]*16,'b':[0.0]*64,
                    'bias':[1.0+2**(-8 if dtype=='bf16' else -11)]*8}]
            for inputs in cases:
                intermediate,_=execute(p,inputs)
                expected,_=execute(e,intermediate)
                observed,trace=execute(r.schedule,inputs)
                self.assertEqual(list(observed.values()),list(expected.values()))
                self.assertEqual(sum(trace.stores.values()),16)
            mutant=r.schedule
            consumer_cast=next(op for op in mutant['operations'] if op['kind']=='cast' and op['reads']==['rounded'])
            consumer_cast['reads']=['shifted']
            actual,_=execute(mutant,cases[-1])
            expected,_=execute(e,execute(p,cases[-1])[0])
            self.assertNotEqual(list(actual.values()),list(expected.values()))

    def assert_refused_valid_pair(self,p,e,reason):
        for document in (p,e):
            assessment=self.compiler.assess(document)
            self.assertTrue(assessment.lowering_eligible,assessment.findings)
        before=deepcopy((p,e));r=self.fuse(p,e)
        self.assertFalse(r.applied)
        self.assertIsNone(r.schedule)
        self.assertEqual(r.reason,reason,r.message)
        self.assertEqual((p,e),before)

    def test_fp32_boundary_is_refused_even_when_both_stages_lower(self):
        self.assert_refused_valid_pair(stage('producer','fp32'),stage('consumer','fp32'),'rounding_boundary')

    def test_target_and_execution_controls_are_not_silently_changed(self):
        self.assert_refused_valid_pair(stage('producer'),stage('consumer',target='sm_103a'),'target_route')
        e=stage('consumer');e['roles'][0]['warps']=[0]
        self.assert_refused_valid_pair(stage('producer'),e,'execution_controls')
        e=stage('consumer');e['residency']={'registers_per_thread':64}
        self.assert_refused_valid_pair(stage('producer'),e,'execution_controls')
        e=stage('consumer');e['program_map']['axes'][0]['axis']=1
        self.assert_refused_valid_pair(stage('producer'),e,'row_ownership')

    def test_extra_epilogue_input_and_shape_changes_are_refused(self):
        e=stage('consumer');e['buffers'].append({'name':'extra','space':'global','dtype':'bf16','shape':[2,8],'mode':'input'})
        self.assert_refused_valid_pair(stage('producer'),e,'composition_boundary')
        e=frontend.parse((ROOT/'examples/python/epilogue_consumer.py').read_text().replace('(2, 8)','(2, 4)')).document
        self.assert_refused_valid_pair(stage('producer'),e,'intermediate_abi')

    def test_private_composition_and_fresh_identity_are_explicit(self):
        self.assertEqual(self.fuse(private_intermediate='another_tensor').reason,'composition_boundary')
        self.assertEqual(self.fuse(schedule_id='epilogue_producer').reason,'result_identity')
        p,e=stage('producer'),stage('consumer')
        p['metadata']={'workload_contract_sha256':'1'*64};e['metadata']={'workload_contract_sha256':'2'*64}
        self.assertEqual(self.fuse(p,e).schedule['metadata'],{})

    def test_malformed_program_is_not_repaired_or_applied(self):
        p=stage('producer');p['operations'][-1]['reads']=['missing']
        before=deepcopy(p);r=self.fuse(p)
        self.assertFalse(r.applied);self.assertEqual(r.reason,'input_refused');self.assertEqual(p,before)

    def test_names_are_alpha_renamed_and_the_new_assessment_checks_resource_cost(self):
        p=stage('producer');p['buffers'].append({'name':'ep_out','space':'register','dtype':'fp32','shape':[8],'mode':'scratch'})
        r=self.fuse(p);self.assertTrue(r.applied,r.message)
        self.assertEqual(r.schedule['outputs'],['ep_ep_out'])
        self.assertEqual(r.assessment.analysis['operation_counts']['store'],1)
        self.assertTrue(r.assessment.lowering_eligible)


    def test_exported_second_output_cannot_be_eliminated(self):
        p=stage('producer')
        b=deepcopy(next(b for b in p['buffers'] if b['name']=='mid'));b['name']='also_visible';p['buffers'].append(b)
        store=deepcopy(p['operations'][-1]);store.update(id='store_visible',writes=['also_visible']);p['operations'].append(store)
        access=deepcopy(p['access_maps'][-1]);access.update(operation='store_visible',buffer='also_visible');p['access_maps'].append(access)
        p['outputs'].append('also_visible')
        self.assert_refused_valid_pair(p,stage('consumer'),'composition_boundary')

    def test_nonpointwise_consumer_is_refused_by_the_pass_domain(self):
        text=(ROOT/'examples/python/epilogue_consumer.py').read_text()
        text=text.replace('negated = values * -1.0',
            'total = lm.reduce(values, op="sum", axis=0, scope="cta", across_loop=False, id="sum_values")\n        negated = values * total')
        self.assert_refused_valid_pair(stage('producer'),frontend.parse(text).document,'operation_domain')
