"""Public native lowering contracts. Source/static evidence, never GPU qualification."""
from __future__ import annotations
import copy
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from open_cake_ir.compiler import Compiler, CompilerError, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda import emit, preflight
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.verifier import verify
from open_cake_ir.compiler import profile_envelope
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.performance.work import work_bound

ROOT=Path(__file__).resolve().parents[2]
EXAMPLES=ROOT/'examples/schedules/native'


def document(name='gemm-bias'):
    return json.loads((EXAMPLES/(name+'.json')).read_text())


def one_tile_document():
    d=document()
    d['tile_loops']=[]
    d['pipelines'][0]['stages']=1
    d['allocations'][0]['size_bytes']=(128+64)*64*2
    for b in d['buffers']:
        if b['name'] in ('a','b'):b['shape'][1]=64
        if b['space']=='shared':b['stages']=1
        if b['name']=='b_stage':b['byte_offset']=128*64*2
    for access in d['access_maps'][:2]:access['indices'][1]={'source':'dimension','dimension':1}
    return d


def nested_rows_document():
    d=document('kmeans')
    d['program_map']['axes']=[axis for axis in d['program_map']['axes'] if axis['name']!='rows']
    d['tile_loops'].insert(0,dict(name='rows_loop',iterator='rr',buffer='tokens',dimension=1,
        tile=128,body=['columns','store'],range_options=dict(d['tile_loops'][0]['range_options'])))
    for access in d['access_maps']:
        for component in access['indices']:
            if component.get('name')=='rows':component.update(source='loop_tile',name='rr')
    return d


def resident_candidates_document(count=4):
    d = document('kmeans')
    reduction = next(op for op in d['operations'] if op['kind'] == 'reduce_argmin')
    loop = next(loop for loop in d['tile_loops'] if reduction['id'] in loop['body'])
    d['tile_loops'].remove(loop)
    d['program_map']['axes'].insert(0, dict(name='candidate_coordinates', axis=2,
        buffer=loop['buffer'], dimension=loop['dimension'], tile=loop['tile']))
    reduction['parameters']['across_loop'] = False
    for access in d['access_maps']:
        for position, component in enumerate(access['indices']):
            if component.get('source') == 'loop_tile' and component.get('name') == loop['iterator']:
                component.update(source='program_tile', name='candidate_coordinates')
                next(b for b in d['buffers'] if b['name'] == access['buffer'])['shape'][position] = count
    return d


def source_loop_scopes(source):
    """Observe lexical loop ownership of emitted statements, without running CUDA.

    Only emitted block delimiters are relevant: braces in inline asm or one-line
    expressions are not blocks. This checks the publication/initialization scope,
    independently of the emitter's Python traversal or intended phase counters.
    """
    stack=[]
    for line in source.splitlines():
        text=line.strip()
        if text in ('}', '};'):
            stack.pop()
        yield text, tuple(item for item in stack if item is not None)
        if text.endswith('{'):
            loop=re.match(r'for \(int (\w+)\s*=\s*0;\s*\1\s*<\s*(\d+);',text)
            stack.append((loop.group(1),int(loop.group(2))) if loop else None)
    if stack:raise AssertionError('unbalanced generated block structure')


class NativeCudaContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')

    def lower(self,d):
        a=self.compiler.assess(d)
        self.assertTrue(a.lowering_eligible,[(f.code,f.path,f.message) for f in a.findings if f.blocks_lowering])
        return self.compiler.lower(a)

    def refuses(self,d,code):
        a=self.compiler.assess(d)
        self.assertFalse(a.lowering_eligible)
        self.assertIn(code,[f.code for f in a.findings if f.blocks_lowering],[(f.code,f.path) for f in a.findings])
        with self.assertRaises(CompilerError):self.compiler.lower(a)
        f=next(f for f in a.findings if f.code==code)
        self.assertTrue(f.path)
        return f

    def test_distinct_uses_and_two_mmas_pass_public_path(self):
        for name in ('gemm-bias','gemm-bias-k65-tail','two-mma','kmeans'):
            with self.subTest(name=name):
                d=document(name);l=self.lower(d)
                self.assertTrue(l.generated)
                self.assertEqual(l.toolchain_requirements['compiler'],'nvcc')
                self.assertEqual(l.toolchain_requirements['source_language'],'cuda_cpp')
                self.assertTrue(all(op['id'] in l.source_map for op in d['operations']))
                self.assertIn('tcgen05.mma.cta_group::1.kind::f16',l.source)
                self.assertIn('tcgen05.alloc',l.source)
                self.assertIn('tcgen05.dealloc',l.source)
                self.assertNotIn('import cutlass',l.source)
                self.assertNotIn('triton',l.source)

    def test_operation_names_do_not_dispatch_kernels(self):
        d=document();d['schedule_id']='unrelated-operator'
        for i,op in enumerate(d['operations']):
            previous=op['id'];op['id']=f'node_{i}'
            for consumer in d['operations']:
                consumer['depends_on']=[op['id'] if x==previous else x for x in consumer.get('depends_on',[])]
            for loop in d['tile_loops']:loop['body']=[op['id'] if x==previous else x for x in loop['body']]
            for access in d['access_maps']:
                if access['operation']==previous:access['operation']=op['id']
        self.lower(d)

    def test_both_targets_preserve_exact_compilation_and_device_admission(self):
        for target,macro in [('sm_100a','1000'),('sm_103a','1030')]:
            d=document();d['target']=target;l=self.lower(d)
            self.assertIn('--gpu-architecture='+target.replace('sm_','compute_'),l.toolchain_requirements['nvcc_flags'])
            self.assertIn('--gpu-code='+target,l.toolchain_requirements['nvcc_flags'])
            self.assertFalse(any(flag.startswith('-arch=') for flag in l.toolchain_requirements['nvcc_flags']))
            self.assertIn('__CUDA_ARCH__ != '+macro,l.source)
            self.assertIn('prop.minor != '+('0' if target=='sm_100a' else '3'),l.source)
        d=document();d['target']='sm_90a';self.refuses(d,'TARGET_UNSUPPORTED')

    def test_tail_uses_declared_device_staging_without_host_padding(self):
        l=self.lower(document('gemm-bias-k65-tail'))
        self.assertNotIn('cuTensorMapEncodeTiled(&h->',l.source)
        self.assertIn('const int swizzled',l.source)
        self.assertIn('cake_arrive(bar0+stage)',l.source)
        self.assertIn('< 65',l.source)
        self.assertNotIn('cudaMalloc',l.source)
        self.assertNotIn('cudaMemcpy',l.source)
        self.assertNotIn('cudaDeviceSynchronize',l.source)
        self.assertNotIn('cake_mma(',l.source.split('_create(')[-1])

    def test_tma_tail_stride_is_not_silently_padded(self):
        d=document('gemm-bias-k65-tail')
        for op in d['operations'][:2]:
            op['parameters'].update(movement='tma',descriptor_box=next(b['shape'] for b in d['buffers'] if b['name']==op['writes'][0]))
        self.refuses(d,'NATIVE_TMA_DESCRIPTOR')

    def test_pipeline_roles_overlap_and_release_after_all_mmas(self):
        l=self.lower(document('two-mma'));src=l.source
        self.assertEqual(src.count('cake_mma((*'),2)
        self.assertLess(src.index('// CAKE_OP: mma2'),src.index('cake_commit(free0+stage)'))
        self.assertIn('it0 >= 2',src)
        self.assertIn('(it0/2-1)&1',src)
        self.assertIn('(it0/2)&1',src)
        # Both independent roles share only the ring protocol inside their loops.
        start=src.index('if (warp >= 5');stop=src.index('cake_inval(bar0+stage)',start)
        self.assertEqual(src[start:stop].count('__syncthreads();'),1)
        self.assertIn('it0 != 0 || atom != 0',src)
        self.assertIn('135267472u',src)  # PTX descriptor M128/N64, BF16 A/B, FP32 D.

    def test_stage_storage_changes_real_ring_behavior(self):
        d=document();d['pipelines'][0]['stages']=3
        for b in d['buffers']:
            if b['space']=='shared':b['stages']=3
            if b['name']=='b_stage':b['byte_offset']=128*64*2*3
        d['allocations'][0]['size_bytes']=(128+64)*64*2*3
        d['tile_loops'][0]['range_options']['num_stages']=3
        l=self.lower(d)
        self.assertIn('it0 % 3',l.source)
        self.assertIn('it0 >= 3',l.source)
        self.assertGreater(l.toolchain_requirements['dynamic_shared_bytes'],self.lower(document()).toolchain_requirements['dynamic_shared_bytes'])

    def test_tile_and_aligned_role_variants_change_emission(self):
        d=document()
        d['roles'][0]['warps']=[4,5,6,7]
        d['roles'][1]['warps']=[8];d['roles'][2]['warps']=[9]
        d['allocations'][0]['size_bytes']=(128+128)*32*2*2
        d['allocations'][1].update(size_bytes=128*512,tensor_columns=128)
        for b in d['buffers']:
            if b['space']=='shared':b.update(shape=[128,32],swizzle='swizzle_64b')
            if b['name']=='b_stage':b['byte_offset']=128*32*2*2
            if b['name'] in ('acc','dot','sum'):b['shape']=[128,128]
            if b['name']=='bias_tile':b['shape']=[128]
        for op in d['operations'][:2]:op['parameters']['descriptor_box']=[128,32]
        d['operations'][2]['parameters']['tile_shape']=[128,128,32]
        d['operations'][2]['parameters']['instruction']['shape']=[128,128,16]
        d['program_map']['axes'][1]['tile']=128
        d['tile_loops'][0]['tile']=32
        l=self.lower(d)
        self.assertIn('warp >= 9',l.source)
        self.assertIn('int(threadIdx.x) - 128',l.source)
        self.assertIn('CU_TENSOR_MAP_SWIZZLE_64B',l.source)
        self.assertIn('atom<2',l.source)
        self.assertEqual(l.toolchain_requirements['block'],[320,1,1])

    def test_tmem_columns_and_smem_bounds_are_not_rounded(self):
        d=document();d['allocations'][1].update(size_bytes=48*512,tensor_columns=48)
        self.assertFalse(self.compiler.assess(d).lowering_eligible)
        t=Target.load(ROOT/'compiler/targets/sm_100a.json')
        self.assertIn('NATIVE_TMEM_ALLOCATION',[f.code for f in preflight(Schedule.from_dict(d),t)])
        d=document();d['allocations'][0]['size_bytes']-=16
        a=self.compiler.assess(d)
        self.assertFalse(a.lowering_eligible)
        self.assertTrue(any('BOUND' in f.code or 'ALLOCATION' in f.code for f in a.findings if f.blocks_lowering))

    def test_manual_and_tma_producers_share_one_completion_protocol(self):
        d=document();d['operations'][0]['parameters']={'movement':'global'}
        l=self.lower(d)
        self.assertIn('cake_arrive(bar0+stage)',l.source)
        self.assertIn('cake_expect(bar0+stage',l.source)
        self.assertIn('cake_wait(bar0+stage',l.source)

    def test_tmem_transfer_is_not_reported_as_global_traffic(self):
        s=Schedule.from_dict(document())
        profile=profile_envelope(s,Target.load(ROOT/'compiler/targets/sm_100a.json')).lowering
        self.assertEqual(profile['global_load_operations'],3)
        self.assertNotIn('read_acc',[x.operation for x in work_bound(s).scheduled_transfers])

    def test_tmem_atom_and_shape_contracts(self):
        d=document();d['operations'][3]['parameters']['source_atom']['repetition']=3
        self.refuses(d,'TMEM_LOAD_ATOM')
        d=document();next(b for b in d['buffers'] if b['name']=='dot')['shape']=[64,64]
        self.refuses(d,'TMEM_LOAD_CONTRACT')

    def test_tmem_physical_warp_alignment_is_required(self):
        d=document();d['roles'][0]['warps']=[1,2,3,4];d['roles'][1]['warps']=[5];d['roles'][2]['warps']=[6]
        self.refuses(d,'NATIVE_ROLE_ALIGNMENT')

    def test_noncoalesced_mapping_must_be_explicit(self):
        d=document();d['operations'][-1]['parameters']['coalesced']=True
        self.refuses(d,'NATIVE_STORE_COALESCING')

    def test_each_range_control_has_a_local_refusal(self):
        for field,value in [('num_stages',3),('loop_unroll_factor',2),('flatten',True),('warp_specialize',False),('disallow_acc_multi_buffer',False),('disable_licm',True)]:
            d=document();d['tile_loops'][0]['range_options'][field]=value
            f=self.refuses(d,'NATIVE_RANGE_OPTION_UNSUPPORTED')
            self.assertIn(field,f.path)

    def test_invalid_offset_alignment_is_local(self):
        d=document();next(b for b in d['buffers'] if b['name']=='b_stage')['byte_offset']+=16
        d['allocations'][0]['size_bytes']+=1024
        self.refuses(d,'NATIVE_SMEM_ALIGNMENT')

    def test_barrier_count_must_match_stage_producers(self):
        d=document();d['barriers'][0]['count']=1
        self.refuses(d,'NATIVE_PIPELINE_BARRIER')

    def test_stage_count_mismatch_is_not_ignored(self):
        d=document();next(b for b in d['buffers'] if b['name']=='a_stage')['stages']=1
        self.refuses(d,'NATIVE_PIPELINE_STAGES')

    def test_second_mma_contract_is_checked(self):
        d=document('two-mma');d['operations'][3]['parameters']['instruction']['operand_major']=['mn','k']
        self.refuses(d,'NATIVE_MMA_CONTRACT')

    def test_multiple_mma_issuers_cannot_share_a_release(self):
        d=document('two-mma');d['roles'].append({'name':'second','warps':[6]})
        d['operations'][3]['role']='second'
        d['barriers'][0]['consumers'].append('second');d['barriers'][2]['producers']=['second']
        self.refuses(d,'NATIVE_PIPELINE_ROLES')

    def test_mma_completion_is_required_even_before_same_role_shortcut(self):
        d=document();d['operations'][3]['waits']=[]
        s=Schedule.from_dict(d);target=Target.load(ROOT/'compiler/targets/sm_100a.json')
        self.assertIn('NATIVE_TMEM_COMPLETION',[f.code for f in preflight(s,target)])
        self.assertFalse(self.compiler.assess(d).lowering_eligible)

    def test_register_values_do_not_cross_roles(self):
        d=document();d['roles'].append({'name':'other','warps':[8,9,10,11]});d['operations'][-1]['role']='other'
        self.assertFalse(self.compiler.assess(d).lowering_eligible)
        s=Schedule.from_dict(d);t=Target.load(ROOT/'compiler/targets/sm_100a.json')
        self.assertIn('NATIVE_REGISTER_ROLE_OWNERSHIP',[f.code for f in preflight(s,t)])

    def test_arithmetic_global_access_requires_an_explicit_register_load(self):
        # The original composition loads bias into registers before adding it.
        self.lower(document())
        d=document()
        d['buffers'].append(dict(name='arithmetic_input',space='global',dtype='fp32',
                                 shape=[128,64],mode='input'))
        op=next(op for op in d['operations'] if op['id']=='add_bias')
        op['reads']=['dot','arithmetic_input'];op['parameters']={'op':'add'}
        d['access_maps'].append(dict(operation='add_bias',buffer='arithmetic_input',
            boundary='mask_tiled_axes',indices=[dict(source='dimension',dimension=i) for i in range(2)]))
        s=Schedule.from_dict(d);target=Target.load(ROOT/'compiler/targets/sm_100a.json')
        self.assertFalse([f for f in verify(s,target) if f.blocks_lowering or f.blocks_acceptance])
        self.refuses(d,'NATIVE_ARITHMETIC_STORAGE')
        self.assertIn('NATIVE_ARITHMETIC_STORAGE',[f.code for f in preflight(s,target)])
        with self.assertRaises(EmitError):emit(s,target)

    def test_scalar_literals_are_valid_finite_fp32_in_public_and_direct_lowering(self):
        target=Target.load(ROOT/'compiler/targets/sm_100a.json')
        for value in (0,2,-2,0.25,-0.0,3.4028234663852886e38):
            d=document();op=next(op for op in d['operations'] if op['id']=='add_bias')
            op['reads']=['dot'];op['parameters']={'op':'mul','scalar':value}
            with self.subTest(value=value):
                source=self.lower(d).source
                self.assertIn(f'__fmul_rn(b7[col],{float(value)!r}f)',source)
                self.assertIn(f'{float(value)!r}f',emit(Schedule.from_dict(d),target).source)
        for value in (1e300,-1e300,3.4028236e38):
            d=document();op=next(op for op in d['operations'] if op['id']=='add_bias')
            op['reads']=['dot'];op['parameters']={'op':'mul','scalar':value}
            with self.subTest(value=value):
                s=Schedule.from_dict(d)
                self.assertFalse([f for f in verify(s,target) if f.blocks_lowering or f.blocks_acceptance])
                self.refuses(d,'NATIVE_SCALAR_FINITE')
                self.assertIn('NATIVE_SCALAR_FINITE',[f.code for f in preflight(s,target)])
                with self.assertRaises(EmitError):emit(s,target)

    def test_missing_output_writer_fails_public_contract(self):
        d=document();d['operations'].pop();d['access_maps'].pop()
        self.refuses(d,'OUTPUT_UNWRITTEN')

    def test_tmem_read_cannot_escape_output_tile_lifetime(self):
        d=document();d['tile_loops'].insert(0,dict(name='outer',iterator='rr',buffer='a',dimension=0,tile=128,body=['contraction'],range_options={**d['tile_loops'][0]['range_options'],'num_stages':1}))
        self.refuses(d,'NATIVE_TMEM_LIFETIME')

    def test_source_map_labels_cannot_splice_generated_code(self):
        d=document();d['allocations'][0]['name']='operands\\'
        for b in d['buffers']:
            if b.get('allocation')=='operands':b['allocation']='operands\\'
        self.refuses(d,'NATIVE_NAME_UNSUPPORTED')

    def test_unknown_references_return_public_findings(self):
        for field in ('access_buffer','role','read','global_read'):
            d=document()
            if field=='access_buffer':d['access_maps'][0]['buffer']='undeclared_buffer'
            elif field=='role':d['operations'][0]['role']='undeclared_role'
            else:
                d['operations'][0]['reads']=['undeclared_input']
                if field=='global_read':d['operations'][0]['parameters']={'movement':'global'}
            with self.subTest(field=field):
                a=self.compiler.assess(d)
                self.assertFalse(a.lowering_eligible)
                self.assertTrue(any(f.blocks_lowering and f.path and 'UNKNOWN' in f.code for f in a.findings))
                with self.assertRaises(CompilerError):self.compiler.lower(a)

    def test_completion_has_one_observed_phase_per_output_tile(self):
        many=document('two-mma')
        for b in many['buffers']:
            if b['name'] in ('a','b'):b['shape'][1]=576
            if b['space']=='shared':b['stages']=3
            if b['name']=='b_stage':b['byte_offset']=128*64*2*3
        many['pipelines'][0]['stages']=3
        many['tile_loops'][0]['range_options']['num_stages']=3
        many['allocations'][0]['size_bytes']=(128+64)*64*2*3
        for d in (one_tile_document(),document(),document('gemm-bias-k65-tail'),many,document('kmeans'),nested_rows_document()):
            with self.subTest(schedule=d['schedule_id'],loops=len(d['tile_loops'])):
                lower=self.lower(d)
                scopes=list(source_loop_scopes(lower.source))
                for index,barrier in enumerate(d['barriers']):
                    if barrier.get('pipeline'):continue
                    name=f'bar{index}'
                    events=[]
                    for text,loops in scopes:
                        match=re.fullmatch(r'cake_(init|commit|wait|inval)\('+name+r'(?:, (\d+))?\);',text)
                        if match:events.append((match.group(1),match.group(2),loops))
                    self.assertEqual([kind for kind,_,_ in events],['init','commit','wait','inval'])
                    self.assertEqual(events[0][1],'1')
                    self.assertEqual(events[2][1],'0')
                    # One publication, one observed phase, one invalidation in each
                    # dynamic output tile, independent of K trips or ring wraps.
                    self.assertTrue(all(event[2]==events[0][2] for event in events))
                    k_loops={f'it{i}' for i,loop in enumerate(d['tile_loops']) if 'mma' in loop['body']}
                    self.assertFalse(k_loops & {var for var,_ in events[1][2]})

    def test_one_k_tile_uses_the_canonical_loop_free_schedule(self):
        d=one_tile_document();lower=self.lower(d)
        self.assertFalse(d['tile_loops'])
        self.assertIn('once0<1',lower.source)
        self.assertIn('cake_wait(free0+0, 0);',lower.source)
        d['pipelines'][0]['stages']=2
        self.refuses(d,'NATIVE_PIPELINE_STAGES')

    def test_completion_cannot_be_repurposed_for_other_signal_uses(self):
        d=document();d['operations'][4]['waits']=['done']
        f=self.refuses(d,'NATIVE_MMA_COMPLETION_CONSUMER')
        self.assertEqual(f.path,'operations[4].waits')
        d=document();d['tile_loops'][0]['body'].append('read_acc')
        failures=preflight(Schedule.from_dict(d),Target.load(ROOT/'compiler/targets/sm_100a.json'))
        self.assertTrue(any(f.code=='NATIVE_MMA_COMPLETION_CONSUMER' and f.path=='operations[3].waits' for f in failures))
        self.assertFalse(self.compiler.assess(d).lowering_eligible)

    def test_every_used_ring_slot_is_drained_before_invalidation(self):
        d=document();lower=self.lower(d)
        lines=list(source_loop_scopes(lower.source))
        # K256/tile64 visits two slots twice: both finish primary phase one.
        waits=[(text,loops) for text,loops in lines if re.fullmatch(r'cake_wait\(free0\+[01], 1\);',text)]
        self.assertEqual(len(waits),2)
        self.assertTrue(all(not loops for _,loops in waits))
        for text,_ in waits:self.assertLess(lower.source.index(text),lower.source.index('cake_inval(free0+stage)'))

    def test_nested_argmin_resets_for_each_row_tile(self):
        d=nested_rows_document();lower=self.lower(d)
        index=next(i for i,op in enumerate(d['operations']) if op['kind']=='reduce_argmin')
        resets=[loops for text,loops in source_loop_scopes(lower.source)
                if text.startswith(f'best{index} = INFINITY;')]
        self.assertEqual(resets,[(('it0',3),)])
        # Every nested column reduction owns fresh state; earlier rows cannot win.
        outputs=[];best=float('inf');winner=-1
        for row,scores in enumerate(((0.0,4.0),(8.0,2.0))):
            if resets[0]==(('it0',3),) or row==0:best=float('inf');winner=-1
            for col,value in enumerate(scores):
                if value<best:best=value;winner=col
            outputs.append(winner)
        self.assertEqual(outputs,[0,1])

    def test_resident_argmin_uses_logical_candidate_bound_not_physical_width(self):
        d = resident_candidates_document()
        lowered = self.lower(d)
        typed = Schedule.from_dict(d)
        reduction = next(op for op in typed.operations if op.kind.value == 'reduce_argmin')
        coordinate, extent = typed.argmin_domain(reduction)
        self.assertEqual((coordinate.name, extent), ('candidate_coordinates', 4))
        self.assertEqual(typed.buffer(reduction.reads[0]).shape[1], 64)
        self.assertIn('const int index = 0+col;', lowered.source)
        self.assertIn('if (index < 4 &&', lowered.source)
        self.assertIn('if (index < 257 &&', self.lower(document('kmeans')).source)
        # A positive minimum tied in the valid domain must beat zero-valued padding;
        # the emitted bounds, rather than the frozen duplicate sample, exclude it.
        values = [5, 5, 9, 9] + [0] * 60
        valid = [i for i in range(len(values)) if i < extent]
        self.assertEqual(min(valid, key=lambda i: (values[i], i)), 0)

    def test_argmin_refuses_multiple_candidate_ctas_and_conflicting_column_bounds(self):
        self.refuses(resident_candidates_document(65), 'NATIVE_ARGMIN_DOMAIN')
        d = resident_candidates_document()
        norm = next(op for op in d['operations'] if op['kind'] == 'load'
                    and len(next(b for b in d['buffers'] if b['name'] == op['reads'][0])['shape']) == 2
                    and op['parameters']['movement'] == 'global')
        next(b for b in d['buffers'] if b['name'] == norm['reads'][0])['shape'][1] = 2
        self.refuses(d, 'NATIVE_ARGMIN_DOMAIN')

    def test_runtime_candidate_prefix_is_not_a_static_domain_through_mma_or_broadcast(self):
        for name in ('centroids', 'centroid_sq'):
            with self.subTest(source=name):
                d = resident_candidates_document()
                data = next(b for b in d['buffers'] if b['name'] == name)
                data['valid_extent'] = dict(dimension=1, buffer='candidate_lengths', indexed_by=[0])
                d['buffers'].append(dict(name='candidate_lengths', space='global', dtype='int32',
                    shape=[data['shape'][0]], mode='input'))
                typed = Schedule.from_dict(d)
                operation = next(op for op in typed.operations if op.kind.value == 'reduce_argmin')
                self.assertIsNone(typed.argmin_domain(operation))
                assessment = self.compiler.assess(d)
                self.assertTrue(assessment.accepted)
                # Native also rejects runtime buffer refinements. The domain rule
                # must reject independently, rather than borrow that earlier block.
                finding = next(f for f in assessment.findings if f.code == 'NATIVE_ARGMIN_DOMAIN')
                self.assertTrue(finding.blocks_lowering)
                self.assertFalse(finding.blocks_acceptance)
                self.assertIn(finding, preflight(typed, Target.load(ROOT / 'compiler/targets/sm_103a.json')))
                with self.assertRaises(CompilerError):
                    self.compiler.lower(assessment)

    def test_unknown_loop_domain_is_rejected_by_the_common_verifier(self):
        for field,value,code in [('buffer','absent','LOOP_BUFFER_UNKNOWN'),('dimension',99,'LOOP_DIMENSION_RANGE')]:
            d=document();d['tile_loops'][0][field]=value
            f=self.refuses(d,code)
            self.assertEqual(f.path,'tile_loops[0].'+field)
            self.assertTrue(f.blocks_acceptance)

    def test_shared_load_outside_pipeline_is_an_explicit_refusal(self):
        d=document()
        d['allocations'].append(dict(name='extra',space='shared',size_bytes=16384))
        d['buffers'].append(dict(name='extra_shared',space='shared',dtype='bf16',shape=[128,64],
                                 mode='scratch',allocation='extra',swizzle='swizzle_128b'))
        d['operations'].append(dict(id='extra_load',kind='load',role='copy',reads=['a'],
                                    writes=['extra_shared'],parameters={'movement':'global'}))
        d['access_maps'].append(dict(operation='extra_load',buffer='a',boundary='mask_tiled_axes',
                                    indices=[{'source':'program_tile','name':'rows'},
                                             {'source':'dimension','dimension':1,'extent':64}]))
        self.refuses(d,'NATIVE_SHARED_LOAD_SCOPE')

    def test_tensor_swizzle_cannot_be_silently_ignored(self):
        d=document();d['buffers'][6]['swizzle']='swizzle_128b'
        f=self.refuses(d,'NATIVE_TMEM_SWIZZLE_UNSUPPORTED')
        self.assertEqual(f.path,'buffers[6].swizzle')

    def test_frontend_schema_includes_native_transfer_and_rejects_misapplied_atom(self):
        try:import jsonschema
        except ImportError:self.skipTest('jsonschema optional dependency unavailable')
        schema=schedule_schema();jsonschema.validate(document(),schema)
        d=document();d['operations'][4]['parameters']['source_atom']={'op':'tcgen05.Ld32x32b','repetition':16}
        with self.assertRaises(jsonschema.ValidationError):jsonschema.validate(d,schema)
        self.refuses(d,'SCHEDULE_STRUCTURE')

    def test_public_cli_generates_cuda(self):
        with tempfile.TemporaryDirectory() as scratch:
            output=Path(scratch)/'native.cu'
            result=subprocess.run([sys.executable,'-m','open_cake_ir.cli','--project-root',str(ROOT),'compiler','lower','--revision',str(ROOT/'compiler/revision.json'),str(EXAMPLES/'gemm-bias.json'),'--output',str(output)],capture_output=True,text=True,cwd=ROOT)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('tcgen05.mma',output.read_text())


if __name__=='__main__':unittest.main()
