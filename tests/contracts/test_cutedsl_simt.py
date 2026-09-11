"""Generic CuTe source semantics, eligibility and variable ABI; no GPU claim."""
from copy import deepcopy
import importlib
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler,CompilerError,frontend
from open_cake_ir.compiler.backends import cutedsl
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.cute_toolchain import validate_cute_kernel
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.evaluation.core import compare_tile_outputs
from open_cake_ir.tasks.workloads import create_task,materialize_case,reference_outputs
from tests.contracts._cute_simt_cpu import F32,execute,fmax

ROOT=Path(__file__).resolve().parents[2]
FAMILIES=('activation','rowwise','reductions','optimizers','contraction')


def task(name,family,*,target='sm_103a',columns=8,depth=8):
    kwargs={'backend':'triton-b300' if target=='sm_103a' else 'triton-b200','rows':2,'columns':columns}
    if family=='contraction':kwargs['depth']=depth
    # Non-power-of-two CPU oracle inputs use the existing Metal task constructor;
    # Triton's task constructor rejects them before a CuTe schedule can be assessed.
    if columns & (columns-1) or family=='contraction' and depth & (depth-1):
        kwargs['backend']='metal-m1-pro'
    document,source=create_task(name,**kwargs)
    schedule=frontend.parse(source).document
    schedule['target']=target
    schedule['lowering']['backend']='cutlass_cute_dsl'
    return WorkloadContract(document),schedule


def reduction_tile(shape,axis,op):
    """One complete tile per batch, independent of the task family's column mapping."""
    output=shape[:axis]+shape[axis+1:]
    read='batch'+', :'*len(shape);write='batch'+', :'*len(output)
    return frontend.parse(f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="full-tile-reduction", target="sm_103a",
               backend="cutlass_cute_dsl", entry_point="tile_reduce")
def candidate(lm, x: cake.Tensor({(1,*shape)!r}, "fp32"),
              out: cake.Tensor({(1,*output)!r}, "fp32", mode="output")):
    compute = lm.role(warps=[0])
    batch = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[{read}])
        total = lm.reduce(values, op={op!r}, axis={axis}, scope="cta", across_loop=False)
        lm.store(out[{write}], total, coalesced=False)
''').document


class CuTeSimtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')

    def test_all_twenty_four_tasks_lower_without_mma_or_task_name_dispatch(self):
        seen=[]
        for family in FAMILIES:
            for name in importlib.import_module(f'open_cake_ir.tasks.{family}.workload').TASKS:
                for target in ('sm_100a','sm_103a'):
                    with self.subTest(task=name,target=target):
                        _,s=task(name,family,target=target)
                        s['schedule_id']='unnamed_composition'
                        a=self.compiler.assess(s)
                        self.assertTrue(a.lowering_eligible,a.findings)
                        self.assertTrue(any(f.code=='CUTE_SIMT_EXECUTION' for f in a.findings))
                        lowered=self.compiler.lower(a)
                        validate_cute_kernel(lowered.source.encode(),dict(lowered.toolchain_requirements))
                        self.assertEqual(lowered.toolchain_requirements['target'],target)
                        self.assertNotIn('cute.gemm(',lowered.source)
                seen.append(name)
        self.assertEqual(len(seen),24)

    def test_emitted_source_matches_every_task_oracle_and_writes_all_outputs(self):
        for family in FAMILIES:
            for name in importlib.import_module(f'open_cake_ir.tasks.{family}.workload').TASKS:
                workload,s=task(name,family)
                emission=cutedsl.emit(Schedule.from_dict(s),Target.load(ROOT/'compiler/targets/sm_103a.json'))
                sizes={arg.name:math.prod(arg.shape) for arg in workload.tensor_abi('primary') if arg.mode=='output'}
                for case in ('primary','zeros','alternating'):
                    with self.subTest(task=name,case=case):
                        inputs=materialize_case(workload,case)
                        expected=reference_outputs(workload,case,inputs)
                        actual=execute(emission,inputs,sizes)
                        passed,metrics=compare_tile_outputs(workload,inputs,expected,actual,inputs)
                        self.assertTrue(passed,metrics)

    def test_controls_and_effects_are_refused_before_lowering(self):
        _,base=task('silu','activation')
        variants=[]
        s=deepcopy(base);s['roles'][0]['warps']=[0,1];variants.append((s,'CUTE_SIMT_ROLE'))
        s=deepcopy(base);s['residency']={'registers_per_thread':32};variants.append((s,'CUTE_SIMT_RESIDENCY'))
        s=deepcopy(base);s['operations'][-1]['parameters']['coalesced']=True;variants.append((s,'CUTE_SIMT_STORE'))
        s=deepcopy(base);s['operations'][0]['parameters']['reuse']='streamed';variants.append((s,'CUTE_SIMT_LOAD'))
        for s,code in variants:
            a=self.compiler.assess(s)
            self.assertFalse(a.lowering_eligible)
            self.assertIn(code,[f.code for f in a.findings])
        # Canonical semantics refuses this before backend preflight. Probe both owners
        # explicitly instead of attributing that earlier refusal to the backend.
        s=deepcopy(base);s['program_map']['persistent']=True
        a=self.compiler.assess(s)
        self.assertFalse(a.lowering_eligible)
        self.assertIn('PERSISTENT_WITHOUT_RESIDENCY',[f.code for f in a.findings])
        findings=cutedsl.preflight(Schedule.from_dict(s),Target.load(ROOT/'compiler/targets/sm_103a.json'))
        self.assertIn('CUTE_SIMT_PROGRAM_MAP',[f.code for f in findings])

    def test_non_cuda_targets_are_refused_by_the_exact_target_owner(self):
        _,s=task('silu','activation')
        for name in ('apple_gpu_family7','apple_gpu_family8'):
            with self.subTest(target=name):
                s['target']=name
                target=Target.load(ROOT/f'compiler/targets/{name}.json')
                a=self.compiler.assess(s)
                self.assertTrue(a.accepted,a.findings)
                self.assertFalse(a.lowering_eligible)
                self.assertIn('CUTE_SIMT_TARGET',[f.code for f in a.findings])
                with self.assertRaises(CompilerError):self.compiler.lower(a)
                self.assertIn('CUTE_SIMT_TARGET',[f.code for f in cutedsl.preflight(Schedule.from_dict(s),target)])
                with self.assertRaises(EmitError):cutedsl.emit(Schedule.from_dict(s),target)

    def test_cross_lane_slots_and_partial_final_warp(self):
        # 33 columns forces multiple per-lane slots and a partial last stripe;
        # contraction also reduces an axis with nonunit trailing stride.
        for name,family in (('softmax_backward','rowwise'),('gemm','contraction'),('channel_absmax_scale','reductions')):
            workload,s=task(name,family,columns=33,depth=5)
            emission=cutedsl.emit(Schedule.from_dict(s),Target.load(ROOT/'compiler/targets/sm_103a.json'))
            sizes={arg.name:math.prod(arg.shape) for arg in workload.tensor_abi('primary') if arg.mode=='output'}
            inputs=materialize_case(workload,'primary')
            expected=reference_outputs(workload,'primary',inputs)
            actual=execute(emission,inputs,sizes)
            passed,metrics=compare_tile_outputs(workload,inputs,expected,actual,inputs)
            self.assertTrue(passed,metrics)

    def test_lane_local_contractions_preserve_oracles_and_bound_source_growth(self):
        for name in ('gemm','gemm_silu'):
            source_sizes=[]
            for columns in (32,64,96):
                with self.subTest(task=name,columns=columns):
                    workload,s=task(name,'contraction',columns=columns)
                    a=self.compiler.assess(s)
                    self.assertTrue(a.lowering_eligible,a.findings)
                    lowered=self.compiler.lower(a)
                    validate_cute_kernel(lowered.source.encode(),dict(lowered.toolchain_requirements))
                    self.assertIn('Lane-local reduction:',lowered.source)
                    self.assertNotIn('warp_reduction_',lowered.source)
                    source_sizes.append(len(lowered.source))
                    emission=cutedsl.emit(Schedule.from_dict(s),Target.load(ROOT/'compiler/targets/sm_103a.json'))
                    sizes={'out':2*columns}
                    for case in ('primary','zeros','alternating'):
                        inputs=materialize_case(workload,case)
                        expected=reference_outputs(workload,case,inputs)
                        actual=execute(emission,inputs,sizes)
                        passed,metrics=compare_tile_outputs(workload,inputs,expected,actual,inputs)
                        self.assertTrue(passed,(case,metrics))
            # Fixed-depth contractions must not expand one reduction per output
            # scalar or scan every source slot for every output as width grows.
            self.assertLess(max(source_sizes)-min(source_sizes),256,source_sizes)

    def test_nondivisible_reduction_strides_keep_collectives_and_numerics(self):
        for columns in (31,33):
            with self.subTest(columns=columns):
                workload,s=task('gemm','contraction',columns=columns,depth=5)
                a=self.compiler.assess(s)
                self.assertTrue(a.lowering_eligible,a.findings)
                lowered=self.compiler.lower(a)
                validate_cute_kernel(lowered.source.encode(),dict(lowered.toolchain_requirements))
                self.assertNotIn('Lane-local reduction:',lowered.source)
                self.assertIn('warp_reduction_sum',lowered.source)
                emission=cutedsl.emit(Schedule.from_dict(s),Target.load(ROOT/'compiler/targets/sm_103a.json'))
                inputs=materialize_case(workload,'alternating')
                expected=reference_outputs(workload,'alternating',inputs)
                actual=execute(emission,inputs,{'out':2*columns})
                passed,metrics=compare_tile_outputs(workload,inputs,expected,actual,inputs)
                self.assertTrue(passed,metrics)

    def test_lane_local_sum_max_order_identities_and_outer_axes(self):
        for shape,axis in (((5,32),0),((2,5,64),1),((2,5,3,32),1),((1,64),0)):
            inner=math.prod(shape[axis+1:]);extent=shape[axis]
            count=math.prod(shape)//extent
            for op in ('sum','max'):
                with self.subTest(shape=shape,axis=axis,op=op):
                    s=reduction_tile(shape,axis,op)
                    a=self.compiler.assess(s)
                    self.assertTrue(a.lowering_eligible,a.findings)
                    lowered=self.compiler.lower(a)
                    validate_cute_kernel(lowered.source.encode(),dict(lowered.toolchain_requirements))
                    self.assertNotIn('warp_reduction_',lowered.source)
                    self.assertIn('ordered lane-local folds',next(
                        f.message for f in a.findings if f.code=='CUTE_SIMT_EXECUTION'))
                    emission=cutedsl.emit(Schedule.from_dict(s),Target.load(ROOT/'compiler/targets/sm_103a.json'))
                    patterns=((2**24,1,-2**24,2**-149,-0.0),
                              (-3.4028234663852886e38,-1,-2**-149,-0.0,0.0),
                              (-0.0,)*5,(0.0,)*5)
                    values=[F32(0)]*math.prod(shape);expected=[]
                    for output in range(count):
                        value=F32(0 if op=='sum' else float('-inf'))
                        for k in range(extent):
                            x=F32(patterns[output%len(patterns)][k%5])
                            values[(output//inner)*extent*inner+k*inner+output%inner]=x
                            value=value+x if op=='sum' else fmax(value,x)
                        expected.append(value)
                    actual=execute(emission,{'x':values},{'out':count})['out']
                    self.assertEqual(actual,expected)
                    self.assertEqual([math.copysign(1,x) for x in actual],
                                     [math.copysign(1,x) for x in expected])

    def test_attention_decode_composes_local_and_cross_lane_reductions(self):
        for depth in (32,64):
            with self.subTest(depth=depth):
                workload,s=task('attention_decode','contraction',columns=8,depth=depth)
                a=self.compiler.assess(s)
                self.assertTrue(a.lowering_eligible,a.findings)
                lowered=self.compiler.lower(a)
                validate_cute_kernel(lowered.source.encode(),dict(lowered.toolchain_requirements))
                # Weighted values reduce [sequence, depth] along the first axis;
                # QK dot products and softmax still require their warp reductions.
                self.assertEqual(lowered.source.count('Lane-local reduction:'),1)
                self.assertIn('warp_reduction_sum',lowered.source)
                self.assertIn('warp_reduction_max',lowered.source)
                emission=cutedsl.emit(Schedule.from_dict(s),Target.load(ROOT/'compiler/targets/sm_103a.json'))
                for case in ('primary','zeros','alternating'):
                    inputs=materialize_case(workload,case)
                    expected=reference_outputs(workload,case,inputs)
                    actual=execute(emission,inputs,{'out':2*depth})
                    passed,metrics=compare_tile_outputs(workload,inputs,expected,actual,inputs)
                    self.assertTrue(passed,(case,metrics))

    def test_prefix_collision_is_mangled_without_invalid_double_underscore(self):
        _,s=task('silu','activation')
        old=next(b['name'] for b in s['buffers'] if b['space']=='global' and b['mode']=='input')
        new='_cake_input'
        for b in s['buffers']:
            if b['name']==old:b['name']=new
        for op in s['operations']:op['reads']=[new if n==old else n for n in op['reads']]
        for a in s['access_maps']:
            if a['buffer']==old:a['buffer']=new
        for a in s['program_map']['axes']:
            if a['buffer']==old:a['buffer']=new
        assessed=self.compiler.assess(s)
        self.assertTrue(assessed.lowering_eligible,assessed.findings)
        lowered=self.compiler.lower(assessed)
        validate_cute_kernel(lowered.source.encode(),dict(lowered.toolchain_requirements))
        self.assertIn('_cake1_',lowered.source)
