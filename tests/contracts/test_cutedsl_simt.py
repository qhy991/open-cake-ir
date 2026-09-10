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
from tests.contracts._cute_simt_cpu import execute

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
