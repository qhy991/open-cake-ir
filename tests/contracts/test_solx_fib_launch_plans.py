"""Every remaining FlashInfer task is a typed composition of generated Cake kernels."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib import attention, moe
from open_cake_ir.tasks.solx_fib.catalog import TASK_IDS, task_owner
from open_cake_ir.tasks.workloads import load_workload
from open_cake_ir.evaluation.core import compare_tile_outputs

ROOT=Path(__file__).resolve().parents[2]


class CompletePackPlanTests(unittest.TestCase):
    def test_all_26_tasks_have_one_owning_implementation(self):
        self.assertEqual(len(TASK_IDS),26)
        self.assertEqual(len({t.split('_',1)[0] for t in TASK_IDS}),26)
        self.assertTrue(all(task_owner(t)[1] is not None for t in TASK_IDS))

    def test_attention_and_moe_stages_are_assessed_and_lowered_by_compiler(self):
        compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')
        for owner in (attention,moe):
            for task in owner.TASKS:
                for variant in (['captured','boundary'] if owner is attention else ['captured']):
                    w=WorkloadContract(owner.workload_document(task,variant=variant))
                    plan=owner.launch_plan(w)
                    self.assertEqual(len(plan.stages),4 if owner is attention else 8)
                    for stage in plan.stages:
                        a=compiler.assess(json.loads(stage.schedule_bytes))
                        self.assertEqual(a.findings,(),(task,variant,stage.name))
                        lowering=compiler.lower(a)
                        self.assertTrue(lowering.generated)
                        self.assertEqual(lowering.target,'sm_103a')

    def test_every_frozen_plan_contract_loads_from_common_registry(self):
        for owner in (attention,moe):
            for task in owner.TASKS:
                for variant in (['captured','boundary'] if owner is attention else ['captured']):
                    d=owner.workload_document(task,variant=variant)
                    w=load_workload(ROOT/'contracts/workloads'/f"{d['workload_id']}.json")
                    self.assertEqual(w.document,d)

    def test_moe_retains_complete_geometry_and_runtime_scalars(self):
        w=WorkloadContract(moe.workload_document());abi={a.name:a for a in w.tensor_abi('primary')}
        self.assertEqual(abi['gemm1_weights'].shape,(32,4096,7168))
        self.assertEqual(abi['gemm2_weights'].shape,(32,7168,2048))
        self.assertEqual(abi['hidden_states'].dtype,'fp8_e4m3')
        self.assertEqual(abi['local_expert_offset'].shape,(1,))
        self.assertEqual(abi['routed_scaling_factor'].shape,(1,))
        for key in ('tie_break','definition'):
            bad=deepcopy(w.document);bad['semantics'][key]='changed'
            with self.assertRaises(ValueError):moe.validate_contract(bad)

    def test_ieee_comparison_is_opt_in_and_never_accepts_missing_finite_values(self):
        class Workload:
            document={'validation':{'comparison':'per_output','outputs':{
                'lse':{'comparison':'elementwise_atol_rtol_ieee','atol':0.0,'rtol':0.0}}}}
        for expected,actual,ok in [([-float('inf')],[-float('inf')],True),
                                   ([-float('inf')],[0.0],False),
                                   ([-float('inf')],[float('inf')],False),
                                   ([float('nan')],[float('nan')],True),
                                   ([float('nan')],[12345.0],False),
                                   ([0.0],[float('nan')],False)]:
            passed,_=compare_tile_outputs(Workload(),{}, {'lse':expected},{'lse':actual},{})
            self.assertEqual(passed,ok)
        Workload.document['validation']['outputs']['lse']['comparison']='elementwise_atol_rtol'
        passed,_=compare_tile_outputs(Workload(),{}, {'lse':[-float('inf')]},{'lse':[-float('inf')]},{})
        self.assertFalse(passed)


@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU tensor oracle requires Torch')
class TensorOracleTests(unittest.TestCase):
    def test_empty_and_fully_masked_attention_follow_the_named_definition(self):
        import torch
        for task,spec in attention.SPECS.items():
            w=WorkloadContract(attention.workload_document(task,variant='boundary'))
            inputs=attention.materialize_tensors(w,'segmented')
            result=attention.reference_tensors(w,'segmented',inputs)
            end=1 if spec['decode'] else 3
            self.assertTrue(torch.equal(result['output'][:end],torch.zeros_like(result['output'][:end])))
            self.assertTrue(bool(torch.isneginf(result['lse'][:end]).all()))
            if not spec['decode']:
                inputs=attention.materialize_tensors(w,'primary')
                result=attention.reference_tensors(w,'primary',inputs)
                self.assertTrue(bool(torch.isneginf(result['lse'][0]).all()))
                self.assertEqual(bool(torch.isnan(result['output'][0]).all()), spec['mla'] or not spec['paged'])

    def test_zero_query_has_uniform_weights_and_base_two_lse(self):
        import torch
        task=next(iter(attention.TASKS));w=WorkloadContract(attention.workload_document(task,variant='boundary'))
        inputs=attention.materialize_tensors(w,'zeros');result=attention.reference_tensors(w,'zeros',inputs)
        for batch in range(2):
            lo,hi=inputs['kv_indptr'][batch:batch+2].tolist()
            ids=inputs['kv_indices'][lo:hi].long()
            expected=inputs['v_cache'][ids,0].double().mean(0).repeat_interleave(8,0).to(torch.bfloat16)
            torch.testing.assert_close(result['output'][batch],expected,rtol=0,atol=0)
            torch.testing.assert_close(result['lse'][batch],torch.full((32,),math.log2(hi-lo)),rtol=1e-6,atol=1e-6)

    def test_group_routing_uses_top_two_sum_and_unbiased_normalization(self):
        import torch
        logits=torch.full((1,256),-6.0)
        for group in [0,2,3,4]:logits[0,group*32:group*32+2]=3.0
        logits[0,32]=6.0
        selected,weights=moe.routing_reference(logits,torch.zeros(256,dtype=torch.bfloat16),2.5)
        self.assertEqual(selected.tolist(),[[0,1,64,65,96,97,128,129]])
        torch.testing.assert_close(weights,torch.full((1,8),2.5/8,dtype=torch.float32))
