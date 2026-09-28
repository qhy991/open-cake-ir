"""Exact tensor placement and lifecycle around B300 ranked-tile lowering."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program, RankedTileEffects, Schedule
from open_cake_ir.compiler.ir import DType, ProgramTensor
from open_cake_ir.evaluation.ranked_tile_launch import (
    RankedTileBound, RankedTileExecutable, prepare_ranked_tile_case,
    validate_ranked_tile_case,
)
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.workloads import load_workload


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Tensor:
    spec: ProgramTensor
    device: int
    start: int

    @property
    def end(self):
        return self.start+self.spec.nbytes


class RankedTileLaunchContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=Compiler.load(ROOT)
        if compiler.commit is None:
            raise unittest.SkipTest('ranked-tile launch needs a fixed clean Compiler')
        local=Program.from_dict(json.loads((ROOT/'examples/programs/'
            'weave-model-local-expert-ffn-native-b300.json').read_text()))
        effects=RankedTileEffects.from_dict(json.loads((ROOT/'examples/programs/'
            'weave-model-ranked-tile-effects-b300.json').read_text()))
        combine=Schedule.from_dict(json.loads((ROOT/'examples/schedules/native/'
            'weave-model-weighted-combine-rank512-b300.json').read_text()))
        cls.lowered=compiler.lower_ranked_tiles(effects,local,combine)
        n128=Program.from_dict(json.loads((ROOT/'examples/programs/'
            'weave-model-local-expert-ffn-native-b300-n128.json').read_text()))
        cls.lowered_n128=compiler.lower_ranked_tiles(effects,n128,combine)
        cls.workload=load_workload(ROOT/'contracts/workloads/'
            'weave-model-ep4-bf16-moe-b300-v1.json')

    def fixtures(self, lowered=None):
        req=(lowered or self.lowered).toolchain_requirements
        inputs={};outputs={};plans={}
        for rank in range(4):
            cursor=0x1000
            tensors={}
            for row in req['rank_inputs']:
                spec=ProgramTensor(tuple(row['shape']),DType(row['dtype']))
                tensors[row['name']]=Tensor(spec,rank,cursor)
                cursor+=spec.nbytes+4096
            output_spec=ProgramTensor((512,2048),DType.BF16)
            inputs[rank]=tensors
            outputs[rank]=Tensor(output_spec,rank,cursor)
            plans[rank]={'communication_ctas':74 if rank%2==0 else 1,
                         'chunks':4,
                         'steal_budget':5888 if rank%2==0 else 0}
        return inputs,outputs,plans

    def prepare(self,inputs,outputs,plans,*,isolated=True,launch_status=0,
                calls=None,tile_counts=None,lowered=None):
        selected=lowered or self.lowered
        events=[] if calls is None else calls
        def bind(bound_inputs,bound_outputs,contexts):
            events.append(('bind',tuple(contexts)))
            self.assertEqual(set(bound_inputs),set(range(4)))
            self.assertEqual(set(bound_outputs),set(range(4)))
            return RankedTileBound(
                launch=lambda c,b,k:(events.append(('launch',c,b,k)) or launch_status),
                stolen=lambda rank:5888 if rank%2==0 else 0,
                payloads=lambda rank:(618,312,306,300)[rank],
                tile_counts=lambda rank:tile_counts or (1,)+(0,)*19,
                destroy=lambda:(events.append(('destroy',)) or 0),
            )
        def loader(lowered):
            self.assertIs(lowered,selected)
            events.append(('load',))
            return RankedTileExecutable(4,20,512*2048*2,isolated,bind)
        bound=prepare_ranked_tile_case(
            selected,self.workload,'fanin_v2',inputs,outputs,plans,
            load_source=loader,
            check_tensor=lambda tensor,spec:self.assertEqual(tensor.spec,spec),
            storage_span=lambda tensor:(tensor.device,tensor.start,tensor.end),
            execution_context=lambda rank:f'ctx{rank}',
        )
        return bound,events

    def test_n128_capacity_admits_checked_four_rank_launch(self):
        inputs,outputs,plans=self.fixtures(self.lowered_n128)
        for row in plans.values():
            row['steal_budget']=12750
        bound,events=self.prepare(inputs,outputs,plans,
                                  lowered=self.lowered_n128)
        self.assertIn(('load',),events)
        self.assertEqual(bound.lowering.analysis.stage_task_slots_per_rank,12750)
        bound.close()

    def test_workload_case_binding_and_shard_refusals(self):
        for case_id in self.workload.case_ids:
            validate_ranked_tile_case(self.lowered,self.workload,case_id)
        changed=self.workload.document
        changed['semantics']['tensor_placement']['w_down']='rank_sharded_axis_0'
        with self.assertRaisesRegex(ValueError,'w_down.*shard ABI'):
            validate_ranked_tile_case(self.lowered,WorkloadContract(changed),'fanin_v2')
        changed=self.workload.document
        changed['cases'][0]['shape']['H']=1024
        with self.assertRaisesRegex(ValueError,'shard ABI'):
            validate_ranked_tile_case(self.lowered,WorkloadContract(changed),'fanin_v2')

    def test_rank_local_controls_and_repeated_lifecycle(self):
        inputs,outputs,plans=self.fixtures()
        bound,events=self.prepare(inputs,outputs,plans)
        actual,status=bound.run()
        self.assertEqual(actual,outputs)
        self.assertEqual(status['stolen_by_rank'],(5888,0,5888,0))
        self.assertEqual(status['remote_payloads_by_owner'],(618,312,306,300))
        self.assertEqual(status['tile_counts_by_rank'][0],(1,)+(0,)*19)
        self.assertEqual(bound.launch_calls,1)
        self.assertIn(('launch',(74,1,74,1),(5888,0,5888,0),(4,4,4,4)),events)
        replay={rank:{**plan,'communication_ctas':73 if rank%2==0 else 2}
                for rank,plan in plans.items()}
        _,again=bound.run(replay)
        self.assertEqual(again['stolen_by_rank'],(5888,0,5888,0))
        self.assertEqual(bound.launch_calls,2)
        self.assertIn(('launch',(73,2,73,2),(5888,0,5888,0),(4,4,4,4)),events)
        for chunks in (2,1):
            temporal={rank:{**plan,'chunks':chunks}
                      for rank,plan in plans.items()}
            _,status=bound.run(temporal)
            self.assertEqual(status['chunks_by_rank'],(chunks,)*4)
            self.assertIn(('launch',(74,1,74,1),(5888,0,5888,0),
                           (chunks,)*4),events)
        bound.close()
        self.assertIn(('destroy',),events)
        with self.assertRaisesRegex(ValueError,'closed'):
            bound.run()

    def test_alias_and_wrong_owner_refused_before_loading(self):
        inputs,outputs,plans=self.fixtures()
        outputs[0]=Tensor(outputs[0].spec,0,inputs[0]['hidden'].start)
        with self.assertRaisesRegex(ValueError,'alias'):
            self.prepare(inputs,outputs,plans)
        inputs,outputs,plans=self.fixtures()
        row=inputs[2]['expert_ids']
        inputs[2]['expert_ids']=Tensor(row.spec,1,row.start)
        with self.assertRaisesRegex(ValueError,'owner'):
            self.prepare(inputs,outputs,plans)

    def test_plan_and_process_isolation_refused(self):
        inputs,outputs,plans=self.fixtures()
        plans[1]['chunks']=3
        with self.assertRaisesRegex(ValueError,'controls'):
            self.prepare(inputs,outputs,plans)
        inputs,outputs,plans=self.fixtures()
        plans[1]['chunks']=2
        with self.assertRaisesRegex(ValueError,'chunks must agree'):
            self.prepare(inputs,outputs,plans)
        inputs,outputs,plans=self.fixtures()
        plans[0]['communication_ctas']=96
        with self.assertRaisesRegex(ValueError,'controls'):
            self.prepare(inputs,outputs,plans)
        inputs,outputs,plans=self.fixtures()
        with self.assertRaisesRegex(ValueError,'isolation'):
            self.prepare(inputs,outputs,plans,isolated=False)

    def test_failed_launch_poison_does_not_destroy_live_state(self):
        inputs,outputs,plans=self.fixtures()
        bound,events=self.prepare(inputs,outputs,plans,launch_status=37)
        with self.assertRaisesRegex(ValueError,'status 37'):
            bound.run()
        with self.assertRaisesRegex(ValueError,'isolated process exit'):
            bound.close()
        self.assertNotIn(('destroy',),events)

    def test_inactive_temporal_event_count_poison_is_retained(self):
        inputs,outputs,plans=self.fixtures()
        for row in plans.values():
            row['chunks']=1
        invalid=(1,)+(0,)*18+(1,)
        bound,_=self.prepare(inputs,outputs,plans,tile_counts=invalid)
        with self.assertRaisesRegex(ValueError,'tile status'):
            bound.run()
        with self.assertRaisesRegex(ValueError,'isolated process exit'):
            bound.close()


if __name__ == '__main__':
    unittest.main()
