"""Tile-keyed capacity and stealing resources derive from complete Cake math."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program, RankedTileEffects, Schedule


ROOT=Path(__file__).resolve().parents[2]
EFFECTS=ROOT/'examples/programs/weave-ranked-tile-effects-b300.json'


def local_tile_program() -> Program:
    gemm=json.loads((ROOT/'examples/schedules/native/gemm-bias.json').read_text())
    gemm['target']='sm_103a'
    for buffer in gemm['buffers']:
        if buffer['name'] in ('a','c'):
            buffer['shape'][0]=128
    document={
        'schema_version':1,'program_id':'synthetic-complete-bf16-tile',
        'target':'sm_103a',
        'tensors':{
            'a':{'shape':[128,256],'dtype':'bf16'},
            'b':{'shape':[256,256],'dtype':'bf16'},
            'bias':{'shape':[256],'dtype':'fp32'},
            'c':{'shape':[128,256],'dtype':'fp32'},
        },
        'inputs':['a','b','bias'],'outputs':['c'],
        'stages':[{'name':'math','schedule':gemm,
                   'bindings':{'a':'a','b':'b','bias':'bias','c':'c'}}],
    }
    return Program.from_dict(document)


def combine(tokens: int = 8) -> Schedule:
    document=json.loads((ROOT/'examples/schedules/triton/'
                         'weave-weighted-combine-t8-h16.json').read_text())
    document['target']='sm_103a'
    shapes={'contributions':[8,2,256],'weights':[8,2],
            'output':[8,256],'route_values':[2,256],
            'route_weights':[2],'weighted':[2,256],
            'summed':[256],'rounded':[256]}
    for buffer in document['buffers']:
        buffer['shape']=shapes[buffer['name']]
        if buffer['name'] in ('contributions','weights','output'):
            buffer['shape'][0]=tokens
    return Schedule.from_dict(document)


def two_wave_plan():
    ids=[[[0,1] if token%2==0 else [2,3] for token in range(64)]
         for _ in range(4)]
    tasks=[]
    for expert in range(4):
        for wave in range(2):
            rows=[(source,token,route)
                  for source in range(4)
                  for token in range(wave*32,(wave+1)*32)
                  for route,value in enumerate(ids[source][token])
                  if value==expert]
            tasks.append({'owner_rank':expert//2,'expert_id':expert,
                          'tile_index':wave,'valid_rows':len(rows),
                          'published_after':['wave_end',0] if wave==0
                                            else 'all_dispatch_done',
                          'rows':rows})
    return ids,tasks


class RankedTileEffectsContract(unittest.TestCase):
    def test_complete_tile_math_derives_queue_and_full_steal_resources(self):
        document=json.loads(EFFECTS.read_text())
        effects=RankedTileEffects.from_dict(document)
        self.assertEqual(effects.document,document)
        compiler=Compiler.load(ROOT)
        analysis=compiler.assess_ranked_tiles(effects,local_tile_program(),combine())
        self.assertEqual((analysis.world_size,analysis.items_per_rank,
                          analysis.routes_per_item,analysis.experts,
                          analysis.experts_per_rank,analysis.tile_rows,
                          analysis.feature_width),(4,8,2,8,2,128,256))
        self.assertEqual((analysis.remote_payload_slots_per_rank,
                          analysis.packed_route_rows_per_rank,
                          analysis.rows_per_expert,
                          analysis.tile_task_slots_per_rank,
                          analysis.return_slots_per_rank),(24,64,32,4,16))
        self.assertEqual((analysis.required_execution_groups,
                          analysis.maximum_shared_bytes,
                          analysis.maximum_tensor_bytes),(6,49152,32768))
        if compiler.commit is not None:
            with self.assertRaisesRegex(ValueError,'dedicated backend lowering'):
                compiler.lower_ranked_tiles(effects,local_tile_program(),combine())

    def test_waiting_for_full_tile_has_smaller_task_capacity(self):
        document=json.loads(EFFECTS.read_text())
        document['partial_threshold_rows']=128
        analysis=RankedTileEffects.from_dict(document).analyze(
            local_tile_program(),combine())
        self.assertEqual(analysis.tile_task_slots_per_rank,2)
        self.assertEqual(analysis.return_slots_per_rank,16)

    def test_wrong_owner_publication_domain_or_resource_transition_refuses(self):
        base=json.loads(EFFECTS.read_text())
        changes=(
            (lambda d:d['channels']['task'].__setitem__('owner','source_rank'),
             'channel keys'),
            (lambda d:d.__setitem__('publication','partial_at_every_route'),
             'publication'),
            (lambda d:d.__setitem__('input_domain','arbitrary_ids'),
             'input'),
            (lambda d:d['steal'].__setitem__('resource_transition','one_warp'),
             'steal'),
            (lambda d:d.__setitem__('reset','optional'),'reset'),
            (lambda d:d.__setitem__('schema_version',True),'fields or version'),
        )
        for mutate,reason in changes:
            document=deepcopy(base)
            mutate(document)
            with self.subTest(reason=reason),self.assertRaisesRegex(ValueError,reason):
                RankedTileEffects.from_dict(document)

    def test_mismatched_math_tile_and_unbounded_chunks_refuse(self):
        base=json.loads(EFFECTS.read_text())
        changed=deepcopy(base)
        changed['tile_rows']=64
        with self.assertRaisesRegex(ValueError,'FP32'):
            RankedTileEffects.from_dict(changed).analyze(
                local_tile_program(),combine())
        changed=deepcopy(base)
        changed['maximum_chunks_per_rank']=9
        with self.assertRaisesRegex(ValueError,'chunk extent'):
            RankedTileEffects.from_dict(changed).analyze(
                local_tile_program(),combine())
        wrong=json.loads((ROOT/'examples/schedules/triton/'
                          'weave-weighted-combine-t8-h16.json').read_text())
        wrong['target']='sm_103a'
        with self.assertRaisesRegex(ValueError,'combine shape'):
            RankedTileEffects.from_dict(base).analyze(
                local_tile_program(),Schedule.from_dict(wrong))

    def test_materialized_two_wave_plan_preserves_every_return_key(self):
        effects=RankedTileEffects.from_dict(json.loads(EFFECTS.read_text()))
        analysis=effects.analyze(local_tile_program(),combine(64))
        ids,tasks=two_wave_plan()
        checked=analysis.check_plan(ids,tasks,source_chunk_tokens=32)
        self.assertEqual(checked.tasks_by_owner,(4,4,0,0))
        self.assertEqual(checked.rows_by_owner,(256,256,0,0))
        self.assertEqual((checked.full_tiles,checked.early_partial_tiles,
                          checked.terminal_partial_tiles),(0,4,4))

    def test_plan_refuses_duplicate_route_and_premature_partial(self):
        effects=RankedTileEffects.from_dict(json.loads(EFFECTS.read_text()))
        analysis=effects.analyze(local_tile_program(),combine(64))
        ids,tasks=two_wave_plan()
        duplicate=deepcopy(tasks)
        duplicate[0]['rows'][1]=duplicate[0]['rows'][0]
        with self.assertRaisesRegex(ValueError,'duplicated'):
            analysis.check_plan(ids,duplicate,source_chunk_tokens=32)
        premature=deepcopy(tasks)
        premature[1]['published_after']=['wave_end',0]
        with self.assertRaisesRegex(ValueError,'early partial'):
            analysis.check_plan(ids,premature,source_chunk_tokens=32)
        with self.assertRaisesRegex(ValueError,'chunks exceed'):
            analysis.check_plan(ids,tasks,source_chunk_tokens=1)
        bad_ids=deepcopy(ids)
        bad_ids[0][0]=[0,0]
        with self.assertRaisesRegex(ValueError,'distinct input'):
            analysis.check_plan(bad_ids,tasks,source_chunk_tokens=32)


if __name__=='__main__':
    unittest.main()
