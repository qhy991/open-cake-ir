"""Bind actual four-GPU expert-bin rows to a checked ranked tile plan.

This development bridge is deliberately after the P2P bin job. It checks the
route-key/BF16 row identity again and forms padded M128 inputs for the Cake
worker. Tile publication is performed by the host here, not on the device.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


R,T,K,E,H,B=4,512,8,128,2048,128
LOCAL_E=E//R
WEIGHT_UP_BYTES=E*(2*768)*H*2
WEIGHT_DOWN_BYTES=E*H*768*2


def document(path:Path)->dict:
    return json.loads(path.read_text())


def prepare(p2p:Path,plan_file:Path,bridge:Path,output:Path)->None:
    p2p_report=document(p2p/'report.json')
    bridge_report=document(bridge/'report.json')
    plan=document(plan_file)
    if (not p2p_report.get('passed') or not bridge_report.get('passed')
            or bridge_report.get('routes')!=R*T*K
            or p2p_report.get('source_commit')
               !='a14d758aceec0ee12832d2a72bce89b2fc1d0475'
            or plan.get('schema_version')!=2
            or plan.get('geometry',{})!={
                'world_size':R,'tokens_per_rank':T,'routes_per_token':K,
                'experts':E,'experts_per_rank':LOCAL_E,'tile_rows':B,
                'source_chunk_tokens':128,'early_flush_min_rows':64}
            or len(plan.get('tasks',[]))!=256
            or (bridge/'weights_upgate.bf16').stat().st_size!=WEIGHT_UP_BYTES
            or (bridge/'weights_down.bf16').stat().st_size!=WEIGHT_DOWN_BYTES):
        raise ValueError('P2P bin, bridge weights or threshold-64 plan differs')
    if any(output.iterdir()):
        raise ValueError('worker tile inputs must be create-only')
    hidden=np.fromfile(p2p/'hidden.bf16',dtype='<u2').reshape(R*T,H)
    ids=np.fromfile(p2p/'expert_ids.i32',dtype='<i4').reshape(R*T,K)
    source_bytes=(p2p/'hidden.bf16').read_bytes()
    if (p2p/'device_outputs/observed_hidden.bf16').read_bytes()!=source_bytes:
        raise ValueError('P2P job changed source hidden rows')
    if ((p2p/'device_outputs/observed_expert_ids.i32').read_bytes()
            !=(p2p/'expert_ids.i32').read_bytes()):
        raise ValueError('P2P job changed expert IDs')
    all_keys=set()
    reports=[]
    for owner in range(R):
        tasks=[task for task in plan['tasks'] if task['owner_rank']==owner]
        if len(tasks)!=64:
            raise ValueError(f'owner {owner} tile count differs')
        row_root=p2p/'device_outputs'/f'owner{owner}'
        counts=np.fromfile(row_root/'counts.u32',dtype='<u4')
        keys=np.fromfile(row_root/'keys.i32',dtype='<i4').reshape(LOCAL_E,R*T)
        rows=np.fromfile(row_root/'rows.bf16',dtype='<u2').reshape(-1,H)
        if counts.size!=LOCAL_E or int(counts.sum())!=rows.shape[0]:
            raise ValueError(f'owner {owner} packed bin extent differs')
        location={}
        cursor=0
        for local_expert,count in enumerate(counts):
            for slot in range(int(count)):
                key=int(keys[local_expert,slot])
                if key in location or not 0<=key<R*T*K:
                    raise ValueError(f'owner {owner} duplicate or invalid route key')
                source_token=key//K
                if int(ids[source_token,key%K])!=owner*LOCAL_E+local_expert:
                    raise ValueError(f'owner {owner} route key was misrouted')
                if not np.array_equal(rows[cursor+slot],hidden[source_token]):
                    raise ValueError(f'owner {owner} P2P BF16 row differs')
                location[key]=(local_expert,cursor+slot)
            cursor+=int(count)
        tile_inputs=np.zeros((64,B,H),dtype='<u2')
        tile_experts=np.empty(64,dtype='<i4')
        tile_keys=np.full((64,B),-1,dtype='<i4')
        owner_keys=set()
        for tile_index,task in enumerate(tasks):
            expert=task['expert_id']
            valid=task['valid_rows']
            if (task['owner_rank']!=owner or not owner*LOCAL_E<=expert<(owner+1)*LOCAL_E
                    or not 1<=valid<=B or len(task['rows'])!=valid):
                raise ValueError(f'owner {owner} tile {tile_index} plan differs')
            local_expert=expert-owner*LOCAL_E
            tile_experts[tile_index]=local_expert
            for row_index,(source,item,route) in enumerate(task['rows']):
                key=(source*T+item)*K+route
                if key not in location or key in owner_keys:
                    raise ValueError(f'owner {owner} tile route missing or duplicated')
                observed_expert,bin_index=location[key]
                if observed_expert!=local_expert:
                    raise ValueError(f'owner {owner} tile expert and bin differ')
                tile_inputs[tile_index,row_index]=rows[bin_index]
                tile_keys[tile_index,row_index]=key
                owner_keys.add(key)
        if owner_keys!=set(location):
            raise ValueError(f'owner {owner} tiles fail to cover bin keys')
        all_keys.update(owner_keys)
        rank=output/f'rank{owner}'
        rank.mkdir()
        tile_inputs.tofile(rank/'x_tiles.bf16')
        tile_experts.tofile(rank/'tile_expert.i32')
        tile_keys.tofile(rank/'tile_route_keys.i32')
        reports.append({'owner_rank':owner,'valid_routes':len(owner_keys),
                        'logical_tiles':len(tasks),'stage_work_units':len(tasks)*184,
                        'partial_tiles':sum(task['valid_rows']<B for task in tasks)})
    if all_keys!=set(range(R*T*K)):
        raise ValueError('four-rank tile plan lost a route key')
    (output/'case.json').write_text(json.dumps({
        'geometry':{'R':R,'T':T,'K':K,'E':E,'H':H,'tile_rows':B},
        'p2p_root':str(p2p.resolve()),
        'plan':str(plan_file.resolve()),
        'bridge_weights':str(bridge.resolve()),
        'owners':reports,
        'scope':'host-materialized M128 tiles from actual EP4 P2P bins; no GPU tile publication',
    },indent=2)+'\n')
    print(json.dumps({'owners':reports},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--p2p-root',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--bridge-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    a=parser.parse_args()
    prepare(a.p2p_root.expanduser().resolve(strict=True),
            a.plan.expanduser().resolve(strict=True),
            a.bridge_root.expanduser().resolve(strict=True),
            a.output_root.expanduser().resolve(strict=True))
