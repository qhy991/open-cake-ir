"""Derive a bounded development view of one profiled EP4 run per spatial plan.

Nsight kernel activity is diagnostic only. This script intentionally emits no
speedup, qualified latency, or plan ranking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


SOURCE='4900b162cfbec0c4fc48c175db50f82b8dd0afaf'
COMBINE='1b353de868c781fbf611cc524b19ee5856d4f615'
CONTROLS=((1,0),(74,5888),(147,5888),(148,11776))
PHASES=('dispatch_routes','gather_tiles','tile_schedule_probe',
        'scatter_returns','wait_returns','cake_weave_rank512_combine_kernel')


def doc(path:Path)->dict:
    return json.loads(path.read_text())


def summarize(roots:list[Path])->dict:
    if len(roots)!=4:
        raise ValueError('spatial view needs four exact plan evidence roots')
    cases=[]
    reference=None
    for root in roots:
        manifest=doc(root/'manifest.json')
        case=doc(root/'case.json')
        report=doc(root/'report.json')
        activity=doc(root/'activity.json')
        device=doc(root/'device_report.json')
        receipt=doc(root/'gpuq-admission.json')
        control=(case['communication_ctas'],case['steal_budget_per_owner'])
        if (manifest.get('source_commit')!=SOURCE
                or manifest.get('combine_compiler_commit')!=COMBINE
                or control not in CONTROLS
                or (manifest['communication_ctas'],manifest['steal_budget'])
                   !=control
                or case['geometry']!={'R':4,'T':512,'K':8,'E':128,
                                      'H':2048,'tile_rows':128}
                or report.get('passed') is not True
                or report.get('route_contribution_bit_mismatches')!=0
                or report.get('failing_elements')!=0
                or report.get('routes')!=16384
                or report.get('source_commit')!=SOURCE
                or report.get('actual_stolen_by_owner')
                   !=device.get('stolen_by_owner')
                or activity.get('source_commit')!=SOURCE
                or activity.get('broker_job')!=report.get('broker_job')
                or receipt.get('job_id')!=report.get('broker_job')
                or receipt.get('mode')!='exclusive'
                or receipt.get('gpu_count')!=4
                or device.get('no_interphase_host_sync') is not True):
            raise ValueError(f'{root.name} source, oracle, receipt or control differs')
        binding=(case['bin_root'],case['prepared_root'],case['bridge_root'])
        if reference is None:reference=binding
        elif binding!=reference:
            raise ValueError('spatial cases do not share exact Workload inputs')
        rank_rows=activity['ranks']
        if [row['rank'] for row in rank_rows]!=list(range(4)):
            raise ValueError('spatial activity omits one logical rank')
        durations=[];waits=[]
        for row in rank_rows:
            phases=row['phases']
            if [phase['name'] for phase in phases]!=list(PHASES):
                raise ValueError('spatial activity phase order differs')
            durations.append(next(phase['duration_ns'] for phase in phases
                                  if phase['name']=='tile_schedule_probe'))
            waits.append(next(phase['duration_ns'] for phase in phases
                              if phase['name']=='wait_returns'))
        stolen=report['actual_stolen_by_owner']
        if (len(stolen)!=4 or any(type(value) is not int
                                  or not 0<=value<=control[1]
                                  for value in stolen)
                or (control[0]==148 and stolen!=[11776]*4)):
            raise ValueError('actual steal does not fit its admitted budget')
        cases.append({
            'communication_ctas':control[0],
            'steal_budget':control[1],
            'actual_stolen_by_owner':stolen,
            'broker_job':report['broker_job'],
            'physical_gpus':receipt['gpu_ids'],
            'ffn_activity_ms_by_rank':[round(value/1e6,6)
                                       for value in durations],
            'ffn_activity_median_ms':round(median(durations)/1e6,6),
            'return_wait_activity_ms_by_rank':[round(value/1e6,6)
                                               for value in waits],
            'kernel_activity_union_ms':round(
                activity['kernel_activity_union_ns']/1e6,6),
            'cross_rank_phase_overlap_pair_counts':{
                part['earlier']+'__'+part['later']:len(part['pairs'])
                for part in activity['cross_rank_phase_overlap']},
        })
    if ([(row['communication_ctas'],row['steal_budget'])
         for row in sorted(cases,key=lambda row:row['communication_ctas'])]
            !=list(CONTROLS)):
        raise ValueError('spatial view duplicated or omitted a control')
    return {
        'schema_version':1,
        'source_commit':SOURCE,
        'combine_compiler_commit':COMBINE,
        'workload_binding':{'bin_root':reference[0],
                            'prepared_root':reference[1],
                            'bridge_root':reference[2]},
        'scope':'four correct one-shot Nsight CUDA-activity development observations; no qualified timer or speedup',
        'cases':sorted(cases,key=lambda row:row['communication_ctas']),
    }


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('spatial summary must be create-only')
    result=summarize([root.expanduser().resolve(strict=True)
                      for root in args.root])
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
