"""Development-only Nsight CUDA activity for the one-allocation EP4 chain.

The profile is a single trace, not the target's CUPTI/L2-reset timer. Require
an independent correctness report before interpreting kernel activity.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time


LABEL='cake-weave-ep4-live-chain-nsys-b300'
NSYS='/usr/local/cuda-13.1/bin/nsys'
PHASES=('dispatch_routes','gather_tiles','tile_schedule_probe',
        'scatter_returns','wait_returns','cake_weave_rank512_combine_kernel')


def doc(path:Path)->dict:
    return json.loads(path.read_text())


def root_contract(root:Path)->tuple[dict,dict]:
    manifest=doc(root/'manifest.json')
    case=doc(root/'case.json')
    build=doc(root/'build_report.json')
    if (manifest.get('target')!='sm_103a'
            or len(manifest.get('source_commit',''))!=40
            or case.get('source_commit')!=manifest['source_commit']
            or build.get('source_commit')!=manifest['source_commit']
            or build.get('exit_code')!=0
            or not (root/'live_probe').is_file()
            or (root/'live_probe').read_bytes()[:4]!=b'\x7fELF'):
        raise ValueError('profile target or fixed source/build differs')
    return manifest,case


def admitted()->str:
    path=Path(os.environ.get('BROKER_RECEIPT','/nonexistent'))
    for _ in range(30):
        if path.is_file():break
        time.sleep(.1)
    else:raise RuntimeError('broker admission receipt missing')
    receipt=doc(path)
    if (receipt.get('label')!=LABEL or receipt.get('mode')!='exclusive'
            or receipt.get('gpu_count')!=4
            or len(receipt.get('gpu_ids',[]))!=4
            or not isinstance(receipt.get('job_id'),str)):
        raise RuntimeError('exact exclusive four-GPU profiler admission differs')
    return receipt['job_id']


def run(root:Path)->None:
    manifest,case=root_contract(root)
    if any((root/name).exists() for name in
           ('device.json','launch_failure.json','profile_plan.json',
            'nsys-live-chain.nsys-rep')):
        raise ValueError('profile device evidence must be create-only')
    job=admitted()
    command=[NSYS,'profile','--trace=cuda','--sample=none',
             '--cpuctxsw=none','--force-overwrite=false',
             '--output='+str(root/'nsys-live-chain'),
             str(root/'live_probe'),str(root),case['bin_root'],
             case['prepared_root'],case['bridge_root'],
             str(case['communication_ctas']),
             str(case['steal_budget_per_owner'])]
    (root/'profile_plan.json').write_text(json.dumps(command,indent=2)+'\n')
    try:
        result=subprocess.run(command,cwd=root,capture_output=True,
                              text=True,timeout=300,check=False)
        (root/'profile.log').write_text(result.stdout+result.stderr or
                                        '(no profiler output)\n')
        if result.returncode or not (root/'nsys-live-chain.nsys-rep').is_file():
            raise RuntimeError(f'Nsight CUDA activity exited {result.returncode}')
        (root/'device.json').write_text(json.dumps({
            'broker_job':job,'source_commit':manifest['source_commit'],
            'combine_compiler_commit':manifest['combine_compiler_commit'],
            'target':'sm_103a','world_size':4,
            'scope':'one profiled EP4 development correctness run; no admitted latency',
        },indent=2)+'\n')
    except Exception as error:
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job':job,'error':f'{type(error).__name__}: {error}'},
            indent=2)+'\n')
        raise


def export(root:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('Nsight export and oracle require released GPU lease')
    manifest,_=root_contract(root)
    if (not doc(root/'report.json').get('passed')
            or doc(root/'device.json')['source_commit']!=manifest['source_commit']
            or (root/'nsys-live-chain.sqlite').exists()):
        raise ValueError('profile export needs a passing exact source oracle')
    command=[NSYS,'export','--type=sqlite',
             '--output='+str(root/'nsys-live-chain.sqlite'),
             str(root/'nsys-live-chain.nsys-rep')]
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,
                          timeout=180,check=False)
    (root/'export.log').write_text(result.stdout+result.stderr or
                                   '(no export output)\n')
    if result.returncode or not (root/'nsys-live-chain.sqlite').is_file():
        raise RuntimeError('Nsight SQLite export failed')


def read_rows(database:Path)->list[dict]:
    with closing(sqlite3.connect(f'file:{database.resolve()}?mode=ro',uri=True)) as db:
        try:
            rows=db.execute('''SELECT k.deviceId,k.start,k.end,
                                      k.registersPerThread,k.gridX,k.blockX,
                                      s.value
                               FROM CUPTI_ACTIVITY_KIND_KERNEL k
                               JOIN StringIds s ON s.id=k.shortName''').fetchall()
        except sqlite3.DatabaseError as error:
            raise ValueError('Nsight CUDA kernel activity schema differs') from error
    return [dict(zip(('device_id','start_ns','end_ns','registers',
                      'grid_x','block_x','name'),row,strict=True)) for row in rows]


def analyze(root:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('activity analysis requires released GPU lease')
    if (root/'activity.json').exists():
        raise ValueError('activity report must be create-only')
    manifest,case=root_contract(root)
    receipt=doc(root/'gpuq-admission.json')
    device=doc(root/'device.json')
    report=doc(root/'report.json')
    device_report=doc(root/'device_report.json')
    if (receipt.get('label')!=LABEL or receipt.get('mode')!='exclusive'
            or receipt.get('gpu_count')!=4
            or device.get('broker_job')!=receipt.get('job_id')
            or report.get('broker_job')!=receipt.get('job_id')
            or report.get('passed') is not True
            or device.get('source_commit')!=manifest['source_commit']
            or device_report.get('communication_ctas')
               !=case['communication_ctas']
            or device_report.get('steal_budget')
               !=case['steal_budget_per_owner']
            or device_report.get('no_interphase_host_sync') is not True):
        raise ValueError('profile broker/source or complete oracle differs')
    rows=read_rows(root/'nsys-live-chain.sqlite')
    observations={}
    for row in rows:
        found=[phase for phase in PHASES if phase in row['name']]
        if len(found)!=1 or row['device_id'] not in range(4):
            raise ValueError(f'unexpected profiled kernel {row["name"]!r}')
        phase=found[0]
        key=(row['device_id'],phase)
        if (key in observations or row['start_ns']<=0
                or row['end_ns']<=row['start_ns']
                or not 0<row['registers']<=255):
            raise ValueError('profile kernel identity or timestamp differs')
        observations[key]=row
    if len(observations)!=4*len(PHASES):
        raise ValueError('profile needs exactly six kernel phases on each rank')
    for rank in range(4):
        sequence=[observations[(rank,phase)] for phase in PHASES]
        if any(later['start_ns']<earlier['end_ns']
               for earlier,later in zip(sequence,sequence[1:])):
            raise ValueError('rank-local kernel order differs')
    first=min(row['start_ns'] for row in rows)
    last=max(row['end_ns'] for row in rows)
    cross_rank=[]
    for earlier,later in ((PHASES[0],PHASES[1]),
                          (PHASES[1],PHASES[2]),
                          (PHASES[2],PHASES[3])):
        pairs=[]
        for a in range(4):
            for b in range(4):
                if a==b:continue
                x=observations[(a,earlier)];y=observations[(b,later)]
                overlap=max(0,min(x['end_ns'],y['end_ns'])
                             -max(x['start_ns'],y['start_ns']))
                if overlap:pairs.append({'earlier_rank':a,'later_rank':b,
                                         'overlap_ns':overlap})
        cross_rank.append({'earlier':earlier,'later':later,'pairs':pairs})
    physical=receipt['gpu_ids']
    result={'schema_version':1,
            'scope':'single Nsight CUDA-activity development trace; no CUPTI/L2-reset qualification or speedup',
            'source_commit':manifest['source_commit'],
            'broker_job':receipt['job_id'],
            'communication_ctas':case['communication_ctas'],
            'steal_budget_per_owner':case['steal_budget_per_owner'],
            'kernel_activity_union_ns':last-first,
            'cross_rank_phase_overlap':cross_rank,
            'ranks':[
                {'rank':rank,'physical_gpu':physical[rank],
                 'phases':[
                    {'name':phase,
                     'start_offset_ns':observations[(rank,phase)]['start_ns']-first,
                     'end_offset_ns':observations[(rank,phase)]['end_ns']-first,
                     'duration_ns':observations[(rank,phase)]['end_ns']
                                   -observations[(rank,phase)]['start_ns'],
                     'registers_per_thread':observations[(rank,phase)]['registers']}
                    for phase in PHASES]}
                for rank in range(4)]}
    (root/'activity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'source_commit':result['source_commit'],
                      'broker_job':result['broker_job'],
                      'kernel_activity_union_ns':result['kernel_activity_union_ns'],
                      'cross_rank_pairs':[(part['earlier'],part['later'],
                                           len(part['pairs']))
                                          for part in cross_rank]},indent=2))


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('run','export','analyze'))
    parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args();root=a.root.expanduser().resolve(strict=True)
    {'run':run,'export':export,'analyze':analyze}[a.phase](root)


if __name__=='__main__':main()
