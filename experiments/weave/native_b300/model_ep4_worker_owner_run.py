"""Four-rank Cake FFN worker over host-materialized real EP4 peer-bin rows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


LABEL='cake-weave-ep4-worker-bridge-b300'
NVCC='/usr/local/cuda-13.1/bin/nvcc'
FLAGS=['-std=c++17','--gpu-architecture=compute_103a',
       '--gpu-code=sm_103a','-O3','--fmad=false','-lineinfo','-Xptxas=-v']
R,T,K,H,B,TILES=4,512,8,2048,128,64
ROUTES=R*T*K
TASKS=[[0,408,360,768],[0,2176,1920,4096],[0,544,480,1024]]
CASE='near_full_c147_budget5888'
UPGATE_BYTES=TILES*B*1536*4
ACTIVATED_BYTES=TILES*B*768*2
DOWN_BYTES=TILES*B*H*4
INPUT_BYTES=TILES*B*H*2
KEY_BYTES=TILES*B*4


def doc(path:Path)->dict:
    return json.loads(path.read_text())


def require_root(root:Path)->dict:
    m=doc(root/'manifest.json')
    if (m.get('target')!='sm_103a'
            or len(m.get('source_commit',''))!=40
            or m.get('world_size')!=R
            or m.get('generation')!='cake_ep4_owner_stages'
            or not (root/'model_tile_ready_ffn_capped.cu').is_file()):
        raise ValueError('four-rank Cake worker source or manifest differs')
    return m


def prepare(root:Path,prepared:Path,bridge:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('worker preparation must run outside a GPU lease')
    m=require_root(root)
    if (root/'case.json').exists() or any((root/f'rank{r}').exists() for r in range(R)):
        raise ValueError('worker binding must be create-only')
    p=doc(prepared/'case.json')
    b=doc(bridge/'report.json')
    if (not b.get('passed') or b.get('routes')!=ROUTES
            or p.get('geometry')!={'R':R,'T':T,'K':K,'E':128,'H':H,'tile_rows':B}
            or [row['logical_tiles'] for row in p['owners']]!=[TILES]*R
            or [row['stage_work_units'] for row in p['owners']]!=[11776]*R
            or not (bridge/'device_outputs/contributions.fp32').is_file()
            or (bridge/'device_outputs/contributions.fp32').stat().st_size
               !=ROUTES*H*4
            or (bridge/'weights_upgate.bf16').stat().st_size!=128*1536*H*2
            or (bridge/'weights_down.bf16').stat().st_size!=128*H*768*2):
        raise ValueError('actual bin tile plan or reference bridge differs')
    for owner in range(R):
        src=prepared/f'rank{owner}'
        dst=root/f'rank{owner}'
        dst.mkdir()
        for name,size in (('x_tiles.bf16',INPUT_BYTES),
                          ('tile_expert.i32',TILES*4),
                          ('tile_route_keys.i32',KEY_BYTES)):
            path=src/name
            if path.stat().st_size!=size:
                raise ValueError(f'owner {owner} {name} extent differs')
            (dst/name).symlink_to(path.resolve(strict=True))
        experts=np.fromfile(src/'tile_expert.i32',dtype='<i4')
        keys=np.fromfile(src/'tile_route_keys.i32',dtype='<i4')
        if (experts.size!=TILES or np.any((experts<0)|(experts>=32))
                or keys.size!=TILES*B
                or np.count_nonzero(keys>=0)!=p['owners'][owner]['valid_routes']):
            raise ValueError(f'owner {owner} tile metadata differs')
    (root/'case.json').write_text(json.dumps({
        'source_commit':m['source_commit'],
        'prepared_root':str(prepared.resolve()),
        'bridge_root':str(bridge.resolve()),
        'geometry':p['geometry'],'owners':p['owners'],
        'case':CASE,'tasks':TASKS,'communication_ctas':147,'steal_budget':5888,
        'scope':'four-rank Cake worker over host-materialized P2P bin rows; no live GPU tile publication',
    },indent=2)+'\n')


def build(root:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must run outside a GPU lease')
    m=require_root(root);c=doc(root/'case.json')
    if (c.get('source_commit')!=m['source_commit'] or c.get('tasks')!=TASKS
            or c.get('case')!=CASE or c.get('communication_ctas')!=147
            or c.get('steal_budget')!=5888):
        raise ValueError('four-rank worker case binding differs')
    if not Path(NVCC).is_file() or any((root/name).exists() for name in
                                         ('owner_probe','build_plan.json',
                                          'build_report.json','build_failure.json')):
        raise ValueError('four-rank worker build must be exact and create-only')
    command=[NVCC,*FLAGS,'model_tile_ready_ffn_capped.cu','-o',
             'owner_probe','-lcuda','-lcudart']
    (root/'build_plan.json').write_text(json.dumps(command,indent=2)+'\n')
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,
                          timeout=180,check=False)
    (root/'compile.log').write_text(result.stdout+result.stderr or
                                    '(no compiler output)\n')
    if result.returncode:
        (root/'build_failure.json').write_text(json.dumps({
            'exit_code':result.returncode},indent=2)+'\n')
        raise RuntimeError('four-rank worker nvcc failed')
    if (root/'owner_probe').read_bytes()[:4]!=b'\x7fELF':
        raise RuntimeError('worker executable is not ELF')
    (root/'build_report.json').write_text(json.dumps({
        'source_commit':m['source_commit'],'target':'sm_103a','exit_code':0,
        'scope':'CPU nvcc/PTXAS only; no four-GPU correctness or timing',
    },indent=2)+'\n')


def admitted()->str:
    path=Path(os.environ.get('BROKER_RECEIPT','/nonexistent'))
    for _ in range(30):
        if path.is_file():break
        time.sleep(.1)
    else:raise RuntimeError('broker admission receipt missing')
    row=doc(path)
    if (row.get('label')!=LABEL or row.get('mode')!='exclusive'
            or row.get('gpu_count')!=R or len(row.get('gpu_ids',[]))!=R
            or not isinstance(row.get('job_id'),str)):
        raise RuntimeError('exact exclusive four-GPU admission differs')
    return row['job_id']


def run(root:Path)->None:
    m=require_root(root);c=doc(root/'case.json')
    build_report=doc(root/'build_report.json')
    if (build_report.get('source_commit')!=m['source_commit']
            or build_report.get('exit_code')!=0
            or (root/'device.json').exists()
            or (root/'launch_failure.json').exists()):
        raise ValueError('worker source/build or device evidence differs')
    job=admitted()
    procs=[];logs=[]
    try:
        for owner in range(R):
            handle=(root/f'rank{owner}'/'device.log').open('x')
            logs.append(handle)
            procs.append(subprocess.Popen([
                str(root/'owner_probe'),str(root/f'rank{owner}'),
                str(owner),c['bridge_root']],cwd=root,
                stdout=handle,stderr=subprocess.STDOUT))
        for owner,proc in enumerate(procs):
            result=proc.wait(timeout=240)
            if result:
                raise RuntimeError(f'owner {owner} worker exited {result}')
        (root/'device.json').write_text(json.dumps({
            'broker_job':job,'source_commit':m['source_commit'],
            'target':'sm_103a','world_size':R,'case':CASE,
            'scope':'four-GPU Cake FFN over P2P bin rows; host tile formation, no P2P return or timing',
        },indent=2)+'\n')
    except Exception as error:
        for proc in procs:
            if proc.poll() is None:proc.kill()
        for proc in procs:proc.wait()
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job':job,'error':f'{type(error).__name__}: {error}'},
            indent=2)+'\n')
        raise
    finally:
        for handle in logs:handle.close()


def bf16_round(values:np.ndarray)->np.ndarray:
    bits=np.ascontiguousarray(values,dtype='<f4').view('<u4')
    rounded=bits+np.uint32(0x7fff)+((bits>>16)&1)
    return (rounded&np.uint32(0xffff0000)).view('<f4')


def verify(root:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('CPU oracle must run after broker release')
    if (root/'report.json').exists():
        raise ValueError('worker report must be create-only')
    m=require_root(root);c=doc(root/'case.json')
    d=doc(root/'device.json');receipt=doc(root/'gpuq-admission.json')
    if (d['broker_job']!=receipt['job_id']
            or d['source_commit']!=m['source_commit']
            or d['world_size']!=R or d['case']!=CASE):
        raise ValueError('worker broker/source evidence differs')
    bridge=Path(c['bridge_root']).resolve(strict=True)
    reference=np.memmap(bridge/'device_outputs/contributions.fp32',
                        dtype='<f4',mode='r',shape=(ROUTES,H))
    actual=np.zeros((ROUTES,H),dtype='<f4')
    seen=np.zeros(ROUTES,dtype=np.uint8)
    owner_reports=[]
    bit_mismatches=0;max_route_abs=0.0
    for owner in range(R):
        rank=root/f'rank{owner}'
        row=doc(rank/f'{CASE}.json')
        resource=doc(rank/'resource.json')
        if (row['name']!=CASE or row['tasks']!=TASKS
                or row['processed']!=TASKS
                or row['communication_ctas']!=147
                or row['steal_budget']!=5888
                or row['permit_count']!=5888
                or sum(map(sum,row['stolen']))!=5888
                or row['dispatched']!=[147]*4
                or row['tile_completed']!=[[24]*TILES,[128]*TILES,[32]*TILES]
                or row['stage_overlap']!=[1,1]
                or resource['owner_rank']!=owner
                or resource['target']!='sm_103a'
                or resource['threads_per_cta']!=192
                or resource['dynamic_shared_bytes']!=49200
                or resource['active_blocks_per_sm']<1):
            raise ValueError(f'owner {owner} task, resource or steal report differs')
        for stage,waves in enumerate(row['owner_blocks']):
            for wave,owners in enumerate(waves):
                if (len(owners)!=TASKS[stage][wave]
                        or any(not 0<=block<148 for block in owners)):
                    raise ValueError(f'owner {owner} task owner observation differs')
        for name,size in ((f'{CASE}.upgate.actual.fp32',UPGATE_BYTES),
                          (f'{CASE}.activated.actual.bf16',ACTIVATED_BYTES),
                          (f'{CASE}.down.actual.fp32',DOWN_BYTES)):
            if (rank/name).stat().st_size!=size:
                raise ValueError(f'owner {owner} {name} byte extent differs')
        keys=np.fromfile(rank/'tile_route_keys.i32',dtype='<i4').reshape(TILES,B)
        out=np.memmap(rank/f'{CASE}.down.actual.fp32',dtype='<f4',
                      mode='r',shape=(TILES,B,H))
        used=0
        for tile in range(TILES):
            valid=keys[tile]>=0
            selected=keys[tile,valid]
            if np.any(seen[selected]):
                raise ValueError(f'owner {owner} repeated a route contribution')
            actual[selected]=out[tile,valid]
            seen[selected]=1
            difference=np.abs(out[tile,valid].astype(np.float64)
                              -reference[selected].astype(np.float64))
            bit_mismatches+=int(np.count_nonzero(
                out[tile,valid].view('<u4')!=reference[selected].view('<u4')))
            if difference.size:max_route_abs=max(max_route_abs,float(np.max(difference)))
            used+=len(selected)
        if used!=c['owners'][owner]['valid_routes']:
            raise ValueError(f'owner {owner} valid route count differs')
        owner_reports.append({'owner_rank':owner,'routes':used,
                              'stolen_by_stage':[sum(stage) for stage in row['stolen']]})
    if not np.all(seen==1) or not np.isfinite(actual).all():
        raise ValueError('worker lost or invalidated a route contribution')
    weights=np.fromfile(bridge/'route_weights.fp32',dtype='<f4')
    if weights.size!=ROUTES:raise ValueError('route weight extent differs')
    combined=np.zeros((R*T,H),dtype=np.float64)
    for key in range(ROUTES):
        combined[key//K]+=actual[key].astype(np.float64)*float(weights[key])
    rounded=bf16_round(combined.astype('<f4')).reshape(R,T,H)
    expected=np.load(bridge/'expected_output.npy',mmap_mode='r')
    error=np.abs(rounded.astype(np.float64)-expected.astype(np.float64))
    failing=error>0.01+0.01*np.abs(expected.astype(np.float64))
    report={'passed':not np.any(failing),'target':'sm_103a',
            'source_commit':m['source_commit'],'broker_job':d['broker_job'],
            'owners':owner_reports,'route_bit_mismatches_vs_prior_cake':bit_mismatches,
            'max_route_abs_error_vs_prior_cake':max_route_abs,
            'failing_elements':int(np.count_nonzero(failing)),
            'per_rank_failing_elements':[
                int(np.count_nonzero(failing[rank])) for rank in range(R)],
            'max_abs_error':float(np.max(error)),'atol':0.01,'rtol':0.01,
            'scope':'four-GPU Cake FFN on actual peer bin rows with host tile formation and CPU combine; no live return or qualified timing'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','build','run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prepared-root',type=Path)
    parser.add_argument('--bridge-root',type=Path)
    a=parser.parse_args()
    root=a.root.expanduser().resolve(strict=True)
    if a.phase=='prepare':
        if a.prepared_root is None or a.bridge_root is None:
            parser.error('prepare needs --prepared-root and --bridge-root')
        prepare(root,a.prepared_root.expanduser().resolve(strict=True),
                a.bridge_root.expanduser().resolve(strict=True))
    else:
        {'build':build,'run':run,'verify':verify}[a.phase](root)


if __name__=='__main__':main()
