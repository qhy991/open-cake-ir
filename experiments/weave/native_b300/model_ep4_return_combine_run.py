"""Brokered EP4 P2P return and Cake rank-local GPU combine oracle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


LABEL='cake-weave-ep4-return-combine-b300'
NVCC='/usr/local/cuda-13.1/bin/nvcc'
FLAGS=['-std=c++17','--gpu-architecture=compute_103a',
       '--gpu-code=sm_103a','-O3','--fmad=false','-lineinfo','-Xptxas=-v']
R,T,K,H,TILES,ROWS=4,512,8,2048,64,128
ROUTES=R*T*K
DOWN_BYTES=TILES*ROWS*H*4
KEY_BYTES=TILES*ROWS*4
CONTRIBUTION_BYTES=ROUTES*H*4
OUTPUT_BYTES=R*T*H*2


def doc(path:Path)->dict:
    return json.loads(path.read_text())


def require_root(root:Path)->dict:
    m=doc(root/'manifest.json')
    if (m.get('target')!='sm_103a' or len(m.get('source_commit',''))!=40
            or len(m.get('combine_compiler_commit',''))!=40
            or m.get('world_size')!=R
            or not (root/'model_ep4_return_combine.cu').is_file()
            or not (root/'combine/kernel.cu').is_file()):
        raise ValueError('EP4 return source or manifest differs')
    return m


def prepare(root:Path,worker:Path,bridge:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('return preparation must run outside the GPU lease')
    m=require_root(root)
    if (root/'case.json').exists() or (root/'device_report.json').exists():
        raise ValueError('return evidence must be create-only')
    w=doc(worker/'report.json')
    b=doc(bridge/'report.json')
    if (not w.get('passed') or not b.get('passed')
            or w.get('source_commit')!='014d1f558f20c1374c1d4dba053d35b77c3495e0'
            or w.get('failing_elements')!=0
            or b.get('routes')!=ROUTES
            or (bridge/'route_weights.fp32').stat().st_size!=ROUTES*4
            or (bridge/'device_outputs/contributions.fp32').stat().st_size
               !=CONTRIBUTION_BYTES):
        raise ValueError('retained Cake worker or route-weight oracle differs')
    seen=np.zeros(ROUTES,dtype=np.uint8)
    counts=[]
    remote=0
    for owner in range(R):
        rank=worker/f'rank{owner}'
        keys=np.fromfile(rank/'tile_route_keys.i32',dtype='<i4')
        if keys.size!=TILES*ROWS or (rank/'near_full_c147_budget5888.down.actual.fp32').stat().st_size!=DOWN_BYTES:
            raise ValueError(f'owner {owner} route map or down extent differs')
        valid=keys[keys>=0]
        if np.any(valid>=ROUTES) or np.any(seen[valid]):
            raise ValueError(f'owner {owner} return route key repeats or exceeds bound')
        seen[valid]=1
        counts.append(int(valid.size))
        remote+=int(np.count_nonzero(valid//(T*K)!=owner))
    if (not np.all(seen==1) or counts!=[4039,4196,4016,4133]
            or remote!=12271):
        raise ValueError('EP4 return keys do not cover the exact synthetic route set')
    (root/'case.json').write_text(json.dumps({
        'source_commit':m['source_commit'],
        'combine_compiler_commit':m['combine_compiler_commit'],
        'worker_root':str(worker.resolve()),
        'bridge_root':str(bridge.resolve()),
        'geometry':{'R':R,'T':T,'K':K,'H':H,'tiles_per_owner':TILES},
        'routes':ROUTES,'valid_routes_by_owner':counts,
        'remote_owner_routes':remote,
        'contribution_bytes':CONTRIBUTION_BYTES,'output_bytes':OUTPUT_BYTES,
        'scope':'deterministic-key P2P return and Cake rank-local GPU combine; no full-layer timing',
    },indent=2)+'\n')


def build(root:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must run outside a broker GPU lease')
    m=require_root(root);c=doc(root/'case.json')
    if (c.get('source_commit')!=m['source_commit']
            or c.get('combine_compiler_commit')!=m['combine_compiler_commit']
            or c.get('routes')!=ROUTES
            or c.get('contribution_bytes')!=CONTRIBUTION_BYTES
            or c.get('output_bytes')!=OUTPUT_BYTES):
        raise ValueError('return source/geometry binding differs')
    if not Path(NVCC).is_file() or any((root/name).exists() for name in
                                         ('return_probe','build_plan.json',
                                          'build_report.json','build_failure.json')):
        raise ValueError('return build must be exact and create-only')
    command=[NVCC,*FLAGS,'model_ep4_return_combine.cu','-o',
             'return_probe','-lcuda','-lcudart']
    (root/'build_plan.json').write_text(json.dumps(command,indent=2)+'\n')
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,
                          timeout=180,check=False)
    (root/'compile.log').write_text(result.stdout+result.stderr or
                                    '(no compiler output)\n')
    if result.returncode:
        (root/'build_failure.json').write_text(json.dumps({
            'exit_code':result.returncode},indent=2)+'\n')
        raise RuntimeError('EP4 return/combination nvcc failed')
    if (root/'return_probe').read_bytes()[:4]!=b'\x7fELF':
        raise RuntimeError('return executable is not ELF')
    (root/'build_report.json').write_text(json.dumps({
        'source_commit':m['source_commit'],'target':'sm_103a','exit_code':0,
        'scope':'CPU nvcc/PTXAS only; no return/combine device result or timing',
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
        raise RuntimeError('exact exclusive four-GPU return admission differs')
    return row['job_id']


def run(root:Path)->None:
    m=require_root(root);c=doc(root/'case.json')
    build_report=doc(root/'build_report.json')
    if (build_report.get('source_commit')!=m['source_commit']
            or build_report.get('exit_code')!=0
            or (root/'device.json').exists()
            or (root/'launch_failure.json').exists()):
        raise ValueError('return source/build or create-only device evidence differs')
    job=admitted()
    try:
        result=subprocess.run([str(root/'return_probe'),str(root),
                               c['worker_root'],c['bridge_root']],cwd=root,
                              capture_output=True,text=True,timeout=240,
                              check=False)
        (root/'device.log').write_text(result.stdout+result.stderr or
                                       '(no device output)\n')
        if result.returncode:
            raise RuntimeError(f'EP4 return probe exited {result.returncode}')
        (root/'device.json').write_text(json.dumps({
            'broker_job':job,'source_commit':m['source_commit'],
            'combine_compiler_commit':m['combine_compiler_commit'],
            'target':'sm_103a','world_size':R,
            'scope':'four-GPU P2P return with GPU acquire and Cake combine; no layer timing',
        },indent=2)+'\n')
    except Exception as error:
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job':job,'error':f'{type(error).__name__}: {error}'},
            indent=2)+'\n')
        raise


def bf16_as_fp32(raw:np.ndarray)->np.ndarray:
    return (raw.astype('<u4')<<16).view('<f4')


def verify(root:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('return oracle must run after broker release')
    if (root/'report.json').exists():
        raise ValueError('return oracle report must be create-only')
    m=require_root(root);c=doc(root/'case.json')
    d=doc(root/'device.json');receipt=doc(root/'gpuq-admission.json')
    device=doc(root/'device_report.json')
    if (d['broker_job']!=receipt['job_id']
            or d['source_commit']!=m['source_commit']
            or d['combine_compiler_commit']!=m['combine_compiler_commit']
            or d['world_size']!=R
            or device['target']!='sm_103a' or device['ranks']!=R
            or device['valid_routes_by_owner']!=c['valid_routes_by_owner']
            or device['sm_counts']!=[148]*R
            or any(name not in ('NVIDIA B300','NVIDIA B300 SXM6 AC')
                   for name in device['device_names'])):
        raise ValueError('return broker, device or source evidence differs')
    worker=Path(c['worker_root']).resolve(strict=True)
    bridge=Path(c['bridge_root']).resolve(strict=True)
    expected_keys=b''.join((worker/f'rank{owner}'/'tile_route_keys.i32').read_bytes()
                           for owner in range(R))
    if ((root/'observed_keys.i32').read_bytes()!=expected_keys
            or (root/'ready.i32').stat().st_size!=ROUTES*4
            or not np.all(np.fromfile(root/'ready.i32',dtype='<i4')==1)):
        raise ValueError('return key inputs or acquired readiness differs')
    if ((root/'contributions.fp32').stat().st_size!=CONTRIBUTION_BYTES
            or (root/'output.bf16').stat().st_size!=OUTPUT_BYTES):
        raise ValueError('return contribution or BF16 output extent differs')
    actual=np.memmap(root/'contributions.fp32',dtype='<f4',mode='r',
                     shape=(ROUTES,H))
    prior=np.memmap(bridge/'device_outputs/contributions.fp32',
                    dtype='<f4',mode='r',shape=(ROUTES,H))
    bit_mismatch=int(np.count_nonzero(actual.view('<u4')!=prior.view('<u4')))
    if not np.isfinite(actual).all():
        raise ValueError('return contribution contains a nonfinite value')
    output_bits=np.fromfile(root/'output.bf16',dtype='<u2').reshape(R,T,H)
    actual_output=bf16_as_fp32(output_bits)
    expected=np.load(bridge/'expected_output.npy',mmap_mode='r')
    error=np.abs(actual_output.astype(np.float64)-expected.astype(np.float64))
    failing=error>0.01+0.01*np.abs(expected.astype(np.float64))
    report={'passed':bit_mismatch==0 and not np.any(failing),
            'target':'sm_103a','source_commit':m['source_commit'],
            'combine_compiler_commit':m['combine_compiler_commit'],
            'broker_job':d['broker_job'],'routes':ROUTES,
            'remote_owner_routes':c['remote_owner_routes'],
            'route_contribution_bit_mismatches':bit_mismatch,
            'failing_elements':int(np.count_nonzero(failing)),
            'per_rank_failing_elements':[
                int(np.count_nonzero(failing[rank])) for rank in range(R)],
            'max_abs_error':float(np.max(error)),'atol':0.01,'rtol':0.01,
            'scope':'four-GPU deterministic P2P return and Cake T512 GPU combine; earlier bin/FFN jobs and host tile formation remain separate; no qualified timing'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','build','run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--worker-root',type=Path)
    parser.add_argument('--bridge-root',type=Path)
    a=parser.parse_args();root=a.root.expanduser().resolve(strict=True)
    if a.phase=='prepare':
        if a.worker_root is None or a.bridge_root is None:
            parser.error('prepare needs --worker-root and --bridge-root')
        prepare(root,a.worker_root.expanduser().resolve(strict=True),
                a.bridge_root.expanduser().resolve(strict=True))
    else:{'build':build,'run':run,'verify':verify}[a.phase](root)


if __name__=='__main__':main()
