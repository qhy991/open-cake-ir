"""Brokered B300 EP4 peer-bin transport and GPU tile materialization."""
from __future__ import annotations

import argparse
from array import array
import json
import os
from pathlib import Path
import subprocess
import sys
import time


LABEL='cake-weave-model-ep4-p2p-tile-gather-b300'
NVCC='/usr/local/cuda-13.1/bin/nvcc'
FLAGS=['-std=c++17','--gpu-architecture=compute_103a',
       '--gpu-code=sm_103a','-O3','--fmad=false','-lineinfo','-Xptxas=-v']
R,T,K,E,H=4,512,8,128,2048
HIDDEN_BYTES=R*T*H*2
IDS_BYTES=R*T*K*4
TILE_BYTES=64*128*H*2
TILE_KEY_BYTES=64*128*4


def document(path:Path)->dict:
    return json.loads(path.read_text())


def require_root(root:Path)->dict:
    manifest=document(root/'manifest.json')
    if (manifest.get('target')!='sm_103a'
            or len(manifest.get('source_commit',''))!=40
            or manifest.get('world_size')!=R
            or not (root/'model_ep4_p2p_tile_gather.cu').is_file()
            or not (root/'bin_pack_oracle.py').is_file()):
        raise ValueError('EP4 peer-bin source or manifest differs')
    return manifest


def prepare(root:Path,prior:Path,prepared:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('EP4 input preparation must be outside the GPU lease')
    manifest=require_root(root)
    if any((root/name).exists() for name in
           ('hidden.bf16','expert_ids.i32','case.json','device_outputs')):
        raise ValueError('EP4 peer-bin inputs must be create-only')
    prior_report=document(prior/'report.json')
    prior_case=document(prior/'case.json')
    prepared_case=document(prepared/'case.json')
    hidden=(prior/'hidden.bf16').read_bytes()
    ids=(prior/'expert_ids.i32').read_bytes()
    if (not prior_report.get('passed')
            or prior_case.get('geometry')!={'R':R,'T':T,'K':K,'E':E,'H':H}
            or prepared_case.get('geometry')
            !={'R':R,'T':T,'K':K,'E':E,'H':H,'tile_rows':128}
            or [row['logical_tiles'] for row in prepared_case['owners']]!=[64]*R
            or len(hidden)!=HIDDEN_BYTES or len(ids)!=IDS_BYTES):
        raise ValueError('retained Cake bin oracle input differs')
    values=array('i');values.frombytes(ids)
    if sys.byteorder!='little' or values.itemsize!=4 or len(values)!=R*T*K:
        raise ValueError('exact little-endian INT32 route domain required')
    owner_counts=[0]*R
    for token in range(R*T):
        row=values[token*K:(token+1)*K]
        if any(not 0<=expert<E for expert in row) or len(set(row))!=K:
            raise ValueError('expert IDs violate the declared route domain')
        for expert in row:owner_counts[expert//(E//R)]+=1
    (root/'hidden.bf16').write_bytes(hidden)
    (root/'expert_ids.i32').write_bytes(ids)
    for owner in range(R):
        (root/'device_outputs'/f'owner{owner}').mkdir(parents=True)
        for name,size in (('x_tiles.bf16',TILE_BYTES),
                          ('tile_route_keys.i32',TILE_KEY_BYTES),
                          ('tile_expert.i32',64*4)):
            if (prepared/f'rank{owner}'/name).stat().st_size!=size:
                raise ValueError(f'prepared owner {owner} tile contract differs')
    (root/'case.json').write_text(json.dumps({
        'source_commit':manifest['source_commit'],
        'source_case':str(prior.resolve()),
        'geometry':{'R':R,'T':T,'K':K,'E':E,'H':H},
        'hidden_bytes':HIDDEN_BYTES,'ids_bytes':IDS_BYTES,
        'owner_route_counts':owner_counts,
        'prepared_root':str(prepared.resolve()),
        'tile_bytes_per_owner':TILE_BYTES,
        'scope':'EP4 peer bin system release/acquire into GPU-materialized model tiles',
    },indent=2)+'\n')


def build(root:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must be outside a broker GPU lease')
    manifest=require_root(root)
    case=document(root/'case.json')
    if (case.get('source_commit')!=manifest['source_commit']
            or case.get('geometry')!={'R':R,'T':T,'K':K,'E':E,'H':H}
            or case.get('hidden_bytes')!=HIDDEN_BYTES
            or case.get('ids_bytes')!=IDS_BYTES
            or case.get('tile_bytes_per_owner')!=TILE_BYTES
            or (root/'hidden.bf16').stat().st_size!=HIDDEN_BYTES
            or (root/'expert_ids.i32').stat().st_size!=IDS_BYTES):
        raise ValueError('EP4 input extent or source identity differs')
    if not Path(NVCC).is_file() or any((root/name).exists() for name in
                                         ('bin_probe','build_plan.json',
                                          'build_report.json','build_failure.json')):
        raise ValueError('EP4 build must be exact and create-only')
    command=[NVCC,*FLAGS,'model_ep4_p2p_tile_gather.cu','-o','bin_probe',
             '-lcuda','-lcudart']
    (root/'build_plan.json').write_text(json.dumps(command,indent=2)+'\n')
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,
                          timeout=180,check=False)
    (root/'compile.log').write_text(result.stdout+result.stderr or
                                    '(no compiler output)\n')
    if result.returncode:
        (root/'build_failure.json').write_text(json.dumps({
            'exit_code':result.returncode},indent=2)+'\n')
        raise RuntimeError('EP4 peer-bin nvcc failed')
    if (root/'bin_probe').read_bytes()[:4]!=b'\x7fELF':
        raise RuntimeError('EP4 peer-bin executable is not ELF')
    (root/'build_report.json').write_text(json.dumps({
        'source_commit':manifest['source_commit'],'target':'sm_103a',
        'exit_code':0,'scope':'CPU nvcc/PTXAS only; no device result or timing',
    },indent=2)+'\n')


def admitted()->str:
    path=Path(os.environ.get('BROKER_RECEIPT','/nonexistent'))
    for _ in range(30):
        if path.is_file():break
        time.sleep(.1)
    else:raise RuntimeError('broker admission receipt missing')
    receipt=document(path)
    if (receipt.get('label')!=LABEL or receipt.get('mode')!='exclusive'
            or receipt.get('gpu_count')!=R
            or len(receipt.get('gpu_ids',[]))!=R
            or not isinstance(receipt.get('job_id'),str)):
        raise RuntimeError('exact exclusive four-GPU admission differs')
    return receipt['job_id']


def run(root:Path)->None:
    manifest=require_root(root)
    build_report=document(root/'build_report.json')
    if (build_report.get('source_commit')!=manifest['source_commit']
            or build_report.get('exit_code')!=0
            or (root/'device.json').exists()
            or (root/'launch_failure.json').exists()):
        raise ValueError('EP4 source/build or create-only device evidence differs')
    job=admitted()
    try:
        prepared=document(root/'case.json')['prepared_root']
        result=subprocess.run([str(root/'bin_probe'),str(root),prepared],cwd=root,
                              capture_output=True,text=True,timeout=240,
                              check=False)
        (root/'device.log').write_text(result.stdout+result.stderr or
                                       '(no device output)\n')
        if result.returncode:
            raise RuntimeError(f'EP4 peer-bin probe exited {result.returncode}')
        (root/'device.json').write_text(json.dumps({
            'broker_job':job,'source_commit':manifest['source_commit'],
            'target':'sm_103a','world_size':R,
            'scope':'four-GPU BF16 peer-bin and GPU tile gather; no FFN or latency',
        },indent=2)+'\n')
    except Exception as error:
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job':job,'error':f'{type(error).__name__}: {error}'},
            indent=2)+'\n')
        raise


def verify(root:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('CPU oracle must run after broker release')
    if (root/'report.json').exists():
        raise ValueError('EP4 oracle report must be create-only')
    manifest=require_root(root)
    case=document(root/'case.json')
    device=document(root/'device.json')
    receipt=document(root/'gpuq-admission.json')
    if (device['broker_job']!=receipt['job_id']
            or device['source_commit']!=manifest['source_commit']
            or device['world_size']!=R):
        raise ValueError('EP4 broker/source evidence differs')
    sys.path.insert(0,str(root))
    from bin_pack_oracle import verify_rows
    owners=verify_rows(root)
    if ([row['routes'] for row in owners]!=case['owner_route_counts']
            or sum(row['routes'] for row in owners)!=R*T*K):
        raise ValueError('EP4 owner count oracle differs')
    prepared=Path(case['prepared_root']).resolve(strict=True)
    for owner in range(R):
        actual=root/'device_outputs'/f'owner{owner}'/'tiles.bf16'
        expected=prepared/f'rank{owner}'/'x_tiles.bf16'
        if (actual.stat().st_size!=TILE_BYTES
                or expected.stat().st_size!=TILE_BYTES
                or actual.read_bytes()!=expected.read_bytes()):
            raise ValueError(f'owner {owner} GPU tile rows differ from checked route plan')
    report={'passed':True,'target':'sm_103a',
            'source_commit':manifest['source_commit'],
            'broker_job':device['broker_job'],'owners':owners,
            'tile_rows_bitwise_equal_to_checked_plan':True,
            'scope':'four-GPU P2P expert bins and GPU tile materialization; no FFN, latency or speedup'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','build','run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--prior-root',type=Path)
    parser.add_argument('--prepared-root',type=Path)
    args=parser.parse_args()
    root=args.root.expanduser().resolve(strict=True)
    if args.phase=='prepare':
        if args.prior_root is None or args.prepared_root is None:
            parser.error('prepare needs --prior-root and --prepared-root')
        prepare(root,args.prior_root.expanduser().resolve(strict=True),
                args.prepared_root.expanduser().resolve(strict=True))
    else:
        {'build':build,'run':run,'verify':verify}[args.phase](root)


if __name__=='__main__':main()
