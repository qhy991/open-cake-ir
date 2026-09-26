"""One brokered B300 EP4 chain with Cake FFN and GPU return/combine."""
from __future__ import annotations

import argparse
from array import array
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


LABEL='cake-weave-ep4-live-chain-b300'
NVCC='/usr/local/cuda-13.1/bin/nvcc'
FLAGS=['-std=c++17','--gpu-architecture=compute_103a',
       '--gpu-code=sm_103a','-O3','--fmad=false','-lineinfo','-Xptxas=-v']
R,T,K,E,H,TILES,ROWS=4,512,8,128,2048,64,128
ROUTES=R*T*K
HIDDEN_BYTES=R*T*H*2
IDS_BYTES=ROUTES*4
TILE_BYTES=TILES*ROWS*H*2
DOWN_BYTES=TILES*ROWS*H*4
CONTRIBUTION_BYTES=ROUTES*H*4
OUTPUT_BYTES=R*T*H*2
ADMITTED_CONTROLS={(1,0),(74,5888),(147,5888),(147,11776),(148,11776)}


def doc(path:Path)->dict:
    return json.loads(path.read_text())


def require_root(root:Path)->dict:
    m=doc(root/'manifest.json')
    if (m.get('target')!='sm_103a'
            or len(m.get('source_commit',''))!=40
            or len(m.get('combine_compiler_commit',''))!=40
            or m.get('generation')!='cake_ep4_live_chain_stages'
            or m.get('world_size')!=R
            or type(m.get('communication_ctas')) is not int
            or type(m.get('steal_budget')) is not int
            or (m.get('communication_ctas'),m.get('steal_budget'))
               not in ADMITTED_CONTROLS
            or not (root/'model_tile_ready_ffn_capped.cu').is_file()
            or not (root/'combine/kernel.cu').is_file()):
        raise ValueError('live EP4 Cake source or manifest differs')
    return m


def prepare(root:Path,bin_root:Path,prepared:Path,bridge:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('live chain preparation must be outside GPU lease')
    m=require_root(root)
    if (root/'case.json').exists() or (root/'device_outputs').exists():
        raise ValueError('live chain inputs must be create-only')
    bin_report=doc(bin_root/'report.json')
    bridge_report=doc(bridge/'report.json')
    p=doc(prepared/'case.json')
    if (not bin_report.get('passed') or not bridge_report.get('passed')
            or bridge_report.get('routes')!=ROUTES
            or p.get('geometry')!={'R':R,'T':T,'K':K,'E':E,'H':H,
                                   'tile_rows':ROWS}
            or [row['logical_tiles'] for row in p['owners']]!=[TILES]*R
            or [row['stage_work_units'] for row in p['owners']]!=[11776]*R
            or (bin_root/'hidden.bf16').stat().st_size!=HIDDEN_BYTES
            or (bin_root/'expert_ids.i32').stat().st_size!=IDS_BYTES
            or (bridge/'weights_upgate.bf16').stat().st_size!=E*1536*H*2
            or (bridge/'weights_down.bf16').stat().st_size!=E*H*768*2
            or (bridge/'route_weights.fp32').stat().st_size!=ROUTES*4):
        raise ValueError('live chain exact model inputs or checked plan differ')
    ids=array('i');ids.frombytes((bin_root/'expert_ids.i32').read_bytes())
    if sys.byteorder!='little' or ids.itemsize!=4 or len(ids)!=ROUTES:
        raise ValueError('exact little-endian route IDs required')
    seen=bytearray(ROUTES)
    owner_rows=[]
    for owner in range(R):
        rank=prepared/f'rank{owner}'
        keys=array('i');keys.frombytes((rank/'tile_route_keys.i32').read_bytes())
        experts=array('i');experts.frombytes((rank/'tile_expert.i32').read_bytes())
        if (len(keys)!=TILES*ROWS or len(experts)!=TILES
                or (rank/'x_tiles.bf16').stat().st_size!=TILE_BYTES):
            raise ValueError(f'owner {owner} plan input extent differs')
        count=0
        for tile in range(TILES):
            local=int(experts[tile])
            if not 0<=local<E//R:
                raise ValueError(f'owner {owner} local expert differs')
            for row in range(ROWS):
                key=int(keys[tile*ROWS+row])
                if key<0:continue
                if (not 0<=key<ROUTES or seen[key]
                        or ids[key]!=owner*(E//R)+local):
                    raise ValueError(f'owner {owner} tile route key differs')
                seen[key]=1;count+=1
        if count!=p['owners'][owner]['valid_routes']:
            raise ValueError(f'owner {owner} tile route count differs')
        owner_rows.append(count)
    if any(flag!=1 for flag in seen) or owner_rows!=[4039,4196,4016,4133]:
        raise ValueError('live chain plan does not cover all model routes')
    (root/'device_outputs').mkdir()
    for owner in range(R):(root/'device_outputs'/f'rank{owner}').mkdir()
    (root/'case.json').write_text(json.dumps({
        'source_commit':m['source_commit'],
        'combine_compiler_commit':m['combine_compiler_commit'],
        'bin_root':str(bin_root.resolve()),
        'prepared_root':str(prepared.resolve()),
        'bridge_root':str(bridge.resolve()),
        'geometry':{'R':R,'T':T,'K':K,'E':E,'H':H,'tile_rows':ROWS},
        'owner_routes':owner_rows,'logical_tiles_per_owner':TILES,
        'stage_work_units_per_owner':11776,
        'communication_ctas':m['communication_ctas'],
        'steal_budget_per_owner':m['steal_budget'],
        'scope':'one four-GPU allocation, CPU route-key plan, GPU dispatch/gather/FFN/return/combine',
    },indent=2)+'\n')


def build(root:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must be outside GPU lease')
    m=require_root(root);c=doc(root/'case.json')
    if (c.get('source_commit')!=m['source_commit']
            or c.get('combine_compiler_commit')!=m['combine_compiler_commit']
            or c.get('owner_routes')!=[4039,4196,4016,4133]
            or c.get('stage_work_units_per_owner')!=11776
            or (c.get('communication_ctas'),c.get('steal_budget_per_owner'))
               !=(m['communication_ctas'],m['steal_budget'])):
        raise ValueError('live chain source or plan binding differs')
    if not Path(NVCC).is_file() or any((root/name).exists() for name in
                                         ('live_probe','build_plan.json',
                                          'build_report.json','build_failure.json')):
        raise ValueError('live chain build must be exact and create-only')
    command=[NVCC,*FLAGS,'model_tile_ready_ffn_capped.cu','-o',
             'live_probe','-lcuda','-lcudart']
    (root/'build_plan.json').write_text(json.dumps(command,indent=2)+'\n')
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,
                          timeout=240,check=False)
    (root/'compile.log').write_text(result.stdout+result.stderr or
                                    '(no compiler output)\n')
    if result.returncode:
        (root/'build_failure.json').write_text(json.dumps({
            'exit_code':result.returncode},indent=2)+'\n')
        raise RuntimeError('live EP4 CUDA build failed')
    if (root/'live_probe').read_bytes()[:4]!=b'\x7fELF':
        raise RuntimeError('live EP4 executable is not ELF')
    (root/'build_report.json').write_text(json.dumps({
        'source_commit':m['source_commit'],'target':'sm_103a','exit_code':0,
        'scope':'CPU nvcc/PTXAS only; no full-chain GPU result or timing',
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
        raise RuntimeError('exact four-GPU live-chain admission differs')
    return row['job_id']


def run(root:Path)->None:
    m=require_root(root);c=doc(root/'case.json')
    build_report=doc(root/'build_report.json')
    if (build_report.get('source_commit')!=m['source_commit']
            or build_report.get('exit_code')!=0
            or (root/'device.json').exists()
            or (root/'launch_failure.json').exists()):
        raise ValueError('live chain source/build or device evidence differs')
    job=admitted()
    try:
        result=subprocess.run([str(root/'live_probe'),str(root),
                               c['bin_root'],c['prepared_root'],
                               c['bridge_root'],str(c['communication_ctas']),
                               str(c['steal_budget_per_owner'])],cwd=root,
                              capture_output=True,text=True,timeout=300,
                              check=False)
        (root/'device.log').write_text(result.stdout+result.stderr or
                                       '(no device output)\n')
        if result.returncode:
            raise RuntimeError(f'live EP4 chain exited {result.returncode}')
        (root/'device.json').write_text(json.dumps({
            'broker_job':job,'source_commit':m['source_commit'],
            'combine_compiler_commit':m['combine_compiler_commit'],
            'target':'sm_103a','world_size':R,
            'scope':'one allocation GPU dispatch/gather/Cake FFN/P2P return/Cake combine; CPU plan, no qualified timing',
        },indent=2)+'\n')
    except Exception as error:
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job':job,'error':f'{type(error).__name__}: {error}'},
            indent=2)+'\n')
        raise


def bf16_as_fp32(values:np.ndarray)->np.ndarray:
    return (values.astype('<u4')<<16).view('<f4')


def verify(root:Path)->None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('oracle must run after broker release')
    if (root/'report.json').exists():
        raise ValueError('live chain report must be create-only')
    m=require_root(root);c=doc(root/'case.json')
    d=doc(root/'device.json');receipt=doc(root/'gpuq-admission.json')
    device=doc(root/'device_report.json')
    if (d['broker_job']!=receipt['job_id']
            or d['source_commit']!=m['source_commit']
            or d['combine_compiler_commit']!=m['combine_compiler_commit']
            or d['world_size']!=R
            or device['target']!='sm_103a'
            or device['valid_routes_by_owner']!=c['owner_routes']
            or device['bin_rows_by_owner']!=c['owner_routes']
            or device['communication_ctas']!=c['communication_ctas']
            or device['steal_budget']!=c['steal_budget_per_owner']
            or len(device['stolen_by_owner'])!=R
            or any(type(stolen) is not int or not 0<=stolen<=c['steal_budget_per_owner']
                   for stolen in device['stolen_by_owner'])
            or (c['communication_ctas']==148
                and device['stolen_by_owner']!=[11776]*R)
            or device['sm_counts']!=[148]*R
            or device['active_blocks_per_sm']!=[1]*R
            or device['overlap_flags']!=[[1,1]]*R
            or device['no_interphase_host_sync'] is not True):
        raise ValueError('live chain device, broker or task evidence differs')
    output=root/'device_outputs'
    bin_root=Path(c['bin_root']).resolve(strict=True)
    prepared=Path(c['prepared_root']).resolve(strict=True)
    bridge=Path(c['bridge_root']).resolve(strict=True)
    if ((output/'observed_hidden.bf16').read_bytes()
            !=(bin_root/'hidden.bf16').read_bytes()
            or (output/'observed_expert_ids.i32').read_bytes()
            !=(bin_root/'expert_ids.i32').read_bytes()):
        raise ValueError('live chain changed declared route input')
    for rank in range(R):
        actual=output/f'rank{rank}'/'tile_input.bf16'
        expected=prepared/f'rank{rank}'/'x_tiles.bf16'
        if (actual.stat().st_size!=TILE_BYTES
                or actual.read_bytes()!=expected.read_bytes()):
            raise ValueError(f'rank {rank} GPU tile gather differs from checked plan')
        if (output/f'rank{rank}'/'down.fp32').stat().st_size!=DOWN_BYTES:
            raise ValueError(f'rank {rank} down output extent differs')
    if ((output/'return_ready.i32').stat().st_size!=ROUTES*4
            or not np.all(np.fromfile(output/'return_ready.i32',dtype='<i4')==1)
            or (output/'contributions.fp32').stat().st_size
               !=CONTRIBUTION_BYTES
            or (output/'output.bf16').stat().st_size!=OUTPUT_BYTES):
        raise ValueError('live returned contribution or output extent differs')
    actual=np.memmap(output/'contributions.fp32',dtype='<f4',mode='r',
                     shape=(ROUTES,H))
    prior=np.memmap(bridge/'device_outputs/contributions.fp32',
                    dtype='<f4',mode='r',shape=(ROUTES,H))
    contribution_mismatch=int(np.count_nonzero(actual.view('<u4')
                                                !=prior.view('<u4')))
    if not np.isfinite(actual).all():
        raise ValueError('live route contribution contains a nonfinite value')
    output_bits=np.fromfile(output/'output.bf16',dtype='<u2').reshape(R,T,H)
    actual_output=bf16_as_fp32(output_bits)
    expected=np.load(bridge/'expected_output.npy',mmap_mode='r')
    error=np.abs(actual_output.astype(np.float64)-expected.astype(np.float64))
    failing=error>0.01+0.01*np.abs(expected.astype(np.float64))
    report={'passed':contribution_mismatch==0 and not np.any(failing),
            'target':'sm_103a','source_commit':m['source_commit'],
            'combine_compiler_commit':m['combine_compiler_commit'],
            'broker_job':d['broker_job'],'routes':ROUTES,
            'communication_ctas':c['communication_ctas'],
            'steal_budget_per_owner':c['steal_budget_per_owner'],
            'actual_stolen_by_owner':device['stolen_by_owner'],
            'gpu_tile_inputs_bitwise_equal_to_checked_plan':True,
            'route_contribution_bit_mismatches':contribution_mismatch,
            'failing_elements':int(np.count_nonzero(failing)),
            'per_rank_failing_elements':[
                int(np.count_nonzero(failing[rank])) for rank in range(R)],
            'max_abs_error':float(np.max(error)),'atol':0.01,'rtol':0.01,
            'scope':'one four-GPU allocation with GPU dispatch/gather/Cake FFN/P2P return/Cake GPU combine; CPU route-key tile plan and no qualified timing'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','build','run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--bin-root',type=Path)
    parser.add_argument('--prepared-root',type=Path)
    parser.add_argument('--bridge-root',type=Path)
    a=parser.parse_args();root=a.root.expanduser().resolve(strict=True)
    if a.phase=='prepare':
        if any(item is None for item in
               (a.bin_root,a.prepared_root,a.bridge_root)):
            parser.error('prepare needs --bin-root, --prepared-root, --bridge-root')
        prepare(root,a.bin_root.expanduser().resolve(strict=True),
                a.prepared_root.expanduser().resolve(strict=True),
                a.bridge_root.expanduser().resolve(strict=True))
    else:{'build':build,'run':run,'verify':verify}[a.phase](root)


if __name__=='__main__':main()
