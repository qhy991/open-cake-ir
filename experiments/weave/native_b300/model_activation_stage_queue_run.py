"""Brokered B300 cooperative real SwiGLU stage-task development oracle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


LABEL='cake-weave-model-activation-stage-queue-b300'
NVCC='/usr/local/cuda-13.1/bin/nvcc'
FLAGS=['-std=c++17','--gpu-architecture=compute_103a',
       '--gpu-code=sm_103a','-O3','--fmad=false','-lineinfo','-Xptxas=-v']
CASES={
    'subcapacity_no_steal':(64,1,0,[0,2176,1920,4096]),
    'fullgrid_no_steal':(148,1,0,[0,2176,1920,4096]),
    'near_full_c147_budget4096':(148,147,4096,[0,2176,1920,4096]),
    'all_comm_c148_budget8192':(148,148,8192,[0,2176,1920,4096]),
}
LOGICAL_TILES=64
ROWS=128
INTERMEDIATE=768
UPGATE_BYTES=LOGICAL_TILES*ROWS*(2*INTERMEDIATE)*4
ACTIVATED_BYTES=LOGICAL_TILES*ROWS*INTERMEDIATE*2


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def require_root(root: Path) -> dict:
    manifest=document(root/'manifest.json')
    if (manifest.get('target')!='sm_103a'
            or len(manifest.get('source_commit',''))!=40
            or len(manifest.get('activation_compiler_commit',''))!=40
            or not (root/'model_activation_stage_queue.cu').is_file()
            or manifest.get('case_names')!=list(CASES)):
        raise ValueError('cooperative activation source or cases differ')
    return manifest


def prepare(root: Path, activation_root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('activation input preparation must be outside the GPU lease')
    manifest=require_root(root)
    if any((root/name).exists() for name in
           ('up_gate.fp32','expected_activated.bf16','case.json')):
        raise ValueError('activation input evidence must be create-only')
    report=document(activation_root/'report.json')
    compiler=(activation_root/'compiler_revision_id.txt').read_text().strip()
    case=activation_root/'dense_dyadic'
    source=(case/'expected_up_gate.fp32').read_bytes()
    expected=(case/'expected_activated.bf16').read_bytes()
    if (not report.get('passed')
            or compiler!='open-cake-ir@'+manifest['activation_compiler_commit']
            or len(source)!=ROWS*(2*INTERMEDIATE)*4
            or len(expected)!=ROWS*INTERMEDIATE*2):
        raise ValueError('prior Cake activation tile or oracle differs')
    (root/'up_gate.fp32').write_bytes(source*LOGICAL_TILES)
    (root/'expected_activated.bf16').write_bytes(expected*LOGICAL_TILES)
    (root/'case.json').write_text(json.dumps({
        'activation_compiler_commit':manifest['activation_compiler_commit'],
        'source_case':str(case.resolve()),
        'logical_tiles':LOGICAL_TILES,'rows_per_tile':ROWS,
        'stage_work_units_per_tile':ROWS,
        'stage_task_counts':[0,2176,1920,4096],
        'upgate_bytes':UPGATE_BYTES,'activated_bytes':ACTIVATED_BYTES,
        'scope':'repeated Cake dense-dyadic activation case; real stage math, no TMA/MMA or P2P',
    },indent=2)+'\n')


def build(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must run outside a broker GPU lease')
    manifest=require_root(root)
    case=document(root/'case.json')
    if (case.get('upgate_bytes')!=UPGATE_BYTES
            or case.get('activated_bytes')!=ACTIVATED_BYTES
            or (root/'up_gate.fp32').stat().st_size!=UPGATE_BYTES
            or (root/'expected_activated.bf16').stat().st_size!=ACTIVATED_BYTES):
        raise ValueError('activation stage input extent differs')
    if not Path(NVCC).is_file() or any((root/name).exists() for name in
                                         ('tile_probe','build_plan.json',
                                          'build_report.json','build_failure.json')):
        raise ValueError('cooperative tile build must be exact and create-only')
    command=[NVCC,*FLAGS,'model_activation_stage_queue.cu','-o',
             'tile_probe','-lcuda','-lcudart']
    (root/'build_plan.json').write_text(json.dumps(command,indent=2)+'\n')
    result=subprocess.run(command,cwd=root,capture_output=True,text=True,
                          timeout=180,check=False)
    (root/'compile.log').write_text(
        result.stdout+result.stderr or '(no compiler output)\n')
    if result.returncode:
        (root/'build_failure.json').write_text(json.dumps({
            'exit_code':result.returncode},indent=2)+'\n')
        raise RuntimeError('cooperative tile probe failed nvcc')
    if (root/'tile_probe').read_bytes()[:4]!=b'\x7fELF':
        raise RuntimeError('cooperative tile probe executable is not ELF')
    (root/'build_report.json').write_text(json.dumps({
        'source_commit':manifest['source_commit'],'target':'sm_103a',
        'exit_code':0,
        'scope':'CPU nvcc/PTXAS only; no cooperative device result or timing',
    },indent=2)+'\n')


def admitted() -> str:
    path=Path(os.environ.get('BROKER_RECEIPT','/nonexistent'))
    for _ in range(30):
        if path.is_file():break
        time.sleep(.1)
    else:raise RuntimeError('broker admission receipt missing')
    row=document(path)
    if (row.get('label')!=LABEL or row.get('mode')!='exclusive'
            or row.get('gpu_count')!=1 or len(row.get('gpu_ids',[]))!=1
            or not isinstance(row.get('job_id'),str)):
        raise RuntimeError('one exact exclusive B300 GPU admission differs')
    return row['job_id']


def run(root: Path) -> None:
    manifest=require_root(root)
    build_report=document(root/'build_report.json')
    case=document(root/'case.json')
    if (build_report.get('source_commit')!=manifest['source_commit']
            or build_report.get('exit_code')!=0
            or case.get('activation_compiler_commit')
            !=manifest['activation_compiler_commit']
            or (root/'device.json').exists()
            or any((root/f'{name}.json').exists() for name in CASES)):
        raise ValueError('cooperative activation source/build or device evidence differs')
    job=admitted()
    try:
        result=subprocess.run([str(root/'tile_probe'),str(root)],
                              cwd=root,capture_output=True,text=True,
                              timeout=150,check=False)
        (root/'device.log').write_text(
            result.stdout+result.stderr or '(no device output)\n')
        if result.returncode:
            raise RuntimeError(f'cooperative activation probe exited {result.returncode}')
        (root/'device.json').write_text(json.dumps({
            'broker_job':job,'target':'sm_103a',
            'source_commit':manifest['source_commit'],
            'activation_compiler_commit':manifest['activation_compiler_commit'],
            'case_names':list(CASES),
            'scope':'one-GPU cooperative real SwiGLU stage tasks with resource/steal transition; no TMA/MMA, P2P or timing',
        },indent=2)+'\n')
    except Exception as error:
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job':job,'error':f'{type(error).__name__}: {error}'},
            indent=2)+'\n')
        raise


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('CPU report must run after broker lease release')
    if (root/'report.json').exists():
        raise ValueError('cooperative tile report must be create-only')
    manifest=require_root(root)
    device=document(root/'device.json')
    receipt=document(root/'gpuq-admission.json')
    resource=document(root/'resource.json')
    if (device['broker_job']!=receipt['job_id']
            or device['source_commit']!=manifest['source_commit']
            or device['activation_compiler_commit']
            !=manifest['activation_compiler_commit']
            or device['case_names']!=list(CASES)
            or resource['target']!='sm_103a'
            or resource['device_name'] not in
            ('NVIDIA B300','NVIDIA B300 SXM6 AC')
            or resource['threads_per_cta']!=192
            or resource['dynamic_shared_bytes']!=49200
            or resource['active_blocks_per_sm']<1
            or resource['cooperative_grid_cta_upper_bound']
            !=resource['sm_count']*resource['active_blocks_per_sm']):
        raise ValueError('cooperative tile broker or resource result differs')
    cases=[]
    for name,(grid,communication,budget,tasks) in CASES.items():
        row=document(root/f'{name}.json')
        if (row['name']!=name or row['grid_ctas']!=grid
                or row['communication_ctas']!=communication
                or row['steal_budget']!=budget or row['tasks']!=tasks
                or row['processed']!=tasks
                or row['dispatched']!=[communication]*4
                or sum(row['stolen'])!=row['permit_count']
                or row['permit_count']>budget
                or row['math_bit_mismatches']!=0
                or row['output_bytes']!=ACTIVATED_BYTES
                or (root/f'{name}.actual.bf16').stat().st_size!=ACTIVATED_BYTES
                or (root/f'{name}.actual.bf16').read_bytes()
                !=(root/'expected_activated.bf16').read_bytes()
                or len(row['owner_blocks'])!=4
                or any(len(owners)!=tasks[wave]
                       or any(not 0<=owner<grid for owner in owners)
                       for wave,owners in enumerate(row['owner_blocks']))):
            raise ValueError(f'{name} task, dispatch or steal observation differs')
        if budget==0 and row['permit_count']!=0:
            raise ValueError(f'{name} stole without a permit')
        if name=='all_comm_c148_budget8192' and row['permit_count']!=8192:
            raise ValueError('all-communication CTA transition did not cover every stage task')
        cases.append(row)
    report={'passed':True,'broker_job':device['broker_job'],
            'source_commit':manifest['source_commit'],'target':'sm_103a',
            'activation_compiler_commit':manifest['activation_compiler_commit'],
            'resource':resource,'cases':cases,
            'scope':'one-GPU cooperative real Cake SwiGLU stage task/steal probe; no TMA/MMA, cross-device mailbox, latency or speedup'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'broker_job':device['broker_job'],
                      'active_blocks_per_sm':resource['active_blocks_per_sm'],
                      'cooperative_grid_cta_upper_bound':
                      resource['cooperative_grid_cta_upper_bound'],
                      'cases':[(row['name'],row['permit_count'],
                                sum(row['processed'])) for row in cases]},
                     indent=2))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','build','run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--activation-root',type=Path)
    args=parser.parse_args()
    root=args.root.expanduser().resolve(strict=True)
    if args.phase=='prepare':
        if args.activation_root is None:
            parser.error('prepare requires --activation-root')
        prepare(root,args.activation_root.expanduser().resolve(strict=True))
    else:
        {'build':build,'run':run,'verify':verify}[args.phase](root)


if __name__=='__main__':
    main()
