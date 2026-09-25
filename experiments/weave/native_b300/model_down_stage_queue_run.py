"""Brokered B300 cooperative real Cake down stage-task oracle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


LABEL='cake-weave-model-down-stage-queue-b300'
NVCC='/usr/local/cuda-13.1/bin/nvcc'
FLAGS=['-std=c++17','--gpu-architecture=compute_103a',
       '--gpu-code=sm_103a','-O3','--fmad=false','-lineinfo','-Xptxas=-v']
CASES={
    'subcapacity_no_steal':(64,1,0,[0,544,480,1024]),
    'fullgrid_no_steal':(148,1,0,[0,544,480,1024]),
    'near_full_c147_budget1024':(148,147,1024,[0,544,480,1024]),
    'all_comm_c148_budget2048':(148,148,2048,[0,544,480,1024]),
}
LOGICAL_TILES=64
ROWS=128
HIDDEN=768
OUTPUT=2048
X_BYTES=LOGICAL_TILES*ROWS*HIDDEN*2
WEIGHT_BYTES=OUTPUT*HIDDEN*2
OUTPUT_BYTES=LOGICAL_TILES*ROWS*OUTPUT*4


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def require_root(root: Path) -> dict:
    manifest=document(root/'manifest.json')
    if (manifest.get('target')!='sm_103a'
            or len(manifest.get('source_commit',''))!=40
            or len(manifest.get('down_compiler_commit',''))!=40
            or not (root/'model_down_stage_queue.cu').is_file()
            or not (root/'down/kernel.cu').is_file()
            or manifest.get('case_names')!=list(CASES)):
        raise ValueError('cooperative down source or cases differ')
    return manifest


def prepare(root: Path, ffn_root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('down preparation must run outside the GPU lease')
    manifest=require_root(root)
    if any((root/name).exists() for name in
           ('activated_tiles.bf16','w_down.bf16','expected_down.fp32','case.json')):
        raise ValueError('down stage input evidence must be create-only')
    report=document(ffn_root/'report.json')
    compiler=(ffn_root/'compiler_revision_id.txt').read_text().strip()
    case=ffn_root/'dense_dyadic'
    x=(case/'expected_activated.bf16').read_bytes()
    weight=(case/'w_down.bf16').read_bytes()
    expected=(case/'expected_out.fp32').read_bytes()
    if (not report.get('passed')
            or compiler!='open-cake-ir@'+manifest['down_compiler_commit']
            or len(x)!=ROWS*HIDDEN*2
            or len(weight)!=WEIGHT_BYTES
            or len(expected)!=ROWS*OUTPUT*4):
        raise ValueError('prior Cake down tile or oracle differs')
    (root/'activated_tiles.bf16').write_bytes(x*LOGICAL_TILES)
    (root/'w_down.bf16').write_bytes(weight)
    (root/'expected_down.fp32').write_bytes(expected*LOGICAL_TILES)
    (root/'case.json').write_text(json.dumps({
        'down_compiler_commit':manifest['down_compiler_commit'],
        'source_case':str(case.resolve()),
        'logical_tiles':LOGICAL_TILES,'rows_per_tile':ROWS,
        'stage_work_units_per_tile':32,
        'stage_task_counts':[0,544,480,1024],
        'x_bytes':X_BYTES,'weight_bytes':WEIGHT_BYTES,
        'output_bytes':OUTPUT_BYTES,
        'scope':'repeated Cake dense-dyadic down case; real TMA/MMA stage, no P2P',
    },indent=2)+'\n')


def build(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must run outside a broker GPU lease')
    manifest=require_root(root)
    case=document(root/'case.json')
    if (case.get('x_bytes')!=X_BYTES
            or case.get('weight_bytes')!=WEIGHT_BYTES
            or case.get('output_bytes')!=OUTPUT_BYTES
            or (root/'activated_tiles.bf16').stat().st_size!=X_BYTES
            or (root/'w_down.bf16').stat().st_size!=WEIGHT_BYTES
            or (root/'expected_down.fp32').stat().st_size!=OUTPUT_BYTES):
        raise ValueError('down stage input byte extents differ')
    if not Path(NVCC).is_file() or any((root/name).exists() for name in
                                         ('tile_probe','build_plan.json',
                                          'build_report.json','build_failure.json')):
        raise ValueError('cooperative tile build must be exact and create-only')
    command=[NVCC,*FLAGS,'model_down_stage_queue.cu','-o',
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
            or case.get('down_compiler_commit')
            !=manifest['down_compiler_commit']
            or (root/'device.json').exists()
            or any((root/f'{name}.json').exists() for name in CASES)):
        raise ValueError('cooperative down source/build or device evidence differs')
    job=admitted()
    try:
        result=subprocess.run([str(root/'tile_probe'),str(root)],
                              cwd=root,capture_output=True,text=True,
                              timeout=210,check=False)
        (root/'device.log').write_text(
            result.stdout+result.stderr or '(no device output)\n')
        if result.returncode:
            raise RuntimeError(f'cooperative down probe exited {result.returncode}')
        (root/'device.json').write_text(json.dumps({
            'broker_job':job,'target':'sm_103a',
            'source_commit':manifest['source_commit'],
            'down_compiler_commit':manifest['down_compiler_commit'],
            'case_names':list(CASES),
            'scope':'one-GPU cooperative real Cake TMA/MMA down stage tasks; no P2P or timing',
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
            or device['down_compiler_commit']
            !=manifest['down_compiler_commit']
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
                or row['output_bytes']!=OUTPUT_BYTES
                or (root/f'{name}.actual.fp32').stat().st_size!=OUTPUT_BYTES
                or (root/f'{name}.actual.fp32').read_bytes()
                !=(root/'expected_down.fp32').read_bytes()
                or len(row['owner_blocks'])!=4
                or any(len(owners)!=tasks[wave]
                       or any(not 0<=owner<grid for owner in owners)
                       for wave,owners in enumerate(row['owner_blocks']))):
            raise ValueError(f'{name} task, dispatch or steal observation differs')
        if budget==0 and row['permit_count']!=0:
            raise ValueError(f'{name} stole without a permit')
        if name=='all_comm_c148_budget2048' and row['permit_count']!=2048:
            raise ValueError('all-communication CTA transition did not cover every stage task')
        cases.append(row)
    report={'passed':True,'broker_job':device['broker_job'],
            'source_commit':manifest['source_commit'],'target':'sm_103a',
            'down_compiler_commit':manifest['down_compiler_commit'],
            'resource':resource,'cases':cases,
            'scope':'one-GPU cooperative real Cake TMA/MMA down stage task/steal probe; no cross-device mailbox, latency or speedup'}
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
    parser.add_argument('--ffn-root',type=Path)
    args=parser.parse_args()
    root=args.root.expanduser().resolve(strict=True)
    if args.phase=='prepare':
        if args.ffn_root is None:
            parser.error('prepare requires --ffn-root')
        prepare(root,args.ffn_root.expanduser().resolve(strict=True))
    else:
        {'build':build,'run':run,'verify':verify}[args.phase](root)


if __name__=='__main__':
    main()
