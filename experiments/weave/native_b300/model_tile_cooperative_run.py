"""Brokered B300 cooperative tile-grid resource/steal development probe."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


LABEL='cake-weave-model-tile-cooperative-b300'
NVCC='/usr/local/cuda-13.1/bin/nvcc'
FLAGS=['-std=c++17','--gpu-architecture=compute_103a',
       '--gpu-code=sm_103a','-O3','--fmad=false','-lineinfo','-Xptxas=-v']
CASES={
    'subcapacity_no_steal':(64,1,0,[0,17,15,32]),
    'fullgrid_no_steal':(148,1,0,[0,17,15,32]),
    'mixed_c74_budget8':(148,74,8,[0,17,15,32]),
    'near_full_c147_budget32':(148,147,32,[0,17,15,32]),
    'all_comm_c148_budget64':(148,148,64,[0,17,15,32]),
    'terminal_only_c74_budget8':(148,74,8,[0,0,0,47]),
}


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def require_root(root: Path) -> dict:
    manifest=document(root/'manifest.json')
    if (manifest.get('target')!='sm_103a'
            or len(manifest.get('source_commit',''))!=40
            or not (root/'model_tile_cooperative_probe.cu').is_file()
            or manifest.get('case_names')!=list(CASES)):
        raise ValueError('cooperative tile probe source or cases differ')
    return manifest


def build(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must run outside a broker GPU lease')
    manifest=require_root(root)
    if not Path(NVCC).is_file() or any((root/name).exists() for name in
                                         ('tile_probe','build_plan.json',
                                          'build_report.json','build_failure.json')):
        raise ValueError('cooperative tile build must be exact and create-only')
    command=[NVCC,*FLAGS,'model_tile_cooperative_probe.cu','-o',
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
    if (build_report.get('source_commit')!=manifest['source_commit']
            or build_report.get('exit_code')!=0
            or (root/'device.json').exists()
            or any((root/f'{name}.json').exists() for name in CASES)):
        raise ValueError('cooperative tile source/build or device evidence differs')
    job=admitted()
    try:
        result=subprocess.run([str(root/'tile_probe'),str(root)],
                              cwd=root,capture_output=True,text=True,
                              timeout=75,check=False)
        (root/'device.log').write_text(
            result.stdout+result.stderr or '(no device output)\n')
        if result.returncode:
            raise RuntimeError(f'cooperative tile probe exited {result.returncode}')
        (root/'device.json').write_text(json.dumps({
            'broker_job':job,'target':'sm_103a',
            'source_commit':manifest['source_commit'],
            'case_names':list(CASES),
            'scope':'one-GPU cooperative role/steal resource probe; no FFN arithmetic or timing',
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
                or len(row['owner_blocks'])!=4
                or any(len(owners)!=tasks[wave]
                       or any(not 0<=owner<grid for owner in owners)
                       for wave,owners in enumerate(row['owner_blocks']))):
            raise ValueError(f'{name} task, dispatch or steal observation differs')
        if budget==0 and row['permit_count']!=0:
            raise ValueError(f'{name} stole without a permit')
        if name=='all_comm_c148_budget64' and row['permit_count']!=64:
            raise ValueError('all-communication CTA transition did not cover every tile')
        cases.append(row)
    report={'passed':True,'broker_job':device['broker_job'],
            'source_commit':manifest['source_commit'],'target':'sm_103a',
            'resource':resource,'cases':cases,
            'scope':'one-GPU cooperative CTA resource/claim probe; no model FFN, cross-device mailbox, latency or speedup'}
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
    parser.add_argument('phase',choices=('build','run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    {'build':build,'run':run,'verify':verify}[
        args.phase](args.root.expanduser().resolve(strict=True))


if __name__=='__main__':
    main()
