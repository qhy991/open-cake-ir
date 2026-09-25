"""CPU build and brokered residency query for exact Cake B300 FFN stages."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


LABEL='cake-weave-model-ffn-residency-b300'
NVCC='/usr/local/cuda-13.1/bin/nvcc'
FLAGS=['-std=c++17','--gpu-architecture=compute_103a',
       '--gpu-code=sm_103a','-O3','--fmad=false','-lineinfo','-Xptxas=-v']
STAGES=('up_gate','down')


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def require_root(root: Path) -> dict:
    manifest=document(root/'manifest.json')
    if (manifest.get('target')!='sm_103a'
            or len(manifest.get('probe_commit',''))!=40
            or len(manifest.get('ffn_compiler_commit',''))!=40
            or manifest.get('stages')!=list(STAGES)
            or not (root/'probe.cu').is_file()
            or any(not (root/stage/'kernel.cu').is_file()
                   for stage in STAGES)):
        raise ValueError('exact B300 Cake FFN residency source differs')
    return manifest


def build(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must run outside a GPU lease')
    manifest=require_root(root)
    if not Path(NVCC).is_file() or any((root/name).exists() for name in
                                         ('build_plan.json','build_report.json',
                                          'probe_up_gate','probe_down')):
        raise ValueError('exact CUDA 13.1 build must be create-only')
    commands={stage:[NVCC,*FLAGS,'-DCAKE_'+
                     ('UPGATE' if stage=='up_gate' else 'DOWN'),
                     'probe.cu','-o','probe_'+stage,'-lcuda','-lcudart']
              for stage in STAGES}
    (root/'build_plan.json').write_text(json.dumps(commands,indent=2)+'\n')
    exits={}
    for stage,command in commands.items():
        result=subprocess.run(command,cwd=root,capture_output=True,text=True,
                              timeout=180,check=False)
        (root/f'{stage}.compile.log').write_text(
            result.stdout+result.stderr or '(no compiler output)\n')
        exits[stage]=result.returncode
        if result.returncode:
            (root/'build_failure.json').write_text(json.dumps({
                'stage':stage,'exit_codes':exits},indent=2)+'\n')
            raise RuntimeError(f'{stage} occupancy source failed nvcc')
    if any((root/f'probe_{stage}').read_bytes()[:4]!=b'\x7fELF'
           for stage in STAGES):
        raise RuntimeError('occupancy executable missing or not ELF')
    (root/'build_report.json').write_text(json.dumps({
        'target':'sm_103a','probe_commit':manifest['probe_commit'],
        'ffn_compiler_commit':manifest['ffn_compiler_commit'],
        'exit_codes':exits,
        'scope':'CPU nvcc/PTXAS build only; no device query or latency',
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
    report=document(root/'build_report.json')
    if (report.get('exit_codes')!={stage:0 for stage in STAGES}
            or report.get('probe_commit')!=manifest['probe_commit']
            or report.get('ffn_compiler_commit')!=manifest['ffn_compiler_commit']
            or (root/'device.json').exists()
            or any((root/f'{stage}.occupancy.json').exists()
                   for stage in STAGES)):
        raise ValueError('compiled probe or device evidence differs')
    job=admitted()
    try:
        for stage in STAGES:
            result=subprocess.run([str(root/f'probe_{stage}'),
                                   str(root/f'{stage}.occupancy.json')],
                                  cwd=root,capture_output=True,text=True,
                                  timeout=30,check=False)
            (root/f'{stage}.device.log').write_text(
                result.stdout+result.stderr or '(no device output)\n')
            if result.returncode:
                raise RuntimeError(f'{stage} residency query exited {result.returncode}')
        (root/'device.json').write_text(json.dumps({
            'broker_job':job,'target':'sm_103a',
            'probe_commit':manifest['probe_commit'],
            'ffn_compiler_commit':manifest['ffn_compiler_commit'],
            'stages':list(STAGES),
            'scope':'one-GPU exact-kernel occupancy query; no kernel latency',
        },indent=2)+'\n')
    except Exception as error:
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job':job,'error':f'{type(error).__name__}: {error}'},
            indent=2)+'\n')
        raise


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('CPU report must run after lease release')
    if (root/'report.json').exists():
        raise ValueError('occupancy report must be create-only')
    manifest=require_root(root)
    device=document(root/'device.json')
    receipt=document(root/'gpuq-admission.json')
    if (device['broker_job']!=receipt['job_id']
            or device['probe_commit']!=manifest['probe_commit']
            or device['ffn_compiler_commit']!=manifest['ffn_compiler_commit']
            or device['stages']!=list(STAGES)):
        raise ValueError('probe source, device or broker binding differs')
    results=[]
    for stage in STAGES:
        row=document(root/f'{stage}.occupancy.json')
        if (row['stage']!=stage or row['target']!='sm_103a'
                or row['device_name'] not in
                ('NVIDIA B300','NVIDIA B300 SXM6 AC')
                or row['cooperative_launch']!=1
                or row['threads_per_cta']!=192
                or row['dynamic_shared_bytes']!=49200
                or row['active_blocks_per_sm']<1
                or row['cooperative_grid_cta_upper_bound']
                !=row['sm_count']*row['active_blocks_per_sm']):
            raise ValueError(f'{stage} residency result differs')
        results.append(row)
    report={'passed':True,'target':'sm_103a',
            'broker_job':device['broker_job'],
            'probe_commit':manifest['probe_commit'],
            'ffn_compiler_commit':manifest['ffn_compiler_commit'],
            'stages':results,
            'minimum_queried_cooperative_cta_bound':min(
                row['cooperative_grid_cta_upper_bound'] for row in results),
            'scope':'CUDA occupancy API on exact FFN stage kernels; TMEM and live ranked queue residency not independently qualified; no latency'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('build','run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    {'build':build,'run':run,'verify':verify}[
        args.phase](args.root.expanduser().resolve(strict=True))


if __name__=='__main__':
    main()
