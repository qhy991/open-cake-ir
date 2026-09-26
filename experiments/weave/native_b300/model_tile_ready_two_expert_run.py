"""Brokered B300 tile-ready Cake FFN with two selected expert weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import struct
import time


LABEL='cake-weave-model-tile-ready-two-expert-b300'
NVCC='/usr/local/cuda-13.1/bin/nvcc'
FLAGS=['-std=c++17','--gpu-architecture=compute_103a',
       '--gpu-code=sm_103a','-O3','--fmad=false','-lineinfo','-Xptxas=-v']
CASES={
    'subcapacity_no_steal':(64,1,0,
                            [[0,408,360,768],[0,2176,1920,4096],
                             [0,544,480,1024]]),
    'fullgrid_no_steal':(148,1,0,
                         [[0,408,360,768],[0,2176,1920,4096],
                          [0,544,480,1024]]),
    'near_full_c147_budget5888':(148,147,5888,
                                  [[0,408,360,768],[0,2176,1920,4096],
                                   [0,544,480,1024]]),
    'all_comm_c148_budget11776':(148,148,11776,
                                  [[0,408,360,768],[0,2176,1920,4096],
                                   [0,544,480,1024]]),
}
LOGICAL_TILES=64
EXPERTS=2
ROWS=128
INPUT=2048
HIDDEN=768
OUTPUT=2048
INPUT_BYTES=LOGICAL_TILES*ROWS*INPUT*2
UP_WEIGHT_BYTES=(2*HIDDEN)*INPUT*2
UP_WEIGHT_TOTAL_BYTES=EXPERTS*UP_WEIGHT_BYTES
UPGATE_BYTES=LOGICAL_TILES*ROWS*(2*HIDDEN)*4
ACTIVATED_BYTES=LOGICAL_TILES*ROWS*HIDDEN*2
DOWN_WEIGHT_BYTES=OUTPUT*HIDDEN*2
DOWN_WEIGHT_TOTAL_BYTES=EXPERTS*DOWN_WEIGHT_BYTES
OUTPUT_BYTES=LOGICAL_TILES*ROWS*OUTPUT*4
EXPERT_BYTES=LOGICAL_TILES*4


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def require_root(root: Path) -> dict:
    manifest=document(root/'manifest.json')
    if (manifest.get('target')!='sm_103a'
            or len(manifest.get('source_commit',''))!=40
            or len(manifest.get('ffn_compiler_commit',''))!=40
            or not (root/'model_tile_ready_ffn_capped.cu').is_file()
            or not (root/'down/kernel.cu').is_file()
            or manifest.get('case_names')!=list(CASES)):
        raise ValueError('cooperative full FFN source or cases differ')
    return manifest


def prepare(root: Path, ffn_root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('full FFN preparation must run outside the GPU lease')
    manifest=require_root(root)
    if any((root/name).exists() for name in
           ('x_tiles.bf16','w_up_gate.bf16','expected_up_gate.fp32',
            'expected_activated.bf16','w_down.bf16','expected_down.fp32',
            'tile_expert.i32','case.json')):
        raise ValueError('full FFN input evidence must be create-only')
    report=document(ffn_root/'report.json')
    compiler=(ffn_root/'compiler_revision_id.txt').read_text().strip()
    cases=(ffn_root/'dense_dyadic',ffn_root/'sparse_one_hot')
    inputs=[(case/'x.bf16').read_bytes() for case in cases]
    up_weights=[(case/'w_up_gate.bf16').read_bytes() for case in cases]
    up_gates=[(case/'expected_up_gate.fp32').read_bytes() for case in cases]
    activations=[(case/'expected_activated.bf16').read_bytes() for case in cases]
    down_weights=[(case/'w_down.bf16').read_bytes() for case in cases]
    expecteds=[(case/'expected_out.fp32').read_bytes() for case in cases]
    if (not report.get('passed')
            or {(row.get('case'),row.get('passed')) for row in report.get('cases',[])}
            != {('dense_dyadic',True),('sparse_one_hot',True)}
            or compiler!='open-cake-ir@'+manifest['ffn_compiler_commit']
            or any(len(value)!=extent for values,extent in (
                (inputs,ROWS*INPUT*2),
                (up_weights,UP_WEIGHT_BYTES),
                (up_gates,ROWS*(2*HIDDEN)*4),
                (activations,ROWS*HIDDEN*2),
                (down_weights,DOWN_WEIGHT_BYTES),
                (expecteds,ROWS*OUTPUT*4)) for value in values)
            or up_weights[0]==up_weights[1]
            or down_weights[0]==down_weights[1]
            or up_gates[0]==up_gates[1]
            or expecteds[0]==expecteds[1]):
        raise ValueError('two independent Cake FFN oracle cases differ')
    experts=[tile%EXPERTS for tile in range(LOGICAL_TILES)]
    (root/'tile_expert.i32').write_bytes(struct.pack('<64i',*experts))
    for filename,values in (
        ('x_tiles.bf16',inputs),
        ('expected_up_gate.fp32',up_gates),
        ('expected_activated.bf16',activations),
        ('expected_down.fp32',expecteds)):
        (root/filename).write_bytes(b''.join(values[expert] for expert in experts))
    (root/'w_up_gate.bf16').write_bytes(b''.join(up_weights))
    (root/'w_down.bf16').write_bytes(b''.join(down_weights))
    (root/'case.json').write_text(json.dumps({
        'ffn_compiler_commit':manifest['ffn_compiler_commit'],
        'source_cases':[str(case.resolve()) for case in cases],
        'expert_count':EXPERTS,'tile_expert':experts,
        'logical_tiles':LOGICAL_TILES,'rows_per_tile':ROWS,
        'stage_work_units_per_tile':[24,128,32],
        'stage_task_counts':[[0,408,360,768],[0,2176,1920,4096],
                             [0,544,480,1024]],
        'input_bytes':INPUT_BYTES,'expert_bytes':EXPERT_BYTES,
        'up_weight_bytes':UP_WEIGHT_TOTAL_BYTES,
        'upgate_bytes':UPGATE_BYTES,'activated_bytes':ACTIVATED_BYTES,
        'down_weight_bytes':DOWN_WEIGHT_TOTAL_BYTES,'output_bytes':OUTPUT_BYTES,
        'scope':'alternating independent Cake expert FFN cases; no P2P',
    },indent=2)+'\n')


def build(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must run outside a broker GPU lease')
    manifest=require_root(root)
    case=document(root/'case.json')
    if (case.get('input_bytes')!=INPUT_BYTES
            or case.get('expert_count')!=EXPERTS
            or case.get('tile_expert')!=[tile%EXPERTS for tile in range(LOGICAL_TILES)]
            or case.get('expert_bytes')!=EXPERT_BYTES
            or case.get('up_weight_bytes')!=UP_WEIGHT_TOTAL_BYTES
            or case.get('upgate_bytes')!=UPGATE_BYTES
            or case.get('activated_bytes')!=ACTIVATED_BYTES
            or case.get('down_weight_bytes')!=DOWN_WEIGHT_TOTAL_BYTES
            or case.get('output_bytes')!=OUTPUT_BYTES
            or (root/'x_tiles.bf16').stat().st_size!=INPUT_BYTES
            or (root/'tile_expert.i32').stat().st_size!=EXPERT_BYTES
            or (root/'tile_expert.i32').read_bytes()
            !=struct.pack('<64i',*[tile%EXPERTS for tile in range(LOGICAL_TILES)])
            or (root/'w_up_gate.bf16').stat().st_size!=UP_WEIGHT_TOTAL_BYTES
            or (root/'expected_up_gate.fp32').stat().st_size!=UPGATE_BYTES
            or (root/'expected_activated.bf16').stat().st_size!=ACTIVATED_BYTES
            or (root/'w_down.bf16').stat().st_size!=DOWN_WEIGHT_TOTAL_BYTES
            or (root/'expected_down.fp32').stat().st_size!=OUTPUT_BYTES):
        raise ValueError('full FFN input byte extents differ')
    if not Path(NVCC).is_file() or any((root/name).exists() for name in
                                         ('tile_probe','build_plan.json',
                                          'build_report.json','build_failure.json')):
        raise ValueError('cooperative tile build must be exact and create-only')
    command=[NVCC,*FLAGS,'model_tile_ready_ffn_capped.cu','-o',
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
            or case.get('ffn_compiler_commit')
            !=manifest['ffn_compiler_commit']
            or (root/'device.json').exists()
            or any((root/f'{name}.json').exists() for name in CASES)):
        raise ValueError('cooperative full FFN source/build or device evidence differs')
    job=admitted()
    try:
        result=subprocess.run([str(root/'tile_probe'),str(root)],
                              cwd=root,capture_output=True,text=True,
                              timeout=210,check=False)
        (root/'device.log').write_text(
            result.stdout+result.stderr or '(no device output)\n')
        if result.returncode:
            raise RuntimeError(f'cooperative full FFN probe exited {result.returncode}')
        (root/'device.json').write_text(json.dumps({
            'broker_job':job,'target':'sm_103a',
            'source_commit':manifest['source_commit'],
            'ffn_compiler_commit':manifest['ffn_compiler_commit'],
            'case_names':list(CASES),
            'scope':'one-GPU tile-ready Cake FFN with two selected expert weights; no P2P or timing',
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
    case=document(root/'case.json')
    device=document(root/'device.json')
    receipt=document(root/'gpuq-admission.json')
    resource=document(root/'resource.json')
    if (case['expert_count']!=EXPERTS
            or case['tile_expert']!=[tile%EXPERTS for tile in range(LOGICAL_TILES)]
            or device['broker_job']!=receipt['job_id']
            or device['source_commit']!=manifest['source_commit']
            or device['ffn_compiler_commit']
            !=manifest['ffn_compiler_commit']
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
                or sum(map(sum,row['stolen']))!=row['permit_count']
                or row['permit_count']>budget
                or row['upgate_bit_mismatches']!=0
                or row['activation_bit_mismatches']!=0
                or row['down_bit_mismatches']!=0
                or row['upgate_bytes']!=UPGATE_BYTES
                or row['activation_bytes']!=ACTIVATED_BYTES
                or row['down_bytes']!=OUTPUT_BYTES
                or (root/f'{name}.upgate.actual.fp32').stat().st_size
                !=UPGATE_BYTES
                or (root/f'{name}.upgate.actual.fp32').read_bytes()
                !=(root/'expected_up_gate.fp32').read_bytes()
                or (root/f'{name}.activated.actual.bf16').stat().st_size
                !=ACTIVATED_BYTES
                or (root/f'{name}.activated.actual.bf16').read_bytes()
                !=(root/'expected_activated.bf16').read_bytes()
                or (root/f'{name}.down.actual.fp32').stat().st_size
                !=OUTPUT_BYTES
                or (root/f'{name}.down.actual.fp32').read_bytes()
                !=(root/'expected_down.fp32').read_bytes()
                or len(row['owner_blocks'])!=3
                or row['tile_completed']!=[[24]*LOGICAL_TILES,
                                           [128]*LOGICAL_TILES,
                                           [32]*LOGICAL_TILES]
                or len(row['stage_overlap'])!=2
                or any(value not in (0,1) for value in row['stage_overlap'])
                or any(len(owners)!=tasks[stage][wave]
                       or any(not 0<=owner<grid for owner in owners)
                       for stage,waves in enumerate(row['owner_blocks'])
                       for wave,owners in enumerate(waves))):
            raise ValueError(f'{name} task, dispatch or steal observation differs')
        if budget==0 and row['permit_count']!=0:
            raise ValueError(f'{name} stole without a permit')
        if name=='all_comm_c148_budget11776' and row['permit_count']!=11776:
            raise ValueError('all-communication CTA transition did not cover every stage task')
        if name in ('subcapacity_no_steal','fullgrid_no_steal') and row['stage_overlap']!=[1,1]:
            raise ValueError(f'{name} did not observe both tile-local stage overlaps')
        cases.append(row)
    report={'passed':True,'broker_job':device['broker_job'],
            'source_commit':manifest['source_commit'],'target':'sm_103a',
            'ffn_compiler_commit':manifest['ffn_compiler_commit'],
            'resource':resource,'cases':cases,
            'scope':'one-GPU tile-ready Cake FFN with two selected expert weights; no cross-device mailbox, latency or speedup'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'broker_job':device['broker_job'],
                      'active_blocks_per_sm':resource['active_blocks_per_sm'],
                      'cooperative_grid_cta_upper_bound':
                      resource['cooperative_grid_cta_upper_bound'],
                      'cases':[(row['name'],row['permit_count'],
                                sum(map(sum,row['processed']))) for row in cases]},
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
