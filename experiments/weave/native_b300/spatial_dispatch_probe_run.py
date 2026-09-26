"""Standalone four-GPU spatial dispatch/compute CTA proof with CPU oracles."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


LABEL = 'cake-weave-spatial-dispatch-probe-b300'
NVCC = '/usr/local/cuda-13.1/bin/nvcc'
FLAGS = ['-std=c++17', '--gpu-architecture=compute_103a',
         '--gpu-code=sm_103a', '-O3', '--fmad=false', '-lineinfo', '-Xptxas=-v']
R, T, K, E, H = 4, 512, 8, 128, 2048


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def contract(root: Path) -> dict:
    manifest = read(root/'manifest.json')
    if (manifest.get('target') != 'sm_103a'
            or manifest.get('world_size') != R
            or type(manifest.get('communication_ctas')) is not int
            or not 1 <= manifest['communication_ctas'] < 96
            or len(manifest.get('source_commit', '')) != 40
            or not (root/'spatial_dispatch_probe.cu').is_file()
            or not (root/'bin_pack_oracle.py').is_file()):
        raise ValueError('spatial dispatch source or target contract differs')
    return manifest


def prepare(root: Path, input_root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('spatial input preparation belongs outside the GPU lease')
    manifest = contract(root)
    if any((root/name).exists() for name in
           ('hidden.bf16', 'expert_ids.i32', 'case.json', 'device_outputs')):
        raise ValueError('spatial input root is not create-only')
    hidden = (input_root/'hidden.bf16').read_bytes()
    ids = (input_root/'expert_ids.i32').read_bytes()
    if len(hidden) != R*T*H*2 or len(ids) != R*T*K*4:
        raise ValueError('model hidden or route input extent differs')
    routes = np.frombuffer(ids, dtype='<i4').reshape(R*T, K)
    if (np.any(routes < 0) or np.any(routes >= E)
            or any(len(set(map(int, row))) != K for row in routes)):
        raise ValueError('model route IDs violate the distinct expert domain')
    (root/'hidden.bf16').write_bytes(hidden)
    (root/'expert_ids.i32').write_bytes(ids)
    for owner in range(R):
        (root/'device_outputs'/f'owner{owner}').mkdir(parents=True)
    (root/'case.json').write_text(json.dumps({
        'source_commit': manifest['source_commit'],
        'source_inputs': str(input_root),
        'communication_ctas': manifest['communication_ctas'],
        'geometry': {'R':R,'T':T,'K':K,'E':E,'H':H},
        'owner_route_counts': np.bincount(routes.ravel()//(E//R),
                                          minlength=R).tolist(),
        'scope': 'resident CTA communication and independent tensor work; no FFN',
    }, indent=2)+'\n')


def build(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('spatial NVCC build belongs outside the GPU lease')
    manifest = contract(root)
    case = read(root/'case.json')
    if (case['source_commit'] != manifest['source_commit']
            or case['communication_ctas'] != manifest['communication_ctas']
            or any((root/name).exists() for name in
                   ('spatial_probe','build_plan.json','build_report.json',
                    'build_failure.json'))):
        raise ValueError('spatial build source or create-only state differs')
    command = [NVCC, *FLAGS, 'spatial_dispatch_probe.cu', '-o',
               'spatial_probe', '-lcuda', '-lcudart']
    (root/'build_plan.json').write_text(json.dumps(command,indent=2)+'\n')
    result = subprocess.run(command,cwd=root,capture_output=True,text=True,
                            timeout=240,check=False)
    (root/'compile.log').write_text(result.stdout+result.stderr or
                                    '(no compiler output)\n')
    if result.returncode:
        (root/'build_failure.json').write_text(json.dumps({
            'exit_code': result.returncode},indent=2)+'\n')
        raise RuntimeError('spatial dispatch NVCC build failed')
    if (root/'spatial_probe').read_bytes()[:4] != b'\x7fELF':
        raise RuntimeError('spatial probe executable is not ELF')
    (root/'build_report.json').write_text(json.dumps({
        'source_commit': manifest['source_commit'], 'target': 'sm_103a',
        'exit_code': 0, 'scope': 'CPU-only NVCC/PTXAS compilation',
    }, indent=2)+'\n')


def admitted() -> str:
    path = Path(os.environ.get('BROKER_RECEIPT', '/nonexistent'))
    for _ in range(30):
        if path.is_file():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('spatial probe broker receipt missing')
    receipt = read(path)
    if (receipt.get('label') != LABEL or receipt.get('mode') != 'exclusive'
            or receipt.get('gpu_count') != R
            or len(receipt.get('gpu_ids', [])) != R):
        raise RuntimeError('spatial probe four-GPU admission differs')
    return receipt['job_id']


def run(root: Path) -> None:
    manifest = contract(root)
    case = read(root/'case.json')
    if (read(root/'build_report.json')['source_commit']
            != manifest['source_commit']
            or (root/'device.json').exists()
            or (root/'launch_failure.json').exists()):
        raise ValueError('spatial device evidence is not create-only')
    job = admitted()
    try:
        result = subprocess.run([str(root/'spatial_probe'),str(root),
                                 str(case['communication_ctas'])],
                                cwd=root,capture_output=True,text=True,
                                timeout=300,check=False)
        (root/'device.log').write_text(result.stdout+result.stderr or
                                       '(no device output)\n')
        if result.returncode:
            raise RuntimeError(f'spatial probe exited {result.returncode}')
        (root/'device.json').write_text(json.dumps({
            'broker_job': job, 'source_commit': manifest['source_commit'],
            'target': 'sm_103a', 'communication_ctas': case['communication_ctas'],
            'scope': 'four-GPU CTA transport/computation proof; no model FFN or timing',
        },indent=2)+'\n')
    except Exception as error:
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job': job, 'error': f'{type(error).__name__}: {error}'},
            indent=2)+'\n')
        raise


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('spatial oracle belongs after broker release')
    if (root/'report.json').exists():
        raise ValueError('spatial report is create-only')
    manifest = contract(root)
    case = read(root/'case.json')
    device = read(root/'device.json')
    receipt = read(root/'gpuq-admission.json')
    if (device['broker_job'] != receipt['job_id']
            or device['source_commit'] != manifest['source_commit']
            or device['communication_ctas'] != case['communication_ctas']):
        raise ValueError('spatial source or broker identity differs')
    sys.path.insert(0,str(root))
    from bin_pack_oracle import verify_rows
    owners = verify_rows(root)
    if [item['routes'] for item in owners] != case['owner_route_counts']:
        raise ValueError('spatial P2P expert-bin route counts differ')
    hidden = np.fromfile(root/'hidden.bf16',dtype='<u2').reshape(R,T,H)
    expected = (hidden.astype('<u4') << 16).view('<f4') * 2.0
    observations = []
    for rank in range(R):
        row = root/'device_outputs'/f'owner{rank}'
        actual = np.fromfile(row/'compute.fp32',dtype='<f4').reshape(T,H)
        phase = read(row/'phase.json')
        if (not np.array_equal(actual,expected[rank])
                or phase['communication_ctas'] != case['communication_ctas']
                or phase['comm_ctas_done'] != case['communication_ctas']
                or phase['compute_ctas_done'] != 96-case['communication_ctas']
                or phase['active_comm'] != 0 or phase['active_compute'] != 0
                or not phase['comm_start'] < phase['comm_end']
                or not phase['compute_start'] < phase['compute_end']):
            raise ValueError(f'rank {rank} compute, CTA ownership or timestamps differ')
        intersection = max(0,min(phase['comm_end'],phase['compute_end'])
                           -max(phase['comm_start'],phase['compute_start']))
        observations.append({'rank':rank,
            'comm_ctas_done':phase['comm_ctas_done'],
            'compute_ctas_done':phase['compute_ctas_done'],
            'overlap_flag':phase['overlap'],
            'group_window_intersection_ns':intersection})
    report = {'passed':True,'source_commit':manifest['source_commit'],
              'broker_job':device['broker_job'],
              'communication_ctas':case['communication_ctas'],
              'owner_routes':owners,'rank_observations':observations,
              'scope':'exact BF16 P2P rows and FP32 transform; CTA group overlap diagnostic; no FFN or qualified latency'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'communication_ctas':case['communication_ctas'],
                      'overlap_flags':[row['overlap_flag'] for row in observations]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','build','run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--input-root',type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve(strict=True)
    if args.phase == 'prepare':
        if args.input_root is None:
            parser.error('prepare needs --input-root')
        prepare(root,args.input_root.expanduser().resolve(strict=True))
    else:
        {'build':build,'run':run,'verify':verify}[args.phase](root)


if __name__ == '__main__':
    main()
