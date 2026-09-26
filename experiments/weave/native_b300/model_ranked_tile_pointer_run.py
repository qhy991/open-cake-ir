"""Brokered B300 probe for the native ranked-tile tensor-pointer ABI."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


R, T, K, E, H = 4, 512, 8, 128, 2048
LABEL = 'cake-weave-pointer-abi-b300'
NVCC = '/usr/local/cuda-13.1/bin/nvcc'
CUDART = '/usr/local/cuda-13.1/lib64/libcudart.so'
ENTRY = 'cake_ranked_tile_b300'
H2D, D2H = 1, 2


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def contract(root: Path) -> dict:
    manifest = read(root / 'manifest.json')
    lowering = read(root / 'lowering_report.json')
    if (manifest.get('target') != 'sm_103a'
            or manifest.get('generation') != 'cake_ranked_tile_pointer_abi'
            or len(manifest.get('source_commit', '')) != 40
            or lowering.get('source_commit') != manifest['source_commit']
            or lowering.get('entry_point') != ENTRY
            or lowering.get('logical_tile_capacity') != 255
            or lowering.get('stage_task_capacity') != 46920
            or not (root / 'ranked_tile.cu').is_file()):
        raise ValueError('ranked-tile library source or Compiler binding differs')
    return manifest


def prepare(root: Path, bin_root: Path, bridge: Path,
            oracle_root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('pointer ABI preparation must be outside the lease')
    manifest = contract(root)
    if (root / 'case.json').exists() or (root / 'device_outputs').exists():
        raise ValueError('pointer ABI inputs are create-only')
    ids = np.fromfile(bin_root / 'expert_ids.i32', dtype='<i4')
    expected = np.load(oracle_root / 'expected_output.npy', mmap_mode='r')
    oracle = read(oracle_root / 'oracle_report.json')
    if (ids.size != R*T*K or expected.shape != (R,T,H)
            or expected.dtype != np.float32 or not np.all(np.isfinite(expected))
            or (bin_root / 'hidden.bf16').stat().st_size != R*T*H*2
            or (bridge / 'route_weights.fp32').stat().st_size != R*T*K*4
            or (bridge / 'weights_upgate.bf16').stat().st_size
            != E*1536*H*2
            or (bridge / 'weights_down.bf16').stat().st_size
            != E*H*768*2
            or Path(oracle['ids_path']).resolve(strict=True)
            != (bin_root / 'expert_ids.i32').resolve(strict=True)):
        raise ValueError('pointer ABI inputs or external oracle differ')
    ids = ids.reshape(R,T,K)
    if (np.any(ids < 0) or np.any(ids >= E)
            or any(len(set(map(int,row))) != K for rank in ids for row in rank)):
        raise ValueError('pointer ABI route IDs violate the distinct domain')
    payloads_by_owner=[0]*R
    remote_route_rows=0
    for source in range(R):
        for token in range(T):
            owners=(ids[source,token]//(E//R)).tolist()
            remote_route_rows+=sum(owner!=source for owner in owners)
            for owner in set(owners):
                if owner!=source:
                    payloads_by_owner[owner]+=1
    if max(payloads_by_owner)>(R-1)*T:
        raise ValueError('remote payload plan exceeds ranked-tile capacity')
    (root / 'device_outputs').mkdir()
    (root / 'case.json').write_text(json.dumps({
        'source_commit': manifest['source_commit'],
        'bin_root': str(bin_root.resolve()),
        'bridge_root': str(bridge.resolve()),
        'oracle_root': str(oracle_root.resolve()),
        'geometry': {'R':R,'T':T,'K':K,'E':E,'H':H},
        'controls': [
            {'name':'uniform_c1','communication_ctas':[1]*R,
             'steal_budget':[0]*R},
            {'name':'uniform_c74','communication_ctas':[74]*R,
             'steal_budget':[5888]*R},
            {'name':'alternating','communication_ctas':[74,1,74,1],
             'steal_budget':[5888,0,5888,0]},
        ],
        'remote_route_rows':remote_route_rows,
        'remote_payloads_by_owner':payloads_by_owner,
        'scope': 'same external GPU tensors, two complete ABI launches with reset',
    },indent=2)+'\n')


def build(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('nvcc build must be outside the GPU lease')
    manifest = contract(root)
    case = read(root / 'case.json')
    if (case['source_commit'] != manifest['source_commit']
            or any((root / name).exists() for name in
                   ('libranked_tile.so','build_plan.json','build_report.json',
                    'build_failure.json'))):
        raise ValueError('pointer ABI build source or create-only state differs')
    command = [NVCC,'-std=c++17','--gpu-architecture=compute_103a',
               '--gpu-code=sm_103a','-O3','--fmad=false','-lineinfo',
               '-Xptxas=-v','-shared','-Xcompiler','-fPIC',
               'ranked_tile.cu','-o','libranked_tile.so','-lcuda','-lcudart']
    (root / 'build_plan.json').write_text(json.dumps(command,indent=2)+'\n')
    result = subprocess.run(command,cwd=root,capture_output=True,text=True,
                            timeout=240,check=False)
    (root / 'compile.log').write_text(result.stdout+result.stderr or
                                      '(no compiler output)\n')
    if result.returncode:
        (root / 'build_failure.json').write_text(json.dumps({
            'exit_code':result.returncode},indent=2)+'\n')
        raise RuntimeError('pointer ABI CUDA build failed')
    if (root / 'libranked_tile.so').read_bytes()[:4] != b'\x7fELF':
        raise RuntimeError('pointer ABI library is not ELF')
    (root / 'build_report.json').write_text(json.dumps({
        'source_commit':manifest['source_commit'],'target':'sm_103a',
        'exit_code':0,'scope':'CPU-only exact B300 NVCC/PTXAS compilation',
    },indent=2)+'\n')


def admitted() -> str:
    path = Path(os.environ.get('BROKER_RECEIPT','/nonexistent'))
    for _ in range(30):
        if path.is_file():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('broker admission receipt missing')
    receipt = read(path)
    if (receipt.get('label') != LABEL or receipt.get('mode') != 'exclusive'
            or receipt.get('gpu_count') != R
            or len(receipt.get('gpu_ids',[])) != R):
        raise RuntimeError('pointer ABI needs one exact four-GPU lease')
    return receipt['job_id']


def _cuda() -> ctypes.CDLL:
    lib = ctypes.CDLL(CUDART)
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p,ctypes.c_void_p,
                               ctypes.c_size_t,ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    return lib


def _call(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f'{operation} returned CUDA status {status}')


def validate_stolen(stolen: list[int], budget: int | list[int],
                    owner_routes: list[int]) -> None:
    """Require exercised steal only where the route plan has stage tasks."""
    budgets=[budget]*R if type(budget) is int else budget
    if (len(stolen) != R or len(owner_routes) != R
            or len(budgets) != R
            or any(type(value) is not int or type(cap) is not int
                   or value < 0 or cap < 0 or value > cap
                   for value,cap in zip(stolen,budgets,strict=True))
            or any((routes == 0 and value != 0)
                   or (routes > 0 and cap > 0 and value == 0)
                   for value,routes,cap in zip(
                       stolen,owner_routes,budgets,strict=True))):
        raise ValueError('pointer ABI steal cap or nonempty-owner coverage differs')


def _copy_weights(cuda,rank:int,path:Path,extent:int,pointer:ctypes.c_void_p) -> None:
    with path.open('rb') as file:
        file.seek(rank*extent)
        data = file.read(extent)
    if len(data) != extent:
        raise ValueError('expert weight shard extent differs')
    host = np.frombuffer(data,dtype=np.uint8)
    _call(cuda.cudaMemcpy(pointer,ctypes.c_void_p(host.ctypes.data),
                          extent,H2D),'expert weight H2D')


def run(root: Path) -> None:
    manifest=contract(root)
    case=read(root / 'case.json')
    if (read(root / 'build_report.json')['source_commit']
            != manifest['source_commit']
            or (root / 'device.json').exists()
            or (root / 'launch_failure.json').exists()):
        raise ValueError('pointer ABI run source or create-only state differs')
    job=admitted()
    cuda=_cuda()
    library=ctypes.CDLL(str(root / 'libranked_tile.so'))
    prefix=ENTRY+'_'
    void_array=ctypes.POINTER(ctypes.c_void_p)
    create=getattr(library,prefix+'create')
    create.argtypes=[void_array]*6+[ctypes.POINTER(ctypes.c_void_p)]
    create.restype=ctypes.c_int
    launch=getattr(library,prefix+'launch')
    launch.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int),
                     ctypes.POINTER(ctypes.c_int)]
    launch.restype=ctypes.c_int
    destroy=getattr(library,prefix+'destroy')
    destroy.argtypes=[ctypes.c_void_p]
    destroy.restype=ctypes.c_int
    stolen=getattr(library,prefix+'stolen')
    stolen.argtypes=[ctypes.c_void_p,ctypes.c_int,
                     ctypes.POINTER(ctypes.c_int)]
    stolen.restype=ctypes.c_int
    payloads=getattr(library,prefix+'payloads')
    payloads.argtypes=[ctypes.c_void_p,ctypes.c_int,
                       ctypes.POINTER(ctypes.c_int)]
    payloads.restype=ctypes.c_int
    ranks=getattr(library,prefix+'ranks');ranks.restype=ctypes.c_int
    events=getattr(library,prefix+'source_events');events.restype=ctypes.c_int
    bin_bytes=getattr(library,prefix+'bin_bytes')
    bin_bytes.restype=ctypes.c_size_t
    output_bytes=getattr(library,prefix+'output_bytes')
    output_bytes.restype=ctypes.c_size_t
    bin_extent=bin_bytes()
    if (ranks()!=R or events()!=20 or output_bytes()!=T*H*2
            or not (R-1)*T*H*2 < bin_extent < 16*1024*1024):
        raise ValueError('compiled pointer ABI rank/event/output facts differ')
    bin_root=Path(case['bin_root'])
    bridge=Path(case['bridge_root'])
    hidden=np.fromfile(bin_root / 'hidden.bf16',dtype='<u2').reshape(R,T,H)
    ids=np.fromfile(bin_root / 'expert_ids.i32',dtype='<i4').reshape(R,T,K)
    weights=np.fromfile(bridge / 'route_weights.fp32',dtype='<f4').reshape(R,T,K)
    inputs=[hidden,ids,weights]
    extents=[T*H*2,T*K*4,T*K*4,(E//R)*1536*H*2,
             (E//R)*H*768*2,T*H*2]
    buffers=[(ctypes.c_void_p*R)() for _ in range(6)]
    handle=ctypes.c_void_p()
    created=False
    stolen_by_control=[]
    payloads_by_control=[]
    try:
        for rank in range(R):
            _call(cuda.cudaSetDevice(rank),'select rank')
            for index,extent in enumerate(extents):
                pointer=ctypes.c_void_p()
                _call(cuda.cudaMalloc(ctypes.byref(pointer),extent),
                      f'rank {rank} external tensor malloc')
                buffers[index][rank]=pointer
            for index,host in enumerate(inputs):
                _call(cuda.cudaMemcpy(buffers[index][rank],
                                      ctypes.c_void_p(host[rank].ctypes.data),
                                      extents[index],H2D),
                      f'rank {rank} input H2D')
            _copy_weights(cuda,rank,bridge/'weights_upgate.bf16',
                          extents[3],buffers[3][rank])
            _copy_weights(cuda,rank,bridge/'weights_down.bf16',
                          extents[4],buffers[4][rank])
        _call(create(*buffers,ctypes.byref(handle)),'ranked-tile create')
        created=True
        for plan in case['controls']:
            communication=(ctypes.c_int*R)(*plan['communication_ctas'])
            budget=(ctypes.c_int*R)(*plan['steal_budget'])
            _call(launch(handle,communication,budget),
                  f'ranked-tile launch {plan["name"]}')
            counts=[]
            for rank in range(R):
                value=ctypes.c_int(-1)
                _call(stolen(handle,rank,ctypes.byref(value)),
                      f'rank {rank} actual stolen read')
                counts.append(value.value)
            stolen_by_control.append(counts)
            payload_counts=[]
            for rank in range(R):
                value=ctypes.c_int(-1)
                _call(payloads(handle,rank,ctypes.byref(value)),
                      f'rank {rank} remote payload count read')
                payload_counts.append(value.value)
            payloads_by_control.append(payload_counts)
            output=np.empty((R,T,H),dtype='<u2')
            for rank in range(R):
                _call(cuda.cudaSetDevice(rank),'select result rank')
                _call(cuda.cudaMemcpy(
                    ctypes.c_void_p(output[rank].ctypes.data),
                    buffers[5][rank],T*H*2,D2H),
                    'ranked-tile output D2H')
            (root / 'device_outputs' /
             f'output_{plan["name"]}.bf16').write_bytes(
                output.tobytes())
        _call(destroy(handle),'ranked-tile destroy')
        created=False
        (root / 'device.json').write_text(json.dumps({
            'source_commit':manifest['source_commit'],
            'broker_job':job,'target':'sm_103a','world_size':R,
            'source_events':20,'controls':case['controls'],
            'bin_bytes_per_rank':bin_extent,
            'stolen_by_control':stolen_by_control,
            'payloads_by_control':payloads_by_control,
            'scope':'two launches on one pointer ABI state; no qualified timing',
        },indent=2)+'\n')
    except Exception as error:
        (root / 'launch_failure.json').write_text(json.dumps({
            'broker_job':job,'error':f'{type(error).__name__}: {error}'},
            indent=2)+'\n')
        raise
    finally:
        # A failed launch may leave waiters live; the broker owns process exit.
        # Never free scratch underneath potentially active device kernels.
        if not created:
            for rank in range(R):
                cuda.cudaSetDevice(rank)
                for group in buffers:
                    if group[rank]:
                        cuda.cudaFree(group[rank])


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('pointer ABI oracle requires released GPU lease')
    if (root / 'report.json').exists():
        raise ValueError('pointer ABI report must be create-only')
    manifest=contract(root)
    case=read(root / 'case.json')
    device=read(root / 'device.json')
    receipt=read(root / 'gpuq-admission.json')
    if (device['source_commit']!=manifest['source_commit']
            or device['broker_job']!=receipt['job_id']
            or device['source_events']!=20
            or not (R-1)*T*H*2<device['bin_bytes_per_rank']<16*1024*1024
            or device['controls']!=case['controls']
            or len(device['stolen_by_control'])!=len(case['controls'])
            or device['payloads_by_control']
            != [case['remote_payloads_by_owner']]*len(case['controls'])):
        raise ValueError('pointer ABI source, broker or repeated launch differs')
    expected=np.load(Path(case['oracle_root']) / 'expected_output.npy',
                     mmap_mode='r')
    ids=np.fromfile(Path(case['bin_root'])/'expert_ids.i32',dtype='<i4')
    if ids.size != R*T*K or np.any(ids < 0) or np.any(ids >= E):
        raise ValueError('pointer ABI route IDs differ during post-lease verification')
    owner_routes=np.bincount(ids//(E//R),minlength=R).tolist()
    results=[]
    output_bits=[]
    for plan,stolen,payload_counts in zip(
            case['controls'],device['stolen_by_control'],
            device['payloads_by_control'],strict=True):
        validate_stolen(stolen,plan['steal_budget'],owner_routes)
        raw=np.fromfile(root/'device_outputs'/
                        f'output_{plan["name"]}.bf16',dtype='<u2')
        if raw.size!=R*T*H:
            raise ValueError('pointer ABI output extent differs')
        actual=(raw.astype('<u4')<<16).view('<f4').reshape(R,T,H)
        if not np.all(np.isfinite(actual)):
            raise ValueError('pointer ABI output contains nonfinite values')
        difference=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
        failing=difference>.01+.01*np.abs(expected.astype(np.float64))
        results.append({'name':plan['name'],
                        'communication_ctas':plan['communication_ctas'],
                        'steal_budget':plan['steal_budget'],
                        'actual_stolen_by_rank':stolen,
                        'remote_payloads_by_owner':payload_counts,
                        'failing_elements':int(np.count_nonzero(failing)),
                        'max_abs_error':float(np.max(difference))})
        output_bits.append(raw)
    mismatches=sum(int(np.count_nonzero(output_bits[0]!=other))
                   for other in output_bits[1:])
    report={'passed':all(row['failing_elements']==0 for row in results)
                     and mismatches==0,
            'source_commit':manifest['source_commit'],
            'broker_job':device['broker_job'],
            'source_events':20,'controls':results,
            'repeated_launch_bit_mismatches':mismatches,
            'remote_route_rows':case['remote_route_rows'],
            'remote_payloads_per_launch':sum(case['remote_payloads_by_owner']),
            'owner_routes':owner_routes,
            'atol':.01,'rtol':.01,
            'scope':'pointer ABI correctness and state reset against independent FP64 oracle; no qualified timing'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:
        raise SystemExit(1)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','build','run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--bin-root',type=Path)
    parser.add_argument('--bridge-root',type=Path)
    parser.add_argument('--oracle-root',type=Path)
    args=parser.parse_args()
    root=args.root.expanduser().resolve(strict=True)
    if args.phase=='prepare':
        if any(value is None for value in
               (args.bin_root,args.bridge_root,args.oracle_root)):
            parser.error('prepare needs --bin-root, --bridge-root, --oracle-root')
        prepare(root,args.bin_root.expanduser().resolve(strict=True),
                args.bridge_root.expanduser().resolve(strict=True),
                args.oracle_root.expanduser().resolve(strict=True))
    else:
        {'build':build,'run':run,'verify':verify}[args.phase](root)


if __name__=='__main__':
    main()
