"""One-GPU bridge from Cake expert bins to Cake tensor-core expert FFN.

The GPU run retains one FP32 expert contribution per route key. Final weighted
combine and the independent FP64 comparison run on the CPU after lease release.
This is a routed arithmetic bridge, not four-device EP4 or a timing result.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import mmap
import os
from pathlib import Path
import sys
import time

import numpy as np

from bin_pack_oracle import R,T,K,E,H,LOCAL_E,MAX_ROWS,ROUTES,verify_rows
from model_routed_ffn_prepare import I,UP_BYTES,DOWN_BYTES


LABEL='cake-weave-model-routed-ffn-bridge-b300'
CUDA_RUNTIME='/usr/local/cuda-13.1/lib64/libcudart.so'
TILE=128
CONTRIBUTION_BYTES=ROUTES*H*4
BIN_ORDER=('expert_ids_local','token_index','route_key_input',
           'hidden','counts','route_keys','packed')
SIZES={'expert_ids_local':ROUTES*4,'token_index':ROUTES*4,
       'route_key_input':ROUTES*4,'hidden':R*T*H*2,
       'counts':LOCAL_E*4,'route_keys':LOCAL_E*MAX_ROWS*4,
       'packed':LOCAL_E*MAX_ROWS*H*2,
       'w_up_gate':UP_BYTES,'w_down':DOWN_BYTES,
       'up_gate':TILE*2*I*4,'activated':TILE*I*2,
       'out':TILE*H*4}


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def checked(code: int, operation: str) -> None:
    if code:
        raise RuntimeError(f'{operation}: CUDA status {code}')


def roots(root: Path) -> tuple[dict,Path,Path]:
    contract=document(root/'contract.json')
    if (contract.get('input_contract')
            != 'weave-ep4-qwen3-30b-fanin-b300-v2'
            or contract.get('geometry')
            != {'R':R,'T':T,'K':K,'E':E,'H':H,'I':I}
            or contract.get('target')!='sm_103a'
            or contract.get('up_gate_order')!='up_then_gate'
            or (root/'weights_upgate.bf16').stat().st_size!=E*UP_BYTES
            or (root/'weights_down.bf16').stat().st_size!=E*DOWN_BYTES
            or (root/'route_weights.fp32').stat().st_size!=ROUTES*4):
        raise ValueError('routed FFN contract or model weight extents differ')
    bin_root=Path(contract['bin_evidence_root']).resolve(strict=True)
    ffn_root=Path(contract['ffn_evidence_root']).resolve(strict=True)
    if (document(bin_root/'report.json').get('passed') is not True
            or document(ffn_root/'report.json').get('passed') is not True
            or (bin_root/'source_commit.txt').read_text().strip()
            !=contract['bin_compiler_commit']
            or (ffn_root/'compiler_revision_id.txt').read_text().strip()
            !='open-cake-ir@'+contract['ffn_compiler_commit']):
        raise ValueError('bin or FFN Compiler/evidence binding differs')
    return contract,bin_root,ffn_root


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


class Cuda:
    def __init__(self):
        self.runtime=C.CDLL(CUDA_RUNTIME,mode=C.RTLD_GLOBAL)
        api=self.runtime
        api.cudaGetDeviceCount.argtypes=[C.POINTER(C.c_int)]
        api.cudaMalloc.argtypes=[C.POINTER(C.c_void_p),C.c_size_t]
        api.cudaFree.argtypes=[C.c_void_p]
        api.cudaMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
        api.cudaMemset.argtypes=[C.c_void_p,C.c_int,C.c_size_t]
        api.cudaDeviceSynchronize.argtypes=[]
        count=C.c_int()
        checked(api.cudaGetDeviceCount(C.byref(count)),'device count')
        if count.value!=1:
            raise RuntimeError('broker must expose one logical GPU')
        self.ptrs={}

    def allocate(self,name: str) -> None:
        pointer=C.c_void_p()
        checked(self.runtime.cudaMalloc(C.byref(pointer),SIZES[name]),
                f'{name} cudaMalloc')
        self.ptrs[name]=pointer

    def pointer(self,name: str,offset: int=0) -> C.c_void_p:
        if not 0<=offset<SIZES[name]:
            raise ValueError(f'{name} pointer offset outside allocated extent')
        return C.c_void_p(self.ptrs[name].value+offset)

    def write(self,name: str,data: bytes) -> None:
        if len(data)!=SIZES[name]:
            raise ValueError(f'{name} input extent differs')
        source=C.create_string_buffer(data)
        checked(self.runtime.cudaMemcpy(self.ptrs[name],
                                         C.cast(source,C.c_void_p),len(data),1),
                f'{name} H2D')

    def read(self,name: str,size: int,offset: int=0) -> bytes:
        if size<0 or offset<0 or offset+size>SIZES[name]:
            raise ValueError(f'{name} D2H extent differs')
        if size==0:return b''
        host=C.create_string_buffer(size)
        checked(self.runtime.cudaMemcpy(C.cast(host,C.c_void_p),
                                         self.pointer(name,offset),size,2),
                f'{name} D2H')
        return host.raw

    def copy_file(self,name: str,path: Path) -> None:
        if path.stat().st_size!=SIZES[name]:
            raise ValueError(f'{name} weight file extent differs')
        self.write(name,path.read_bytes())

    def clear(self,name: str,value: int=0,size: int|None=None,offset: int=0) -> None:
        extent=SIZES[name]-offset if size is None else size
        if extent<0 or offset+extent>SIZES[name]:
            raise ValueError(f'{name} memset extent differs')
        checked(self.runtime.cudaMemset(self.pointer(name,offset),value,extent),
                f'{name} cudaMemset')

    def synchronize(self,operation: str) -> None:
        checked(self.runtime.cudaDeviceSynchronize(),operation)

    def close(self) -> None:
        for name,pointer in reversed(tuple(self.ptrs.items())):
            checked(self.runtime.cudaFree(pointer),f'{name} cudaFree')


class BinHandle:
    def __init__(self,root: Path,cuda: Cuda):
        req=document(root/'requirements.json')
        report=document(root/'build_report.json')
        if (req.get('argument_order')!=list(BIN_ORDER)
                or req.get('grid')!=[R*T,K,1]
                or req.get('block')!=[256,1,1]
                or not req.get('input_domain_runtime_check')
                or report.get('exit_codes')
                !={'host_wrapper':0,'cubin':0}):
            raise ValueError('typed Cake bin ABI or build differs')
        self.library=C.CDLL(str((root/'kernel.so').resolve()))
        abi=req['host_abi']
        self.create=getattr(self.library,abi['create'])
        self.create.argtypes=[C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        self.create.restype=C.c_int
        self.launch=getattr(self.library,abi['launch'])
        self.launch.argtypes=[C.c_void_p,C.c_void_p]
        self.launch.restype=C.c_int
        self.destroy=getattr(self.library,abi['destroy'])
        self.destroy.argtypes=[C.c_void_p]
        self.destroy.restype=C.c_int
        self.handle=C.c_void_p()
        pointers=(C.c_void_p*len(BIN_ORDER))(
            *(cuda.ptrs[name].value for name in BIN_ORDER))
        checked(self.create(pointers,C.byref(self.handle)),'Cake bin create')

    def run(self) -> None:
        checked(self.launch(self.handle,C.c_void_p()),'Cake bin launch')

    def close(self) -> None:
        if self.handle.value:
            checked(self.destroy(self.handle),'Cake bin destroy')
            self.handle=C.c_void_p()


def run(root: Path) -> None:
    contract,bin_root,ffn_root=roots(root)
    if (root/'device.json').exists() or (root/'device_outputs').exists():
        raise ValueError('routed FFN device evidence must be create-only')
    job=admitted()
    cuda=Cuda()
    handle=None
    output=root/'device_outputs'
    try:
        for name in SIZES:
            cuda.allocate(name)
        for name,filename in (('hidden','hidden.bf16'),
                              ('token_index','token_index.i32'),
                              ('route_key_input','route_key_input.i32')):
            cuda.write(name,(bin_root/filename).read_bytes())
        handle=BinHandle(bin_root,cuda)
        sys.path.insert(0,str(ffn_root))
        from model_local_ffn_run import NativeStage, STAGES, prepare as ffn_prepare
        ffn_req=ffn_prepare(ffn_root)
        output.mkdir()
        contribution_path=output/'contributions.fp32'
        seen=set()
        tasks=0
        with contribution_path.open('xb') as file:
            file.truncate(CONTRIBUTION_BYTES)
            with mmap.mmap(file.fileno(),CONTRIBUTION_BYTES,
                           access=mmap.ACCESS_WRITE) as contributions, \
                 (root/'weights_upgate.bf16').open('rb') as up_weights, \
                 (root/'weights_down.bf16').open('rb') as down_weights:
                for owner in range(R):
                    owner_dir=output/f'owner{owner}'
                    owner_dir.mkdir()
                    cuda.write('expert_ids_local',
                               (bin_root/f'owner{owner}-local_ids.i32').read_bytes())
                    cuda.clear('counts')
                    cuda.clear('route_keys',255)
                    handle.run()
                    cuda.synchronize(f'owner {owner} bin completion')
                    counts_raw=cuda.read('counts',SIZES['counts'])
                    counts=np.frombuffer(counts_raw,dtype='<u4')
                    if np.any(counts>MAX_ROWS):
                        raise RuntimeError(f'owner {owner} bin exceeds row capacity')
                    keys_raw=cuda.read('route_keys',SIZES['route_keys'])
                    keys=np.frombuffer(keys_raw,dtype='<i4').reshape(LOCAL_E,MAX_ROWS)
                    (owner_dir/'counts.u32').write_bytes(counts_raw)
                    (owner_dir/'keys.i32').write_bytes(keys_raw)
                    (owner_dir/'observed_local_ids.i32').write_bytes(
                        cuda.read('expert_ids_local',SIZES['expert_ids_local']))
                    packed_rows=bytearray()
                    for local,raw_count in enumerate(counts):
                        packed_rows.extend(cuda.read('packed',int(raw_count)*H*2,
                                                     local*MAX_ROWS*H*2))
                    (owner_dir/'rows.bf16').write_bytes(packed_rows)
                    (owner_dir/'device.json').write_text(json.dumps({
                        'owner_rank':owner,'used_rows':int(sum(map(int,counts))),
                        'error_flag':0},indent=2)+'\n')
                    for local,raw_count in enumerate(counts):
                        count=int(raw_count)
                        expert=owner*LOCAL_E+local
                        if count==0:continue
                        up_weights.seek(expert*UP_BYTES)
                        down_weights.seek(expert*DOWN_BYTES)
                        up_bytes=up_weights.read(UP_BYTES)
                        down_bytes=down_weights.read(DOWN_BYTES)
                        cuda.write('w_up_gate',up_bytes)
                        cuda.write('w_down',down_bytes)
                        for tile_start in range(0,count,TILE):
                            valid=min(TILE,count-tile_start)
                            if valid<TILE:
                                cuda.clear('packed',0,(TILE-valid)*H*2,
                                           (local*MAX_ROWS+tile_start+valid)*H*2)
                            pointers={
                                'x':cuda.pointer('packed',
                                                 (local*MAX_ROWS+tile_start)*H*2),
                                'w_up_gate':cuda.pointer('w_up_gate'),
                                'w_down':cuda.pointer('w_down'),
                                'up_gate':cuda.pointer('up_gate'),
                                'activated':cuda.pointer('activated'),
                                'out':cuda.pointer('out'),
                            }
                            stages=[]
                            try:
                                for name,bindings,_,_,_ in STAGES:
                                    stages.append(NativeStage(ffn_root,name,
                                                              ffn_req[name],bindings,
                                                              pointers))
                                for stage in stages:stage.run()
                                cuda.synchronize(f'expert {expert} tile {tile_start//TILE} FFN completion')
                                actual=cuda.read('out',valid*H*4)
                            finally:
                                for stage in reversed(stages):stage.close()
                            for row in range(valid):
                                key=int(keys[local,tile_start+row])
                                if not 0<=key<ROUTES or key in seen:
                                    raise RuntimeError(f'expert {expert} duplicate/invalid route key {key}')
                                seen.add(key)
                                contributions[key*H*4:(key+1)*H*4]=actual[row*H*4:(row+1)*H*4]
                            tasks+=1
                        if cuda.read('w_up_gate',UP_BYTES)!=up_bytes or \
                           cuda.read('w_down',DOWN_BYTES)!=down_bytes:
                            raise RuntimeError(f'expert {expert} weight input changed')
        if seen!=set(range(ROUTES)):
            raise RuntimeError('routed FFN lost one or more route contributions')
        (output/'observed_hidden.bf16').write_bytes(
            cuda.read('hidden',SIZES['hidden']))
        (output/'observed_token_index.i32').write_bytes(
            cuda.read('token_index',SIZES['token_index']))
        (output/'observed_route_key_input.i32').write_bytes(
            cuda.read('route_key_input',SIZES['route_key_input']))
        (root/'device.json').write_text(json.dumps({
            'broker_job':job,'target':'sm_103a',
            'bin_compiler_commit':contract['bin_compiler_commit'],
            'ffn_compiler_commit':contract['ffn_compiler_commit'],
            'routes':ROUTES,'expert_tiles':tasks,
            'contribution_bytes':CONTRIBUTION_BYTES,
            'all_weight_inputs_unchanged':True,
            'scope':'one-GPU Cake bin→three-stage FFN bridge; CPU weighted combine after lease',
        },indent=2)+'\n')
    except Exception as error:
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job':job,'error':f'{type(error).__name__}: {error}'},
            indent=2)+'\n')
        raise
    finally:
        if handle is not None:handle.close()
        cuda.close()


def bf16_round(values: np.ndarray) -> np.ndarray:
    bits=np.ascontiguousarray(values,dtype='<f4').view('<u4')
    rounded=bits+np.uint32(0x7fff)+((bits>>16)&1)
    return (rounded&np.uint32(0xffff0000)).view('<f4')


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('CPU oracle must run after broker lease release')
    if (root/'report.json').exists():
        raise ValueError('routed FFN report must be create-only')
    contract,bin_root,_=roots(root)
    device=document(root/'device.json')
    receipt=document(root/'gpuq-admission.json')
    if (device['broker_job']!=receipt['job_id']
            or device['target']!='sm_103a'
            or device['bin_compiler_commit']!=contract['bin_compiler_commit']
            or device['ffn_compiler_commit']!=contract['ffn_compiler_commit']
            or device['routes']!=ROUTES
            or not device['all_weight_inputs_unchanged']):
        raise ValueError('routed FFN device/Compiler/broker binding differs')
    output=root/'device_outputs'
    for name in ('token_index','route_key_input'):
        if ((output/f'observed_{name}.i32').read_bytes()
                !=(bin_root/f'{name}.i32').read_bytes()):
            raise ValueError(f'routed FFN changed {name}')
    ids=np.frombuffer((bin_root/'expert_ids.i32').read_bytes(),
                      dtype='<i4').reshape(R*T,K)
    for owner in range(R):
        expected=np.where((ids//LOCAL_E)==owner,
                          ids-owner*LOCAL_E,-1).astype('<i4').tobytes()
        if ((output/f'owner{owner}'/'observed_local_ids.i32').read_bytes()
                !=expected):
            raise ValueError(f'owner {owner} local IDs changed')
    owner_reports=verify_rows(root,observed_global_ids=False,
                              input_root=bin_root)
    contributions=output/'contributions.fp32'
    if contributions.stat().st_size!=CONTRIBUTION_BYTES:
        raise ValueError('FP32 route contribution byte extent differs')
    values=np.memmap(contributions,dtype='<f4',mode='r',shape=(ROUTES,H))
    if not np.isfinite(values).all():
        raise ValueError('one or more expert contributions are nonfinite')
    weights=np.fromfile(root/'route_weights.fp32',dtype='<f4')
    combined=np.zeros((R*T,H),dtype=np.float64)
    for key in range(ROUTES):
        combined[key//K]+=values[key].astype(np.float64)*float(weights[key])
    actual=bf16_round(combined.astype('<f4')).reshape(R,T,H)
    expected=np.load(root/'expected_output.npy',mmap_mode='r')
    error=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    tolerance=0.01+0.01*np.abs(expected.astype(np.float64))
    failing=error>tolerance
    per_rank=[int(np.count_nonzero(failing[rank])) for rank in range(R)]
    passed=not np.any(failing)
    np.save(output/'actual_output.npy',actual)
    report={'passed':passed,'target':'sm_103a',
            'broker_job':device['broker_job'],
            'bin_compiler_commit':contract['bin_compiler_commit'],
            'ffn_compiler_commit':contract['ffn_compiler_commit'],
            'routes':ROUTES,'expert_tiles':device['expert_tiles'],
            'owner_reports':owner_reports,
            'failing_elements':int(np.count_nonzero(failing)),
            'per_rank_failing_elements':per_rank,
            'max_abs_error':float(np.max(error)),
            'atol':0.01,'rtol':0.01,
            'scope':'one-GPU Cake routed FFN contributions with CPU weighted combine; no EP4 transport or qualified timing'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.expanduser().resolve(strict=True)
    {'run':run,'verify':verify}[args.phase](root)


if __name__=='__main__':
    main()
