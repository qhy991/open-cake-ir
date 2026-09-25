"""Brokered B300 correctness oracle for Cake's model-scale top-8 combine."""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import time

import numpy as np


TOKENS,ROUTES,H = 2048,8,2048
VALUE_BYTES=TOKENS*ROUTES*H*4
WEIGHT_BYTES=TOKENS*ROUTES*4
OUTPUT_BYTES=TOKENS*H*2
LABEL='cake-weave-model-combine-fd698ba9'
CUDA_RUNTIME='/usr/local/cuda-13.1/lib64/libcudart.so'


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def checked(code: int,operation: str) -> None:
    if code:
        raise RuntimeError(f'{operation}: CUDA status {code}')


def prepare(root: Path,bridge: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('combine preparation must be outside the GPU lease')
    if (root/'case.json').exists():
        raise ValueError('combine case must be create-only')
    prior=document(bridge/'report.json')
    if (prior.get('passed') is not True
            or prior.get('routes')!=TOKENS*ROUTES
            or prior.get('failing_elements')!=0
            or (bridge/'device_outputs/contributions.fp32').stat().st_size
            !=VALUE_BYTES
            or (bridge/'route_weights.fp32').stat().st_size!=WEIGHT_BYTES):
        raise ValueError('saved Cake routed FFN contribution boundary differs')
    expected=np.load(bridge/'expected_output.npy',mmap_mode='r')
    if expected.shape!=(4,512,H) or expected.dtype!=np.float32:
        raise ValueError('independent final oracle geometry differs')
    req=document(root/'requirements.json')
    build=document(root/'build_report.json')
    commit=(root/'source_commit.txt').read_text().strip()
    if (req.get('target')!='sm_103a'
            or req.get('grid')!=[TOKENS,H//256,1]
            or req.get('block')!=[256,1,1]
            or req.get('argument_order')
            !=['contributions','weights','output']
            or build.get('compiler_commit')!=commit
            or build.get('exit_codes')
            !={'host_wrapper':0,'cubin':0}):
        raise ValueError('Cake model combine ABI or build differs')
    (root/'case.json').write_text(json.dumps({
        'source_bridge':str(bridge.resolve()),
        'bridge_bin_commit':prior['bin_compiler_commit'],
        'bridge_ffn_commit':prior['ffn_compiler_commit'],
        'combine_compiler_commit':commit,
        'target':'sm_103a',
        'shape':{'tokens':TOKENS,'routes':ROUTES,'hidden':H},
        'atol':prior['atol'],'rtol':prior['rtol'],
        'scope':'one-GPU Cake top-8 combine of retained routed FFN contributions',
    },indent=2)+'\n')


def require_case(root: Path) -> tuple[dict,Path]:
    case=document(root/'case.json')
    bridge=Path(case['source_bridge']).resolve(strict=True)
    if (case['target']!='sm_103a'
            or case['shape']!={'tokens':TOKENS,'routes':ROUTES,'hidden':H}
            or case['atol']!=0.01 or case['rtol']!=0.01
            or case['combine_compiler_commit']
            !=(root/'source_commit.txt').read_text().strip()
            or document(bridge/'report.json').get('passed') is not True
            or (bridge/'device_outputs/contributions.fp32').stat().st_size
            !=VALUE_BYTES):
        raise ValueError('retained Cake combine source case differs')
    return case,bridge


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


class Runtime:
    def __init__(self,root: Path):
        self.root=root
        self.case,self.bridge=require_case(root)
        self.job=admitted()
        self.cuda=C.CDLL(CUDA_RUNTIME,mode=C.RTLD_GLOBAL)
        api=self.cuda
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
        self.library=C.CDLL(str((root/'kernel.so').resolve()))
        abi=document(root/'requirements.json')['host_abi']
        self.create=getattr(self.library,abi['create'])
        self.create.argtypes=[C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        self.create.restype=C.c_int
        self.launch=getattr(self.library,abi['launch'])
        self.launch.argtypes=[C.c_void_p,C.c_void_p]
        self.launch.restype=C.c_int
        self.destroy=getattr(self.library,abi['destroy'])
        self.destroy.argtypes=[C.c_void_p]
        self.destroy.restype=C.c_int
        self.ptrs={}
        self.handle=C.c_void_p()

    def allocate(self,name: str,size: int) -> None:
        pointer=C.c_void_p()
        checked(self.cuda.cudaMalloc(C.byref(pointer),size),f'{name} cudaMalloc')
        self.ptrs[name]=(pointer,size)

    def copy_in(self,name: str,path: Path) -> None:
        pointer,size=self.ptrs[name]
        if path.stat().st_size!=size:
            raise ValueError(f'{name} host input extent differs')
        with path.open('rb') as source:
            offset=0
            while data:=source.read(1<<26):
                host=C.create_string_buffer(data)
                dest=C.c_void_p(pointer.value+offset)
                checked(self.cuda.cudaMemcpy(dest,C.cast(host,C.c_void_p),
                                             len(data),1),f'{name} H2D')
                offset+=len(data)
        if offset!=size:raise ValueError(f'{name} input copy truncated')

    def read(self,name: str,size: int,offset: int=0) -> bytes:
        pointer,extent=self.ptrs[name]
        if size<0 or offset<0 or offset+size>extent:
            raise ValueError(f'{name} D2H extent differs')
        host=C.create_string_buffer(size)
        checked(self.cuda.cudaMemcpy(C.cast(host,C.c_void_p),
                                     C.c_void_p(pointer.value+offset),size,2),
                f'{name} D2H')
        return host.raw

    def inputs_unchanged(self) -> bool:
        for name,path in (
            ('contributions',self.bridge/'device_outputs/contributions.fp32'),
            ('weights',self.bridge/'route_weights.fp32')):
            with path.open('rb') as source:
                offset=0
                while data:=source.read(1<<26):
                    if self.read(name,len(data),offset)!=data:return False
                    offset+=len(data)
        return True

    def run(self) -> None:
        if (self.root/'device.json').exists() or (self.root/'actual.bf16').exists():
            raise ValueError('combine device evidence must be create-only')
        for name,size in (('contributions',VALUE_BYTES),('weights',WEIGHT_BYTES),
                          ('output',OUTPUT_BYTES)):
            self.allocate(name,size)
        self.copy_in('contributions',
                     self.bridge/'device_outputs/contributions.fp32')
        self.copy_in('weights',self.bridge/'route_weights.fp32')
        checked(self.cuda.cudaMemset(self.ptrs['output'][0],255,OUTPUT_BYTES),
                'output sentinel')
        arguments=(C.c_void_p*3)(*(self.ptrs[name][0].value for name in
                                    ('contributions','weights','output')))
        checked(self.create(arguments,C.byref(self.handle)),'Cake combine create')
        checked(self.launch(self.handle,C.c_void_p()),'Cake combine launch')
        checked(self.cuda.cudaDeviceSynchronize(),'Cake combine completion')
        (self.root/'actual.bf16').write_bytes(self.read('output',OUTPUT_BYTES))
        unchanged=self.inputs_unchanged()
        if not unchanged:raise RuntimeError('combine modified a declared input')
        (self.root/'device.json').write_text(json.dumps({
            'broker_job':self.job,'target':'sm_103a',
            'compiler_commit':self.case['combine_compiler_commit'],
            'input_bridge_job':document(self.bridge/'device.json')['broker_job'],
            'inputs_unchanged':unchanged,'output_bytes':OUTPUT_BYTES,
            'scope':'one-GPU model top-8 combine correctness; no timing claim',
        },indent=2)+'\n')

    def close(self) -> None:
        if self.handle.value:
            checked(self.destroy(self.handle),'Cake combine destroy')
            self.handle=C.c_void_p()
        for name,(pointer,_) in reversed(tuple(self.ptrs.items())):
            checked(self.cuda.cudaFree(pointer),f'{name} cudaFree')


def device_run(root: Path) -> None:
    runtime=Runtime(root)
    try:runtime.run()
    except Exception as error:
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job':runtime.job,
            'error':f'{type(error).__name__}: {error}'},indent=2)+'\n')
        raise
    finally:runtime.close()


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('CPU oracle must run after GPU lease release')
    if (root/'report.json').exists():
        raise ValueError('combine oracle report must be create-only')
    case,bridge=require_case(root)
    device=document(root/'device.json')
    admission=document(root/'gpuq-admission.json')
    if (device['broker_job']!=admission['job_id']
            or device['target']!='sm_103a'
            or device['compiler_commit']!=case['combine_compiler_commit']
            or device['output_bytes']!=OUTPUT_BYTES
            or not device['inputs_unchanged']):
        raise ValueError('combine device, Compiler or broker binding differs')
    actual_bits=np.frombuffer((root/'actual.bf16').read_bytes(),
                              dtype='<u2')
    if actual_bits.size!=TOKENS*H:
        raise ValueError('combine BF16 output extent differs')
    actual=(actual_bits.astype('<u4')<<16).view('<f4').reshape(4,512,H)
    expected=np.load(bridge/'expected_output.npy',mmap_mode='r')
    cpu=np.load(bridge/'device_outputs/actual_output.npy',mmap_mode='r')
    if expected.shape!=actual.shape or cpu.shape!=actual.shape:
        raise ValueError('independent or CPU-combine output shape differs')
    finite=np.isfinite(actual)
    error=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    tolerance=case['atol']+case['rtol']*np.abs(expected.astype(np.float64))
    failing=(~finite)|(error>tolerance)
    finite_error=error[finite]
    report={'passed':not np.any(failing),'target':'sm_103a',
            'broker_job':device['broker_job'],
            'compiler_commit':device['compiler_commit'],
            'input_bridge_job':device['input_bridge_job'],
            'elements':int(actual.size),
            'failing_elements':int(np.count_nonzero(failing)),
            'nonfinite_elements':int(np.count_nonzero(~finite)),
            'per_rank_failing_elements':[
                int(np.count_nonzero(failing[rank])) for rank in range(4)],
            'max_abs_error':float(np.max(finite_error)) if finite_error.size else None,
            'bitwise_equal_to_cpu_combine':bool(np.array_equal(
                actual.view('<u4'),cpu.view('<u4'))),
            'cpu_combine_bit_differences':int(np.count_nonzero(
                actual.view('<u4')!=cpu.view('<u4'))),
            'atol':case['atol'],'rtol':case['rtol'],
            'scope':'one-GPU Cake model top-8 combine on saved routed contributions; no EP4 transport or latency claim'}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--bridge',type=Path)
    args=parser.parse_args()
    root=args.root.expanduser().resolve(strict=True)
    if args.phase=='prepare':
        if args.bridge is None:parser.error('prepare requires --bridge')
        prepare(root,args.bridge.expanduser().resolve(strict=True))
    else:{'run':device_run,'verify':verify}[args.phase](root)


if __name__=='__main__':
    main()
