"""Brokered correctness runner for Cake's typed B300 expert-bin Schedule."""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import time

import numpy as np

from bin_pack_oracle import R, T, K, E, H, LOCAL_E, MAX_ROWS, ROUTES, verify_rows


LABEL = 'cake-weave-model-expert-bin-cake-66e97f3a'
CUDA_RUNTIME = '/usr/local/cuda-13.1/lib64/libcudart.so'
ARGUMENT_ORDER = ('expert_ids_local', 'token_index', 'route_key_input',
                  'hidden', 'counts', 'route_keys', 'packed')
SIZES = {'expert_ids_local': ROUTES*4, 'token_index': ROUTES*4,
         'route_key_input': ROUTES*4, 'hidden': R*T*H*2,
         'counts': LOCAL_E*4, 'route_keys': LOCAL_E*MAX_ROWS*4,
         'packed': LOCAL_E*MAX_ROWS*H*2}


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def require_root(root: Path) -> dict:
    req = document(root / 'requirements.json')
    commit = (root / 'source_commit.txt').read_text().strip()
    case = document(root / 'case.json')
    arguments = [(row['name'], row['dtype'], row['shape'], row['mode'])
                 for row in req['arguments']]
    expected = [
        ('expert_ids_local', 'int32', [R*T,K], 'input'),
        ('token_index', 'int32', [R*T,K], 'input'),
        ('route_key_input', 'int32', [R*T,K], 'input'),
        ('hidden', 'bf16', [R*T,H], 'input'),
        ('counts', 'int32', [LOCAL_E], 'state'),
        ('route_keys', 'int32', [LOCAL_E,MAX_ROWS], 'output'),
        ('packed', 'bf16', [LOCAL_E,MAX_ROWS,H], 'output'),
    ]
    if (len(commit) != 40 or req.get('target') != 'sm_103a'
            or req.get('argument_order') != list(ARGUMENT_ORDER)
            or arguments != expected
            or req.get('grid') != [R*T,K,1]
            or req.get('block') != [256,1,1]
            or not req.get('input_domain_runtime_check')
            or req.get('state_reset') != 'zero_counts_before_launch'
            or req.get('bin_capacity_rows_per_expert') != MAX_ROWS
            or case.get('input_contract') != 'weave-ep4-qwen3-30b-fanin-b300-v2'
            or case.get('geometry') != {'R':R,'T':T,'K':K,'E':E,'H':H}
            or (root / 'hidden.bf16').stat().st_size != SIZES['hidden']
            or (root / 'expert_ids.i32').stat().st_size != ROUTES*4
            or (root / 'token_index.i32').stat().st_size != ROUTES*4
            or (root / 'route_key_input.i32').stat().st_size != ROUTES*4
            or any((root / f'owner{owner}-local_ids.i32').stat().st_size
                   != ROUTES*4 for owner in range(R))):
        raise ValueError('Cake bin-pack source, typed ABI or input contract differs')
    return req


def prepare(root: Path, source_case: Path) -> None:
    if (root / 'case.json').exists() or (root / 'hidden.bf16').exists():
        raise ValueError('Cake bin-pack input material must be create-only')
    original = document(source_case / 'case.json')
    if (original.get('geometry') != {'R':R,'T':T,'K':K,'E':E,'H':H}
            or original.get('input_contract')
            != 'weave-ep4-qwen3-30b-fanin-b300-v2'):
        raise ValueError('source input is not the retained model fan-in case')
    hidden = (source_case / 'hidden.bf16').read_bytes()
    ids_bytes = (source_case / 'expert_ids.i32').read_bytes()
    if len(hidden) != SIZES['hidden'] or len(ids_bytes) != ROUTES*4:
        raise ValueError('source BF16/route bytes differ')
    ids = np.frombuffer(ids_bytes, dtype='<i4').reshape(R*T,K)
    if (np.any(ids < 0) or np.any(ids >= E)
            or any(len(set(map(int,row))) != K for row in ids)):
        raise ValueError('source expert domain or per-token uniqueness differs')
    (root / 'hidden.bf16').write_bytes(hidden)
    (root / 'expert_ids.i32').write_bytes(ids_bytes)
    tokens = np.broadcast_to(np.arange(R*T,dtype='<i4')[:,None],
                             (R*T,K)).copy()
    keys = np.arange(ROUTES,dtype='<i4').reshape(R*T,K)
    (root / 'token_index.i32').write_bytes(tokens.tobytes())
    (root / 'route_key_input.i32').write_bytes(keys.tobytes())
    for owner in range(R):
        local = np.where((ids // LOCAL_E) == owner,
                         ids - owner*LOCAL_E, -1).astype('<i4')
        (root / f'owner{owner}-local_ids.i32').write_bytes(local.tobytes())
    (root / 'case.json').write_text(json.dumps({
        'geometry': {'R':R,'T':T,'K':K,'E':E,'H':H},
        'input_contract': 'weave-ep4-qwen3-30b-fanin-b300-v2',
        'source_case': str(source_case.resolve()),
        'scope': 'Cake native one-GPU dynamic expert-bin Schedule correctness',
    }, indent=2)+'\n')
    require_root(root)


def checked(code: int, operation: str) -> None:
    if code:
        raise RuntimeError(f'{operation}: CUDA status {code}')


def admitted() -> str:
    path = Path(os.environ.get('BROKER_RECEIPT','/nonexistent'))
    for _ in range(30):
        if path.is_file():
            break
        time.sleep(.1)
    else:
        raise RuntimeError('broker admission receipt missing')
    row = document(path)
    if (row.get('label') != LABEL or row.get('mode') != 'exclusive'
            or row.get('gpu_count') != 1 or len(row.get('gpu_ids',[])) != 1
            or not isinstance(row.get('job_id'),str)):
        raise RuntimeError('one exact exclusive B300 GPU admission differs')
    return row['job_id']


class Runtime:
    def __init__(self, root: Path):
        self.root = root
        req = require_root(root)
        report = document(root / 'build_report.json')
        if (report.get('compiler_commit')
            != (root / 'source_commit.txt').read_text().strip()
            or report.get('target') != 'sm_103a'
            or report.get('exit_codes')
            != {'host_wrapper':0,'cubin':0}
            or (root / 'kernel.so').read_bytes()[:4] != b'\x7fELF'
            or (root / 'kernel.cubin').read_bytes()[:4] != b'\x7fELF'):
            raise ValueError('Cake compiled source or exact target differs')
        self.job = admitted()
        self.cuda = C.CDLL(CUDA_RUNTIME, mode=C.RTLD_GLOBAL)
        cuda = self.cuda
        cuda.cudaGetDeviceCount.argtypes = [C.POINTER(C.c_int)]
        cuda.cudaMalloc.argtypes = [C.POINTER(C.c_void_p),C.c_size_t]
        cuda.cudaFree.argtypes = [C.c_void_p]
        cuda.cudaMemcpy.argtypes = [C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
        cuda.cudaMemset.argtypes = [C.c_void_p,C.c_int,C.c_size_t]
        cuda.cudaDeviceSynchronize.argtypes = []
        count=C.c_int()
        checked(cuda.cudaGetDeviceCount(C.byref(count)), 'device count')
        if count.value != 1:
            raise RuntimeError('broker must expose one logical GPU')
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
        self.pointers={}
        self.handle=C.c_void_p()

    def allocate(self,name: str) -> None:
        pointer=C.c_void_p()
        checked(self.cuda.cudaMalloc(C.byref(pointer),SIZES[name]),f'{name} cudaMalloc')
        self.pointers[name]=pointer

    def write(self,name: str,data: bytes) -> None:
        if len(data)!=SIZES[name]:
            raise ValueError(f'{name} input byte extent differs')
        source=C.create_string_buffer(data)
        checked(self.cuda.cudaMemcpy(self.pointers[name],
                                     C.cast(source,C.c_void_p),len(data),1),
                f'{name} H2D')

    def read(self,name: str,size: int,offset: int=0) -> bytes:
        result=C.create_string_buffer(size)
        source=C.c_void_p(self.pointers[name].value+offset)
        checked(self.cuda.cudaMemcpy(C.cast(result,C.c_void_p),source,size,2),
                f'{name} D2H')
        return result.raw

    def run(self) -> None:
        if (self.root/'device.json').exists() or (self.root/'device_outputs').exists():
            raise ValueError('Cake bin-pack device evidence must be create-only')
        for name in ARGUMENT_ORDER:
            self.allocate(name)
        for name,filename in (('token_index','token_index.i32'),
                              ('route_key_input','route_key_input.i32'),
                              ('hidden','hidden.bf16')):
            self.write(name,(self.root/filename).read_bytes())
        pointers=(C.c_void_p*len(ARGUMENT_ORDER))(
            *(self.pointers[name].value for name in ARGUMENT_ORDER))
        checked(self.create(pointers,C.byref(self.handle)),'Cake create')
        output=self.root/'device_outputs'
        output.mkdir()
        for owner in range(R):
            row=output/f'owner{owner}'
            row.mkdir()
            self.write('expert_ids_local',
                       (self.root/f'owner{owner}-local_ids.i32').read_bytes())
            checked(self.cuda.cudaMemset(self.pointers['counts'],0,SIZES['counts']),
                    'counts reset')
            checked(self.cuda.cudaMemset(self.pointers['route_keys'],255,
                                         SIZES['route_keys']), 'route keys reset')
            checked(self.launch(self.handle,C.c_void_p()),f'owner {owner} launch')
            checked(self.cuda.cudaDeviceSynchronize(),f'owner {owner} completion')
            counts_raw=self.read('counts',SIZES['counts'])
            counts=np.frombuffer(counts_raw,dtype='<u4')
            if np.any(counts>MAX_ROWS):
                raise RuntimeError(f'owner {owner} expert count exceeds capacity')
            (row/'counts.u32').write_bytes(counts_raw)
            (row/'keys.i32').write_bytes(
                self.read('route_keys',SIZES['route_keys']))
            packed_rows=bytearray()
            for local_expert,count in enumerate(counts):
                packed_rows.extend(self.read('packed',int(count)*H*2,
                                             local_expert*MAX_ROWS*H*2))
            (row/'rows.bf16').write_bytes(packed_rows)
            (row/'observed_local_ids.i32').write_bytes(
                self.read('expert_ids_local',SIZES['expert_ids_local']))
            (row/'device.json').write_text(json.dumps({
                'owner_rank':owner,'used_rows':int(sum(map(int,counts))),
                'error_flag':0},indent=2)+'\n')
        local=np.frombuffer((self.root/'owner0-local_ids.i32').read_bytes(),
                            dtype='<i4').copy().reshape(R*T,K)
        duplicate=local.copy()
        selected=None
        for token in range(R*T):
            active=np.flatnonzero(local[token]>=0)
            if len(active)>=2:
                selected=(token,int(active[0]),int(active[1]))
                break
        if selected is None:
            raise RuntimeError('retained owner0 routes lack a duplicate-domain negative control')
        token,first,second=selected
        duplicate[token,second]=duplicate[token,first]
        self.write('expert_ids_local',duplicate.tobytes())
        checked(self.cuda.cudaMemset(self.pointers['counts'],0,SIZES['counts']),
                'duplicate control counts reset')
        duplicate_status=self.launch(self.handle,C.c_void_p())
        if duplicate_status!=1 or any(self.read('counts',SIZES['counts'])):
            raise RuntimeError('duplicate local expert was not refused before launch')
        self.write('expert_ids_local',local.tobytes())
        nonzero=bytearray(SIZES['counts'])
        nonzero[:4]=(1).to_bytes(4,'little',signed=True)
        self.write('counts',bytes(nonzero))
        nonzero_status=self.launch(self.handle,C.c_void_p())
        if nonzero_status!=1 or self.read('counts',SIZES['counts'])!=bytes(nonzero):
            raise RuntimeError('nonzero count state was not refused before launch')
        negatives={'duplicate_expert':{'status':duplicate_status,
                                       'token':token,'routes':[first,second],
                                       'counts_unchanged':True},
                   'nonzero_count':{'status':nonzero_status,
                                    'counts_unchanged':True}}
        (output/'observed_hidden.bf16').write_bytes(
            self.read('hidden',SIZES['hidden']))
        (output/'observed_token_index.i32').write_bytes(
            self.read('token_index',SIZES['token_index']))
        (output/'observed_route_key_input.i32').write_bytes(
            self.read('route_key_input',SIZES['route_key_input']))
        (self.root/'device.json').write_text(json.dumps({
            'broker_job':self.job,'target':'sm_103a',
            'compiler_commit':(self.root/'source_commit.txt').read_text().strip(),
            'owners':R,'route_ctas_per_owner':ROUTES,
            'host_domain_negative_controls':negatives,
            'scope':'Cake typed one-GPU expert-bin Schedule; no EP4 transport',
        },indent=2)+'\n')

    def close(self) -> None:
        if self.handle.value:
            checked(self.destroy(self.handle),'Cake destroy')
            self.handle=C.c_void_p()
        for name,pointer in reversed(tuple(self.pointers.items())):
            checked(self.cuda.cudaFree(pointer),f'{name} cudaFree')


def device_run(root: Path) -> None:
    runtime=Runtime(root)
    try:
        runtime.run()
    except Exception as error:
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job':runtime.job,
            'error':f'{type(error).__name__}: {error}'},indent=2)+'\n')
        raise
    finally:
        runtime.close()


def verify(root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('CPU oracle must run after broker lease release')
    if (root/'report.json').exists():
        raise ValueError('Cake bin-pack report must be create-only')
    require_root(root)
    device=document(root/'device.json')
    receipt=document(root/'gpuq-admission.json')
    if (device['broker_job']!=receipt['job_id']
            or device['compiler_commit']
            !=(root/'source_commit.txt').read_text().strip()
            or device['target']!='sm_103a' or device['owners']!=R
            or device['route_ctas_per_owner']!=ROUTES
            or device.get('host_domain_negative_controls',{}).get(
                'duplicate_expert',{}).get('status')!=1
            or device.get('host_domain_negative_controls',{}).get(
                'nonzero_count',{}).get('status')!=1):
        raise ValueError('Cake device/Compiler/broker binding differs')
    output=root/'device_outputs'
    for name in ('token_index','route_key_input'):
        if ((output/f'observed_{name}.i32').read_bytes()
                !=(root/f'{name}.i32').read_bytes()):
            raise ValueError(f'Cake device modified {name}')
    ids=np.frombuffer((root/'expert_ids.i32').read_bytes(),
                      dtype='<i4').reshape(R*T,K)
    for owner in range(R):
        expected=np.where((ids//LOCAL_E)==owner,
                          ids-owner*LOCAL_E,-1).astype('<i4').tobytes()
        local=root/f'owner{owner}-local_ids.i32'
        observed=output/f'owner{owner}'/'observed_local_ids.i32'
        if local.read_bytes()!=expected or observed.read_bytes()!=expected:
            raise ValueError(f'Cake owner {owner} local route IDs differ')
    reports=verify_rows(root,observed_global_ids=False)
    result={'passed':True,'target':'sm_103a',
            'compiler_commit':device['compiler_commit'],
            'broker_job':device['broker_job'],'routes':ROUTES,
            'input_bytes_unchanged':True,'owner_reports':reports,
            'scope':'Cake native one-GPU expert-bin Schedule; no EP4 mailbox, tile publication, FFN or latency claim'}
    (root/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','run','verify'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source-case',type=Path)
    args=parser.parse_args()
    root=args.root.expanduser().resolve(strict=True)
    if args.phase=='prepare':
        if args.source_case is None:
            parser.error('prepare requires --source-case')
        prepare(root,args.source_case.expanduser().resolve(strict=True))
    else:
        {'run':device_run,'verify':verify}[args.phase](root)


if __name__=='__main__':
    main()
