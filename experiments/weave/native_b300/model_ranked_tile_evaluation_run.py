"""Brokered Evaluation-adapter replay of public B300 ranked-tile lowering."""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np

import model_ranked_tile_pointer_run as base


LABEL='cake-weave-evaluation-ranked-tile-b300'
R,T,K,E,H=base.R,base.T,base.K,base.E,base.H


@dataclass(frozen=True)
class CudaTensor:
    pointer: int
    device: int
    spec: object


def admitted() -> str:
    path=Path(__import__('os').environ.get('BROKER_RECEIPT','/nonexistent'))
    for _ in range(30):
        if path.is_file():break
        time.sleep(.1)
    else:raise RuntimeError('Evaluation broker receipt missing')
    receipt=base.read(path)
    if (receipt.get('label')!=LABEL or receipt.get('mode')!='exclusive'
            or receipt.get('gpu_count')!=R
            or len(receipt.get('gpu_ids',[]))!=R):
        raise RuntimeError('Evaluation four-GPU admission differs')
    return receipt['job_id']


def _lowered(root: Path):
    sys.path.insert(0,str(root/'src'))
    from open_cake_ir.compiler import Program,RankedTileEffects,Schedule
    from open_cake_ir.compiler.backends.native_cuda_ranked_tile import (
        NativeRankedTileLowering,
    )
    from types import MappingProxyType
    report=base.read(root/'lowering_report.json')
    local=Program.from_dict(base.read(root/'local_program.json'))
    effects=RankedTileEffects.from_dict(base.read(root/'effects.json'))
    combine=Schedule.from_dict(base.read(root/'combine.json'))
    analysis=effects.analyze(local,combine)
    lowered=NativeRankedTileLowering(
        local,combine,effects,analysis,report['compiler_revision_id'],
        (root/'ranked_tile.cu').read_text(),
        MappingProxyType({name:tuple(span) for name,span in
                          report['source_map'].items()}),
        MappingProxyType(report['toolchain_requirements']))
    lowered.validate_binding()
    return lowered


def run(root: Path) -> None:
    manifest=base.contract(root)
    case=base.read(root/'case.json')
    if (base.read(root/'build_report.json')['source_commit']
            !=manifest['source_commit']
            or (root/'device.json').exists()
            or (root/'launch_failure.json').exists()):
        raise ValueError('Evaluation replay source or create-only state differs')
    job=admitted()
    lowered=_lowered(root)
    if lowered.toolchain_requirements['compiler_commit']!=manifest['source_commit']:
        raise ValueError('Evaluation Compiler commit differs from source')
    from open_cake_ir.compiler.ir import DType,ProgramTensor
    from open_cake_ir.evaluation.ranked_tile_launch import prepare_ranked_tiles
    from open_cake_ir.evaluation.ranked_tile_ctypes import load_ranked_tile_ctypes
    cuda=base._cuda()
    bin_root=Path(case['bin_root']);bridge=Path(case['bridge_root'])
    hidden=np.fromfile(bin_root/'hidden.bf16',dtype='<u2').reshape(R,T,H)
    ids=np.fromfile(bin_root/'expert_ids.i32',dtype='<i4').reshape(R,T,K)
    weights=np.fromfile(bridge/'route_weights.fp32',dtype='<f4').reshape(R,T,K)
    hosts=[hidden,ids,weights]
    extents=[T*H*2,T*K*4,T*K*4,(E//R)*1536*H*2,
             (E//R)*H*768*2,T*H*2]
    names=('hidden','expert_ids','route_weights','w_up_gate','w_down')
    buffers=[(ctypes.c_void_p*R)() for _ in range(6)]
    prepared=None
    try:
        for rank in range(R):
            base._call(cuda.cudaSetDevice(rank),'select input rank')
            for index,extent in enumerate(extents):
                pointer=ctypes.c_void_p()
                base._call(cuda.cudaMalloc(ctypes.byref(pointer),extent),
                           f'rank {rank} tensor malloc')
                buffers[index][rank]=pointer
            for index,host in enumerate(hosts):
                base._call(cuda.cudaMemcpy(
                    buffers[index][rank],ctypes.c_void_p(host[rank].ctypes.data),
                    extents[index],base.H2D),f'rank {rank} input H2D')
            base._copy_weights(cuda,rank,bridge/'weights_upgate.bf16',
                               extents[3],buffers[3][rank])
            base._copy_weights(cuda,rank,bridge/'weights_down.bf16',
                               extents[4],buffers[4][rank])
        specs={row['name']:ProgramTensor(tuple(row['shape']),
                                         DType(row['dtype']))
               for row in lowered.toolchain_requirements['rank_inputs']}
        output_spec=ProgramTensor((T,H),DType.BF16)
        tensors={rank:{name:CudaTensor(int(buffers[index][rank]),rank,specs[name])
                       for index,name in enumerate(names)} for rank in range(R)}
        outputs={rank:CudaTensor(int(buffers[5][rank]),rank,output_spec)
                 for rank in range(R)}
        def plans(row):
            return {rank:{'communication_ctas':row['communication_ctas'][rank],
                          'chunks':4,'steal_budget':row['steal_budget'][rank]}
                    for rank in range(R)}
        library_path=root/'libranked_tile.so'
        loader=lambda lower:load_ranked_tile_ctypes(
            lower,library_path,pointer_of=lambda tensor:tensor.pointer,
            isolated_process=True)
        def check_tensor(tensor,spec):
            if tensor.spec!=spec:raise ValueError('Evaluation tensor spec differs')
        prepared=prepare_ranked_tiles(
            lowered,tensors,outputs,plans(case['controls'][0]),
            load_source=loader,check_tensor=check_tensor,
            storage_span=lambda tensor:(tensor.device,tensor.pointer,
                                        tensor.pointer+tensor.spec.nbytes),
            execution_context=lambda rank:('cuda-device',rank))
        library=ctypes.CDLL(str(library_path))
        query=getattr(library,lowered.toolchain_requirements['host_abi']['bin_bytes'])
        query.restype=ctypes.c_size_t
        bin_extent=query()
        stolen_by_control=[];payloads_by_control=[]
        for row in case['controls']:
            _,status=prepared.run(plans(row))
            stolen_by_control.append(list(status['stolen_by_rank']))
            payloads_by_control.append(list(status['remote_payloads_by_owner']))
            output=np.empty((R,T,H),dtype='<u2')
            for rank in range(R):
                base._call(cuda.cudaSetDevice(rank),'select result rank')
                base._call(cuda.cudaMemcpy(
                    ctypes.c_void_p(output[rank].ctypes.data),
                    buffers[5][rank],T*H*2,base.D2H),
                    'Evaluation output D2H')
            (root/'device_outputs'/f'output_{row["name"]}.bf16').write_bytes(
                output.tobytes())
        prepared.close()
        (root/'device.json').write_text(json.dumps({
            'source_commit':manifest['source_commit'],'broker_job':job,
            'target':'sm_103a','world_size':R,'source_events':20,
            'controls':case['controls'],
            'bin_bytes_per_rank':bin_extent,
            'stolen_by_control':stolen_by_control,
            'payloads_by_control':payloads_by_control,
            'evaluation_adapter':'ranked_tile_launch.prepare_ranked_tiles',
            'scope':'three adapter launches in one isolated broker process; no qualified timing',
        },indent=2)+'\n')
    except Exception as error:
        (root/'launch_failure.json').write_text(json.dumps({
            'broker_job':job,'error':f'{type(error).__name__}: {error}'},
            indent=2)+'\n')
        raise
    finally:
        if prepared is None or prepared.closed:
            for rank in range(R):
                cuda.cudaSetDevice(rank)
                for group in buffers:
                    if group[rank]:cuda.cudaFree(group[rank])


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
        base.prepare(root,args.bin_root.expanduser().resolve(strict=True),
                     args.bridge_root.expanduser().resolve(strict=True),
                     args.oracle_root.expanduser().resolve(strict=True))
    elif args.phase=='build':base.build(root)
    elif args.phase=='run':run(root)
    else:base.verify(root)


if __name__=='__main__':main()
