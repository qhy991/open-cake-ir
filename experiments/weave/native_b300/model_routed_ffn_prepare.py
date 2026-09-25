"""CPU-only materialization of model-scale Cake expert weights and oracle."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np


R,T,K,E,H,I = 4,512,8,128,2048,768
LOCAL_E=E//R
UP_BYTES=2*I*H*2
DOWN_BYTES=H*I*2


def document(path: Path) -> dict:
    return json.loads(path.read_text())


def bf16_bytes(values: np.ndarray) -> bytes:
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('expert weight needs finite FP32 source values')
    bits=values.view(np.uint32)
    if np.any(bits & 0xffff):
        raise ValueError('expert weight is not exactly BF16-representable')
    return (bits>>16).astype('<u2').tobytes(order='C')


def prepare(output: Path, baseline: Path, bin_root: Path, ffn_root: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('weight/oracle preparation must be outside the GPU lease')
    if output.exists():
        raise ValueError('routed FFN input root must be create-only')
    oracle_meta=document(baseline/'cpu-oracle-observation.json')
    bin_report=document(bin_root/'report.json')
    ffn_report=document(ffn_root/'report.json')
    bin_commit=(bin_root/'source_commit.txt').read_text().strip()
    ffn_commit=(ffn_root/'compiler_revision_id.txt').read_text().strip()
    if (oracle_meta.get('experiment_id')
            != 'weave-ep4-qwen3-30b-fanin-b300-v2'
            or oracle_meta.get('shape') != [R,T,H]
            or not oracle_meta.get('all_finite')
            or bin_report.get('passed') is not True
            or bin_report.get('routes') != R*T*K
            or bin_report.get('compiler_commit') != bin_commit
            or ffn_report.get('passed') is not True
            or ffn_report.get('compiler_commit') != ffn_commit.removeprefix('open-cake-ir@')
            or len(bin_commit) != 40 or len(ffn_commit) != len('open-cake-ir@')+40):
        raise ValueError('saved oracle, bin or FFN qualification boundary differs')
    expected=np.load(baseline/'oracle-expected.npy',mmap_mode='r')
    if (expected.shape != (R,T,H) or expected.dtype != np.float32
            or not np.isfinite(expected).all()
            or np.any(expected.view(np.uint32)&0xffff)):
        raise ValueError('independent FP64 final oracle extent or BF16 range differs')
    output.mkdir(parents=True)
    weights=[]
    with (output/'weights_upgate.bf16').open('xb') as upfile, \
         (output/'weights_down.bf16').open('xb') as downfile:
        for rank in range(R):
            with np.load(baseline/f'rank{rank}-input.npz') as source:
                ids=source['ids']
                route_weights=source['weights']
                gate=source['gate']
                up=source['up']
                down=source['down']
            if (ids.shape!=(T,K) or ids.dtype!=np.int32
                    or route_weights.shape!=(T,K)
                    or route_weights.dtype!=np.float32
                    or not np.isfinite(route_weights).all()
                    or gate.shape!=(LOCAL_E,I,H)
                    or up.shape!=(LOCAL_E,I,H)
                    or down.shape!=(LOCAL_E,H,I)):
                raise ValueError(f'rank {rank} model input or weight geometry differs')
            for local in range(LOCAL_E):
                # Cake's activation reads up first and gate second.
                upfile.write(bf16_bytes(up[local]))
                upfile.write(bf16_bytes(gate[local]))
                downfile.write(bf16_bytes(down[local]))
            weights.append(route_weights)
    (output/'route_weights.fp32').write_bytes(
        np.concatenate(weights).astype('<f4').tobytes(order='C'))
    shutil.copyfile(baseline/'oracle-expected.npy',
                    output/'expected_output.npy')
    if ((output/'weights_upgate.bf16').stat().st_size != E*UP_BYTES
            or (output/'weights_down.bf16').stat().st_size != E*DOWN_BYTES
            or (output/'route_weights.fp32').stat().st_size != R*T*K*4):
        raise ValueError('materialized expert/route weight byte extent differs')
    (output/'contract.json').write_text(json.dumps({
        'schema_version':1,
        'input_contract':'weave-ep4-qwen3-30b-fanin-b300-v2',
        'geometry':{'R':R,'T':T,'K':K,'E':E,'H':H,'I':I},
        'target':'sm_103a',
        'bin_compiler_commit':bin_commit,
        'ffn_compiler_commit':ffn_commit.removeprefix('open-cake-ir@'),
        'baseline_cpu_oracle':str(baseline.resolve()),
        'bin_evidence_root':str(bin_root.resolve()),
        'ffn_evidence_root':str(ffn_root.resolve()),
        'up_gate_order':'up_then_gate',
        'scope':'one-GPU routed FFN bridge; no distributed EP4 or qualified timing',
    },indent=2)+'\n')
    print(json.dumps({'upgate_bytes':E*UP_BYTES,
                      'down_bytes':E*DOWN_BYTES,
                      'routes':R*T*K,
                      'source_bin_commit':bin_commit,
                      'source_ffn_commit':ffn_commit},indent=2))


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--bin-root',type=Path,required=True)
    parser.add_argument('--ffn-root',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.output.expanduser().absolute(),
            args.baseline.expanduser().resolve(strict=True),
            args.bin_root.expanduser().resolve(strict=True),
            args.ffn_root.expanduser().resolve(strict=True))


if __name__=='__main__':
    main()
