#!/usr/bin/env python3
"""CPU-only compiler replay: vary only C++ standard, preserve CUDA reference text.

The original failure and successor object build are retained separately. No shared
headers are patched, no module is loaded, and this establishes no GPU correctness.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sysconfig


def main():
    root=Path(os.environ['KERNELINFRA_CANDIDATE_DIR'])
    out=Path(os.environ['KERNELINFRA_STAGE_DIR'])
    meta=json.loads((root/'reference/reference.json').read_text())
    if meta['kind']!='cuda_cpp' or len(meta['files'])!=1:
        raise ValueError('probe requires the retained single-source CUDA reference')
    name=Path(meta['files'][0])
    if name.is_absolute() or '..' in name.parts:
        raise ValueError('source must stay in the reference input')
    source=root/'reference'/name
    if source.is_symlink() or not source.is_file():raise ValueError('regular source required')
    torch_root=Path(importlib.util.find_spec('torch').origin).parent
    flags=meta['compile_options']['cuda_cflags']
    if '-std=c++17' not in flags:raise ValueError('original standard is not C++17')
    base=['/usr/local/cuda/bin/nvcc','-c',str(source),'-DTORCH_EXTENSION_NAME=cake_reference_probe',
          '-DTORCH_API_INCLUDE_EXTENSION_H','-D__CUDA_NO_HALF_OPERATORS__',
          '-D__CUDA_NO_HALF_CONVERSIONS__','-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
          '-D__CUDA_NO_HALF2_OPERATORS__','--expt-relaxed-constexpr',
          '-gencode=arch=compute_103,code=sm_103',"--compiler-options=-fPIC"]
    for path in [torch_root/'include',torch_root/'include/torch/csrc/api/include',
                 Path('/usr/local/cuda/include'),Path(sysconfig.get_paths()['include'])]:
        base+=['-isystem',str(path)]
    env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=''
    rows=[]
    for standard in ['c++17','c++20']:
        command=base+[f'-std={standard}' if f=='-std=c++17' else f for f in flags]+['-o',str(out/(standard+'.o'))]
        p=subprocess.run(command,capture_output=True,text=True,timeout=300,env=env)
        (out/(standard+'.stdout')).write_text(p.stdout);(out/(standard+'.stderr')).write_text(p.stderr)
        rows.append({'standard':standard,'returncode':p.returncode,'command':command,
                     'object_exists':(out/(standard+'.o')).is_file()})
    report={'scope':'CPU object build only; unchanged CUDA source; no linked module or GPU launch',
            'hypothesis':'explicit C++20 compilation resolves the observed NVCC/ATen dependent-type rejection',
            'rows':rows}
    (out/'build-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    passed=rows[0]['returncode']!=0 and rows[1]['returncode']==0
    Path(os.environ['KERNELINFRA_RESULT']).write_text(json.dumps({'schema':'kernelinfra.stage-result.v1',
        'status':'passed' if passed else 'failed','validity':'valid' if passed else 'unknown',
        'summary':'C++17 failure / C++20 success reproduced' if passed else 'hypothesis not established',
        'artifacts':{'probe':'build-probe.json','original_error':'c++17.stderr','successor_error':'c++20.stderr'}})+'\n')
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
