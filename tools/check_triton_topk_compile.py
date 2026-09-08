#!/usr/bin/env python3
"""Compile accepted loop-carried top-k Corpus cases with the selected tree's toolchain.

This is an offline CPU compile gate. It neither creates GPU handles nor invokes a
provider, evaluates a Workload, changes expectations or qualifies a device result.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

PINNED_TRITON = '3.7.1'


def load_compiler(root: Path, revision: str):
    source = root / 'src'
    sys.path.insert(0, str(source))
    compiler_module = importlib.import_module('open_cake_ir.compiler')
    toolchain_module = importlib.import_module('open_cake_ir.compiler.toolchain')
    for module in (compiler_module, toolchain_module):
        if not Path(module.__file__).resolve(strict=True).is_relative_to(source):
            raise ValueError('Compiler/toolchain import differs from the selected source tree')
    compiler = compiler_module.Compiler.load(root, root / revision)
    return compiler, toolchain_module


def select_cases(compiler, root: Path):
    selected, refused = [], []
    for case in json.loads((root / 'corpus/manifest.json').read_text())['cases']:
        assessment = compiler.assess_file(root / case['schedule'])
        document = json.loads(assessment.schedule_bytes)
        if (document['lowering']['backend'] != 'triton' or not any(
                op['kind'] == 'top_k' and op['parameters'].get('across_loop') for op in document['operations'])):
            continue
        record = {'case_id':case['case_id'], 'schedule':case['schedule']}
        if assessment.accepted and assessment.lowering_eligible:
            selected.append(record)
        else:
            refused.append({**record,'finding_codes':[finding.code for finding in assessment.findings]})
    return selected, refused


def child_command(root, revision, directory, case_id):
    # -I ignores ambient PYTHONPATH/PYTHONHOME; load_compiler binds this exact src tree.
    return [sys.executable, '-I', str(Path(__file__).resolve()), '--project-root', str(root),
            '--revision', revision, '--output-root', str(directory), '--worker-case', case_id]


def worker(root, revision, directory, case_id):
    compiler, toolchain = load_compiler(root, revision)
    if importlib.metadata.version('triton') != PINNED_TRITON:
        raise ValueError(f'offline compile requires Triton {PINNED_TRITON}')
    selected, _ = select_cases(compiler, root)
    case = next((case for case in selected if case['case_id'] == case_id), None)
    if case is None:
        raise ValueError('worker case is not an accepted loop-carried top-k Corpus case')
    lowered = compiler.lower(compiler.assess_file(root / case['schedule']))
    (directory / 'lowered.py').write_text(lowered.source)
    (directory / 'requirements.json').write_text(json.dumps(dict(lowered.toolchain_requirements), indent=2)+'\n')
    result = toolchain.compile_triton(lowered.source.encode(), lowered.toolchain_requirements)
    for role, payload in result.artifacts.items():
        (directory / ('compiled.' + role)).write_bytes(payload)
    record = {'compiler_module':str(Path(sys.modules['open_cake_ir.compiler'].__file__).resolve()),
              'toolchain_module':str(Path(toolchain.__file__).resolve()),
              'target':result.target, 'entry_point':result.entry_point,
              'triton_version':result.compiler_version, 'threads_per_cta':result.threads_per_cta,
              'shared_memory_bytes':result.dynamic_shared_bytes}
    (directory / 'compilation.json').write_text(json.dumps(record,indent=2)+'\n')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--revision', default='compiler/revision.json')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--worker-case', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    if args.worker_case:
        return worker(root, args.revision, args.output_root.resolve(strict=True), args.worker_case)
    compiler, toolchain = load_compiler(root, args.revision)
    selected, refused = select_cases(compiler, root)
    if args.list:
        print(json.dumps({'selected':selected,'refused':refused,
            'compiler_module':str(Path(sys.modules['open_cake_ir.compiler'].__file__).resolve()),
            'toolchain_module':str(Path(toolchain.__file__).resolve())}, indent=2))
        return 0
    if not selected:
        raise ValueError('no accepted loop-carried top-k Corpus cases were selected')
    if importlib.metadata.version('triton') != PINNED_TRITON:
        raise ValueError(f'offline compile requires Triton {PINNED_TRITON}')
    import torch
    if torch.__version__ != '2.9.1+cpu':
        raise ValueError('this offline CI gate requires the pinned CPU torch2.9.1+cpu environment')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    top = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=root,text=True).strip()).resolve()
    if top != root:
        raise ValueError('selected source tree must be the exact Git checkout root')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise ValueError('offline compile requires a clean selected source tree')
    if args.output_root is None or not args.output_root.is_absolute():
        raise ValueError('--output-root must be an absolute new external directory')
    output = args.output_root.resolve()
    if root == output or root in output.parents or any((p / '.git').exists() for p in output.parents):
        raise ValueError('compile artifacts must remain outside every checkout')
    output.mkdir(parents=True,exist_ok=False)
    records=[]
    for ordinal, case in enumerate(selected):
        directory=output/f'case-{ordinal:03d}'; directory.mkdir()
        argv=child_command(root,args.revision,directory,case['case_id'])
        try:
            completed=subprocess.run(argv,cwd=root,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,timeout=600,env=dict(os.environ))
            stdout,stderr,returncode,timed_out=completed.stdout,completed.stderr,completed.returncode,False
        except subprocess.TimeoutExpired as error:
            stdout,stderr,returncode,timed_out=error.stdout or b'',error.stderr or b'',None,True
        (directory/'stdout.txt').write_bytes(stdout)
        (directory/'stderr.txt').write_bytes(stderr)
        records.append({**case,'directory':directory.name,'returncode':returncode,'timed_out':timed_out})
    report={'schema_version':1,'kind':'offline_triton_compile_gate','source_commit':commit,
        'project_root':str(root),'revision':args.revision,'python':platform.python_version(),
        'triton':PINNED_TRITON,'torch':torch.__version__,
        'compiler_module':str(Path(sys.modules['open_cake_ir.compiler'].__file__).resolve()),
        'toolchain_module':str(Path(toolchain.__file__).resolve()),
        'cases':records,'refused_cases':refused,'passed':all(r['returncode']==0 for r in records),
        'device_validation':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(output/'report.json')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
