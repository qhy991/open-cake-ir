#!/usr/bin/env python3
"""Apply one explicit Compiler pass and retain the candidate outside the checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from open_cake_ir.compiler import Compiler, frontend


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--producer', type=Path, required=True)
    p.add_argument('--epilogue', type=Path, required=True)
    p.add_argument('--private-intermediate', required=True)
    p.add_argument('--schedule-id', required=True)
    p.add_argument('--entry-point', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = args.output.absolute()
    if output != output.resolve() or output.exists() or any((parent/'.git').exists() for parent in output.parents):
        p.error('output must be a new canonical directory outside every Git checkout')
    compiler = Compiler.load(ROOT, ROOT/'compiler/revision.lock.json')
    producer = frontend.read_schedule(args.producer).document
    epilogue = frontend.read_schedule(args.epilogue).document
    result = compiler.fuse_pointwise_epilogue(producer, epilogue,
        private_intermediate=args.private_intermediate, schedule_id=args.schedule_id, entry_point=args.entry_point)
    output.mkdir(parents=True)
    report = {'applied':result.applied, 'reason':result.reason, 'message':result.message,
              'compiler_revision_path':str(ROOT/'compiler/revision.lock.json'),
              'producer':str(args.producer.resolve()),'epilogue':str(args.epilogue.resolve()),
              'scope':'explicit_private_composition; static_only; no_GPU_or_performance_qualification'}
    if result.applied:
        report['compiler_revision_id'] = result.assessment.compiler_revision_id
        lowered = compiler.lower(result.assessment)
        (output/'schedule.json').write_text(json.dumps(result.schedule,indent=2)+'\n')
        (output/'lowered.py').write_text(lowered.source)
        report['workload_binding'] = 'unbound; bind the composed ABI through its task owner before Lab evaluation'
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return 0 if result.applied else 2


if __name__ == '__main__':
    raise SystemExit(main())
