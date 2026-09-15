#!/usr/bin/env python3
"""Materialize one explicitly selected launch-width candidate for common evaluation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from open_cake_ir.compiler import Compiler, frontend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--num-warps', type=int, required=True)
    parser.add_argument('--schedule-id', required=True)
    parser.add_argument('--entry-point', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.absolute()
    if out != out.resolve() or out.exists() or any((p / '.git').exists() for p in out.parents):
        parser.error('output must be a new canonical directory outside Git checkouts')
    compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.lock.json')
    result = compiler.specialize_triton_warps(frontend.read_schedule(args.schedule).document,
        num_warps=args.num_warps, schedule_id=args.schedule_id, entry_point=args.entry_point)
    out.mkdir(parents=True)
    record = {'applied': result.applied, 'reason': result.reason, 'message': result.message,
              'num_warps': args.num_warps, 'scope': 'explicit candidate; common GPU evaluation required'}
    if result.applied:
        record['compiler_revision_id'] = result.assessment.compiler_revision_id
        (out / 'schedule.json').write_text(json.dumps(result.schedule, indent=2) + '\n')
        (out / 'lowered.py').write_text(compiler.lower(result.assessment).source)
    (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    return 0 if result.applied else 2


if __name__ == '__main__':
    raise SystemExit(main())
