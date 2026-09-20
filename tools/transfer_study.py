#!/usr/bin/env python3
"""Prepare or audit preassigned E/P Studies; these commands launch no provider or GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from open_cake_ir.lab import read_study
from open_cake_ir.tasks.runtime import TaskLab


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--plan',type=Path,required=True)
    prepare.add_argument('--output',type=Path,required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--study',type=Path,required=True)
    args = parser.parse_args(argv)
    lab = TaskLab(ROOT)
    if args.command=='prepare':
        study = lab.prepare_study(args.plan,args.output)
        result = {'study':str(study.root),'study_id':study.plan.study_id,
                  'allocated_runs':len(study.plan.allocations()),'executed_runs':0}
    else:
        result = lab.audit_study(read_study(args.study))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
