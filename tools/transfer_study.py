#!/usr/bin/env python3
"""Prepare, execute or audit preassigned E/P Studies through independent Runs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from open_cake_ir.lab import read_study
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.tasks.compose import run_runtime_factory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--plan',type=Path,required=True)
    prepare.add_argument('--output',type=Path,required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--study',type=Path,required=True)
    execute = sub.add_parser('execute')
    execute.add_argument('--study',type=Path,required=True)
    execute.add_argument('--runtime-config',type=Path,required=True)
    args = parser.parse_args(argv)
    lab = TaskLab(ROOT)
    if args.command=='prepare':
        study = lab.prepare_study(args.plan,args.output)
        result = {'study':str(study.root),'study_id':study.plan.study_id,
                  'allocated_runs':len(study.plan.allocations()),'executed_runs':0}
    else:
        study = read_study(args.study)
        if args.command=='execute':
            lab.execute_study(study,runtime_factory=run_runtime_factory(ROOT,args.runtime_config))
        result = lab.audit_study(study)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
