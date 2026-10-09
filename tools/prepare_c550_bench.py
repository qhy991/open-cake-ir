#!/usr/bin/env python3
"""Prepare original-case Workloads and budgets; never start an optimization loop."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from open_cake_ir.lab.bindings import external_file
from open_cake_ir.tasks.c550_bench.binding import BENCH_COMMIT, BenchProblem, validate_input_view_observation, validate_oracle_numerics
from open_cake_ir.tasks.c550_bench.plan import case_budget_plan
from open_cake_ir.tasks.c550_bench.starters.rms_norm import source as rms_norm_source
from open_cake_ir.tasks.c550_bench.workload import BenchWorkload

STARTER_MODULES = {
    'L1/048_fused_gate_up_projection_with_swiglu': 'gate_up',
    'L1/011_rotary_position_embedding': 'rope',
    'L1/058_moe_expert_token_radix_sort_with_prefix_sum': 'stable_routing',
    'L1/001_attention_softmax_dropout_value_matmul_backward': 'gqa_backward',
    'L2/018_cu_seqlens_variable_length_vision_attention': 'varlen_attention',
    'L2/024_moe_expert_parallel_execution': 'expert_execution',
    'L2/035_convnextv2_block_with_grn': 'convnext',
    'L2/060_chunk_gated_delta_rule_linear_attention': 'chunk_delta',
    'L2/056_language_model_decoder_prenorm_attention_ffn_residual_backward': 'decoder_backward',
}


def write_new(path, document):
    with path.open('x') as output:
        json.dump(document, output, indent=2, allow_nan=False)
        output.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--oracle-numerics', type=Path, required=True,
                        help='retained JSON observation of the original oracle matrix policy')
    parser.add_argument('--task', action='append', help='original suite task ID; default is all ten')
    parser.add_argument('--input-views', type=Path, action='append', default=[],
                        help='retained original-factory dense-view observations')
    args = parser.parse_args(argv)
    numerics_path = external_file(ROOT, str(args.oracle_numerics), 'oracle numerics observation')
    oracle_numerics = validate_oracle_numerics(json.loads(numerics_path.read_text()))
    # The original suite owns its task list. A pinned problem opening verifies
    # the source before any private task data are imported.
    first = BenchProblem.open(args.bench_root, 'L1/069_rms_norm')
    tasks = [row['id'] for row in first.api.document('suite.json')['tasks']]
    selected = tasks if args.task is None else args.task
    if len(set(selected)) != len(selected) or not set(selected) <= set(tasks):
        parser.error('select each original suite task at most once')
    if args.output.resolve().is_relative_to(ROOT):
        parser.error('generated Workloads and plans must stay outside the source checkout')
    observed_views = {}
    for path in args.input_views:
        observation = json.loads(path.read_text())
        if observation.get('task') not in selected:
            parser.error('input-view observation belongs to a task outside this preparation')
        view_problem = first if observation['task'] == first.task_id else BenchProblem.open(args.bench_root, observation['task'])
        for uuid, views in validate_input_view_observation(view_problem, observation).items():
            key = (observation['task'], uuid)
            if key in observed_views:
                parser.error('duplicate input-view observation')
            observed_views[key] = views
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'bench_commit': BENCH_COMMIT, 'status': 'prepared_not_launched',
               'search_owner': 'existing_TaskLab_Ralph', 'oracle_numerics_observation': str(numerics_path),
               'oracle_numerics': oracle_numerics, 'tasks': []}
    write_new(args.output / 'preparation.json', summary)
    for task in selected:
        problem = first if task == first.task_id else BenchProblem.open(args.bench_root, task)
        directory = args.output / task.replace('/', '-')
        directory.mkdir()
        plan = case_budget_plan(problem)
        for index, row in enumerate(plan['cases']):
            case_root = directory / f'case-{index:02d}'
            case_root.mkdir()
            document = problem.workload_document(row['workload_uuid'], oracle_numerics=oracle_numerics,
                input_views=observed_views.get((task, row['workload_uuid'])))
            write_new(case_root / 'workload.json', document)
            starter = None
            if task == 'L1/069_rms_norm':
                starter = rms_norm_source(BenchWorkload(document))
            elif task in STARTER_MODULES:
                module = import_module('benchmarks.c550.' + STARTER_MODULES[task])
                starter = module.source_for_workload(BenchWorkload(document), 'primary')
            if starter is not None:
                (case_root / 'starter.py').write_text(starter, encoding='utf-8')
            row.update(workload_path=str(case_root / 'workload.json'),
                       starter_path=str(case_root / 'starter.py') if starter is not None else None,
                       readiness='awaiting_baseline_qualification' if starter is not None else 'awaiting_starter')
        write_new(directory / 'plan.json', plan)
        summary['tasks'].append({'task': task, 'original_cases': len(plan['cases']),
                                  'plan_path': str(directory / 'plan.json')})
    write_new(args.output / 'prepared.json', summary)
    print(json.dumps({'prepared_tasks': len(summary['tasks']),
                      'prepared_cases': sum(row['original_cases'] for row in summary['tasks']),
                      'launched_runs': 0}))


if __name__ == '__main__':
    main()
