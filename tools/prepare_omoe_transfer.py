#!/usr/bin/env python3
"""Prepare one external task and source-bound OMOE authoring material; never launch."""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.tasks.workloads import create_task

# Selection belongs here; original facts remain in OMOE at the requested Git commit.
GROUPS = {
    "fusion": ("gemm-swiglu-epilogue-fusion", "geglu-frontend-fusion-sm87-compute-floor"),
    "residency": ("hybrid-shared-register-state-residency", "register-budget-requires-occupancy-limiter-match"),
    "rounding": ("bitexact-triton-needs-scalar-lowering-sm100", "autoregressive-rollout-compounds-fusion-drift"),
}


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout


def recipe_parts(source: str) -> tuple[dict[str, str], str]:
    if not source.startswith("---\n"):
        raise ValueError("recipe frontmatter missing")
    header, body = source[4:].split("\n---\n", 1)
    fields = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    if not {"id", "verdict", "effect", "source", "evidence", "hardware"} <= fields.keys():
        raise ValueError("recipe lacks its original verdict/evidence boundary")
    return fields, body


def project_steps(fields: dict, body: str) -> str:
    """Reorder quoted source sections; never synthesize new applicability or results."""
    applicability = '\n'.join(f'- **{key}:** {fields[key]}' for key in
        ('hardware', 'regime', 'preconditions', 'failure-modes', 'transfer') if fields.get(key))
    evidence = '\n'.join(f'- **{key}:** {fields[key]}' for key in ('verdict', 'effect', 'evidence', 'source'))
    return (f"## {fields['id']}\n\nCheck the source's conditions before proposing a candidate:\n\n"
            f"{applicability}\n\nOriginal mechanism and procedure (including corrections):\n\n{body.strip()}"
            f"\n\nOriginal reported outcome, not a result for this task:\n\n{evidence}\n")


def prepare(*, omoe_root: Path, omoe_ref: str, output: Path, task: str,
            backend: str, rows: int, columns: int, depth: int | None,
            representation: str) -> dict:
    if representation not in {"case", "steps"}:
        raise ValueError("representation must be case or steps")
    output = output.absolute()
    if output != output.resolve() or output.exists() or any((p / '.git').exists() for p in output.parents):
        raise ValueError("output must be a new canonical directory outside every Git checkout")
    commit = git(omoe_root, 'rev-parse', '--verify', '--end-of-options', f'{omoe_ref}^{{commit}}').strip()
    records, materials = [], []
    for group, ids in GROUPS.items():
        for recipe_id in ids:
            path = f'recipes/{recipe_id}.md'
            source = git(omoe_root, 'show', f'{commit}:{path}')
            fields, body = recipe_parts(source)
            if fields['id'] != recipe_id:
                raise ValueError("recipe filename/id mismatch")
            records.append({'group': group, 'path': path, 'source_commit': commit,
                            'reported_verdict': fields['verdict'],
                            'evidence_status': 'source_record_only_not_requalified'})
            materials.append(source if representation == 'case' else project_steps(fields, body))
    document, source = create_task(task, backend=backend, rows=rows, columns=columns, depth=depth)
    compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.lock.json')
    assessment = compiler.assess(frontend.parse(source).document)
    if not assessment.lowering_eligible:
        codes = ', '.join(f.code for f in assessment.findings)
        raise ValueError(f"task is not lowerable under this Compiler: {codes}")
    lowered = compiler.lower(assessment)
    scaffold = (ROOT / 'contracts/scaffolds/matched-search-v1.md').read_text()
    scaffold += ('\n\n# OMOE experience input\n\n'
                 'Reference access: known_kernel_reproduction only. This material includes low-level '
                 'implementation details and is forbidden in clean_start/direct_low_level authoring. '
                 'Source outcomes are hypotheses for the new task, never acceptance authority. '
                 'Follow the frozen Workload including each output comparison. Never relax its oracle '
                 'or tolerance to fit a recipe. Do not infer model speedup from this tensor task.\n\n'
                 + '\n\n'.join(materials))
    manifest = {'source_repository': 'https://github.com/qhy991/omoe', 'source_commit': commit,
                'representation': representation, 'reference_access': 'known_kernel_reproduction',
                'recipes': records, 'task': task, 'workload_id': document['workload_id'],
                'compiler_revision_id': assessment.compiler_revision_id,
                'status': 'statically_lowered_not_gpu_qualified',
                'experiment_status': 'not_launched',
                'scaffold_binding': 'Bind scaffold.md through the existing arm.scaffold reference before Campaign preflight.',
                'claim_scope': 'artifact_optimization_only; not a representation-comparison study'}
    output.mkdir(parents=True)
    for name, value in [('workload.json', document), ('transfer.json', manifest)]:
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (output/'candidate.py').write_text(source)
    (output/'lowered.py').write_text(lowered.source)
    (output/'scaffold.md').write_text(scaffold)
    # Establish the existing Lab reference boundary once, for these newly produced bytes.
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'),
                                          ensure_ascii=False, allow_nan=False).encode()
    references = {'workload': {'path': str(output/'workload.json'),
                               'canonical_sha256': sha256(canonical(document)).hexdigest()},
                  'arm': {'reference_access': 'known_kernel_reproduction',
                          'scaffold': {'path': str(output/'scaffold.md'),
                                       'sha256': sha256(scaffold.encode()).hexdigest()},
                          'schedule_skeleton': {'path': str(output/'candidate.py'),
                                                'canonical_sha256': sha256(canonical(frontend.parse(source).document)).hexdigest()},
                          'lowering_route': frontend.parse(source).document['lowering'],
                          'input_format': 'schedule_or_python_v1'}}
    (output/'authoring-references.json').write_text(json.dumps(references, indent=2)+'\n')
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--omoe-root', type=Path, required=True)
    p.add_argument('--omoe-ref', required=True, help='source commit/ref; resolved once, working-tree edits ignored')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', default='add_rmsnorm_bf16', help='task already registered by this checkout; gemm_silu follows its own integration')
    p.add_argument('--backend', default='triton-b200')
    p.add_argument('--rows', type=int, default=128)
    p.add_argument('--columns', type=int, default=2560)
    p.add_argument('--depth', type=int)
    p.add_argument('--representation', choices=('case', 'steps'), default='steps')
    args = p.parse_args()
    result = prepare(**vars(args))
    print(json.dumps({'output': str(args.output), 'status': result['status'],
                      'task': result['task'], 'recipe_count': len(result['recipes'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
