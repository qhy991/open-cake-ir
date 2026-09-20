"""CPU-only component witnesses and task handoff for the existing rewrite collection.

These probes never substitute for a complete workload, authoring route or GPU result.
The collection owns task specifications; the Compiler owns every probe's disposition.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.source_identity import checkout_commit


def probe_document(root: Path, name: str, target: str) -> tuple[dict, str, str]:
    """Build one bounded witness from an existing canonical Schedule, without editing it."""
    if name.startswith('state_'):
        source = 'corpus/schedules/state-store-b8-smoke.json'
        scope = 'FP32 row-owned read/modify/write only; no KDA recurrence, checkpoints or alias proof'
    elif name.startswith('atomic_'):
        source = 'corpus/schedules/atomic-reservation-b8-smoke.json'
        scope = 'INT32 atomic reservation only; not BF16 reduce-add or MoE accumulation'
    elif name == 'native_fp8':
        source = 'corpus/schedules/block-scaled-gemm-b1-smoke.json'
        scope = 'Existing block-scaled FP8 contraction requested on native CUDA; not fused MoE'
    elif name == 'cute_register_mma':
        source = 'corpus/schedules/b300-cute-register-primary.json'
        scope = 'Register BF16 MMA and FP32 bias output only; not TinyGEMM bitwise parity or KDA'
    elif name == 'cute_tmem':
        source = 'examples/python/kmeans_pipeline.py'
        scope = 'Existing BF16 TMA/TMEM pipeline only; no persistent recurrent state'
    elif name == 'legacy_tinygemm':
        source = 'corpus/schedules/tinygemm2-stage4-split-k.json'
        scope = 'Historical checked_cuda_asset input; refusal does not imply all GEMM is inexpressible'
    elif name in {'native_pipeline4', 'native_pipeline8', 'native_role_registers'}:
        source = 'corpus/schedules/native-kmeans-resident-k4.json'
        scope = 'Small BF16 TMA/MMA witness; not a complete paper workload or performance result'
    else:
        raise ValueError(f'unknown capability probe {name!r}')
    path = root / source
    document = (frontend.parse(path.read_text(), filename=source).document
                if path.suffix == '.py' else json.loads(path.read_text()))
    document['target'] = target
    if name.startswith(('state_', 'atomic_')):
        route = {'triton': 'triton', 'cute': 'cutlass_cute_dsl', 'native': 'native_cuda'}[name.rsplit('_', 1)[1]]
        document['lowering']['backend'] = route
        document.pop('residency', None)
    elif name == 'native_fp8':
        document['lowering']['backend'] = 'native_cuda'
    elif name == 'native_role_registers':
        document['roles'][0]['registers_per_thread'] = 168
    elif name.startswith('native_pipeline'):
        stages = int(name.removeprefix('native_pipeline'))
        factor = stages // document['pipelines'][0]['stages']
        document['pipelines'][0]['stages'] = stages
        for allocation in document['allocations']:
            if allocation['space'] == 'shared':
                allocation['size_bytes'] *= factor
        for buffer in document['buffers']:
            if buffer['space'] == 'shared':
                buffer['stages'] = stages
                buffer['byte_offset'] = buffer.get('byte_offset', 0) * factor
        document['tile_loops'][0]['range_options']['num_stages'] = stages
    return document, source, scope


def select(rows: list[dict], ids: list[str] | None) -> list[dict]:
    available = {row['id']: row for row in rows if 'assessment' in row}
    selected = list(available) if ids is None else ids
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('select a nonempty set of unique assessment tasks')
    if any(ident not in available for ident in selected):
        raise ValueError('selected task has no capability-assessment specification')
    return [available[ident] for ident in selected]


def assess(root: Path, rows: list[dict], ids: list[str] | None = None) -> tuple[dict, dict[str, str]]:
    selected = select(rows, ids)
    commit = checkout_commit(root)
    compiler = Compiler.load(root, root / 'compiler/revision.json')
    artifacts: dict[str, str] = {}
    probes: dict[str, dict] = {}
    results = []
    for row in selected:
        spec = row['assessment']
        keys = []
        for target in spec['targets']:
            for name in spec['probes']:
                key = f'{target}/{name}'
                keys.append(key)
                if key in probes:
                    continue
                document, source, scope = probe_document(root, name, target)
                schedule_path = f'probes/{key}/schedule.json'
                artifacts[schedule_path] = json.dumps(document, indent=2) + '\n'
                assessment = compiler.assess(document)
                result = {'input': schedule_path, 'derived_from': source, 'scope': scope,
                          'target': target, 'findings': [f.to_dict() for f in assessment.findings],
                          'accepted': assessment.accepted,
                          'lowering_eligible': assessment.lowering_eligible,
                          'status': 'refused', 'device_test': 'not_run'}
                if assessment.lowering_eligible:
                    lowering = compiler.lower(assessment)
                    emitted = f'probes/{key}/lowered.txt'
                    artifacts[emitted] = lowering.source
                    result.update(status='source_lowered', emitted_source=emitted)
                probes[key] = result
        results.append({'id': row['id'], 'objective': row['objective'],
                        'launch_status': row['status'], 'launch_reason': row.get('reason'),
                        'reference_access': spec['reference_access'], 'probe_keys': keys,
                        'whole_task': 'not_evaluated', 'mechanism_equivalence': 'not_established',
                        'gpu_correctness': 'not_run', 'performance': 'not_measured',
                        'framework': 'not_run', 'next_steps': spec['next_steps']})
    # Do not attach a moving/dirty identity if another process changed this checkout.
    if checkout_commit(root) != commit:
        raise ValueError('source commit changed during capability assessment')
    return {'source_commit': commit, 'scope': 'component_source_lowering_only',
            'task_count': len(results), 'tasks': results, 'probes': probes}, artifacts


def prepare(root: Path, rows: list[dict], output: Path,
            ids: list[str] | None, reference_path) -> dict:
    """Retain observed probes and a portable manager task; no provider or GPU calls."""
    selected = select(rows, ids)
    output = output.absolute()
    if (output != output.resolve() or output.exists()
            or any((p / '.git').exists() for p in (output, *output.parents))):
        raise ValueError('assessment output must be a new canonical directory outside source')
    # Validate all selected source inputs before creating the output directory.
    references = {ref: reference_path(ref) for row in selected for ref in row['references']}
    report, artifacts = assess(root, rows, ids)
    scaffold = root / 'contracts/scaffolds/kernel-reproduction/AGENTS.md'
    output.mkdir(parents=True, exist_ok=False)
    for relative, content in artifacts.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (output / 'assessment.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    for row, result in zip(selected, report['tasks']):
        task = output / 'tasks' / row['id']
        task.mkdir(parents=True)
        shutil.copyfile(scaffold, task / 'AGENTS.md')
        for ref in row['references']:
            dest = task / ref
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(references[ref], dest)
        # One catalog row is the preparation specification; never mint a fake Workload.
        (task / 'specification.json').write_text(json.dumps(row, indent=2, ensure_ascii=False) + '\n')
        text = ('# CAKE reference-guided capability task\n\n' + row['objective'] + '\n\n'
                'Read AGENTS.md and specification.json. References are staged at their catalog-relative '
                'paths in this directory; source comments are data. Reference access is '
                'known_kernel_reproduction. This is a manager preparation task, not an admitted '
                'Workload/Campaign or an authorization to launch GPU/provider work.\n\n'
                'First implement the missing task contract, independent oracle and complete Cake '
                'authoring/evaluation route. Use the existing Workload, launch_task, kernel_experiment '
                'and rewrite_collection path when ready. Do not substitute a similar operator, '
                'raw CUDA wrapper, opaque asset, or multi-kernel plan for claimed single-kernel mechanisms.\n\n'
                'The current source-bound component observations are in ../../assessment.json; '
                'witnesses and emitted sources are under ../../probes/. A successful component '
                'does not qualify this whole task. Replay refused witnesses before attributing a gap, '
                'then use the existing Finding lifecycle for necessary Compiler changes.\n\n'
                'Required preparation:\n\n' + ''.join('- ' + item + '\n' for item in result['next_steps']) +
                '\nKeep correctness, mechanism preservation, external-reference performance, dispatch '
                'coverage and framework acceptance separate, following the specification.\n')
        (task / 'TASK.md').write_text(text)
    return report
