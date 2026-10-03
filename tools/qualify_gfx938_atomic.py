"""Bounded atomic fetch-add qualification; outputs stay outside the Compiler."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))

QUALIFICATION_CONTRACT = {
    'operator': 'int32_atomic_fetch_add_reservation', 'target': 'gfx938',
    'shapes': [[8, 8], [256, 8], [2048, 8]], 'counter_shape': [4],
    'order': 'relaxed', 'scope': 'device', 'increment': 1,
    'inputs': 'immutable INT32 expert indices; mutable INT32 counter state',
    'outputs': 'per-expert permutation of consecutive returned old values; invalid indices return zero',
    'state_effect': 'counter increment equals number of valid reservations per expert',
    'numeric_domain': 'bounded counters avoid INT32 overflow',
    'oracle': 'tools/kernel_oracles.py::_measure_atomic_reservation',
}

def write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, indent=2)
        stream.write('\n')

def emit(destination):
    from open_cake_ir.compiler import Compiler
    engine = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
    if engine.commit is None:
        raise ValueError('clean source required')
    destination.mkdir(parents=True, exist_ok=False)
    write(destination / 'qualification-contract.json', QUALIFICATION_CONTRACT)
    base = json.loads((ROOT / 'corpus/schedules/gfx938-atomic-reservation-b8-smoke.json').read_text())
    entries = []
    for rows in (8, 256, 2048):
        document = copy.deepcopy(base)
        document['schedule_id'] += '-' + str(rows)
        document['lowering']['entry_point'] += '_' + str(rows)
        for buffer in document['buffers']:
            if buffer['name'] in ('expert_ids', 'positions'):
                buffer['shape'][0] = rows
        assessment = engine.assess(document)
        if not assessment.lowering_eligible:
            raise ValueError([f.to_dict() for f in assessment.findings])
        result = engine.lower(assessment)
        source = destination / ('source-' + str(rows) + '.py')
        source.write_text(result.source)
        write(destination / ('schedule-' + str(rows) + '.json'), document)
        entries.append({'rows': rows, 'source': str(source), 'entry': result.route.entry_point,
                        'toolchain': dict(result.toolchain_requirements)})
    write(destination / 'manifest.json', {'compiler_commit': engine.commit, 'entries': entries})

def check(destination):
    import torch
    from kernel_oracles import MEASURE_BY_ENTRY_POINT
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx938'
    manifest = json.loads((destination / 'manifest.json').read_text())
    judge = MEASURE_BY_ENTRY_POINT['cake_atomic_reservation_b8_smoke']
    checks = []
    torch.manual_seed(270)
    for entry in manifest['entries']:
        spec = importlib.util.spec_from_file_location('atomic_' + str(entry['rows']), entry['source'])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run = getattr(module, entry['entry'])
        shape = (entry['rows'], 8)
        for scenario in ('uniform', 'contended', 'all_invalid', 'mixed_masked', 'signed_initial'):
            for round_id in range(10):
                indices = torch.randint(0, 4, shape, device='cuda', dtype=torch.int32)
                if scenario == 'contended': indices.zero_()
                if scenario == 'all_invalid': indices.fill_(-1)
                if scenario == 'mixed_masked': indices.flatten()[::3] = -1; indices.flatten()[1::7] = 4
                initial = torch.tensor([-9, 5, -2, 19] if scenario == 'signed_initial' else [0, 5, 2, 19], device='cuda', dtype=torch.int32)
                counts = initial.clone()
                index_snapshot = indices.clone()
                observed = run(indices, counts)
                torch.cuda.synchronize()
                mismatch, measured, passed = judge(observed, index_snapshot, initial,
                                                    (indices, counts, observed), torch, 0)
                if not passed or not torch.equal(indices, index_snapshot):
                    raise ValueError({'scenario': scenario, 'mismatch': mismatch, 'measured': measured})
                checks.append({'rows': entry['rows'], 'scenario': scenario, 'round': round_id, 'passed': True})
        kernel = getattr(module, entry['toolchain']['kernel_entry_point'])
        compiled = kernel.warmup(indices, counts, torch.empty(shape,device='cuda',dtype=torch.int32),
                                 grid=tuple(entry['toolchain']['grid']),
                                 **entry['toolchain']['compile_constants'],
                                 **entry['toolchain']['compile_options'])
        assembly = compiled.asm['amdgcn']
        (destination / ('atomic-' + str(entry['rows']) + '.amdgcn')).write_text(assembly)
        if not any('atomic_add' in line for line in assembly.splitlines()):
            raise ValueError('actual native atomic-add instruction missing')
    write(destination / 'device-result.json', {'status': 'passed', 'compiler_commit': manifest['compiler_commit'],
          'checks': checks, 'scope': 'INT32 relaxed device-scope fetch-add with returned-old-value and bounded indices; not int32 scan, general scatter, stable sort or performance qualification'})
    print(json.dumps({'status': 'passed', 'checks': len(checks)}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('emit', 'check'))
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination == ROOT or ROOT in destination.parents:
        parser.error('evidence must remain outside Compiler checkout')
    emit(destination) if args.mode == 'emit' else check(destination)
