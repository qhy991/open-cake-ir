#!/usr/bin/env python3
"""Compare sealed rewrite/starter artifacts with the unchanged supplied reference.

Uses the existing Workload oracle and paired CUPTI policy. This is a development
comparison, not a Campaign promotion, external leaderboard score or host-latency claim.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from tools.compare_flashinfer_reference import LoadedCallable, admit_judge_source, load_module, write
from open_cake_ir.evaluation.admission import observe_exclusive_cuda
from open_cake_ir.evaluation.benchmark import StrictCuptiBenchmark
from open_cake_ir.evaluation.core import LoadedTorchTensorCandidate, compare_tile_outputs
from open_cake_ir.evaluation.gpuq import observe_allocation
from open_cake_ir.evaluation.paired import paired_protocol, paired_summary, validate_pair_candidates
from open_cake_ir.evaluation.timing import summarize_cohort
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.bindings import load_baseline_bundle
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.evaluate import _fresh_tile_cohort
from open_cake_ir.tasks.normalization.study import evaluation_policy
from open_cake_ir.tasks.workloads import materialize_case, reference_outputs


def regular(root, name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('comparison file name differs')
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('comparison file must stay within its input root')
    path = root / relative
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError('comparison requires canonical regular files')
    return path


def reference_spec(root):
    spec = json.loads(regular(root, 'reference/reference.json').read_bytes())
    if spec.get('kind') not in ('python', 'cuda_cpp'):
        raise ValueError('unsupported reference implementation kind')
    files = spec.get('files')
    if not isinstance(files, list) or not files or len(set(files)) != len(files):
        raise ValueError('reference source files differ')
    paths = [regular(root / 'reference', name) for name in files]
    entry = spec.get('entry_point')
    if not isinstance(entry, str) or entry.count('::') != 1:
        raise ValueError('reference entry must name file::function')
    file, function = entry.split('::')
    if file not in files or not function.isidentifier():
        raise ValueError('reference entry is absent from supplied files')
    if spec['kind'] == 'python' and not file.endswith('.py'):
        raise ValueError('Python reference entry must be a Python file')
    return spec, paths, file, function


def load_reference(root, output, report, cohort_calls):
    spec, paths, file, function = reference_spec(root)
    if spec['kind'] == 'python':
        sys.path.insert(0, str(root / 'reference'))
        module = load_module('comparison_reference', root / 'reference' / file)
    else:
        from torch.utils.cpp_extension import load
        flags = spec['compile_options']
        if not set(flags) <= {'cflags', 'cuda_cflags', 'ld_flags'}:
            raise ValueError('unsupported external compilation option')
        os.environ['TORCH_CUDA_ARCH_LIST'] = '10.3'
        os.environ['MAX_JOBS'] = '4'
        build = output / 'external-build'
        build.mkdir()
        module = load(name='cake_external_reference', sources=[str(p) for p in paths],
                      extra_cflags=flags.get('cflags', []),
                      extra_cuda_cflags=flags.get('cuda_cflags', []),
                      extra_ldflags=flags.get('ld_flags', []),
                      build_directory=str(build), verbose=True)
    modules = [module]
    if spec.get('cached_output', False):
        if spec['kind'] != 'cuda_cpp' or sys.getdlopenflags() & os.RTLD_GLOBAL:
            raise ValueError('cached output isolation requires native modules loaded locally')
        instances = output / 'external-instances'
        instances.mkdir()
        # Distinct loader paths give each unchanged native module independent static
        # Graph/output state. Runtime output-pointer checks, not this assumption, gate timing.
        for index in range(1, cohort_calls):
            directory = instances / str(index)
            directory.mkdir()
            path = directory / Path(module.__file__).name
            shutil.copyfile(module.__file__, path)
            modules.append(load_module(module.__name__, path))
    references = [getattr(item, function) for item in modules]
    if any(not callable(reference) for reference in references):
        raise ValueError('reference entry is not callable')
    report['reference'] = spec
    if spec.get('derived_from'):
        report['roles']['external'] = 'derived external reference; source correction recorded in reference metadata'
    report['external_instance_count'] = len(modules)
    report['external_instance_policy'] = ('Independent copies of one compiled native module; '
        'each cached output is warmed and poisoned before the cohort, then used once. '
        'Original Graph replay is retained; this does not measure natural host/cache lifecycle costs.'
        if len(modules) > 1 else 'Original callable with unique retained outputs per cohort')
    return references



class RetainedExternal(LoadedCallable):
    """Retain distinct outputs; cached native wrappers get one instance per call."""
    def __init__(self, workload, values, launches, torch, cached_output=False):
        super().__init__(workload, values, launches[0], torch)
        self.launches = launches
        self.cached_output = cached_output
        self.seen = set()

    def fresh_argument_sets(self, count):
        arguments = super().fresh_argument_sets(count)
        self.seen = set()
        if self.cached_output:
            if count > len(self.launches):
                raise ValueError('not enough independent reference instances')
            pointers = set()
            for index, argument in enumerate(arguments):
                argument['instance'] = index
                self.launch_function = self.launches[index]
                super().launch(argument)
                result = argument['result']
                pointer = result.data_ptr()
                if pointer in pointers:
                    raise ValueError('external native instances share an output buffer')
                pointers.add(pointer)
                argument['warm_output'] = result
                argument['expected_pointer'] = pointer
                result.fill_(float('nan'))
                argument['result'] = None
            self.torch.cuda.synchronize()
        return arguments

    def launch(self, arguments):
        self.launch_function = self.launches[arguments.get('instance', 0)]
        super().launch(arguments)
        pointer = arguments['result'].data_ptr()
        if self.cached_output and pointer != arguments['expected_pointer']:
            raise ValueError('external timed output differs from poisoned warm buffer')
        if pointer in self.seen:
            raise ValueError('external output reused within a timed cohort')
        self.seen.add(pointer)

def correctness(loaded, workload, inputs, expected):
    import torch
    arguments = loaded.fresh_argument_sets(1)[0]
    loaded.launch(arguments)
    torch.cuda.synchronize()
    observed, after = loaded.snapshot(arguments)
    passed, metrics = compare_tile_outputs(workload, inputs, expected, observed, after)
    return {'passed': passed, 'metrics': metrics}


def main():
    output = Path(os.environ['KERNELINFRA_STAGE_DIR'])
    root = Path(os.environ['KERNELINFRA_CANDIDATE_DIR'])
    report = {'scope': 'sealed_three_way_development_comparison', 'provider_calls': 0,
              'promotion_disposition': 'No promotion', 'cases': [], 'comparisons': {},
              'roles': {'optimized': 'sealed confirmed Cake candidate',
                        'starter': 'sealed original Cake starter',
                        'external': 'unchanged supplied external implementation'}}
    loaded = {}
    try:
        report['judge_commit'] = checkout_commit(ROOT)
        admit_judge_source(report['judge_commit'],
            json.loads(Path(os.environ['KERNELINFRA_TASK']).read_bytes()), os.environ['KERNELINFRA_STAGE_ID'])
        report['input'] = json.loads(regular(root, 'comparison.json').read_bytes())
        workload = WorkloadContract(json.loads(regular(root, 'workload.json').read_bytes()))
        if workload.target != 'sm_103a':
            raise ValueError('comparison binds exact B300 target')
        abi = workload.tensor_abi('primary')
        if any(a.dtype != 'bf16' for a in abi) or [a.name for a in abi if a.mode == 'output'] != ['out']:
            raise ValueError('comparison currently admits the single-output BF16 normalization ABI')
        names = [a.name for a in abi if a.mode == 'input']
        if names not in (['x', 'weight'], ['x', 'residual', 'weight']):
            raise ValueError('reference call ABI differs')
        participants = {role: load_baseline_bundle(ROOT, regular(root, role + '/candidate.json'))
                        for role in ('optimized', 'starter')}
        manifests = validate_pair_candidates(participants['optimized'], participants['starter'], workload, 'primary')
        manifests = {'optimized': manifests['candidate'], 'starter': manifests['baseline']}
        for manifest in manifests.values():
            for case in workload.case_ids:
                manifest.check_validation_case(workload, case)
        report['allocation'] = observe_allocation('sm_103a')
        admission = observe_exclusive_cuda('sm_103a')
        import torch
        import flashinfer.testing as timer
        torch.set_num_threads(4)
        sys.dont_write_bytecode = True
        report['device'] = {'name': admission.device_name, 'job_id': admission.broker_job_id}
        policy = evaluation_policy(workload)
        protocol = paired_protocol(policy)
        external = load_reference(root, output, report, protocol.route_calls_per_cohort)
        cases = {case: materialize_case(workload, case) for case in workload.case_ids}
        expected = {case: reference_outputs(workload, case, values) for case, values in cases.items()}
        reference_launches = [lambda values, out, fn=fn: fn(*(values[name] for name in names)) for fn in external]
        for case, values in cases.items():
            for role in ('optimized', 'starter'):
                loaded[role, case] = LoadedTorchTensorCandidate(participants[role], manifests[role], values, admission)
            loaded['external', case] = RetainedExternal(workload, values, reference_launches, torch,
                cached_output=report['reference'].get('cached_output', False))
        def check_all(phase):
            for case, values in cases.items():
                for role in ('external', 'starter', 'optimized'):
                    result = correctness(loaded[role, case], workload, values, expected[case])
                    report['cases'].append({'phase': phase, 'case': case, 'role': role, **result})
                    if not result['passed']:
                        raise ArithmeticError(f'{role} failed {case}; external source and tolerances unchanged')
        check_all('preflight')
        strict = StrictCuptiBenchmark(timer)
        # A matched pair for every edge of the three-way comparison. The same sealed
        # objects and external callable remain loaded throughout this allocation.
        for left, right in [('optimized', 'external'), ('starter', 'external'), ('optimized', 'starter')]:
            key = left + '_vs_' + right
            raw = {'evaluation_protocol': policy, 'roles': {'candidate': left, 'baseline': right}, 'measurements': []}
            report['comparisons'][key] = raw
            for index, order in enumerate(protocol.pair_order):
                pair = {'pair_index': index, 'order': list(order), 'arms': {}}
                for position, role in enumerate(order):
                    actual = left if role == 'candidate' else right
                    samples, check = _fresh_tile_cohort(loaded[actual, 'primary'], strict,
                        workload, cases['primary'], expected['primary'],
                        samples_per_cohort=protocol.samples_per_cohort,
                        route_calls_per_cohort=protocol.route_calls_per_cohort)
                    if not check['passed']:
                        raise ArithmeticError(f'{actual} timed output validation failed')
                    pair['arms'][role] = {'position': position, 'samples_ms': samples,
                        'summary': summarize_cohort(samples), 'route_calls': check['checked_launches'], 'output_check': check}
                raw['measurements'].append(pair)
                print(json.dumps({'comparison': key, 'pair': index,
                    'medians_ms': {r: x['summary']['median_ms'] for r, x in pair['arms'].items()}}), flush=True)
            raw['timing'] = paired_summary(raw)
        check_all('postflight')
        report['correctness_passed'] = True
        report['measurement_quality_passed'] = all(c['timing']['measurement_quality_passed'] for c in report['comparisons'].values())
        report['status'] = 'passed' if report['measurement_quality_passed'] else 'measurement_quality_failed'
        report['timed_interval'] = ('CUPTI device-activity span of the unmodified call; cold L2 per sample; no timer graph/event fallback. '
            'No validation memcpy/memset is inserted in the timed call. Original external Graph replay is retained. '
            'Compilation, cohort preparation and host time are excluded; device work issued by the reference call is included.')
        validity = 'valid'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        validity = 'invalid' if isinstance(error, ArithmeticError) else 'unknown'
    finally:
        for (role, _), obj in loaded.items():
            if role != 'external':
                obj.close()
    write(output / 'comparison-report.json', report)
    write(Path(os.environ['KERNELINFRA_RESULT']), {'schema': 'kernelinfra.stage-result.v1',
        'status': 'passed' if validity == 'valid' else 'failed', 'validity': validity,
        'summary': report.get('error', report['status']), 'artifacts': {'comparison': 'comparison-report.json'}})
    return 1 if validity == 'unknown' else 0


if __name__ == '__main__':
    raise SystemExit(main())
