#!/usr/bin/env python3
"""GPU replay of every pointer guard; no timing or performance claim."""
from __future__ import annotations

from array import array
import ctypes
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from tools.compare_flashinfer_reference import admit_judge_source, write
from tools.compare_rewrite_artifacts import regular
from open_cake_ir.evaluation.admission import observe_exclusive_cuda
from open_cake_ir.evaluation.core import LoadedTorchTensorCandidate, compare_tile_outputs
from open_cake_ir.evaluation.paired import validate_pair_candidates
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.bindings import load_baseline_bundle
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.workloads import materialize_case, reference_outputs


def case_data(root, workload, case_index, mode):
    """Read native doubles from the preceding trusted CPU stage, never a pickle."""
    values = {}
    for index, arg in enumerate(workload.tensor_abi(workload.case_ids[case_index])):
        if arg.mode != mode:
            continue
        path = root / f'{case_index}-{index}.f64'
        if path.is_symlink():
            raise ValueError('case data must be a regular stage artifact')
        item = array('d')
        if path.stat().st_size != math.prod(arg.shape) * item.itemsize:
            raise ValueError('prepared case extent differs from the Workload ABI')
        item.frombytes(path.read_bytes())
        if len(item) != math.prod(arg.shape):
            raise ValueError('prepared case extent differs from the Workload ABI')
        values[arg.name] = item
    return values


def prepare_cases(workload, output):
    for index, case in enumerate(workload.case_ids):
        values = materialize_case(workload,case)
        expected = reference_outputs(workload,case,values)
        for number, arg in enumerate(workload.tensor_abi(case)):
            result = values[arg.name] if arg.mode == 'input' else expected[arg.name]
            with (output / f'{index}-{number}.f64').open('xb') as stream:
                array('d',result).tofile(stream)
    write(output / 'cases.json',{'workload_sha256':workload.canonical_sha256,
                               'case_ids':list(workload.case_ids),'byteorder':sys.byteorder})


def admit_cases(root, workload):
    metadata = json.loads((root / 'cases.json').read_text())
    if metadata != {'workload_sha256':workload.canonical_sha256,
                    'case_ids':list(workload.case_ids),'byteorder':sys.byteorder}:
        raise ValueError('prepared cases differ from the Workload or host representation')


def verify_capture(workload, prepared, captured):
    admit_cases(prepared,workload)
    report = json.loads((captured / 'capture.json').read_text())
    if (report.get('workload_sha256') != workload.canonical_sha256
            or report.get('capture_complete') is not True):
        raise ValueError('captured Workload differs')
    expected_rows = [(i,index,offset) for i,_ in enumerate(workload.case_ids)
                     for index,offset in [(None,0)] + [(n,b) for n in range(len(workload.tensor_abi(workload.case_ids[i])))
                                                      for b in (2,4,8)]]
    observed_rows = [(r['case_index'],r['offset_index'],r['offset_bytes']) for r in report['checks']]
    if observed_rows != expected_rows:
        raise ValueError('capture does not cover every required case and pointer offset')
    passed = True
    for number, row in enumerate(report['checks']):
        index = row['case_index']
        if (row['case'] != workload.case_ids[index]
                or row['selected'] != ('aligned' if row['offset_bytes'] == 0 else 'generic')):
            raise ValueError('captured guard selection or case differs')
        before = case_data(prepared,workload,index,'input')
        expected = case_data(prepared,workload,index,'output')
        snapshot = captured / str(number)
        after = case_data(snapshot,workload,index,'input')
        observed = {k:list(v) for k,v in case_data(snapshot,workload,index,'output').items()}
        correct, metrics = compare_tile_outputs(workload,before,expected,observed,after)
        row.update(passed=correct,metrics=metrics)
        passed &= correct
    report.update(kind='guarded_alignment_correctness',passed=passed,
                  verification_phase='CPU after GPU worker exit',performance_measured=False)
    return report


def phase_contract(commit, task, *, gpu_stage='guard'):
    if [row['id'] for row in task['stages']] != ['prepare',gpu_stage,'verify']:
        raise ValueError('replay requires ordered prepare, device and verify stages')
    stages = {row['id']:row for row in task['stages']}
    if set(stages) != {'prepare',gpu_stage,'verify'}:
        raise ValueError('guard replay requires separate prepare, guard and verify stages')
    for phase, execution in [('prepare','local'),(gpu_stage,'broker'),('verify','local')]:
        admit_judge_source(commit,task,phase)
        if stages[phase]['execution'] != execution:
            raise ValueError('guard replay resource phase differs')


def offset_arguments(torch, arguments, index, offset_bytes):
    result = []
    for number, original in enumerate(arguments):
        offset = offset_bytes if number == index else 0
        if offset % original.element_size():
            raise ValueError('guard probe offset must preserve element alignment')
        elements = offset // original.element_size()
        storage = torch.empty(original.numel() + elements, dtype=original.dtype,
                              device=original.device)
        view = storage[elements:].reshape(original.shape)
        view.copy_(original)
        result.append(view)
    return result


def main():
    root = Path(os.environ['KERNELINFRA_CANDIDATE_DIR'])
    stage = Path(os.environ['KERNELINFRA_STAGE_DIR'])
    phase = os.environ['KERNELINFRA_STAGE_ID']
    if phase not in {'prepare','guard','verify'}:
        raise ValueError('unknown guard replay phase')
    if phase != 'guard':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    run = Path(os.environ['KERNELINFRA_RUN_DIR'])
    report = {'kind':'guarded_alignment_capture', 'checks':[], 'passed':False,
              'performance_measured':False}
    validity = 'unknown'
    loaded = None
    try:
        commit = checkout_commit(ROOT)
        phase_contract(commit,json.loads(Path(os.environ['KERNELINFRA_TASK']).read_text()))
        report['judge_commit'] = commit
        workload = WorkloadContract(json.loads(regular(root,'workload.json').read_text()))
        if phase == 'prepare':
            prepare_cases(workload,stage)
            write(Path(os.environ['KERNELINFRA_RESULT']),{'schema':'kernelinfra.stage-result.v1',
                'status':'passed','validity':'valid','summary':'CPU inputs and oracle prepared without a GPU lease',
                'artifacts':{'cases':'cases.json'}})
            return 0
        prepared = run / 'stages/prepare'
        admit_cases(prepared,workload)
        candidate = load_baseline_bundle(ROOT,regular(root,'optimized/candidate.json'))
        baseline = load_baseline_bundle(ROOT,regular(root,'starter/candidate.json'))
        manifest = validate_pair_candidates(candidate,baseline,workload,'primary')['candidate']
        if phase == 'verify':
            if candidate.is_program:
                from tools.program_alignment_guard import verify
                report = verify(candidate,workload,prepared,run / 'stages/guard',commit)
            else:
                report = verify_capture(workload,prepared,run / 'stages/guard')
            write(stage / 'guard-report.json',report)
            write(Path(os.environ['KERNELINFRA_RESULT']),{'schema':'kernelinfra.stage-result.v1',
                'status':'passed' if report['passed'] else 'failed',
                'validity':'valid' if report['passed'] else 'invalid',
                'summary':f"{sum(r['passed'] for r in report['checks'])}/{len(report['checks'])} guard checks passed after GPU release",
                'artifacts':{'guard':'guard-report.json'}})
            return 0  # A handled numerical failure is recorded by status/validity.
        if candidate.is_program:
            from tools.program_alignment_guard import capture
            admission = observe_exclusive_cuda(workload.target)
            report = capture(candidate,workload,prepared,stage,admission,commit)
            write(Path(os.environ['KERNELINFRA_RESULT']),{'schema':'kernelinfra.stage-result.v1',
                'status':'passed','validity':'valid','summary':'Complete Program guards captured; CPU verification pending',
                'artifacts':{'capture':'capture.json','snapshots':'snapshots.bin'}})
            return 0
        if not manifest.aligned_variant:
            raise ValueError('guard replay requires a sealed alignment bundle')
        admission = observe_exclusive_cuda(workload.target)
        report['job_id'] = admission.broker_job_id
        report['workload_sha256'] = workload.canonical_sha256
        import torch
        torch.set_num_threads(4)
        report['runtime'] = {'torch':torch.__version__,'cuda':torch.version.cuda}
        for case_index, case in enumerate(workload.case_ids):
            manifest.check_validation_case(workload,case)
            values = case_data(prepared,workload,case_index,'input')
            loaded = LoadedTorchTensorCandidate(candidate,manifest,values,admission)
            variants = [(None,0)] + [(index,offset) for index in range(len(manifest.tensor_abi))
                                   for offset in (2,4,8)]
            for index, offset in variants:
                arguments = offset_arguments(torch,loaded.arguments,index,offset)
                aligns = loaded.loaded.aligned_manifest.pointer_alignments
                pointers = {name:value.data_ptr() for (name,_,_,_),value
                            in zip(manifest.tensor_abi,arguments,strict=True)}
                aligned = all(pointers[name] % alignment == 0 for name,alignment in aligns.items())
                if aligned != (offset == 0):
                    raise ValueError('allocator did not realize the requested guard counterexample')
                leaf_before = loaded.loaded.aligned.launch_calls
                if offset:
                    try:
                        loaded.loaded.aligned.launch(arguments,
                            tensor_contract=loaded.loaded.aligned_manifest,
                            stream=torch.cuda.current_stream().cuda_stream)
                    except ValueError as error:
                        if 'alignment contract' not in str(error):
                            raise
                    else:
                        raise AssertionError('extracted aligned leaf accepted an unaligned pointer')
                    if loaded.loaded.aligned.launch_calls != leaf_before:
                        raise AssertionError('restricted leaf launched before rejecting alignment')
                loaded.launch(arguments)
                selected = loaded.loaded.last_variant
                if selected != ('aligned' if aligned else 'generic'):
                    raise AssertionError('runtime guard selected the wrong implementation')
                snapshot = stage / str(len(report['checks']))
                snapshot.mkdir()
                for number, value in enumerate(arguments):
                    host = value.detach().to(device='cpu',dtype=torch.float64).contiguous()
                    with (snapshot / f'{case_index}-{number}.f64').open('xb') as stream:
                        stream.write(ctypes.string_at(host.data_ptr(),host.numel()*host.element_size()))
                report['checks'].append({'case':case,'case_index':case_index,'offset_index':index,
                    'offset_argument':None if index is None else manifest.tensor_abi[index][0],
                    'offset_bytes':offset,'selected':selected})
            report.setdefault('case_resources',{})[case] = loaded.loaded.resources
            loaded.close()
            loaded = None
        report['capture_complete'] = True
        validity = 'valid'
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        validity = 'invalid' if isinstance(error,(ArithmeticError,AssertionError)) else 'unknown'
    finally:
        if loaded is not None:
            loaded.close()
    write(stage / 'capture.json',report)
    write(Path(os.environ['KERNELINFRA_RESULT']),{'schema':'kernelinfra.stage-result.v1',
        'status':'passed' if report.get('capture_complete') else 'failed','validity':validity,
        'summary':report.get('error',f"{len(report['checks'])} host snapshots retained; CPU verification pending"),
        'artifacts':{'capture':'capture.json'}})
    return 1 if validity == 'unknown' else 0


if __name__ == '__main__':
    raise SystemExit(main())
