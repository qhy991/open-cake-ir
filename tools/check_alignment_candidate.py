#!/usr/bin/env python3
"""GPU replay of every pointer guard; no timing or performance claim."""
from __future__ import annotations

from array import array
import json
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
    report = {'kind':'guarded_alignment_correctness', 'checks':[], 'passed':False,
              'performance_measured':False}
    validity = 'unknown'
    loaded = None
    try:
        commit = checkout_commit(ROOT)
        admit_judge_source(commit, json.loads(Path(os.environ['KERNELINFRA_TASK']).read_text()),
                           os.environ['KERNELINFRA_STAGE_ID'])
        report['judge_commit'] = commit
        workload = WorkloadContract(json.loads(regular(root,'workload.json').read_text()))
        candidate = load_baseline_bundle(ROOT,regular(root,'optimized/candidate.json'))
        baseline = load_baseline_bundle(ROOT,regular(root,'starter/candidate.json'))
        manifest = validate_pair_candidates(candidate,baseline,workload,'primary')['candidate']
        if not manifest.aligned_variant:
            raise ValueError('guard replay requires a sealed alignment bundle')
        admission = observe_exclusive_cuda(workload.target)
        report['job_id'] = admission.broker_job_id
        import torch
        torch.set_num_threads(4)
        report['runtime'] = {'torch':torch.__version__,'cuda':torch.version.cuda}
        for case in workload.case_ids:
            manifest.check_validation_case(workload,case)
            values = materialize_case(workload,case)
            expected = reference_outputs(workload,case,values)
            before = {k:array('d',v) for k,v in values.items()}
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
                observed, after = loaded.snapshot(arguments)
                passed, metrics = compare_tile_outputs(workload,before,expected,observed,after)
                selected = loaded.loaded.last_variant
                if selected != ('aligned' if aligned else 'generic'):
                    raise AssertionError('runtime guard selected the wrong implementation')
                report['checks'].append({'case':case,'offset_argument':None if index is None else manifest.tensor_abi[index][0],
                    'offset_bytes':offset,'selected':selected,'passed':passed,'metrics':metrics})
                if not passed:
                    raise ArithmeticError('guarded candidate failed the unchanged Workload oracle')
            report.setdefault('case_resources',{})[case] = loaded.loaded.resources
            loaded.close()
            loaded = None
        report['passed'] = True
        validity = 'valid'
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        validity = 'invalid' if isinstance(error,(ArithmeticError,AssertionError)) else 'unknown'
    finally:
        if loaded is not None:
            loaded.close()
    write(stage / 'guard-report.json',report)
    write(Path(os.environ['KERNELINFRA_RESULT']),{'schema':'kernelinfra.stage-result.v1',
        'status':'passed' if report['passed'] else 'failed','validity':validity,
        'summary':report.get('error',f"{len(report['checks'])} complete-output guard checks passed; no timing"),
        'artifacts':{'guard':'guard-report.json'}})
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
