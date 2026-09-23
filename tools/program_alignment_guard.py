"""Program-specific observations for the existing three-phase alignment checker.

Only the guard tool instruments the existing Program allocator, to realize a named
private-pointer counterexample. Program binding, launch, guards and teardown stay
with LoadedProgram/LoadedAlignmentCandidate; no second execution graph is created.
"""
from contextlib import contextmanager
import json
import math
import shutil
import sys

from tools.check_alignment_candidate import admit_cases, case_data, offset_arguments
from tools.compare_flashinfer_reference import write
from tools.compare_rewrite_artifacts import regular
from tools.staged_rewrite_comparison import (
    WIDTHS, INPUT_REFERENCES, SnapshotWriter, SnapshotReader, check_snapshot,
)
from open_cake_ir.evaluation.core import LoadedTorchTensorCandidate
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.evaluation.program import program_components


def guard_plan(candidate, workload):
    manifest,_,leaves = program_components(candidate)
    if not manifest.aligned_stages:
        raise ValueError('Program guard requires at least one guarded stage')
    for case in workload.case_ids:manifest.check_validation_case(workload,case)
    program = manifest.program
    # Ask each admitted leaf which of its local bindings owns a restriction.
    from open_cake_ir.evaluation.kernel_bundle import alignment_component
    _,children,_ = program_components(candidate)
    restricted = set()
    for stage in program.stages:
        if stage.name not in manifest.aligned_stages:continue
        _,leaf = alignment_component(children[stage.name],leaves[stage.name])
        if set(leaf.pointer_alignments)!={name for name,_,_,_ in leaf.tensor_abi}:
            raise ValueError('Program guard probe requires all stage tensor pointers to be guarded')
        for local,alignment in leaf.pointer_alignments.items():
            if alignment != 16:raise ValueError('this guard probe covers16-byte stage alignment')
            restricted.add(stage.bindings[local].tensor)
    if any(program.tensors[name].dtype.value not in WIDTHS for name in restricted):
        raise ValueError('Program guard tensor dtype is outside the snapshot domain')
    variants = [(None,0)] + [(name,offset) for name in program.tensors if name in restricted
                            for offset in (2,4,8) if offset % WIDTHS[program.tensors[name].dtype.value] == 0]
    return [{'case':case,'case_index':i,'offset_tensor':name,'offset_bytes':offset}
            for i,case in enumerate(workload.case_ids) for name,offset in variants]


def storage_budget(candidate, workload):
    plan = guard_plan(candidate,workload);seen=set();compact=raw=largest=0
    for row in plan:
        size=0
        for arg in workload.tensor_abi(row['case']):
            extent=math.prod(arg.shape)*WIDTHS[arg.dtype];size+=extent
            if arg.mode=='input':
                compact+=1
                if (row['case'],arg.name) not in seen:
                    compact+=extent;seen.add((row['case'],arg.name))
            else:compact+=extent
        raw+=size;largest=max(largest,size)
    return {'encoding':INPUT_REFERENCES,'checks':len(plan),'raw_bytes':raw,
            'unchanged_input_bytes':compact,'byte_limit':compact+largest}


@contextmanager
def offset_private_allocator(loaded, program, changed, offset, torch):
    """Use the Program's one allocation callback; check its declared call order.

    This is local test instrumentation, not a production allocation mode. Every
    allocation remains checked by Program.prepare_arguments before any launch.
    """
    original = loaded._allocate
    private = [(name,spec) for name,spec in program.tensors.items()
               if name not in program.inputs + program.outputs]
    pending = iter(private)
    consumed = []
    def allocate(spec):
        name,expected = next(pending)
        if spec != expected:raise ValueError('Program allocator order differs')
        value = original(spec)
        consumed.append(name)
        return offset_arguments(torch,[value],0,offset)[0] if name==changed else value
    loaded._allocate = allocate
    try:
        yield
        if consumed != [name for name,_ in private]:
            raise ValueError('Program did not allocate every declared private tensor')
    finally:
        loaded._allocate = original


def check_stage_guards(loaded, manifest, tensors, changed, stream):
    selections={};rejections=[]
    for stage in manifest.program.stages:
        item=loaded._children[stage.name]
        if stage.name not in manifest.aligned_stages:
            selections[stage.name]='generic';continue
        leaf=item.aligned_manifest
        args=[]
        for local,shape,_,_ in leaf.tensor_abi:
            binding=stage.bindings[local];value=tensors[binding.tensor]
            args.append(value.view(shape) if binding.singleton_view else value)
        pointers={name:t.data_ptr() for (name,_,_,_),t in zip(leaf.tensor_abi,args,strict=True)}
        aligned=all(pointers[name]%alignment==0 for name,alignment in leaf.pointer_alignments.items())
        affected=changed in {stage.bindings[name].tensor for name in leaf.pointer_alignments}
        if aligned == affected:
            raise ValueError('allocator did not realize the declared Program offset counterexample')
        selections[stage.name]='aligned' if aligned else 'generic'
        if affected:
            before=item.aligned.launch_calls
            try:
                item.aligned.launch(args,tensor_contract=leaf,stream=stream)
            except ValueError as error:
                if 'alignment contract' not in str(error):raise
            else:raise AssertionError('restricted Program leaf accepted an unaligned pointer')
            if item.aligned.launch_calls!=before:
                raise AssertionError('restricted Program leaf dispatched before refusal')
            rejections.append(stage.name)
    return selections,rejections


def expected_stage_checks(manifest, changed):
    selections={};rejections=[]
    for stage in manifest.program.stages:
        guarded=stage.name in manifest.aligned_stages
        # The checked builder specializes every tensor argument for its stages.
        affected=changed in {binding.tensor for binding in stage.bindings.values()}
        selections[stage.name]='generic' if not guarded or affected else 'aligned'
        if guarded and affected:rejections.append(stage.name)
    return selections,rejections


def capture(candidate, workload, prepared, output, admission, commit):
    import torch
    admit_cases(prepared,workload)
    manifest,_,_=program_components(candidate)
    plan=guard_plan(candidate,workload);budget=storage_budget(candidate,workload)
    if shutil.disk_usage(output).free<budget['byte_limit']:
        raise OSError('insufficient storage for every Program guard observation')
    report={'kind':'program_alignment_capture','judge_commit':commit,'candidate':candidate_identity(candidate),
            'workload_sha256':workload.canonical_sha256,'byteorder':sys.byteorder,
            'snapshot_encoding':INPUT_REFERENCES,'checks':[],'capture_complete':False,
            'performance_measured':False,'job_id':admission.broker_job_id,'storage_budget':budget}
    public=manifest.program.inputs+manifest.program.outputs
    with (output/'snapshots.bin').open('xb') as stream:
        writer=SnapshotWriter(stream,workload,budget['byte_limit'])
        for case_index,case in enumerate(workload.case_ids):
            values=case_data(prepared,workload,case_index,'input')
            loaded=LoadedTorchTensorCandidate(candidate,manifest,values,admission)
            try:
                for row in (r for r in plan if r['case']==case):
                    changed,offset=row['offset_tensor'],row['offset_bytes']
                    index=public.index(changed) if changed in public else None
                    arguments=offset_arguments(torch,loaded.arguments,index,offset)
                    with offset_private_allocator(loaded.loaded,manifest.program,changed,offset,torch):
                        tensors=loaded.loaded.prepare_arguments(arguments)
                    try:
                        selections,rejections=check_stage_guards(loaded.loaded,manifest,tensors,changed,
                                                                torch.cuda.current_stream().cuda_stream)
                        before=loaded.loaded.launch_calls
                        loaded.launch(arguments)
                        torch.cuda.synchronize()
                        if loaded.loaded.launch_calls-before!=manifest.kernels_per_call:
                            raise AssertionError('Program guard did not execute every stage exactly once')
                        actual={stage.name:loaded.loaded._children[stage.name].last_variant
                                if stage.name in manifest.aligned_stages else 'generic' for stage in manifest.program.stages}
                        if actual!=selections:raise AssertionError('Program stage selected the wrong implementation')
                        writer.append(arguments,case)
                        report['checks'].append({**row,'selected':actual,'rejected_leaves':rejections,
                                                 'kernel_calls':manifest.kernels_per_call})
                    finally:
                        loaded.release_argument_sets([arguments])
            finally:loaded.close()
        report.update(snapshot_count=writer.count,snapshot_bytes=writer.encoder.bytes_written,capture_complete=True)
    write(output/'capture.json',report)
    return report


def verify(candidate, workload, prepared, captured, commit):
    admit_cases(prepared,workload);manifest,_,_=program_components(candidate)
    report=json.loads(regular(captured,'capture.json').read_text());plan=guard_plan(candidate,workload)
    if (report.get('kind')!='program_alignment_capture' or report.get('capture_complete') is not True
        or report.get('judge_commit')!=commit
        or report.get('candidate')!=candidate_identity(candidate)
        or report.get('workload_sha256')!=workload.canonical_sha256 or report.get('byteorder')!=sys.byteorder
        or report.get('snapshot_encoding')!=INPUT_REFERENCES
        or report.get('storage_budget')!=storage_budget(candidate,workload)
        or report.get('snapshot_count')!=len(plan)
        or [{key:r.get(key) for key in plan[0]} for r in report['checks']]!=plan):
        raise ValueError('Program guard capture does not cover the bound candidate and complete plan')
    passed=True;first_verdicts={};counts={'literal':0,'reference':0}
    cases={c:case_data(prepared,workload,i,'input') for i,c in enumerate(workload.case_ids)}
    expected={c:case_data(prepared,workload,i,'output') for i,c in enumerate(workload.case_ids)}
    with regular(captured,'snapshots.bin').open('rb') as stream:
        reader=SnapshotReader(stream,workload,INPUT_REFERENCES)
        for row in report['checks']:
            selected,rejected=expected_stage_checks(manifest,row['offset_tensor'])
            if (row.get('selected')!=selected or row.get('rejected_leaves')!=rejected
                or row.get('kernel_calls')!=manifest.kernels_per_call):
                raise ValueError('Program guard selection, leaf refusal or stage count differs')
            correct,metrics=check_snapshot(reader,workload,row['case'],cases[row['case']],expected[row['case']],first_verdicts,counts)
            row.update(passed=correct,metrics=metrics);passed &= correct
        if stream.read(1):raise ValueError('Program guard snapshots contain trailing data')
    report.update(passed=passed,verification_phase='CPU after Program GPU worker exit',input_check_counts=counts)
    return report
