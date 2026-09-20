#!/usr/bin/env python3
"""CPU preparation, one continuous paired GPU capture, then bulk CPU verification."""
from __future__ import annotations

from array import array
import ctypes
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import struct
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'src')]

from tools.check_alignment_candidate import prepare_cases, admit_cases, case_data, phase_contract
from tools.compare_flashinfer_reference import write
from tools.compare_rewrite_artifacts import regular, reference_spec, reference_arguments, load_reference, RetainedExternal
from open_cake_ir.evaluation.admission import observe_exclusive_cuda
from open_cake_ir.evaluation.benchmark import StrictCuptiBenchmark
from open_cake_ir.evaluation.core import LoadedTorchTensorCandidate, compare_tile_outputs
from open_cake_ir.evaluation.paired import candidate_identity, paired_protocol, paired_summary, validate_pair_candidates
from open_cake_ir.evaluation.timing import summarize_cohort
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.bindings import load_baseline_bundle
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.normalization.study import evaluation_policy

EDGES = (('optimized','external'),('starter','external'),('optimized','starter'))
ROLES = ('external','starter','optimized')
WIDTHS = {'bf16':2,'fp16':2,'fp32':4,'int32':4}
TORCH_DTYPES = {'bf16':'torch.bfloat16','fp16':'torch.float16','fp32':'torch.float32','int32':'torch.int32'}


def observation_plan(workload, protocol):
    result = []
    def cases(phase):
        for case in workload.case_ids:
            for role in ROLES:
                result.append(dict(phase=phase,case=case,role=role,count=1))
    cases('preflight')
    for left,right in EDGES:
        for index,order in enumerate(protocol.pair_order):
            for position,arm in enumerate(order):
                result.append(dict(phase='timed',case='primary',role=left if arm=='candidate' else right,
                    count=protocol.route_calls_per_cohort,edge=left+'_vs_'+right,
                    pair_index=index,position=position,arm=arm))
    cases('postflight')
    return result


def decode_values(payload, dtype):
    """Lossless native tensor bytes become the existing host comparison values."""
    if dtype == 'bf16':
        if len(payload) % 2:
            raise ValueError('BF16 snapshot byte extent differs')
        expanded = bytearray(len(payload)*2)
        if sys.byteorder == 'little':
            expanded[2::4],expanded[3::4] = payload[0::2],payload[1::2]
        else:
            expanded[0::4],expanded[1::4] = payload[0::2],payload[1::2]
        values = array('f');values.frombytes(expanded)
        return array('d',values)
    if dtype == 'fp16':
        return array('d',(row[0] for row in struct.iter_unpack('=e',payload)))
    values = array('f' if dtype == 'fp32' else 'i')
    values.frombytes(payload)
    return array('d',values)


class SnapshotWriter:
    def __init__(self, stream, workload):
        self.stream,self.workload = stream,workload
        self.count = 0

    def append(self, arguments, case):
        abi = self.workload.tensor_abi(case)
        if isinstance(arguments,dict):
            tensors = [arguments['inputs'][arg.name] if arg.mode=='input' else arguments['result'] for arg in abi]
        else:
            tensors = arguments
        for arg,value in zip(abi,tensors,strict=True):
            if (tuple(value.shape) != arg.shape or str(value.dtype) != TORCH_DTYPES[arg.dtype]
                    or not value.is_contiguous()):
                raise ValueError('captured tensor ABI differs')
            host = value.detach().to(device='cpu').contiguous()
            size = math.prod(arg.shape)*WIDTHS[arg.dtype]
            if host.numel()*host.element_size() != size:
                raise ValueError('captured tensor byte extent differs')
            self.stream.write(ctypes.string_at(host.data_ptr(),size))
        self.count += 1


def read_snapshot(stream, workload, case):
    before,observed = {},{}
    for arg in workload.tensor_abi(case):
        size = math.prod(arg.shape)*WIDTHS[arg.dtype]
        payload = stream.read(size)
        if len(payload) != size:
            raise ValueError('snapshot stream is truncated')
        values = decode_values(payload,arg.dtype)
        if arg.mode == 'input':before[arg.name] = values
        else:observed[arg.name] = list(values)
    return observed,before


def load_participants(root,workload):
    candidates = {role:load_baseline_bundle(ROOT,regular(root,role+'/candidate.json'))
                  for role in ('optimized','starter')}
    parsed = validate_pair_candidates(candidates['optimized'],candidates['starter'],workload,'primary')
    manifests = {'optimized':parsed['candidate'],'starter':parsed['baseline']}
    for manifest in manifests.values():
        for case in workload.case_ids:manifest.check_validation_case(workload,case)
    return candidates,manifests


def prepare(root,output,workload):
    import torch
    if torch.cuda.is_initialized():raise ValueError('CPU preparation acquired a CUDA context')
    load_participants(root,workload)
    prepare_cases(workload,output)
    spec,_,_,_ = reference_spec(root)
    metadata = {'torch':torch.__version__,'native_library':None,'reference':spec}
    if spec['kind'] == 'cuda_cpp':
        report = {'roles':{}}
        load_reference(root,output,report,1)
        libraries = list((output/'external-build').glob('*.so'))
        if len(libraries) != 1:raise ValueError('one prepared native reference library is required')
        # One executable handoff: refuse a changed library before importing it
        # in the later GPU process. No source-file digest catalogue is added.
        metadata['native_library'] = {'path':str(libraries[0].relative_to(output)),
                                      'sha256':sha256(libraries[0].read_bytes()).hexdigest()}
    if torch.cuda.is_initialized():raise ValueError('reference preparation initialized CUDA')
    write(output/'preparation.json',metadata)
    return {'preparation':'preparation.json','cases':'cases.json'}


def capture(root,output,prepared,workload,commit):
    from open_cake_ir.tasks.evaluate import capture_tile_cohort
    import torch
    import flashinfer.testing as timer
    torch.set_num_threads(4)
    admit_cases(prepared,workload)
    preparation = json.loads(regular(prepared,'preparation.json').read_text())
    if preparation['torch'] != torch.__version__:raise ValueError('prepared native runtime differs')
    policy = evaluation_policy(workload);protocol = paired_protocol(policy)
    candidates,manifests = load_participants(root,workload)
    report = {'scope':'staged_sealed_three_way_comparison','judge_commit':commit,
        'input':json.loads(regular(root,'comparison.json').read_text()),
        'workload_sha256':workload.canonical_sha256,'byteorder':sys.byteorder,
        'participants':{role:candidate_identity(value) for role,value in candidates.items()},
        'evaluation_protocol':policy,'groups':[],'capture_complete':False,
        'roles':{'optimized':'sealed confirmed Cake candidate','starter':'sealed original Cake starter',
                 'external':'unchanged supplied external implementation'}}
    if report['input'].get('kind') == 'explicit_alignment_ablation':
        report['roles'].update(optimized='same-source candidate with guarded AOT alignment variants',
                               starter='unchanged pre-specialization optimized binary')
    admission = observe_exclusive_cuda('sm_103a')
    report['device'] = {'name':admission.device_name,'job_id':admission.broker_job_id,'gpu_uuid':admission.gpu_uuid}
    library = prepared_library(prepared,preparation)
    references = load_reference(root,output,report,protocol.route_calls_per_cohort,prebuilt_library=library)
    if report['reference'] != preparation['reference']:raise ValueError('prepared reference specification differs')
    names = reference_arguments(workload)
    launches = [lambda values,out,fn=fn:fn(*(values[name] for name in names)) for fn in references]
    cases = {case:case_data(prepared,workload,index,'input') for index,case in enumerate(workload.case_ids)}
    expected = {case:case_data(prepared,workload,index,'output') for index,case in enumerate(workload.case_ids)}
    loaded = {}
    try:
        for case,values in cases.items():
            for role in ('optimized','starter'):
                loaded[role,case] = LoadedTorchTensorCandidate(candidates[role],manifests[role],values,admission)
            loaded['external',case] = RetainedExternal(workload,values,launches,torch,
                cached_output=report['reference'].get('cached_output',False))
        assay = StrictCuptiBenchmark(timer)
        with (output/'snapshots.bin').open('xb') as stream:
            writer = SnapshotWriter(stream,workload)
            for observation in observation_plan(workload,protocol):
                role,case = observation['role'],observation['case']
                obj = loaded[role,case]
                group = {'observation':observation}
                if observation['phase'] == 'timed':
                    samples,arguments = capture_tile_cohort(obj,assay,
                        samples_per_cohort=protocol.samples_per_cohort,
                        route_calls_per_cohort=protocol.route_calls_per_cohort)
                    group['samples_ms'] = samples
                else:
                    arguments = obj.fresh_argument_sets(1)
                    obj.launch(arguments[0]);torch.cuda.synchronize()
                    if observation['phase'] == 'preflight':
                        # Preserve correctness-before-timing in this same allocation.
                        # The much larger timed/postflight validation is deferred.
                        observed,after = obj.snapshot(arguments[0])
                        correct,metrics = compare_tile_outputs(workload,cases[case],expected[case],observed,after)
                        if not correct:raise ArithmeticError(f'{role} failed preflight {case}: {metrics}')
                for arguments_ in arguments:writer.append(arguments_,case)
                report['groups'].append(group)
            report['snapshot_count'] = writer.count
        report['compiled_resources'] = {role:loaded[role,'primary'].loaded.resources for role in ('optimized','starter')}
        report['capture_complete'] = True
        write(output/'capture.json',report)
    finally:
        for (role,_),obj in loaded.items():
            if role != 'external':obj.close()
    return {'capture':'capture.json','snapshots':'snapshots.bin'}


def prepared_library(root,metadata):
    record = metadata['native_library']
    if record is None:return None
    if not isinstance(record,dict) or set(record) != {'path','sha256'}:
        raise ValueError('prepared native handoff fields differ')
    path = regular(root,record['path'])
    if sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('prepared native library changed before GPU loading')
    return path


def verify(root,output,prepared,captured,workload):
    admit_cases(prepared,workload)
    raw = json.loads(regular(captured,'capture.json').read_text())
    policy = evaluation_policy(workload);protocol = paired_protocol(policy)
    candidates,_ = load_participants(root,workload)
    if (raw.get('capture_complete') is not True or raw.get('workload_sha256') != workload.canonical_sha256
            or raw.get('byteorder') != sys.byteorder or raw.get('evaluation_protocol') != policy
            or raw.get('participants') != {role:candidate_identity(value) for role,value in candidates.items()}):
        raise ValueError('capture authority differs')
    plan = observation_plan(workload,protocol)
    if ([row['observation'] for row in raw['groups']] != plan
            or raw.get('snapshot_count') != sum(row['count'] for row in plan)):
        raise ValueError('capture must retain every prescribed call in order')
    cases = {case:case_data(prepared,workload,index,'input') for index,case in enumerate(workload.case_ids)}
    expected = {case:case_data(prepared,workload,index,'output') for index,case in enumerate(workload.case_ids)}
    report = {key:value for key,value in raw.items() if key not in ('groups','capture_complete')}
    report.update(cases=[],comparisons={},provider_calls=0,promotion_disposition='No promotion',
                  verification_phase='bulk CPU verification after GPU worker exit')
    for left,right in EDGES:
        report['comparisons'][left+'_vs_'+right] = {'evaluation_protocol':policy,
            'roles':{'candidate':left,'baseline':right},'measurements':[{'pair_index':i,'order':list(order),'arms':{}}
                for i,order in enumerate(protocol.pair_order)]}
    all_correct = True
    with regular(captured,'snapshots.bin').open('rb') as stream:
        for row in raw['groups']:
            observation = row['observation'];case = observation['case']
            check = {'checked_launches':observation['count'],'passed':True,'output_mismatches':0,
                     'max_abs_error':0.0,'inputs_unchanged':True}
            for _ in range(observation['count']):
                observed,after = read_snapshot(stream,workload,case)
                correct,metrics = compare_tile_outputs(workload,cases[case],expected[case],observed,after)
                check['passed'] &= correct
                check['output_mismatches'] += metrics['output_mismatches']
                check['max_abs_error'] = max(check['max_abs_error'],metrics['max_abs_error'])
                check['inputs_unchanged'] &= metrics['inputs_unchanged']
            all_correct &= check['passed']
            if observation['phase'] == 'timed':
                pair = report['comparisons'][observation['edge']]['measurements'][observation['pair_index']]
                pair['arms'][observation['arm']] = {'position':observation['position'],'samples_ms':row['samples_ms'],
                    'summary':summarize_cohort(row['samples_ms']),'route_calls':check['checked_launches'],'output_check':check}
            else:
                report['cases'].append({**observation,**check})
        if stream.read(1):raise ValueError('snapshot stream contains unbound trailing data')
    report['correctness_passed'] = all_correct
    # Failed numerical data never receives an accepted timing summary.
    if all_correct:
        for comparison in report['comparisons'].values():comparison['timing'] = paired_summary(comparison)
        report['measurement_quality_passed'] = all(row['timing']['measurement_quality_passed'] for row in report['comparisons'].values())
        report['status'] = 'passed' if report['measurement_quality_passed'] else 'measurement_quality_failed'
    else:
        report.update(status='correctness_failed',measurement_quality_passed=False)
    report['timed_interval'] = ('Unchanged CUPTI device-activity span, cold L2 per sample, no timer fallback; '
        'all three edges in one exclusive allocation. CPU preparation precedes it; required preflight numerical gates stay before timing. '
        'Every timed/postflight tensor is retained in its original dtype for complete CPU verification after device worker exit.')
    write(output/'comparison-report.json',report)
    return report


def main():
    phase = os.environ['KERNELINFRA_STAGE_ID']
    if phase not in {'prepare','capture','verify'}:raise ValueError('unknown comparison phase')
    if phase != 'capture':os.environ['CUDA_VISIBLE_DEVICES'] = ''
    output = Path(os.environ['KERNELINFRA_STAGE_DIR']);run = Path(os.environ['KERNELINFRA_RUN_DIR'])
    root = Path(os.environ['KERNELINFRA_CANDIDATE_DIR'])
    result = {'schema':'kernelinfra.stage-result.v1','status':'failed','validity':'unknown','artifacts':{}}
    try:
        commit = checkout_commit(ROOT)
        phase_contract(commit,json.loads(Path(os.environ['KERNELINFRA_TASK']).read_text()),gpu_stage='capture')
        workload = WorkloadContract(json.loads(regular(root,'workload.json').read_text()))
        if workload.target != 'sm_103a':raise ValueError('comparison binds exact B300 target')
        reference_arguments(workload)
        if phase == 'prepare':
            artifacts = prepare(root,output,workload)
            summary = 'CPU inputs, oracle and native reference prepared; no GPU lease'
        elif phase == 'capture':
            artifacts = capture(root,output,run/'stages/prepare',workload,commit)
            summary = 'Paired device capture complete; bulk numerical verification remains pending'
        else:
            report = verify(root,output,run/'stages/prepare',run/'stages/capture',workload)
            artifacts = {'comparison':'comparison-report.json'};summary = report['status']
            if not report['correctness_passed']:
                result.update(status='failed',validity='invalid',summary=summary,artifacts=artifacts)
                write(Path(os.environ['KERNELINFRA_RESULT']),result)
                return 0
        result.update(status='passed',validity='valid',summary=summary,artifacts=artifacts)
    except Exception as error:
        result.update(summary=f'{type(error).__name__}: {error}',
                      validity='invalid' if isinstance(error,ArithmeticError) else 'unknown')
        write(output/'failure.json',{'error':str(error),'traceback':traceback.format_exc()})
        result['artifacts']['failure'] = 'failure.json'
    write(Path(os.environ['KERNELINFRA_RESULT']),result)
    return 1 if result['validity']=='unknown' else 0


if __name__ == '__main__':
    raise SystemExit(main())
