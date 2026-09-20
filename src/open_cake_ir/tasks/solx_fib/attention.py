"""GQA/MLA contracts and independent reference semantics, authored entirely as Cake IR."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from open_cake_ir.evaluation.workload import WorkloadContract
from .attention_specs import SPECS
from .plan_authoring import PlanAuthor, plan_workload_target

TASKS = {name:('solx_'+name+'_bf16','1') for name in SPECS}
VARIANTS = ('captured','boundary')
CASES = ('primary','zeros','permuted','scaled','segmented')


def geometry(task, variant='captured'):
    spec = SPECS[task]
    if variant == 'captured':
        return dict(spec['captured_axes'])
    if variant != 'boundary':
        raise ValueError('attention variant must be captured or boundary')
    shape = {'len_indptr':3, ('batch_size' if spec['decode'] else 'total_q'):2 if spec['decode'] else 6}
    if spec['paged']:
        shape.update(num_pages=11,num_kv_indices=5)
    else:
        shape['total_kv'] = 5
    return shape


def workload_document(task, *, variant='captured', backend='triton-b300'):
    if task not in TASKS:
        raise ValueError('unknown FlashInfer attention task')
    spec=SPECS[task]; axes={**spec['constants'],**geometry(task,variant)}
    operator,revision=TASKS[task]
    target = plan_workload_target(backend)
    predecessor_id = operator.replace('_','-')+'-b300-'+variant+'-v1'
    successor = backend != 'triton-b300'
    if successor:
        revision = '2'
    tensors={}
    for name,decl in {**spec['inputs'],**spec['outputs']}.items():
        tensors[name]={'shape':[axes.get(v,v) if isinstance(v,str) else v for v in decl['shape']],
                       'dtype':decl['dtype'],'layout':'contiguous_row_major','finite_only':name not in spec['outputs']}
    for name in spec['inputs']:
        if tensors[name]['dtype'] != 'int32':
            tensors[name]['max_abs'] = 1.0 if name == 'sm_scale' else 0.5
    nan_masked = not spec['decode'] and (spec['mla'] or not spec['paged'])
    return {
        'schema_version':1,'workload_id':(operator.replace('_','-')+'-'+backend+'-'+variant+'-v2'
                                       if successor else predecessor_id),
        'revision':revision,'state':'frozen','operator':operator,
        'provenance':[{'kind':'solx_pack_task','upstream':'flashinfer-bench','path':'flashinfer-bench-tasks/tasks/'+spec['upstream'],
                       'scope':'mathematical_definition_only_no_candidate_or_timing_import'},
                      {'kind':'upstream_workload','path':'flashinfer_trace/workloads/'+spec['family']+'/'+task[4:]+'.jsonl',
                       'scope':'first_record_axes_only' if variant=='captured' else 'synthetic_boundary_shape',
                       'workload_uuid':spec['workload_uuid']},
                      {'kind':'restricted_artifact','path':'flashinfer-bench-tasks/tasks/'+spec['upstream']+'/baseline/',
                       'scope':'complete_target_implementation'}]
                     + ([{'kind':'workload_successor','workload_id':predecessor_id,
                           'scope':'new_exact_target_binding; original_semantics_inputs_oracle_tolerances_preserved'}]
                        if successor else []),
        'cases':[{'case_id':case,'shape':axes,'seed':2801+i,'mode':case} for i,case in enumerate(CASES)],
        'tensors':tensors,
        'semantics':{'target':target,'variant':variant,'task':task,
            'candidate_abi':{'inputs':list(spec['inputs']),'outputs':list(spec['outputs'])},
            'input_effects':'unchanged','output_storage':'fresh_contiguous_nonaliasing',
            'definition':'scaled QK dot plus optional positional dot; bottom-right causal softmax; weighted V and base2 logsumexp',
            'scalar_abi':'upstream scalar values are one-element input tensors, never specialization constants',
            'fully_masked_nonempty_segment_output':'nan' if nan_masked else 'zero',
            'empty_segment_output':'zero','empty_or_masked_lse':'negative_infinity',
            'materialization':'deterministic CPU torch generator; BF16 uniform[-0.5,0.5]; runtime monotone indptr and in-range pages; captured scale as FP32',
            'exclusions':['one_fixed_shape_per_contract','new_input_distributions_not_original_blob_bytes','no_upstream_score_or_latency_claim']},
        'oracle':{'kind':'independent_cpu_attention','callable':'open_cake_ir.tasks.solx_fib.attention.reference_tensors',
                  'implementation':'CPU_float64_dot_softmax_then_declared_output_rounding'},
        'validation':{'primary_case':'primary','all_cases_required':True,'equal_nan':nan_masked,
            'comparison':'per_output','atol':2**-16,'rtol':2**-7,
            'outputs':{'output':{'comparison':'elementwise_atol_rtol_ieee' if nan_masked else 'elementwise_atol_rtol',
                                 'atol':2**-16,'rtol':2**-7},
                       'lse':{'comparison':'elementwise_atol_rtol_ieee','atol':1e-4,'rtol':1e-5}},
            'qualification':'all_elements_and_both_outputs_at_this_fixed_shape; ordered_Cake_plan; performance_unqualified',
            'tolerance_rationale':'BF16 output permits FP32 reduction rounding; LSE is FP32. IEEE comparison matches only corresponding NaNs and signed infinities required by the source definition.'},
    }


def validate_contract(document: Mapping):
    from open_cake_ir.tasks.devices import backend_for_target
    task=document.get('semantics',{}).get('task')
    variant=document.get('semantics',{}).get('variant')
    if task not in TASKS or variant not in {'captured','boundary'}:
        raise ValueError('FlashInfer attention task/variant differs')
    backend=backend_for_target(document.get('semantics',{}).get('target'))
    if backend is None:
        raise ValueError('FlashInfer attention target is not registered')
    expected=workload_document(task,variant=variant,backend=backend)
    if json.dumps(document,sort_keys=True,allow_nan=False)!=json.dumps(expected,sort_keys=True):
        raise ValueError('FlashInfer attention frozen semantic/ABI contract differs')
    workload=WorkloadContract(document)
    for case in CASES:workload.tensor_abi(case)


def materialize_tensors(workload,case_id):
    """Create reference inputs on CPU only; candidate host orchestration computes no math."""
    import torch
    validate_contract(workload.document)
    spec=SPECS[workload.document['semantics']['task']]
    axes=workload.case(case_id)['shape'];batch=axes['len_indptr']-1
    qcount=axes.get('batch_size',axes.get('total_q'))
    kvcount=axes.get('num_kv_indices',axes.get('total_kv'))
    generator=torch.Generator(device='cpu').manual_seed(workload.case(case_id)['seed'])
    inputs={}
    for arg in workload.tensor_abi(case_id):
        if arg.mode=='input' and arg.dtype=='bf16':
            inputs[arg.name]=(torch.rand(arg.shape,generator=generator,dtype=torch.float32)-0.5).to(torch.bfloat16)
    qptr=[i*qcount//batch for i in range(batch+1)]
    kptr=[i*kvcount//batch for i in range(batch+1)]
    if case_id=='segmented' and batch>1:kptr[1]=0
    inputs['kv_indptr']=torch.tensor(kptr,dtype=torch.int32)
    if not spec['decode']:inputs['qo_indptr']=torch.tensor(qptr,dtype=torch.int32)
    if spec['paged']:
        pages=axes['num_pages']
        ids=(torch.arange(kvcount,dtype=torch.int64)*3+1)%pages
        if case_id=='permuted':ids=ids.flip(0)
        inputs['kv_indices']=ids.to(torch.int32)
    scale=spec['captured_scale']*(1.75 if case_id=='scaled' else 1.0)
    inputs['sm_scale']=torch.tensor([scale],dtype=torch.float32)
    if case_id=='zeros':
        for name in ('q','q_nope','q_pe'):
            if name in inputs:inputs[name].zero_()
    return inputs


def reference_tensors(workload,case_id,inputs):
    import torch
    validate_contract(workload.document)
    spec=SPECS[workload.document['semantics']['task']]
    abi=workload.tensor_abi(case_id)
    expected={a.name:a for a in abi if a.mode=='input'}
    if set(inputs)!=set(expected):raise ValueError('attention input ABI differs')
    dtypes={'bf16':torch.bfloat16,'fp32':torch.float32,'int32':torch.int32}
    for name,arg in expected.items():
        value=inputs[name]
        if tuple(value.shape)!=arg.shape or value.dtype!=dtypes[arg.dtype] or value.device.type!='cpu' or not value.is_contiguous():
            raise ValueError(f'attention CPU oracle input {name} shape/dtype/device differs')
        if arg.dtype!='int32' and (not bool(torch.isfinite(value).all()) or bool((value.abs()>workload.document['tensors'][name]['max_abs']).any())):
            raise ValueError('attention inputs must be bounded and finite')
    q=inputs['q_nope'] if spec['mla'] else inputs['q']
    count,heads,_=q.shape
    kvptr=inputs['kv_indptr'].tolist()
    qptr=list(range(count+1)) if spec['decode'] else inputs['qo_indptr'].tolist()
    kvcount=inputs['kv_indices'].numel() if spec['paged'] else inputs['k'].shape[0]
    if len(qptr)!=len(kvptr) or qptr[0]!=0 or qptr[-1]!=count or kvptr[0]!=0 or kvptr[-1]!=kvcount or any(a>b for ptr in (qptr,kvptr) for a,b in zip(ptr,ptr[1:])):
        raise ValueError('attention indptr bounds/monotonicity differ')
    if spec['paged']:
        pages=inputs['ckv_cache' if spec['mla'] else 'k_cache'].shape[0]
        if bool(((inputs['kv_indices']<0)|(inputs['kv_indices']>=pages)).any()):raise ValueError('attention page IDs exceed cache')
    out_arg=next(a for a in abi if a.name=='output')
    result=torch.zeros(out_arg.shape,dtype=torch.bfloat16)
    lse=torch.full((count,heads),-float('inf'),dtype=torch.float32)
    scale=float(inputs['sm_scale'][0])
    for b in range(len(kvptr)-1):
        qs,qe=qptr[b:b+2];ks,ke=kvptr[b:b+2]
        if qs==qe or ks==ke:continue
        ids=inputs['kv_indices'][ks:ke].long() if spec['paged'] else torch.arange(ks,ke)
        if spec['mla']:
            keys=inputs['ckv_cache'][ids,0].double();positions=inputs['kpe_cache'][ids,0].double();values=keys
        else:
            keys=(inputs['k_cache'][ids,0] if spec['paged'] else inputs['k'][ids]).double()
            values=(inputs['v_cache'][ids,0] if spec['paged'] else inputs['v'][ids]).double()
        for qi in range(qs,qe):
            valid=(ke-ks) if spec['decode'] else min(ke-ks,qi-qs+1+(ke-ks)-(qe-qs))
            if valid<=0:
                if workload.document['semantics']['fully_masked_nonempty_segment_output']=='nan':result[qi].fill_(float('nan'))
                continue
            for h in range(heads):
                if spec['mla']:
                    logits=(keys[:valid] @ inputs['q_nope'][qi,h].double() + positions[:valid] @ inputs['q_pe'][qi,h].double())*scale
                    vals=values[:valid]
                else:
                    kh=h//(heads//spec['constants']['num_kv_heads'])
                    logits=(keys[:valid,kh] @ q[qi,h].double())*scale;vals=values[:valid,kh]
                lse[qi,h]=torch.logsumexp(logits,0)/math.log(2)
                result[qi,h]=(torch.softmax(logits,0) @ vals).to(torch.bfloat16)
    return {'output':result,'lse':lse}


def _power_two(value):
    return 1 << (value-1).bit_length()


def author_plan(workload,case_id='primary'):
    """Four complete Cake Schedules; host-side work is allocation and ordered launch."""
    validate_contract(workload.document)
    spec=SPECS[workload.document['semantics']['task']]
    axes=workload.case(case_id)['shape']
    q=axes.get('batch_size',axes.get('total_q'));h=spec['constants']['num_qo_heads']
    batch=axes['len_indptr']-1;length=_power_two(axes.get('num_kv_indices',axes.get('total_kv')))
    nan_masked=workload.document['semantics']['fully_masked_nonempty_segment_output']=='nan'
    plan=PlanAuthor(workload,case_id)
    for name in ['kv_start','valid_count']+(['kv_length'] if nan_masked else []):plan.tensor(name,(q,),'int32')
    plan.tensor('logits',(q,h,length),'fp32');plan.tensor('probabilities',(q,h,length),'fp32')
    body=['with compute:', '    qi = lm.coordinate(source="program", name="q_row")']
    if spec['decode']:
        body += ['    batch_index = lm.coordinate(source="program", name="q_row")']
    else:
        body += [f'    segments = lm.coordinate(source="range", start=0, extent={_power_two(batch)})',
                 '    starts = lm.load(qo_indptr[segments])',
                 '    preceding = lm.compare(starts, qi, op="le")',
                 f'    inside = lm.compare(segments, {batch}, op="lt")',
                 '    flags = preceding * inside',
                 '    count = lm.reduce(flags, op="sum", axis=0, across_loop=False)',
                 '    batch_index = count - 1']
    body += ['    next_batch = batch_index + 1',
             '    start_value = lm.load(kv_indptr[lm.scalar_index(batch_index)])',
             '    end_value = lm.load(kv_indptr[lm.scalar_index(next_batch)])',
             '    length_value = end_value - start_value']
    valid='length_value'
    if not spec['decode']:
        body += ['    query_start = lm.load(qo_indptr[lm.scalar_index(batch_index)])',
                 '    query_end = lm.load(qo_indptr[lm.scalar_index(next_batch)])',
                 '    query_length = query_end - query_start',
                 '    local_query = qi - query_start',
                 '    prefix = length_value - query_length',
                 '    causal_count = local_query + prefix + 1',
                 '    shorter = lm.compare(causal_count, length_value, op="lt")',
                 '    bounded = lm.select(shorter, causal_count, length_value)',
                 '    positive = lm.compare(bounded, 0, op="gt")',
                 '    valid_value = lm.select(positive, bounded, 0)']
        valid='valid_value'
    body += ['    lm.store(kv_start[q_row], start_value, coalesced=False)',
             f'    lm.store(valid_count[q_row], {valid}, coalesced=False)']
    if nan_masked:body.append('    lm.store(kv_length[q_row], length_value, coalesced=False)')
    plan.stage('attention_metadata', ['kv_indptr']+([] if spec['decode'] else ['qo_indptr']),
               ['kv_start','valid_count']+(['kv_length'] if nan_masked else []), [('q_row','kv_start',0,1)],body)
    score_inputs=['kv_start','valid_count','sm_scale']
    score_inputs += ['q_nope','q_pe','ckv_cache','kpe_cache'] if spec['mla'] else ['q','k_cache' if spec['paged'] else 'k']
    if spec['paged']:score_inputs+=['kv_indices']
    body=['with compute:',
          '    positions = lm.coordinate(source="program_tile", name="key_block")',
          '    start = lm.load(kv_start[q_row])',
          '    valid = lm.load(valid_count[q_row])',
          '    absolute = positions + start',
          '    scale = lm.load(sm_scale[:])']
    if spec['paged']:
        body += ['    pages = lm.load(kv_indices[absolute])',
                 '    zero = lm.coordinate(source="range", start=0, extent=1)']
    if spec['mla']:
        body += ['    qc = lm.load(q_nope[q_row, h_head, :])',
                 '    qp = lm.load(q_pe[q_row, h_head, :])',
                 '    kc = lm.load(ckv_cache[pages, lm.scalar_index(zero), :])',
                 '    kp = lm.load(kpe_cache[pages, lm.scalar_index(zero), :])',
                 '    qc32 = lm.cast(qc, to="fp32")','    qp32 = lm.cast(qp, to="fp32")',
                 '    kc32 = lm.cast(kc, to="fp32")','    kp32 = lm.cast(kp, to="fp32")',
                 '    products_c = kc32 * lm.broadcast(qc32, axis=1)',
                 '    products_p = kp32 * lm.broadcast(qp32, axis=1)',
                 '    dot_c = lm.reduce(products_c, op="sum", axis=1, across_loop=False)',
                 '    dot_p = lm.reduce(products_p, op="sum", axis=1, across_loop=False)',
                 '    dot = dot_c + dot_p']
    else:
        body += ['    head = lm.coordinate(source="program", name="h_head")',
                 f'    kv_head = head // {h//spec["constants"]["num_kv_heads"]}',
                 '    query = lm.load(q[q_row, h_head, :])',
                 ('    key = lm.load(k_cache[pages, lm.scalar_index(zero), lm.scalar_index(kv_head), :])' if spec['paged'] else
                  '    key = lm.load(k[absolute, lm.scalar_index(kv_head), :])'),
                 '    query32 = lm.cast(query, to="fp32")','    key32 = lm.cast(key, to="fp32")',
                 '    products = key32 * lm.broadcast(query32, axis=1)',
                 '    dot = lm.reduce(products, op="sum", axis=1, across_loop=False)']
    body += ['    scaled = dot * scale','    keep = lm.compare(positions, valid, op="lt")',
             '    masked = lm.select(keep, scaled, "negative_infinity")',
             '    lm.store(logits[q_row, h_head, key_block], masked, coalesced=False)']
    plan.stage('attention_scores',score_inputs,['logits'],
               [('q_row','logits',0,1),('h_head','logits',1,1),('key_block','logits',2,min(32,length))],body)
    nan_masked=workload.document['semantics']['fully_masked_nonempty_segment_output']=='nan'
    body=['with compute:', '    scores = lm.load(logits[q_row, h_head, :])',
          '    count = lm.load(valid_count[q_row])', '    valid_row = lm.compare(count, 0, op="gt")',
          '    maximum = lm.reduce(scores, op="max", axis=0, across_loop=False)',
          '    safe_maximum = lm.select(valid_row, maximum, 0.0)',
          '    shifted = scores - safe_maximum','    exponentials = lm.exp(shifted)',
          '    total = lm.reduce(exponentials, op="sum", axis=0, across_loop=False)',
          '    safe_total = lm.select(valid_row, total, 1.0)']
    if nan_masked:
        body += ['    length_value = lm.load(kv_length[q_row])',
                 '    nonempty = lm.compare(length_value, 0, op="gt")',
                 '    denominator = lm.select(nonempty, total, 1.0)',
                 '    weights = exponentials / denominator']
    else:body += ['    weights = exponentials / safe_total']
    body += ['    log_total = lm.log2(safe_total)',f'    log_max = safe_maximum * {math.log2(math.e)!r}',
             '    logarithm = log_total + log_max',
             '    final_lse = lm.select(valid_row, logarithm, "negative_infinity")',
             '    lm.store(probabilities[q_row, h_head, :], weights, coalesced=False)',
             '    lm.store(lse[q_row, h_head], final_lse, coalesced=False)']
    plan.stage('attention_normalize',['logits','valid_count']+(['kv_length'] if nan_masked else []),
               ['probabilities','lse'], [('q_row','lse',0,1),('h_head','lse',1,1)],body)
    value_inputs=['kv_start','probabilities']+(['ckv_cache'] if spec['mla'] else ['v_cache' if spec['paged'] else 'v'])
    if spec['paged']:value_inputs+=['kv_indices']
    body=['with compute:', '    start = lm.load(kv_start[q_row])']
    if spec['paged']:body+=['    zero = lm.coordinate(source="range", start=0, extent=1)']
    if not spec['mla']:
        body+=['    head = lm.coordinate(source="program", name="h_head")',
               f'    kv_head = head // {h//spec["constants"]["num_kv_heads"]}']
    body += [f'for key in lm.range(probabilities, name="key_loop", dimension=2, tile={min(16,length)}, num_stages=1):',
             '    with compute:', '        positions = lm.coordinate(source="loop_tile", name="key")',
             '        absolute = positions + start']
    if spec['paged']:body+=['        pages = lm.load(kv_indices[absolute])']
    if spec['mla']:value_load='ckv_cache[pages, lm.scalar_index(zero), :]'
    elif spec['paged']:value_load='v_cache[pages, lm.scalar_index(zero), lm.scalar_index(kv_head), :]'
    else:value_load='v[absolute, lm.scalar_index(kv_head), :]'
    body += [f'        values = lm.load({value_load})','        values32 = lm.cast(values, to="fp32")',
             '        weights = lm.load(probabilities[q_row, h_head, key])',
             '        products = values32 * lm.broadcast(weights, axis=0)',
             '        accum = lm.reduce(products, op="sum", axis=0)',
             'with compute:','    rounded = lm.cast(accum, to="bf16")',
             '    lm.store(output[q_row, h_head, :], rounded, coalesced=False)']
    if length <= 16:
        single = []
        for line in body:
            if line.startswith('for key in '):
                single.append('with compute:')
            elif line == '    with compute:':
                continue
            elif line.startswith('        '):
                line = line[4:]
                line = line.replace('source="loop_tile", name="key"', f'source="range", start=0, extent={length}')
                line = line.replace('probabilities[q_row, h_head, key]', 'probabilities[q_row, h_head, :]')
                line = line.replace('op="sum", axis=0)', 'op="sum", axis=0, across_loop=False)')
                single.append(line)
            else:
                single.append(line)
        body = single
    plan.stage('attention_values',value_inputs,['output'],
               [('q_row','output',0,1),('h_head','output',1,1)],body)
    return plan


def launch_plan(workload,case_id='primary'):
    return author_plan(workload,case_id).finish()
