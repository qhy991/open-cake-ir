"""DeepSeek FP8 MoE as ordinary Cake stages; no routing or dequantization on the host."""
from __future__ import annotations

import json
from collections.abc import Mapping
from open_cake_ir.evaluation.workload import WorkloadContract
from .plan_authoring import PlanAuthor, plan_workload_target

TASK = 'fib_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048'
TASKS = {TASK:('solx_'+TASK+'_bf16','1')}
VARIANTS = ('captured',)
CASES = ('primary','zero_hidden','all_nonlocal','mixed_local','scaled','ties')
TENSORS = {
 'routing_logits':([1,256],'fp32'), 'routing_bias':([256],'bf16'),
 'hidden_states':([1,7168],'fp8_e4m3'), 'hidden_states_scale':([56,1],'fp32'),
 'gemm1_weights':([32,4096,7168],'fp8_e4m3'), 'gemm1_weights_scale':([32,32,56],'fp32'),
 'gemm2_weights':([32,7168,2048],'fp8_e4m3'), 'gemm2_weights_scale':([32,56,16],'fp32'),
 'local_expert_offset':([1],'int32'), 'routed_scaling_factor':([1],'fp32'),
 'output':([1,7168],'bf16'),
}


def _bound(name):
    if name == 'hidden_states': return 0.25
    if name in {'gemm1_weights','gemm2_weights'}: return 0.0625
    if name.endswith('_scale'): return 1.5
    if name == 'routing_bias': return 0.02
    if name == 'routing_logits': return 3.0
    if name == 'routed_scaling_factor': return 4.0
    raise ValueError('unknown MoE bounded input')


def workload_document(task=TASK,*,variant='captured',backend='triton-b300'):
    if task!=TASK or variant!='captured':raise ValueError('unsupported FlashInfer MoE task/variant')
    operator,revision=TASKS[task]
    target=plan_workload_target(backend,top_k=True,fp8=True)
    predecessor_id=operator.replace('_','-')+'-b300-t1-v1'
    successor=backend!='triton-b300'
    if successor:revision='2'
    return {
        'schema_version':1,'workload_id':(operator.replace('_','-')+'-'+backend+'-t1-v2'
                                       if successor else predecessor_id),
        'revision':revision,'state':'frozen','operator':operator,
        'provenance':[{'kind':'solx_pack_task','upstream':'flashinfer-bench',
                      'path':'flashinfer-bench-tasks/tasks/020_'+TASK[4:],
                      'scope':'mathematical_definition_only_no_candidate_or_timing_import'},
                     {'kind':'upstream_workload','path':'flashinfer_trace/workloads/moe/'+TASK[4:]+'.jsonl',
                      'workload_uuid':'e05c6c03-5603-4a1c-b34c-dcce0ecaeea4','scope':'smallest_declared_seq_len_1'},
                     {'kind':'restricted_artifact','path':'flashinfer-bench-tasks/tasks/020_'+TASK[4:]+'/baseline/',
                      'scope':'complete_target_implementation'}]
                     + ([{'kind':'workload_successor','workload_id':predecessor_id,
                           'scope':'new_exact_target_binding; original_semantics_inputs_oracle_tolerances_preserved'}]
                        if successor else []),
        'cases':[{'case_id':case,'shape':{'T':1,'E':256,'LOCAL_E':32,'H':7168,'I':2048},
                  'seed':2901+i,'mode':case} for i,case in enumerate(CASES)],
        'tensors':{name:{'shape':shape,'dtype':dtype,'layout':'contiguous_row_major','finite_only':True,
                        **({'max_abs':_bound(name)} if name not in {'output','local_expert_offset'} else {})}
                   for name,(shape,dtype) in TENSORS.items()},
        'semantics':{'target':target,'task':TASK,'variant':variant,
            'candidate_abi':{'inputs':[n for n in TENSORS if n!='output'],'outputs':['output']},
            'input_effects':'unchanged','output_storage':'fresh_contiguous_nonaliasing',
            'definition':'sigmoid + bias for group/expert choice; group top2 sum -> top4 groups -> top8 experts; normalize unbiased sigmoid over all selected experts; local FP8 scaled GEMM1 -> X1*silu(X2) -> scaled GEMM2 -> weighted sum -> BF16',
            'tie_break':'lowest_global_index_for_group_and_expert_cutoffs',
            'tie_scope':'deterministic refinement of upstream unspecified topk ties; no claim of matching an arbitrary torch tie outcome',
            'quantization':'FP8_E4M3FN_values_times_FP32_scales_in_128_element_blocks',
            'local_offset_domain':[0,224],
            'scalar_abi':'offset and factor are one-element input tensors, never specialization constants',
            'materialization':'CPU torch seeded finite bounded values; nonuniform positive scales; structured routing exercises local/nonlocal ownership; ties case has zero hidden so task output is tie-independent',
            'exclusions':['seq_len_1_only','new_input_distributions_not_original_blob_bytes','no_speedup_or_framework_claim']},
        'oracle':{'kind':'independent_cpu_fp8_moe','callable':'open_cake_ir.tasks.solx_fib.moe.reference_tensors',
                  'implementation':'CPU_stable_sort_routing_and_per_expert_FP64_dequantized_matmul'},
        'validation':{'primary_case':'primary','all_cases_required':True,'equal_nan':False,
            'comparison':'elementwise_atol_rtol','atol':1e-2,'rtol':1e-2,
            'qualification':'all_elements_all_six_input_cases; ordered_Cake_plan; performance_unqualified',
            'tolerance_rationale':'Predeclared pack tolerance, requiring all output elements instead of matched_ratio 0.99; allows FP32 accumulation order and BF16 output rounding.'},
    }


def validate_contract(document: Mapping):
    from open_cake_ir.tasks.devices import backend_for_target
    backend=backend_for_target(document.get('semantics',{}).get('target'))
    if backend is None:
        raise ValueError('FlashInfer MoE target is not registered')
    if json.dumps(document,sort_keys=True,allow_nan=False)!=json.dumps(workload_document(backend=backend),sort_keys=True):
        raise ValueError('FlashInfer MoE frozen semantic/ABI contract differs')
    workload=WorkloadContract(document)
    for case in CASES:workload.tensor_abi(case)


def materialize_tensors(workload,case_id):
    import torch
    validate_contract(workload.document)
    g=torch.Generator(device='cpu').manual_seed(workload.case(case_id)['seed'])
    values={}
    for name,(shape,dtype) in TENSORS.items():
        if dtype=='fp8_e4m3':
            # Generation and quantization belong to the Workload input, never the candidate.
            scale=0.5 if name=='hidden_states' else 0.125
            value=torch.empty(shape,dtype=torch.float8_e4m3fn)
            flat=value.reshape(-1)
            for start in range(0,flat.numel(),4*1024*1024):
                end=min(start+4*1024*1024,flat.numel())
                flat[start:end]=((torch.rand((end-start,),generator=g)-0.5)*scale).to(torch.float8_e4m3fn)
            values[name]=value
        elif name.endswith('_scale'):
            values[name]=0.5+torch.rand(shape,generator=g)
    values['routing_logits']=torch.linspace(-2,2,256,dtype=torch.float32).reshape(1,256)
    values['routing_bias']=(torch.rand((256,),generator=g)*0.015).to(torch.bfloat16)
    offset=224
    if case_id=='all_nonlocal':offset=0
    if case_id=='mixed_local':
        # The high-scoring experts straddle local [224,256) and nonlocal [192,224).
        values['routing_logits'][0,216:224]+=0.5
    if case_id in {'zero_hidden','ties'}:values['hidden_states'].zero_()
    if case_id=='ties':values['routing_logits'].zero_();values['routing_bias'].zero_()
    factor=0.75 if case_id=='scaled' else 2.5
    values['local_expert_offset']=torch.tensor([offset],dtype=torch.int32)
    values['routed_scaling_factor']=torch.tensor([factor],dtype=torch.float32)
    return values


def reference_tensors(workload,case_id,inputs):
    import torch
    validate_contract(workload.document)
    dtype={'fp32':torch.float32,'bf16':torch.bfloat16,'fp8_e4m3':torch.float8_e4m3fn,'int32':torch.int32}
    args={a.name:a for a in workload.tensor_abi(case_id) if a.mode=='input'}
    if set(inputs)!=set(args):raise ValueError('MoE input ABI differs')
    for name,arg in args.items():
        value=inputs[name]
        if tuple(value.shape)!=arg.shape or value.dtype!=dtype[arg.dtype] or value.device.type!='cpu' or not value.is_contiguous():
            raise ValueError(f'MoE CPU oracle input {name} shape/dtype/device differs')
        bound=workload.document['tensors'][name].get('max_abs')
        for chunk in value.reshape(-1).split(4*1024*1024):
            checked=chunk.float()
            if not bool(torch.isfinite(checked).all()) or (bound is not None and bool((checked.abs()>bound).any())):
                raise ValueError('MoE inputs must be bounded and finite')
            if name.endswith('_scale') and bool((checked<0).any()):
                raise ValueError('MoE block scales must be nonnegative')
    offset=int(inputs['local_expert_offset'][0]);factor=float(inputs['routed_scaling_factor'][0])
    if not 0<=offset<=224:raise ValueError('local expert interval must fit 256 experts')
    selected, weights = routing_reference(inputs['routing_logits'],inputs['routing_bias'],factor)
    output=torch.zeros((1,7168),dtype=torch.float64)
    a=inputs['hidden_states'].double()*inputs['hidden_states_scale'].T.double().repeat_interleave(128,1)
    for slot in range(8):
        local=int(selected[0,slot])-offset
        if not 0<=local<32:continue
        w1=inputs['gemm1_weights'][local].double()
        w1*=inputs['gemm1_weights_scale'][local].double().repeat_interleave(128,0).repeat_interleave(128,1)
        first=a @ w1.T
        x1,x2=first[:,:2048],first[:,2048:]
        activation=x1*(x2/(1+torch.exp(-x2)))
        w2=inputs['gemm2_weights'][local].double()
        w2*=inputs['gemm2_weights_scale'][local].double().repeat_interleave(128,0).repeat_interleave(128,1)
        output+=(activation @ w2.T)*weights[0,slot]
    return {'output':output.to(torch.bfloat16)}


def author_plan(workload,case_id='primary'):
    validate_contract(workload.document)
    plan=PlanAuthor(workload,case_id)
    for name,shape,dtype in [
        ('group_scores',(1,8),'fp32'),('selected_groups',(1,4),'int32'),
        ('expert_ids',(1,8),'int32'),('route_weights',(1,8),'fp32'),
        ('first_projection',(1,8,4096),'fp32'),('activations',(1,8,2048),'fp32'),
        ('contributions',(1,8,7168),'fp32'),
    ]:plan.tensor(name,shape,dtype)
    body=['with compute:',
          '    group_id = lm.coordinate(source="program", name="group")',
          '    lanes = lm.coordinate(source="range", start=0, extent=32)',
          '    base = group_id * 32','    experts = lanes + base',
          '    logits = lm.load(routing_logits[token, experts])',
          '    bias = lm.load(routing_bias[experts])','    bias32 = lm.cast(bias, to="fp32")',
          '    negative = logits * -1.0','    exp_negative = lm.exp(negative)',
          '    denominator = exp_negative + 1.0','    sigmoid = lm.reciprocal(denominator)',
          '    scores = sigmoid + bias32',
          '    best = lm.buffer(shape=(2,), dtype="fp32")',
          '    chosen = lm.buffer(shape=(2,), dtype="int32")',
          '    lm.top_k(scores, k=2, tie_break="lowest_index", nan_policy="reject_input", out=[best, chosen])',
          '    total = lm.reduce(best, op="sum", axis=0, across_loop=False)',
          '    lm.store(group_scores[token, group], total, coalesced=False)']
    plan.stage('moe_group_scores',['routing_logits','routing_bias'],['group_scores'],
               [('token','group_scores',0,1),('group','group_scores',1,1)],body)
    body=['with compute:','    scores = lm.load(group_scores[token, :])',
          '    best = lm.buffer(shape=(4,), dtype="fp32")',
          '    chosen = lm.buffer(shape=(4,), dtype="int32")',
          '    lm.top_k(scores, k=4, tie_break="lowest_index", nan_policy="reject_input", out=[best, chosen])',
          '    lm.store(selected_groups[token, :], chosen, coalesced=False)']
    plan.stage('moe_group_selection',['group_scores'],['selected_groups'],[('token','selected_groups',0,1)],body)
    body=['with compute:', '    ids = lm.coordinate(source="range", start=0, extent=256)',
          '    group_ids = ids // 32','    logits = lm.load(routing_logits[token, :])',
          '    bias = lm.load(routing_bias[:])','    bias32 = lm.cast(bias, to="fp32")',
          '    negative = logits * -1.0','    exp_negative = lm.exp(negative)',
          '    denominator = exp_negative + 1.0','    sigmoid = lm.reciprocal(denominator)',
          '    scores = sigmoid + bias32']
    for i in range(4):
        body += [f'    group_{i} = lm.load(selected_groups[token, {i}:{i+1}])',
                 f'    allowed_{i} = lm.compare(group_ids, group_{i}, op="eq")']
    body += ['    allowed = allowed_0 + allowed_1 + allowed_2 + allowed_3',
             '    pruned = lm.select(allowed, scores, "negative_infinity")',
             '    best = lm.buffer(shape=(8,), dtype="fp32")',
             '    chosen = lm.buffer(shape=(8,), dtype="int32")',
             '    lm.top_k(pruned, k=8, tie_break="lowest_index", nan_policy="reject_input", out=[best, chosen])',
             '    lm.store(expert_ids[token, :], chosen, coalesced=False)']
    plan.stage('moe_expert_selection',['routing_logits','routing_bias','selected_groups'],['expert_ids'],
               [('token','expert_ids',0,1)],body)
    body=['with compute:', '    chosen = lm.load(expert_ids[token, :])',
          '    logits = lm.load(routing_logits[token, chosen])',
          '    negative = logits * -1.0','    exp_negative = lm.exp(negative)',
          '    denominator = exp_negative + 1.0','    sigmoid = lm.reciprocal(denominator)',
          '    total = lm.reduce(sigmoid, op="sum", axis=0, across_loop=False)',
          '    safe_total = total + 1e-20','    normalized = sigmoid / safe_total',
          '    factor = lm.load(routed_scaling_factor[:])','    weights = normalized * factor',
          '    lm.store(route_weights[token, :], weights, coalesced=False)']
    plan.stage('moe_route_weights',['routing_logits','expert_ids','routed_scaling_factor'],['route_weights'],
               [('token','route_weights',0,1)],body)

    def projection(first):
        prefix='gemm1' if first else 'gemm2'
        activation='hidden_states' if first else 'activations'
        weights=prefix+'_weights';scales=prefix+'_weights_scale'
        output='first_projection' if first else 'contributions'
        inputs=['expert_ids','local_expert_offset',activation,weights,scales]
        inputs+=['hidden_states_scale'] if first else ['route_weights']
        body=['with compute:', '    expert = lm.load(expert_ids[token, slot])',
              '    offset = lm.load(local_expert_offset[:])','    local = expert - offset',
              '    feature_block = lm.coordinate(source="program", name="feature")',
              '    feature_start = feature_block * 16','    scale_row = feature_start // 128',
              f'for k in lm.range({weights}, name="contraction", dimension=2, tile=128, num_stages=1):',
              '    with compute:', '        k_start = lm.coordinate(source="loop", name="k")',
              '        scale_column = k_start // 128']
        if first:
            body += ['        a_raw = lm.load(hidden_states[token, k])','        a32 = lm.cast(a_raw, to="fp32")',
                     '        a_scale = lm.load(hidden_states_scale[lm.scalar_index(scale_column), token])',
                     '        activation_values = a32 * a_scale']
        else:body+=['        activation_values = lm.load(activations[token, slot, k])']
        body += [f'        w_raw = lm.load({weights}[lm.scalar_index(local), feature, k])',
                 '        w32 = lm.cast(w_raw, to="fp32")',
                 f'        w_scale = lm.load({scales}[lm.scalar_index(local), lm.scalar_index(scale_row), lm.scalar_index(scale_column)])',
                 '        scaled_weights = w32 * w_scale',
                 '        products = scaled_weights * lm.broadcast(activation_values, axis=1)',
                 '        accum = lm.reduce(products, op="sum", axis=1)',
                 'with compute:']
        result='accum'
        if not first:
            body += ['    route_weight = lm.load(route_weights[token, slot])','    weighted = accum * route_weight']
            result='weighted'
        body += [f'    lm.store({output}[token, slot, feature], {result}, coalesced=False)']
        plan.stage('moe_'+prefix,inputs,[output],
                   [('token',output,0,1),('slot',output,1,1),('feature',output,2,16)],body)
    projection(True)
    body=['with compute:', '    columns = lm.coordinate(source="program_tile", name="feature")',
          '    second_columns = columns + 2048',
          '    x1 = lm.load(first_projection[token, slot, columns])',
          '    x2 = lm.load(first_projection[token, slot, second_columns])',
          '    negative = x2 * -1.0','    exp_negative = lm.exp(negative)',
          '    denominator = exp_negative + 1.0','    silu = x2 / denominator',
          '    activation = x1 * silu',
          '    lm.store(activations[token, slot, feature], activation, coalesced=False)']
    plan.stage('moe_swiglu',['first_projection'],['activations'],
               [('token','activations',0,1),('slot','activations',1,1),('feature','activations',2,128)],body)
    projection(False)
    body=['with compute:', '    values = lm.load(contributions[token, :, feature])',
          '    total = lm.reduce(values, op="sum", axis=0, across_loop=False)',
          '    rounded = lm.cast(total, to="bf16")',
          '    lm.store(output[token, feature], rounded, coalesced=False)']
    plan.stage('moe_combine',['contributions'],['output'],
               [('token','output',0,1),('feature','output',1,128)],body)
    return plan


def launch_plan(workload,case_id='primary'):
    return author_plan(workload,case_id).finish()


def routing_reference(logits,bias,factor):
    """Independent CPU ordering; explicit stable ties do not inherit a GPU topk kernel."""
    import torch
    s=1/(1+torch.exp(-logits.float()))
    biased=(s+bias.float()).reshape(1,8,32)
    group_scores=torch.sort(biased,dim=2,descending=True,stable=True).values[:,:,:2].sum(2)
    groups=torch.argsort(group_scores,dim=1,descending=True,stable=True)[:,:4]
    mask=torch.zeros((1,8),dtype=torch.bool).scatter_(1,groups,True).repeat_interleave(32,1)
    scores=(s+bias.float()).masked_fill(~mask,-float('inf'))
    selected=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:8]
    selected_s=s.gather(1,selected)
    weights=selected_s/(selected_s.sum(1,keepdim=True)+1e-20)*factor
    return selected, weights
