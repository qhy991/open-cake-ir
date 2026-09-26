"""Model-width EP4 Workload and independent CPU FP64 mathematics.

The synthetic fan-in inputs are independent of ShareGPT routing and measured
Qwen weights. The four cases retain one hidden/weight/route-weight seed and
change only expert IDs. No Compiler or NVIDIA backend code is imported here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from math import sqrt

from open_cake_ir.evaluation.workload import WorkloadContract
from . import workload as small


WORKLOAD_ID = 'weave-model-ep4-bf16-moe-b300-v1'
OPERATOR = 'bf16_model_expert_parallel_moe'
SHAPE = {'R':4,'T':512,'E':128,'H':2048,'I':768,'F':1536,'K':8}
CASES = (
    ('fanin_v2','retained_fanin_v2'),
    ('hot8_full_tiles','override_hot8_ids'),
    ('mixed_full_early_terminal','override_mixed_ids'),
    ('dense_full_and_partial','override_dense_ids'),
)
SEED = 260921483


def workload_document() -> dict:
    result = deepcopy(small.workload_document())
    result['workload_id'] = WORKLOAD_ID
    result['operator'] = OPERATOR
    result['provenance'] = [{
        'kind':'retained_baseline_input',
        'source':'open-cake-ir@4d2ebd4dfa8867e8ec5a9bbc474c4fbb79f6d645:experiments/weave_open_baseline/model_scale_inputs_fanin_v2.json',
        'scope':'fan-in-scaled deterministic synthetic inputs; not measured Qwen weights or ShareGPT routing',
    }]
    result['cases'] = [
        {'case_id':name,'shape':dict(SHAPE),'seed':SEED,'mode':mode}
        for name,mode in CASES
    ]
    result['semantics']['top_k'] = 8
    result['semantics']['definition'] = (
        'for each rank/token/route: dispatch BF16 activation to its expert owner; '
        'the independent FP64 reference computes up and gate projections, '
        'up*silu(gate), down projection, and a source-rank FP32-weighted route '
        'sum; round the final output to BF16')
    result['semantics']['exclusions'] = [
        'no_model_forward','no_serving','no_automatic_single_device_fallback',
        'synthetic_inputs_not_paper_routing_or_real_model_weights',
    ]
    result['oracle'] = {
        'kind':'independent_cpu_ep4_model_bf16_moe',
        'callable':'open_cake_ir.tasks.weave_ep.model_workload.reference_tensors',
        'implementation':'CPU_FP64_per_expert_linear_layers_and_weighted_combine',
    }
    result['validation'] = {
        'primary_case':'fanin_v2','all_cases_required':True,
        'comparison':'elementwise_atol_rtol','atol':0.01,'rtol':0.01,
        'equal_nan':False,
        'qualification':'CPU_oracle_and_development_B300_correctness_only; qualified_timing_pending',
    }
    return result


def validate_contract(document: Mapping) -> None:
    if json.dumps(document,sort_keys=True,allow_nan=False) != json.dumps(
            workload_document(),sort_keys=True,allow_nan=False):
        raise ValueError('model EP4 Workload semantic or placement contract differs')
    workload = WorkloadContract(document)
    if not workload.requires_distributed_execution or workload.target != 'sm_103a':
        raise ValueError('model EP4 requires exact distributed B300 execution')
    for name,_ in CASES:
        rows=workload.tensor_abi(name)
        if [row.name for row in rows] != [
                'hidden_states','expert_ids','route_weights',
                'w_up_gate','w_down','output']:
            raise ValueError('model EP4 tensor ABI differs')


def round_bf16(values):
    import numpy as np
    values=np.ascontiguousarray(values,dtype=np.float32)
    bits=values.view(np.uint32)
    rounded=bits+np.uint32(0x7fff)+((bits>>16)&1)
    return (rounded&np.uint32(0xffff0000)).view(np.float32)


def _route_ids(mode: str, original):
    import numpy as np
    if mode=='retained_fanin_v2':
        return original
    ids=np.empty_like(original)
    for token in range(SHAPE['T']):
        if mode=='override_hot8_ids':
            values=np.arange(8,dtype=np.int32)
        elif mode=='override_mixed_ids':
            values=(np.arange(8,dtype=np.int32) if token<128 else
                    8+((token-128)*8+np.arange(8,dtype=np.int32))%120)
        elif mode=='override_dense_ids':
            local=token%128
            low=np.array([2*(local%8),2*(local%8)+1],dtype=np.int32)
            high=16+((6*local+np.arange(6,dtype=np.int32))%16)
            values=np.concatenate((low,high))
        else:
            raise ValueError('model EP4 route mode differs')
        ids[:,token,:]=values
    return ids


def materialize_tensors(workload: WorkloadContract, case_id: str):
    """Recreate the baseline seed and BF16 inputs before any device work."""
    import numpy as np
    import torch
    validate_contract(workload.document)
    mode=workload.case(case_id)['mode']
    hidden=torch.empty((4,512,2048),dtype=torch.bfloat16)
    ids=torch.empty((4,512,8),dtype=torch.int32)
    weights=torch.empty((4,512,8),dtype=torch.float32)
    up_gate=torch.empty((128,1536,2048),dtype=torch.bfloat16)
    down_weight=torch.empty((128,2048,768),dtype=torch.bfloat16)
    for rank in range(4):
        input_rng=np.random.default_rng(SEED+rank)
        weight_rng=np.random.default_rng(SEED+1000+rank)
        x=round_bf16(input_rng.standard_normal((512,2048),dtype=np.float32))
        scores=input_rng.random((512,128),dtype=np.float32)
        chosen=np.argsort(-scores,axis=1,kind='stable')[:,:8].astype(np.int32)
        w=input_rng.random((512,8),dtype=np.float32)+np.float32(0.1)
        w/=w.sum(axis=1,keepdims=True)
        gate=round_bf16(weight_rng.standard_normal((32,768,2048),
                                                    dtype=np.float32)*(1.0/sqrt(2048)))
        up=round_bf16(weight_rng.standard_normal((32,768,2048),
                                                  dtype=np.float32)*(1.0/sqrt(2048)))
        down=round_bf16(weight_rng.standard_normal((32,2048,768),
                                                    dtype=np.float32)*(1.0/sqrt(768)))
        hidden[rank]=torch.from_numpy(x).to(torch.bfloat16)
        ids[rank]=torch.from_numpy(chosen)
        weights[rank]=torch.from_numpy(w)
        up_gate[rank*32:(rank+1)*32]=torch.from_numpy(
            np.concatenate((up,gate),axis=1)).to(torch.bfloat16)
        down_weight[rank*32:(rank+1)*32]=torch.from_numpy(down).to(torch.bfloat16)
    ids[:]=torch.from_numpy(_route_ids(mode,ids.numpy()))
    return {'hidden_states':hidden,'expert_ids':ids,'route_weights':weights,
            'w_up_gate':up_gate,'w_down':down_weight}


def reference_values(shape: Mapping[str,int], hidden, expert_ids,
                     route_weights, w_up_gate, w_down):
    """FP64 expert-grouped oracle with final BF16 rounding only."""
    import numpy as np
    ranks,tokens,top_k,experts,width,intermediate=(shape[name] for name in
        ('R','T','K','E','H','I'))
    if (experts%ranks or hidden.shape!=(ranks,tokens,width)
            or expert_ids.shape!=(ranks,tokens,top_k)
            or route_weights.shape!=(ranks,tokens,top_k)
            or w_up_gate.shape!=(experts,2*intermediate,width)
            or w_down.shape!=(experts,width,intermediate)
            or np.any(expert_ids<0) or np.any(expert_ids>=experts)
            or any(len(set(map(int,row)))!=top_k
                   for rank in expert_ids for row in rank)
            or not all(np.isfinite(value).all() for value in
                       (hidden,route_weights,w_up_gate,w_down))
            or np.any(route_weights<0)
            or not np.allclose(route_weights.sum(axis=2),1,atol=1e-6,rtol=0)):
        raise ValueError('model EP4 FP64 reference input domain differs')
    output=np.zeros((ranks,tokens,width),dtype=np.float64)
    for expert in range(experts):
        source,token,slot=np.nonzero(expert_ids==expert)
        if not source.size:continue
        x=hidden[source,token].astype(np.float64)
        up=x@w_up_gate[expert,:intermediate].astype(np.float64).T
        gate=x@w_up_gate[expert,intermediate:].astype(np.float64).T
        activated=up*(gate/(1.0+np.exp(-gate)))
        projected=activated@w_down[expert].astype(np.float64).T
        weight=route_weights[source,token,slot].astype(np.float64)
        np.add.at(output,(source,token),projected*weight[:,None])
    return round_bf16(output.astype(np.float32))


def reference_tensors(workload: WorkloadContract, case_id: str,
                      inputs: Mapping):
    import numpy as np
    import torch
    validate_contract(workload.document)
    specs={row.name:row for row in workload.tensor_abi(case_id)
           if row.mode=='input'}
    if set(inputs)!=set(specs):
        raise ValueError('model EP4 input tensor set differs')
    dtypes={'bf16':torch.bfloat16,'fp32':torch.float32,'int32':torch.int32}
    for name,spec in specs.items():
        value=inputs[name]
        if (not isinstance(value,torch.Tensor) or value.device.type!='cpu'
                or value.dtype!=dtypes[spec.dtype] or not value.is_contiguous()
                or tuple(value.shape)!=spec.shape):
            raise ValueError(f'model EP4 input {name!r} dtype or extent differs')
    def values(name):
        tensor=inputs[name]
        return (tensor.numpy() if tensor.dtype==torch.int32 else
                tensor.to(torch.float32).numpy())
    result=reference_values(workload.case(case_id)['shape'],
        values('hidden_states'),values('expert_ids'),values('route_weights'),
        values('w_up_gate'),values('w_down'))
    if not np.all(np.isfinite(result)):
        raise ValueError('model EP4 oracle output is nonfinite')
    return {'output':torch.from_numpy(result).to(torch.bfloat16)}
