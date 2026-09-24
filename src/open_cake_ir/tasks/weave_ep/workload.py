"""A rank-explicit BF16 MoE Workload Contract and independent CPU oracle.

This is a small development workload, not an implementation of the paper's six
models. Its four logical ranks and expert owners are semantic facts; no
single-device launch manifest may claim the distributed computation.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from open_cake_ir.evaluation.workload import WorkloadContract


WORKLOAD_ID = 'weave-ep4-bf16-moe-b300-v1'
CASE_SHAPES = (
    ('balanced', 8, 16, 32),
    ('skew_to_rank0', 8, 16, 32),
    ('local_only', 8, 16, 32),
    ('remote_only', 8, 16, 32),
    ('tail_tokens', 7, 16, 32),
)


def workload_document() -> dict:
    return {
        'schema_version': 1, 'workload_id': WORKLOAD_ID, 'revision': '1',
        'state': 'frozen', 'operator': 'bf16_expert_parallel_moe',
        'provenance': [{
            'kind': 'research_design', 'source': 'https://arxiv.org/abs/2609.21483',
            'scope': 'five-stage EP MoE and dynamic scheduling motivation; '
                     'this development geometry and oracle are independent',
        }],
        'cases': [
            {'case_id': name, 'shape': {'R': 4, 'T': tokens, 'E': 8,
                                       'H': hidden, 'I': intermediate,
                                       'F': 2 * intermediate, 'K': 2},
             'seed': 6201 + index, 'mode': name}
            for index, (name, tokens, hidden, intermediate) in enumerate(CASE_SHAPES)
        ],
        'tensors': {
            'hidden_states': {'shape': ['R', 'T', 'H'], 'dtype': 'bf16',
                              'layout': 'contiguous_row_major'},
            'expert_ids': {'shape': ['R', 'T', 'K'], 'dtype': 'int32',
                           'layout': 'contiguous_row_major'},
            'route_weights': {'shape': ['R', 'T', 'K'], 'dtype': 'fp32',
                              'layout': 'contiguous_row_major'},
            'w_up_gate': {'shape': ['E', 'F', 'H'], 'dtype': 'bf16',
                          'layout': 'contiguous_row_major'},
            'w_down': {'shape': ['E', 'H', 'I'], 'dtype': 'bf16',
                       'layout': 'contiguous_row_major'},
            'output': {'shape': ['R', 'T', 'H'], 'dtype': 'bf16',
                       'layout': 'contiguous_row_major'},
        },
        'semantics': {
            'target': 'sm_103a',
            'execution_topology': {'kind': 'expert_parallel', 'world_size': 4},
            'expert_placement': 'contiguous_equal_ranges_by_rank',
            'tensor_placement': {
                'hidden_states': 'rank_sharded_axis_0',
                'expert_ids': 'rank_sharded_axis_0',
                'route_weights': 'rank_sharded_axis_0',
                'w_up_gate': 'expert_sharded_axis_0',
                'w_down': 'expert_sharded_axis_0',
                'output': 'rank_sharded_axis_0',
            },
            'candidate_abi': {
                'inputs': ['hidden_states', 'expert_ids', 'route_weights',
                           'w_up_gate', 'w_down'],
                'outputs': ['output'],
            },
            'definition': 'for each rank/token/route: dispatch BF16 activation '
                          'to expert owner; FP64 reference computes up+gate, '
                          'x1*silu(x2), down, weighted combine to source rank; '
                          'round final output to BF16',
            'top_k': 2,
            'route_ids': 'distinct_in_range_per_token',
            'route_weights': 'finite_nonnegative_sum_one_per_token',
            'input_effects': 'unchanged',
            'output_storage': 'fresh_contiguous_nonaliasing',
            'exclusions': ['no_model_forward', 'no_serving',
                           'no_automatic_single_device_fallback'],
        },
        'oracle': {
            'kind': 'independent_cpu_ep4_bf16_moe',
            'callable': 'open_cake_ir.tasks.weave_ep.workload.reference_tensors',
            'implementation': 'CPU_FP64_per_route_linear_layers_and_weighted_combine',
        },
        'validation': {
            'primary_case': 'balanced', 'all_cases_required': True,
            'comparison': 'elementwise_atol_rtol', 'atol': 1e-2, 'rtol': 1e-2,
            'equal_nan': False,
            'qualification': 'CPU_oracle_only; distributed_CUDA_candidate_and_timing_pending',
        },
    }


def validate_contract(document: Mapping) -> None:
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(
            workload_document(), sort_keys=True, allow_nan=False):
        raise ValueError('Weave EP4 Workload semantic or placement contract differs')
    workload = WorkloadContract(document)
    if not workload.requires_distributed_execution:
        raise ValueError('Weave EP4 Workload requires distributed execution')
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def _shape(workload: WorkloadContract, case_id: str) -> dict:
    validate_contract(workload.document)
    return workload.case(case_id)['shape']


def _check_routes(expert_ids, shape):
    import torch
    ranks, tokens, top_k, experts = (shape[name] for name in ('R', 'T', 'K', 'E'))
    if (expert_ids.device.type != 'cpu' or expert_ids.dtype != torch.int32
            or not expert_ids.is_contiguous()
            or tuple(expert_ids.shape) != (ranks, tokens, top_k)
            or not bool(((0 <= expert_ids) & (expert_ids < experts)).all())
            or not bool((expert_ids[:, :, 0] != expert_ids[:, :, 1]).all())):
        raise ValueError('EP4 expert ids must be distinct, in range and rank-sharded')


def _check_route_values(expert_ids, shape):
    ranks, tokens, top_k, experts = (shape[name] for name in ('R', 'T', 'K', 'E'))
    if (len(expert_ids) != ranks or any(len(rows) != tokens for rows in expert_ids)):
        raise ValueError('EP4 route rank/token dimensions differ')
    for rows in expert_ids:
        for chosen in rows:
            if (len(chosen) != top_k
                    or any(type(expert) is not int or not 0 <= expert < experts
                           for expert in chosen)
                    or len(set(chosen)) != top_k):
                raise ValueError('EP4 expert ids must be distinct and in range')


def materialize_tensors(workload: WorkloadContract, case_id: str):
    import torch
    shape = _shape(workload, case_id)
    ranks, tokens, experts = (shape[name] for name in ('R', 'T', 'E'))
    hidden, intermediate = shape['H'], shape['I']
    generator = torch.Generator(device='cpu').manual_seed(workload.case(case_id)['seed'])
    def bf16(dims, scale):
        return ((torch.rand(dims, generator=generator) - 0.5) * scale).to(torch.bfloat16)
    ids = torch.empty((ranks, tokens, 2), dtype=torch.int32)
    experts_per_rank = experts // ranks
    for rank in range(ranks):
        for token in range(tokens):
            if case_id == 'skew_to_rank0':
                first, second = token % experts_per_rank, (token + 1) % experts_per_rank
            elif case_id == 'local_only':
                first, second = rank * experts_per_rank, rank * experts_per_rank + 1
            elif case_id == 'remote_only':
                first = ((rank + 1) % ranks) * experts_per_rank
                second = ((rank + 2) % ranks) * experts_per_rank
            else:
                first = (rank * experts_per_rank + token) % experts
                second = (first + 3) % experts
            ids[rank, token] = torch.tensor([first, second], dtype=torch.int32)
    route_weights = torch.empty((ranks, tokens, 2), dtype=torch.float32)
    route_weights[:, :, 0] = 0.625
    route_weights[:, :, 1] = 0.375
    return {
        'hidden_states': bf16((ranks, tokens, hidden), 0.25),
        'expert_ids': ids,
        'route_weights': route_weights,
        'w_up_gate': bf16((experts, 2 * intermediate, hidden), 0.125),
        'w_down': bf16((experts, hidden, intermediate), 0.125),
    }


@dataclass(frozen=True)
class RankVolume:
    local_routes: int
    remote_in_routes: int
    unique_remote_in_tokens: int
    unique_remote_out_tokens: int


def routed_volume_values(shape, expert_ids) -> tuple[RankVolume, ...]:
    _check_route_values(expert_ids, shape)
    ranks, tokens, top_k, experts = (shape[name] for name in ('R', 'T', 'K', 'E'))
    local = [0] * ranks
    incoming = [0] * ranks
    incoming_unique = [set() for _ in range(ranks)]
    outgoing_unique = [set() for _ in range(ranks)]
    for source in range(ranks):
        for token in range(tokens):
            for slot in range(top_k):
                destination = expert_ids[source][token][slot] // (experts // ranks)
                if destination == source:
                    local[source] += 1
                else:
                    incoming[destination] += 1
                    incoming_unique[destination].add((source, token))
                    outgoing_unique[source].add(token)
    return tuple(RankVolume(local[rank], incoming[rank], len(incoming_unique[rank]),
                            len(outgoing_unique[rank])) for rank in range(ranks))


def routed_volumes(workload: WorkloadContract, case_id: str, expert_ids) -> tuple[RankVolume, ...]:
    shape = _shape(workload, case_id)
    _check_routes(expert_ids, shape)
    return routed_volume_values(shape, expert_ids.tolist())


def reference_values(shape, hidden, expert_ids, route_weights, w_up_gate, w_down):
    """Pure CPU math; nested rank/expert values have no device or kernel dependency."""
    _check_route_values(expert_ids, shape)
    ranks, tokens, top_k = (shape[name] for name in ('R', 'T', 'K'))
    hidden_width, intermediate, experts = (shape[name] for name in ('H', 'I', 'E'))
    if (len(hidden) != ranks or len(route_weights) != ranks
            or len(w_up_gate) != experts or len(w_down) != experts):
        raise ValueError('EP4 reference rank or expert count differs')
    output = [[[0.0] * hidden_width for _ in range(tokens)] for _ in range(ranks)]
    for source in range(ranks):
        if len(hidden[source]) != tokens or len(route_weights[source]) != tokens:
            raise ValueError('EP4 reference token count differs')
        for token in range(tokens):
            x = hidden[source][token]
            weights = route_weights[source][token]
            if len(x) != hidden_width or len(weights) != top_k:
                raise ValueError('EP4 reference activation or route width differs')
            for slot in range(top_k):
                expert = expert_ids[source][token][slot]
                up = w_up_gate[expert]
                down = w_down[expert]
                if (len(up) != 2 * intermediate or len(down) != hidden_width
                        or any(len(row) != hidden_width for row in up)
                        or any(len(row) != intermediate for row in down)):
                    raise ValueError('EP4 reference expert matrix shape differs')
                projected = [math.fsum(float(weight) * float(value)
                                       for weight, value in zip(row, x, strict=True))
                             for row in up]
                activated = [a * (b / (1 + math.exp(-b)))
                             for a, b in zip(projected[:intermediate],
                                             projected[intermediate:], strict=True)]
                contribution = [math.fsum(float(weight) * value
                                          for weight, value in zip(row, activated, strict=True))
                                for row in down]
                scale = float(weights[slot])
                output[source][token] = [old + scale * value
                                         for old, value in zip(output[source][token],
                                                               contribution, strict=True)]
    return output


def reference_tensors(workload: WorkloadContract, case_id: str, inputs: Mapping):
    import torch
    shape = _shape(workload, case_id)
    args = {arg.name: arg for arg in workload.tensor_abi(case_id) if arg.mode == 'input'}
    if set(inputs) != set(args):
        raise ValueError('EP4 MoE input ABI differs')
    dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32, 'int32': torch.int32}
    for name, spec in args.items():
        value = inputs[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.dtype != dtype[spec.dtype] or not value.is_contiguous()
                or tuple(value.shape) != spec.shape):
            raise ValueError(f'EP4 MoE input {name!r} shape/dtype/device differs')
        if spec.dtype != 'int32' and not bool(torch.isfinite(value.float()).all()):
            raise ValueError(f'EP4 MoE input {name!r} must be finite')
    _check_routes(inputs['expert_ids'], shape)
    weights = inputs['route_weights']
    if (not bool((weights >= 0).all())
            or not bool(torch.allclose(weights.sum(dim=2),
                                      torch.ones_like(weights[:, :, 0]), atol=1e-6, rtol=0))):
        raise ValueError('EP4 route weights must be nonnegative and sum to one')
    output = reference_values(shape, inputs['hidden_states'].tolist(),
                              inputs['expert_ids'].tolist(), weights.tolist(),
                              inputs['w_up_gate'].tolist(), inputs['w_down'].tolist())
    return {'output': torch.tensor(output, dtype=torch.float64).to(torch.bfloat16)}
