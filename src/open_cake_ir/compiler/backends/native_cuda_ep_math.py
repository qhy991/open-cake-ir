"""Inline expert and combine math for an explicit ranked worker rewrite.

This bounded emitter consumes complete, verified Cake math. It emits no
dispatch, mailbox or synchronization effect and is not called by ordered
Program lowering. A future ranked-worker lowering must opt into this source
after proving the remote queue and handoff contract separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import re
from types import MappingProxyType

from .native_cuda_activation import preflight as activation_preflight
from .native_cuda_combine import preflight as combine_preflight
from .native_cuda_row_dot import preflight as row_dot_preflight
from ..ir import DType, MemorySpace, Program, Schedule
from ..target import CodeObject, Target
from ..verifier import verify


_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
# This bounded source has only the exact B300 development evidence. Hardware
# width and architecture values still come from that Target document.
RANKED_ROUTE_EVIDENCE = frozenset({'sm_103a'})


@dataclass(frozen=True)
class InlineEpMath:
    source: str
    source_map: Mapping[str, tuple[int, int]]
    hidden: int
    intermediate: int
    local_experts: int
    tokens: int
    activated_shared_bytes: int


def _refuse(message):
    raise ValueError(f'ranked inline expert math refused: {message}')


def _check_schedule(schedule: Schedule, target: Target, preflight):
    failures = [finding for check in (verify, preflight)
                for finding in check(schedule, target)
                if finding.blocks_lowering or finding.blocks_acceptance]
    if failures:
        _refuse(f'{schedule.schedule_id}: {[finding.code for finding in failures]}')


def _bound_shape(program: Program, stage_index: int, local: str,
                 expected_shape: tuple[int, ...], expected_dtype: DType):
    stage = program.stages[stage_index]
    buffer = stage.schedule.buffer(local)
    binding = stage.bindings.get(local)
    if (buffer is None or binding is None or buffer.shape != expected_shape
            or buffer.dtype is not expected_dtype
            or program.tensors[binding.tensor].nbytes != buffer.size_bytes):
        _refuse(f'{stage.name} buffer {local!r} geometry or binding differs')
    return binding.tensor


def lower_ep_math(program: Program, combine: Schedule, target: Target, *,
                  entry: str, rewrite: str) -> InlineEpMath:
    """Fuse local leaf math only as material for a declared worker rewrite."""
    if Program.from_dict(program.document) != program:
        _refuse('local Program typed view differs from its canonical document')
    if (rewrite != 'ranked_mailbox' or _IDENTIFIER.fullmatch(entry) is None
            or program.execution is not None
            or len(program.stages) != 3 or program.target != target.target_id
            or combine.target != target.target_id
            or target.code_object is not CodeObject.CUBIN
            or target.target_id not in RANKED_ROUTE_EVIDENCE
            or target.compute_capability is None or target.warp_size != 32):
        _refuse('exact B300 target, entry or three-stage local Program differs')
    first, activation, down = program.stages
    for stage, gate in ((first, row_dot_preflight),
                        (activation, activation_preflight),
                        (down, row_dot_preflight)):
        _check_schedule(stage.schedule, target, gate)
    _check_schedule(combine, target, combine_preflight)
    x = _bound_shape(program, 0, 'x', (16,), DType.BF16)
    up_weights = _bound_shape(program, 0, 'weight', (2, 64, 16), DType.BF16)
    expert = _bound_shape(program, 0, 'expert_id', (1,), DType.INT32)
    up_gate = _bound_shape(program, 0, 'y', (64,), DType.FP32)
    if (_bound_shape(program, 1, 'up_gate', (1, 64), DType.FP32) != up_gate
            or not activation.bindings['up_gate'].singleton_view):
        _refuse('activation must consume the up/gate singleton view')
    activated = _bound_shape(program, 1, 'activated', (1, 32), DType.FP32)
    if (not activation.bindings['activated'].singleton_view
            or _bound_shape(program, 2, 'x', (32,), DType.FP32) != activated
            or _bound_shape(program, 2, 'expert_id', (1,), DType.INT32) != expert):
        _refuse('down projection must consume the FP32 activation and same expert')
    down_weights = _bound_shape(program, 2, 'weight', (2, 16, 32), DType.BF16)
    output = _bound_shape(program, 2, 'y', (16,), DType.FP32)
    contribution_buffer = combine.buffer('contributions')
    weight_buffer = combine.buffer('weights')
    output_buffer = combine.buffer('output')
    if (len({x, up_weights, down_weights, expert, up_gate, activated, output}) != 7
            or contribution_buffer is None or weight_buffer is None
            or output_buffer is None
            or contribution_buffer.shape[1:] != (2, 16)
            or weight_buffer.shape[1:] != (2,)
            or output_buffer.shape[1:] != (16,)):
        _refuse('expert bindings or origin combine geometry differs')
    tokens = output_buffer.shape[0]
    if tokens not in (7, 8):
        _refuse('development combine supports the exact T=7 or T=8 domain')
    labels = {(stage.name, op.op_id) for stage in program.stages
              for op in stage.schedule.operations}
    labels.update(('combine', op.op_id) for op in combine.operations)
    if (len(labels) != sum(len(stage.schedule.operations) for stage in program.stages)
            + len(combine.operations)
            or any('\n' in stage.name or '\r' in stage.name or '\\' in stage.name
                   for stage in program.stages)):
        _refuse('source-map stage or operation names collide')
    label = lambda stage, op: f'// CAKE_OP: {stage}.{op}'
    lines = [
        '#include <cuda_bf16.h>',
        '#include <cmath>',
        f'__device__ __forceinline__ void {entry}_expert(',
        '    const __nv_bfloat16* hidden, const __nv_bfloat16* w_up_gate,',
        '    const __nv_bfloat16* w_down, int local_expert,',
        '    float* activated, float* contribution, int lane) {',
        f'  {label(first.name, first.schedule.operations[0].op_id)}',
        '  const bool cake_valid_expert = 0 <= local_expert && local_expert < 2;',
        '  if (lane < 32) {',
        '    float cake_up = 0.0f, cake_gate = 0.0f;',
        '    for (int h = 0; h < 16; ++h) {',
        f'      {label(first.name, first.schedule.operations[1].op_id)}',
        '      const float cake_x = __bfloat162float(hidden[h]);',
        f'      {label(first.name, first.schedule.operations[2].op_id)}',
        '      const int cake_base = local_expert * 64 * 16;',
        '      const float cake_w_up = cake_valid_expert ?',
        '          __bfloat162float(w_up_gate[cake_base + lane * 16 + h]) : 0.0f;',
        '      const float cake_w_gate = cake_valid_expert ?',
        '          __bfloat162float(w_up_gate[cake_base + (32 + lane) * 16 + h]) : 0.0f;',
        f'      {label(first.name, first.schedule.operations[3].op_id)}',
        f'      {label(first.name, first.schedule.operations[4].op_id)}',
        f'      {label(first.name, first.schedule.operations[5].op_id)}',
        '      const float cake_up_product = cake_x * cake_w_up;',
        '      const float cake_gate_product = cake_x * cake_w_gate;',
        f'      {label(first.name, first.schedule.operations[6].op_id)}',
        '      cake_up += cake_up_product;',
        '      cake_gate += cake_gate_product;',
        '    }',
        f'    {label(first.name, first.schedule.operations[7].op_id)}',
        f'    {label(activation.name, activation.schedule.operations[0].op_id)}',
        f'    {label(activation.name, activation.schedule.operations[1].op_id)}',
        f'    {label(activation.name, activation.schedule.operations[2].op_id)}',
        '    const float cake_neg = cake_gate * -1.0f;',
        f'    {label(activation.name, activation.schedule.operations[3].op_id)}',
        '    const float cake_exp = expf(cake_neg);',
        f'    {label(activation.name, activation.schedule.operations[4].op_id)}',
        '    const float cake_denom = cake_exp + 1.0f;',
        f'    {label(activation.name, activation.schedule.operations[5].op_id)}',
        '    const float cake_silu = cake_gate / cake_denom;',
        f'    {label(activation.name, activation.schedule.operations[6].op_id)}',
        '    const float cake_value = cake_up * cake_silu;',
        f'    {label(activation.name, activation.schedule.operations[7].op_id)}',
        '    activated[lane] = cake_value;',
        '  }',
        '  __syncthreads();',
        '  if (lane < 16) {',
        f'    {label(down.name, down.schedule.operations[0].op_id)}',
        '    float cake_down = 0.0f;',
        '    for (int i = 0; i < 32; ++i) {',
        f'      {label(down.name, down.schedule.operations[1].op_id)}',
        '      const float cake_activation = activated[i];',
        f'      {label(down.name, down.schedule.operations[2].op_id)}',
        '      const float cake_weight = cake_valid_expert ?',
        '          __bfloat162float(w_down[local_expert * 16 * 32 + lane * 32 + i]) : 0.0f;',
        f'      {label(down.name, down.schedule.operations[3].op_id)}',
        f'      {label(down.name, down.schedule.operations[4].op_id)}',
        '      const float cake_product = cake_activation * cake_weight;',
        f'      {label(down.name, down.schedule.operations[5].op_id)}',
        '      cake_down += cake_product;',
        '    }',
        f'    {label(down.name, down.schedule.operations[6].op_id)}',
        '    contribution[lane] = cake_down;',
        '  }',
        '}',
        f'__device__ __forceinline__ void {entry}_combine(',
        '    const float* contributions, const float* weights,',
        '    __nv_bfloat16* output, int token, int lane) {',
        '  if (lane < 16) {',
        f'    {label("combine", combine.operations[0].op_id)}',
        '    const float cake_route0 = contributions[(token * 2) * 16 + lane];',
        '    const float cake_route1 = contributions[(token * 2 + 1) * 16 + lane];',
        f'    {label("combine", combine.operations[1].op_id)}',
        '    const float cake_weight0 = weights[token * 2];',
        '    const float cake_weight1 = weights[token * 2 + 1];',
        f'    {label("combine", combine.operations[2].op_id)}',
        '    const float cake_product0 = cake_route0 * cake_weight0;',
        '    const float cake_product1 = cake_route1 * cake_weight1;',
        f'    {label("combine", combine.operations[3].op_id)}',
        '    const float cake_sum = cake_product0 + cake_product1;',
        f'    {label("combine", combine.operations[4].op_id)}',
        '    const __nv_bfloat16 cake_rounded = __float2bfloat16_rn(cake_sum);',
        f'    {label("combine", combine.operations[5].op_id)}',
        '    output[token * 16 + lane] = cake_rounded;',
        '  }',
        '}',
    ]
    source = '\n'.join(lines) + '\n'
    source_map = {}
    for line_number, line in enumerate(lines, start=1):
        if line.lstrip().startswith('// CAKE_OP: '):
            key = line.split('// CAKE_OP: ', 1)[1]
            if key in source_map:
                _refuse(f'duplicate emitted operation {key!r}')
            source_map[key] = (line_number, line_number)
    if set(source_map) != {f'{stage.name}.{op.op_id}' for stage in program.stages
                           for op in stage.schedule.operations} | {
                               f'combine.{op.op_id}' for op in combine.operations}:
        _refuse('inline source map omits mathematical operations')
    return InlineEpMath(source, MappingProxyType(source_map), 16, 32, 2, tokens,
                        32 * DType.FP32.itemsize)
