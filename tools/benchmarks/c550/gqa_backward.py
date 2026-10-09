"""Complete GQA backward CAKE baseline with one row-sized FP32 intermediate.

The fixed original ABI has 80 query heads, 8 KV heads and head dimension 128.
Score dot products are recomputed to avoid a full score-sized scratch tensor.
The source owns no Torch arithmetic or runtime allocation.
"""
from __future__ import annotations

TASK = 'L1/001_attention_softmax_dropout_value_matmul_backward'
TARGET = 'xcore1002'
QUERY_HEADS = 80
KV_HEADS = 8
HEAD_DIM = 128
KEY_TILE = 32
QUERY_TILE = 32


def _score_stage(batch, query, keys, dropout, *, output_scores):
    name = 'score_gradient' if output_scores else 'softmax_row_sum'
    go_shape = (batch, query, QUERY_HEADS, HEAD_DIM)
    scores_shape = (batch, QUERY_HEADS, query, keys)
    value_shape = (batch, KV_HEADS, keys, HEAD_DIM)
    rows_shape = (batch, QUERY_HEADS, query)
    extra = (f'row_sum: cake.Tensor({rows_shape!r}, "fp32"),\n'
             f'                  grad_attn_scores: cake.Tensor({scores_shape!r}, "bf16", mode="output")'
             if output_scores else f'row_sum: cake.Tensor({rows_shape!r}, "fp32", mode="output")')
    source = f'''@cake.schedule(name="{name}", target="xcore1002", backend="triton", entry_point="cake_{name}")
def {name}(lm, grad_attn_output: cake.Tensor({go_shape!r}, "bf16"),
                  attn_weights: cake.Tensor({scores_shape!r}, "bf16"),
                  value_states: cake.Tensor({value_shape!r}, "bf16"),
                  dropout_mask: cake.Tensor({scores_shape!r}, "bool"),
                  {extra}):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    batch = lm.program(row_sum, axis=0, dimension=0, tile=1)
    head = lm.program(row_sum, axis=1, dimension=1, tile=1)
    query = lm.program(row_sum, axis=2, dimension=2, tile=1)
    with compute:
        head_index = lm.coordinate(source="program", name="head", id="head_index")
        kv_head = head_index // {QUERY_HEADS // KV_HEADS}
        go_stored = lm.load(grad_attn_output[batch, query, head, :], id="load_go")
        go = lm.cast(go_stored, to="fp32", id="widen_go")
'''
    if output_scores:
        source += '        row_term = lm.load(row_sum[batch, head, query], id="load_row_sum")\n'
    loop = keys > KEY_TILE
    if loop:
        source += f'''    for key in lm.range(value_states, name="keys_loop", dimension=2, tile={KEY_TILE},
                        num_stages=1, loop_unroll_factor=1):
        with compute:
'''
    else:
        source += '    with compute:\n'
    key = 'key' if loop else ':'
    body = [f'values_stored = lm.load(value_states[batch, lm.scalar_index(kv_head), {key}, :], id="load_values")',
            'values = lm.cast(values_stored, to="fp32", id="widen_values")',
            'products = values * lm.broadcast(go, axis=1)',
            'dot = lm.reduce(products, op="sum", axis=1, across_loop=False, id="dot")',
            f'mask_stored = lm.load(dropout_mask[batch, head, query, {key}], id="load_mask")',
            'mask = lm.cast(mask_stored, to="fp32", id="mask_to_fp32")',
            'masked = dot * mask',
            f'gradient = masked / {1.0 - dropout!r}',
            f'weights_stored = lm.load(attn_weights[batch, head, query, {key}], id="load_weights")',
            'weights = lm.cast(weights_stored, to="fp32", id="widen_weights")']
    if output_scores:
        body += ['scores = weights * (gradient - row_term)',
                 'rounded_scores = lm.cast(scores, to="bf16", id="round_score_gradient")',
                 f'lm.store(grad_attn_scores[batch, head, query, {key}], rounded_scores, coalesced=False, id="store_scores")']
    else:
        scope = '' if loop else ', across_loop=False'
        body += ['weighted_gradient = gradient * weights',
                 f'total = lm.reduce(weighted_gradient, op="sum", axis=0{scope}, id="sum_term")']
    pad = '            ' if loop else '        '
    source += '\n'.join(pad + line for line in body) + '\n'
    if not output_scores:
        source += '''    with compute:
        lm.store(row_sum[batch, head, query], total, coalesced=False, id="store_row_sum")
'''
    return source


def _value_stage(batch, query, keys):
    groups = QUERY_HEADS // KV_HEADS
    source = f'''@cake.schedule(name="value_gradient", target="xcore1002", backend="triton", entry_point="cake_value_gradient")
def value_gradient(lm, grad_attn_output: cake.Tensor(({batch}, {query}, {QUERY_HEADS}, {HEAD_DIM}), "bf16"),
                   attn_weights_dropped: cake.Tensor(({batch}, {QUERY_HEADS}, {query}, {keys}), "bf16"),
                   grad_value_states: cake.Tensor(({batch}, {KV_HEADS}, {keys}, {HEAD_DIM}), "bf16", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    batch = lm.program(grad_value_states, axis=0, dimension=0, tile=1)
    kv = lm.program(grad_value_states, axis=1, dimension=1, tile=1)
    key = lm.program(grad_value_states, axis=2, dimension=2, tile=1)
    with compute:
        kv_index = lm.coordinate(source="program", name="kv", id="kv_index")
        head_base = kv_index * {groups}
'''
    for group in range(groups):
        source += f'        head_{group} = head_base + {group}\n'
    loop = query > QUERY_TILE
    for group in range(groups):
        if loop:
            source += f'''    for query_{group} in lm.range(grad_attn_output, name="queries_{group}", dimension=1, tile={QUERY_TILE},
                                num_stages=1, loop_unroll_factor=1):
        with compute:
'''
        else:
            source += '    with compute:\n'
        index = f'query_{group}' if loop else ':'
        scope = '' if loop else ', across_loop=False'
        body = [f'go_stored_{group} = lm.load(grad_attn_output[batch, {index}, lm.scalar_index(head_{group}), :], id="load_go_{group}")',
                f'go_{group} = lm.cast(go_stored_{group}, to="fp32", id="widen_go_{group}")',
                f'weights_stored_{group} = lm.load(attn_weights_dropped[batch, lm.scalar_index(head_{group}), {index}, key], id="load_dropped_{group}")',
                f'weights_{group} = lm.cast(weights_stored_{group}, to="fp32", id="widen_dropped_{group}")',
                f'products_{group} = go_{group} * lm.broadcast(weights_{group}, axis=0)',
                f'group_sum_{group} = lm.reduce(products_{group}, op="sum", axis=0{scope}, id="sum_queries_{group}")']
        pad = '            ' if loop else '        '
        source += '\n'.join(pad + line for line in body) + '\n'
    source += '    with compute:\n'
    source += '        grouped = ' + ' + '.join(f'group_sum_{index}' for index in range(groups)) + '\n'
    source += '''        rounded_value = lm.cast(grouped, to="bf16", id="round_value_gradient")
        lm.store(grad_value_states[batch, kv, key, :], rounded_value, coalesced=False, id="store_value_gradient")
'''
    return source


def source_for(batch_size: int, seq_len_q: int, seq_len_kv: int, *, attention_dropout: float = 0.1) -> str:
    if any(type(value) is not int or value <= 0 for value in (batch_size, seq_len_q, seq_len_kv)):
        raise ValueError('GQA backward requires positive original dimensions')
    if type(attention_dropout) is not float or attention_dropout != 0.1:
        raise ValueError('GQA backward requires the original checked factory dropout value 0.1')
    shapes = (batch_size, seq_len_q, seq_len_kv)
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            + _score_stage(*shapes, attention_dropout, output_scores=False) + '\n'
            + _score_stage(*shapes, attention_dropout, output_scores=True) + '\n'
            + _value_stage(*shapes) + f'''
cake.program(program_id="c550_gqa_backward_b{batch_size}_q{seq_len_q}_k{seq_len_kv}",
    inputs=("grad_attn_output", "attn_weights", "attn_weights_dropped", "value_states", "dropout_mask"),
    outputs=("grad_attn_scores", "grad_value_states"), stages=(
        cake.stage(name="softmax_row_sum", schedule=softmax_row_sum,
                   bindings={{"grad_attn_output": "grad_attn_output", "attn_weights": "attn_weights",
                             "value_states": "value_states", "dropout_mask": "dropout_mask", "row_sum": "row_sum"}}),
        cake.stage(name="score_gradient", schedule=score_gradient,
                   bindings={{"grad_attn_output": "grad_attn_output", "attn_weights": "attn_weights",
                             "value_states": "value_states", "dropout_mask": "dropout_mask", "row_sum": "row_sum",
                             "grad_attn_scores": "grad_attn_scores"}}),
        cake.stage(name="value_gradient", schedule=value_gradient,
                   bindings={{"grad_attn_output": "grad_attn_output", "attn_weights_dropped": "attn_weights_dropped",
                             "grad_value_states": "grad_value_states"}}),
))
''')


def program_for(batch_size: int, seq_len_q: int, seq_len_kv: int):
    from open_cake_ir.compiler.program_frontend import parse_program
    return parse_program(source_for(batch_size, seq_len_q, seq_len_kv)).program


def source_for_workload(workload, case_id: str) -> str:
    abi = workload.tensor_abi(case_id)
    if workload.target != TARGET or len(abi) != 7 or len(abi[0].shape) != 4:
        raise ValueError('GQA backward target or original public ABI differs')
    batch, query, _, _ = abi[0].shape
    if len(abi[3].shape) != 4:
        raise ValueError('GQA value-state rank differs')
    keys = abi[3].shape[2]
    score_shape = (batch, QUERY_HEADS, query, keys)
    value_shape = (batch, KV_HEADS, keys, HEAD_DIM)
    expected = (('grad_attn_output', (batch, query, QUERY_HEADS, HEAD_DIM), 'bf16', 'input'),
                ('attn_weights', score_shape, 'bf16', 'input'),
                ('attn_weights_dropped', score_shape, 'bf16', 'input'),
                ('value_states', value_shape, 'bf16', 'input'),
                ('dropout_mask', score_shape, 'bool', 'input'),
                ('grad_attn_scores', score_shape, 'bf16', 'output'),
                ('grad_value_states', value_shape, 'bf16', 'output'))
    if tuple((arg.name, tuple(arg.shape), arg.dtype, arg.mode) for arg in abi) != expected:
        raise ValueError('GQA backward original ordered tensor ABI differs')
    scalars = workload.document['semantics'].get('fixed_scalar_inputs')
    if scalars != {'attention_dropout': {'dtype': 'float32', 'value': 0.1,
                                        'binding': 'original_factory_literal'}}:
        raise ValueError('GQA backward requires the checked custom-factory dropout binding')
    return source_for(batch, query, keys, attention_dropout=scalars['attention_dropout']['value'])
