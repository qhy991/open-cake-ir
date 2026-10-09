"""Untuned pure CAKE Program for original segmented vision attention.

The public binder keeps the original 1152/16/72 dimensions. Smaller dimensions
are an explicit CPU control domain, never a substitute for an original case.
"""
from __future__ import annotations

TASK = 'L2/018_cu_seqlens_variable_length_vision_attention'
TARGET = 'xcore1002'
DIMENSIONS = (1152, 16, 72)


def _header(name, arguments):
    return (f'@cake.schedule(name="{name}", target="{TARGET}", backend="triton", entry_point="cake_{name}")\n'
            f'def {name}(lm, ' + ', '.join(arguments) + '):\n'
            '    compute = lm.role(execution_groups=[0, 1, 2, 3])\n')


def _tensor(name, shape, dtype='bf16', output=False):
    return f'{name}: cake.Tensor({shape!r}, "{dtype}"' + (', mode="output")' if output else ')')


def _projection(tokens, embed, heads, depth, part, output):
    feature_tile = min(8, depth)
    loop = embed > 128
    inner = '        ' if loop else '    '
    source = _header('project_' + output, [_tensor('hidden_states', (tokens, embed)),
        _tensor('qkv_weight', (3 * embed, embed)), _tensor('qkv_bias', (3 * embed,)),
        _tensor(output, (tokens, heads, depth), output=True)])
    source += f'''    token = lm.program({output}, axis=0, dimension=0, tile=1)
    head = lm.program({output}, axis=1, dimension=1, tile=1)
    feature = lm.program({output}, axis=2, dimension=2, tile={feature_tile})
    with compute:
        head_index = lm.coordinate(source="program", name="head", id="head_index")
        features = lm.coordinate(source="program_tile", name="feature", id="features")
        weight_rows = head_index * {depth} + features + {part * embed}
'''
    if loop:
        source += '    for hidden in lm.range(hidden_states, name="hidden_loop", dimension=1, tile=128, num_stages=1, loop_unroll_factor=1):\n'
    k = 'hidden' if loop else ':'
    source += inner + 'with compute:\n'
    source += '\n'.join(inner + '    ' + line for line in [
        f'x_stored = lm.load(hidden_states[token, {k}], id="load_x")',
        'x = lm.cast(x_stored, to="fp32", id="widen_x")',
        f'w_stored = lm.load(qkv_weight[weight_rows, {k}], id="load_weight")',
        'w = lm.cast(w_stored, to="fp32", id="widen_weight")',
        'products = w * lm.broadcast(x, axis=1)',
        f'projection = lm.reduce(products, op="sum", axis=1, scope="cta", across_loop={loop}, id="dot")',
    ]) + '\n'
    source += f'''    with compute:
        bias_stored = lm.load(qkv_bias[weight_rows], id="load_bias")
        bias = lm.cast(bias_stored, to="fp32", id="widen_bias")
        with_bias = projection + bias
        rounded = lm.cast(with_bias, to="bf16", id="round_linear")
        lm.store({output}[token, head, feature], rounded, coalesced=False, id="store_projection")
'''
    return source


def _rotary(tokens, heads, depth):
    names = ('q0', 'k0', 'cos', 'sin', 'q', 'k')
    source = _header('rotary', [_tensor(name, (tokens, heads, depth), output=name in ('q', 'k')) for name in names])
    source += f'''    token = lm.program(q, axis=0, dimension=0, tile=1)
    head = lm.program(q, axis=1, dimension=1, tile=1)
    feature = lm.program(q, axis=2, dimension=2, tile={min(8, depth)})
    with compute:
        features = lm.coordinate(source="program_tile", name="feature", id="features")
        first_half = lm.compare(features, {depth // 2}, op="lt", id="first_half")
        second_indices = features + {depth // 2}
        first_indices = features - {depth // 2}
        rotated_indices = lm.select(first_half, second_indices, first_indices, id="rotation_indices")
        cosine_stored = lm.load(cos[token, head, feature], id="load_cos")
        sine_stored = lm.load(sin[token, head, feature], id="load_sin")
        cosine = lm.cast(cosine_stored, to="fp32", id="widen_cos")
        sine = lm.cast(sine_stored, to="fp32", id="widen_sin")
'''
    for name in ('q', 'k'):
        source += f'''        {name}_stored = lm.load({name}0[token, head, feature], id="load_{name}")
        {name}_rotated_stored = lm.load({name}0[token, head, rotated_indices], id="load_rotated_{name}")
        {name}_value = lm.cast({name}_stored, to="fp32", id="widen_{name}")
        {name}_rotation = lm.cast({name}_rotated_stored, to="fp32", id="widen_rotated_{name}")
        {name}_negative = {name}_rotation * -1.0
        {name}_signed = lm.select(first_half, {name}_negative, {name}_rotation, id="sign_{name}")
        {name}_cos_product = {name}_value * cosine
        {name}_sin_product = {name}_signed * sine
        {name}_cos_bf16 = lm.cast({name}_cos_product, to="bf16", id="round_{name}_cos")
        {name}_sin_bf16 = lm.cast({name}_sin_product, to="bf16", id="round_{name}_sin")
        {name}_cos_fp32 = lm.cast({name}_cos_bf16, to="fp32", id="widen_{name}_cos")
        {name}_sin_fp32 = lm.cast({name}_sin_bf16, to="fp32", id="widen_{name}_sin")
        {name}_sum = {name}_cos_fp32 + {name}_sin_fp32
        {name}_rounded = lm.cast({name}_sum, to="bf16", id="round_{name}_sum")
        lm.store({name}[token, head, feature], {name}_rounded, coalesced=False, id="store_{name}")
'''
    return source


def _segments(tokens, sequences):
    extent = 1 << (sequences - 1).bit_length()
    return _header('segment_ids', [_tensor('cu_seqlens', (sequences,), 'int64'),
        _tensor('segments', (tokens,), 'int32', output=True)]) + f'''    token = lm.program(segments, axis=0, dimension=0, tile=1)
    with compute:
        token_index = lm.coordinate(source="program", name="token", id="token_index")
        token_i64 = lm.cast(token_index, to="int64", id="widen_token")
        boundaries = lm.coordinate(source="range", start=0, extent={extent}, id="boundaries")
        ends = lm.load(cu_seqlens[boundaries], id="load_ends")
        before = lm.compare(ends, token_i64, op="le", id="count_ends")
        valid = lm.compare(boundaries, {sequences}, op="lt", id="valid_ends")
        contributions = before * valid
        segment = lm.reduce(contributions, op="sum", axis=0, scope="cta", across_loop=False, id="segment")
        lm.store(segments[token], segment, coalesced=False, id="store_segment")
'''


def _scores(tokens, heads, depth):
    loop = depth > 8
    inner = '        ' if loop else '    '
    source = _header('attention_scores', [_tensor('q', (tokens, heads, depth)),
        _tensor('k', (tokens, heads, depth)), _tensor('segments', (tokens,), 'int32'),
        _tensor('scores', (tokens, heads, tokens), output=True)])
    source += '''    query = lm.program(scores, axis=0, dimension=0, tile=1)
    head = lm.program(scores, axis=1, dimension=1, tile=1)
    key = lm.program(scores, axis=2, dimension=2, tile=8)
'''
    if loop:
        source += '    for feature in lm.range(q, name="features", dimension=2, tile=8, num_stages=1, loop_unroll_factor=1):\n'
    feature = 'feature' if loop else ':'
    source += inner + 'with compute:\n'
    source += '\n'.join(inner + '    ' + line for line in [
        f'query_stored = lm.load(q[query, head, {feature}], id="load_q")',
        f'key_stored = lm.load(k[key, head, {feature}], id="load_k")',
        'query_value = lm.cast(query_stored, to="fp32", id="widen_q")',
        'key_value = lm.cast(key_stored, to="fp32", id="widen_k")',
        'products = key_value * lm.broadcast(query_value, axis=1)',
        f'dot = lm.reduce(products, op="sum", axis=1, scope="cta", across_loop={loop}, id="dot")',
    ]) + '\n'
    source += f'''    with compute:
        dot_bf16 = lm.cast(dot, to="bf16", id="round_qk")
        dot_fp32 = lm.cast(dot_bf16, to="fp32", id="widen_qk")
        scaled = dot_fp32 * {depth ** -0.5!r}
        scaled_bf16 = lm.cast(scaled, to="bf16", id="round_scale")
        scaled_fp32 = lm.cast(scaled_bf16, to="fp32", id="widen_scale")
        query_segment = lm.load(segments[query], id="query_segment")
        key_segment = lm.load(segments[key], id="key_segment")
        same_segment = lm.compare(query_segment, key_segment, op="eq", id="same_segment")
        masked = lm.select(same_segment, scaled_fp32, "negative_infinity", id="mask_segments")
        masked_bf16 = lm.cast(masked, to="bf16", id="round_masked")
        lm.store(scores[query, head, key], masked_bf16, coalesced=False, id="store_scores")
'''
    return source


def _statistics(tokens, heads):
    extent = 1 << (tokens - 1).bit_length()
    return _header('softmax_statistics', [_tensor('scores', (tokens, heads, tokens)),
        _tensor('row_max', (tokens, heads), 'fp32', output=True),
        _tensor('denominator', (tokens, heads), 'fp32', output=True)]) + f'''    query = lm.program(row_max, axis=0, dimension=0, tile=1)
    head = lm.program(row_max, axis=1, dimension=1, tile=1)
    with compute:
        keys = lm.coordinate(source="range", start=0, extent={extent}, id="keys")
        valid = lm.compare(keys, {tokens}, op="lt", id="valid_keys")
        stored = lm.load(scores[query, head, keys], id="load_scores")
        values = lm.cast(stored, to="fp32", id="widen_scores")
        masked = lm.select(valid, values, "negative_infinity", id="mask_tail")
        maximum = lm.reduce(masked, op="max", axis=0, scope="cta", across_loop=False, id="maximum")
        shifted = masked - maximum
        exponentials = lm.exp(shifted, id="exp")
        total = lm.reduce(exponentials, op="sum", axis=0, scope="cta", across_loop=False, id="denominator")
        lm.store(row_max[query, head], maximum, coalesced=False, id="store_max")
        lm.store(denominator[query, head], total, coalesced=False, id="store_denominator")
'''


def _values(tokens, heads, depth):
    loop = tokens > 128
    extent = 128 if loop else 1 << (tokens - 1).bit_length()
    inner = '        ' if loop else '    '
    source = _header('attention_values', [_tensor('scores', (tokens, heads, tokens)),
        _tensor('v', (tokens, heads, depth)), _tensor('row_max', (tokens, heads), 'fp32'),
        _tensor('denominator', (tokens, heads), 'fp32'), _tensor('attention', (tokens, heads, depth), output=True)])
    source += f'''    query = lm.program(attention, axis=0, dimension=0, tile=1)
    head = lm.program(attention, axis=1, dimension=1, tile=1)
    feature = lm.program(attention, axis=2, dimension=2, tile={min(8, depth)})
    with compute:
        maximum = lm.load(row_max[query, head], id="load_max")
        total = lm.load(denominator[query, head], id="load_denominator")
'''
    if loop:
        source += '    for key in lm.range(scores, name="keys", dimension=2, tile=128, num_stages=1, loop_unroll_factor=1):\n'
    coordinate = 'source="loop_tile", name="key"' if loop else f'source="range", start=0, extent={extent}'
    access = 'key' if loop else 'key_indices'
    source += inner + 'with compute:\n'
    source += '\n'.join(inner + '    ' + line for line in [
        f'key_indices = lm.coordinate({coordinate}, id="key_indices")',
        f'valid = lm.compare(key_indices, {tokens}, op="lt", id="valid_keys")',
        f'stored = lm.load(scores[query, head, {access}], id="load_scores")',
        'values = lm.cast(stored, to="fp32", id="widen_scores")',
        'masked = lm.select(valid, values, "negative_infinity", id="mask_tail")',
        'shifted = masked - maximum',
        'exponentials = lm.exp(shifted, id="exp")',
        'probability = exponentials / total',
        'probability_bf16 = lm.cast(probability, to="bf16", id="round_probability")',
        'probability_fp32 = lm.cast(probability_bf16, to="fp32", id="widen_probability")',
        f'v_stored = lm.load(v[{access}, head, feature], id="load_v")',
        'v_value = lm.cast(v_stored, to="fp32", id="widen_v")',
        'products = v_value * lm.broadcast(probability_fp32, axis=0)',
        f'weighted = lm.reduce(products, op="sum", axis=0, scope="cta", across_loop={loop}, id="weighted_sum")',
    ]) + '\n'
    source += '''    with compute:
        rounded = lm.cast(weighted, to="bf16", id="round_pv")
        lm.store(attention[query, head, feature], rounded, coalesced=False, id="store_attention")
'''
    return source


def _output(tokens, sequences, embed, heads, depth):
    head_loop, feature_loop = heads > 1, depth > 8
    outer = '    ' if head_loop else ''
    inner = outer + ('    ' if feature_loop else '')
    source = _header('output_projection', [_tensor('attention', (tokens, heads, depth)),
        _tensor('proj_weight', (embed, embed)), _tensor('proj_bias', (embed,)),
        _tensor('cu_seqlens', (sequences,), 'int64'), _tensor('output', (tokens, embed), output=True)])
    source += '''    token = lm.program(output, axis=0, dimension=0, tile=1)
    column = lm.program(output, axis=1, dimension=1, tile=1)
'''
    if head_loop:
        source += '    for head in lm.range(attention, name="heads", dimension=1, tile=1, num_stages=1, loop_unroll_factor=1):\n'
    head_coordinate = 'source="loop_tile", name="head"' if head_loop else 'source="range", start=0, extent=1'
    source += outer + f'    with compute:\n{outer}        head_index = lm.coordinate({head_coordinate}, id="head_index")\n'
    if feature_loop:
        source += outer + '    for feature in lm.range(attention, name="features", dimension=2, tile=8, num_stages=1, loop_unroll_factor=1):\n'
    feature_coordinate = 'source="loop_tile", name="feature"' if feature_loop else f'source="range", start=0, extent={depth}'
    head_access = 'head' if head_loop else '0:1'
    feature_access = 'feature' if feature_loop else ':'
    source += inner + '    with compute:\n'
    source += '\n'.join(inner + '        ' + line for line in [
        f'feature_indices = lm.coordinate({feature_coordinate}, id="feature_indices")',
        f'weight_columns = head_index * {depth} + feature_indices',
        f'stored = lm.load(attention[token, {head_access}, {feature_access}], id="load_attention")',
        'values = lm.cast(stored, to="fp32", id="widen_attention")',
        'weight_stored = lm.load(proj_weight[column, weight_columns], id="load_weight")',
        'weight = lm.cast(weight_stored, to="fp32", id="widen_weight")',
        'products = values * lm.broadcast(weight, axis=1)',
        f'per_head = lm.reduce(products, op="sum", axis=1, scope="cta", across_loop={feature_loop}, id="sum_features")',
    ]) + '\n'
    source += outer + f'    with compute:\n{outer}        projection = lm.reduce(per_head, op="sum", axis=0, scope="cta", across_loop={head_loop}, id="sum_heads")\n'
    source += f'''    with compute:
        bias_stored = lm.load(proj_bias[column], id="load_bias")
        bias = lm.cast(bias_stored, to="fp32", id="widen_bias")
        result = projection + bias
        last_endpoint = lm.load(cu_seqlens[{sequences - 1}:{sequences}], id="last_endpoint")
        nonempty = lm.compare(last_endpoint, 0, op="gt", id="nonempty")
        selected = lm.select(nonempty, result, 0.0, id="empty_sequences")
        rounded = lm.cast(selected, to="bf16", id="round_projection")
        lm.store(output[token, column], rounded, coalesced=False, id="store_output")
'''
    return source


def source_for(tokens: int, sequences: int, *, dimensions=DIMENSIONS) -> str:
    embed, heads, depth = dimensions
    if (any(type(n) is not int or n <= 0 for n in (tokens, sequences, embed, heads, depth))
            or embed != heads * depth or depth % 2 or depth % min(8, depth)):
        raise ValueError('Vision attention requires positive dimensions and complete even head features')
    stages = []
    for part, name in enumerate(('q0', 'k0', 'v')):
        stages.append(('project_' + name, _projection(tokens, embed, heads, depth, part, name),
                       ('hidden_states', 'qkv_weight', 'qkv_bias', name)))
    stages += [('rotary', _rotary(tokens, heads, depth), ('q0', 'k0', 'cos', 'sin', 'q', 'k')),
               ('segment_ids', _segments(tokens, sequences), ('cu_seqlens', 'segments')),
               ('attention_scores', _scores(tokens, heads, depth), ('q', 'k', 'segments', 'scores')),
               ('softmax_statistics', _statistics(tokens, heads), ('scores', 'row_max', 'denominator')),
               ('attention_values', _values(tokens, heads, depth), ('scores', 'v', 'row_max', 'denominator', 'attention')),
               ('output_projection', _output(tokens, sequences, embed, heads, depth), ('attention', 'proj_weight', 'proj_bias', 'cu_seqlens', 'output'))]
    declarations = '\n'.join(stage[1] for stage in stages)
    bindings = ',\n'.join(f'        cake.stage(name="{name}", schedule={name}, bindings={dict((n,n) for n in names)!r})'
                           for name, _, names in stages)
    return ('from open_cake_ir.compiler import frontend as cake\n\n' + declarations
        + f'\ncake.program(program_id="varlen_vision_t{tokens}_s{sequences}", '
        'inputs=("hidden_states", "cu_seqlens", "cos", "sin", "qkv_weight", "qkv_bias", "proj_weight", "proj_bias"), '
        'outputs=("output",), stages=(\n' + bindings + ',\n))\n')


def program_for(tokens, sequences, *, dimensions=DIMENSIONS):
    from open_cake_ir.compiler.program_frontend import parse_program
    return parse_program(source_for(tokens, sequences, dimensions=dimensions)).program


def source_for_workload(workload, case_id='primary'):
    abi = workload.tensor_abi(case_id)
    if workload.target != TARGET or len(abi) != 9 or len(abi[0].shape) != 2 or len(abi[1].shape) != 1:
        raise ValueError('Variable-length vision attention original target or public ABI differs')
    tokens, embed = abi[0].shape
    sequences, = abi[1].shape
    e, h, d = DIMENSIONS
    expected = [('hidden_states', (tokens,e), 'bf16','input'), ('cu_seqlens',(sequences,),'int64','input'),
                ('cos',(tokens,h,d),'bf16','input'), ('sin',(tokens,h,d),'bf16','input'),
                ('qkv_weight',(3*e,e),'bf16','input'), ('qkv_bias',(3*e,),'bf16','input'),
                ('proj_weight',(e,e),'bf16','input'), ('proj_bias',(e,),'bf16','input'),
                ('output',(tokens,e),'bf16','output')]
    if [(a.name, tuple(a.shape), a.dtype, a.mode) for a in abi] != expected:
        raise ValueError('Variable-length vision attention changed an original tensor shape, dtype or order')
    return source_for(tokens, sequences)
