"""Reference-shaped chunk delta source with explicit immutable state versions.

This source deliberately recomputes a resident triangular transform per output
column. It is a correctness-first starting design, not a tuned kernel or a
replacement of the reference by a standard token-level delta recurrence.
"""
from __future__ import annotations

import math

TASK = 'L2/060_chunk_gated_delta_rule_linear_attention'
TARGET = 'xcore1002'
KEY_HEADS = 4
VALUE_HEADS = 16
WIDTH = 128
CHUNK = 64
COLUMN_TILE = 32
EPSILON = 1e-6


class _Author:
    def __init__(self, batch, sequence, heads_k, heads_v, width):
        self.inputs = ('query', 'key', 'value', 'g', 'beta')
        self.outputs = ('output',)
        self.tensors = {
            'query': ((batch, heads_k, sequence, width), 'bf16'),
            'key': ((batch, heads_k, sequence, width), 'bf16'),
            'value': ((batch, heads_v, sequence, width), 'bf16'),
            'g': ((batch, heads_v, sequence), 'bf16'),
            'beta': ((batch, heads_v, sequence), 'bf16'),
            'output': ((batch, sequence, heads_v, width), 'bf16'),
        }
        self.name = f'chunk_delta_b{batch}_s{sequence}'
        self.sources = []
        self.stages = []

    def tensor(self, name, shape, dtype='fp32'):
        if name in self.tensors:
            raise ValueError('A chunk-delta tensor has one declaration')
        self.tensors[name] = (tuple(shape), dtype)

    def stage(self, name, inputs, outputs, body):
        declarations = []
        names = (*inputs, *outputs)
        for mode, fields in (('input', inputs), ('output', outputs)):
            for field in fields:
                shape, dtype = self.tensors[field]
                declarations.append(f'{field}: cake.Tensor({shape!r}, "{dtype}", mode="{mode}")')
        self.sources.append(f'@cake.schedule(name="{name}", target="{TARGET}", backend="triton", entry_point="cake_{name}")\n'
                            f'def {name}(lm, ' + ',\n        '.join(declarations) + '):\n'
                            '    compute = lm.role(execution_groups=[0, 1, 2, 3])\n'
                            + '\n'.join('    ' + line for line in body.strip().splitlines()) + '\n')
        self.stages.append(f'cake.stage(name="{name}", schedule={name}, bindings={dict(zip(names,names))!r})')

    def render(self):
        return ('from open_cake_ir.compiler import frontend as cake\n\n' + '\n'.join(self.sources)
                + f'\ncake.program(program_id="{self.name}", inputs={self.inputs!r}, outputs={self.outputs!r}, stages=(\n    '
                + ',\n    '.join(self.stages) + ',\n))\n')


def _normalization(author, batch, sequence, heads_k, width, scale):
    author.tensor('q_normalized', (batch, heads_k, sequence, width))
    author.tensor('k_normalized', (batch, heads_k, sequence, width))
    body = '''batch = lm.program(q_normalized, axis=0, dimension=0, tile=1)
head = lm.program(q_normalized, axis=1, dimension=1, tile=1)
position = lm.program(q_normalized, axis=2, dimension=2, tile=1)
with compute:
'''
    for name, out in (('query', 'q_normalized'), ('key', 'k_normalized')):
        body += f'''    {name}_stored = lm.load({name}[batch, head, position, :], id="load_{name}")
    {name}_fp32 = lm.cast({name}_stored, to="fp32", id="widen_{name}")
    {name}_squares_fp32 = {name}_fp32 * {name}_fp32
    {name}_squares_bf16 = lm.cast({name}_squares_fp32, to="bf16", id="round_{name}_square")
    {name}_squares = lm.cast({name}_squares_bf16, to="fp32", id="widen_{name}_square")
    {name}_sum = lm.reduce({name}_squares, op="sum", axis=0, across_loop=False, id="sum_{name}_square")
    {name}_sum_bf16 = lm.cast({name}_sum, to="bf16", id="round_{name}_sum")
    {name}_sum_fp32 = lm.cast({name}_sum_bf16, to="fp32", id="widen_{name}_sum")
    {name}_epsilon = {name}_sum_fp32 + {EPSILON!r}
    {name}_epsilon_bf16 = lm.cast({name}_epsilon, to="bf16", id="round_{name}_epsilon")
    {name}_epsilon_fp32 = lm.cast({name}_epsilon_bf16, to="fp32", id="widen_{name}_epsilon")
    {name}_inverse = lm.rsqrt({name}_epsilon_fp32, id="rsqrt_{name}")
    {name}_inverse_bf16 = lm.cast({name}_inverse, to="bf16", id="round_{name}_inverse")
    {name}_inverse_fp32 = lm.cast({name}_inverse_bf16, to="fp32", id="widen_{name}_inverse")
    {name}_normalized = {name}_fp32 * {name}_inverse_fp32
    {name}_normalized_bf16 = lm.cast({name}_normalized, to="bf16", id="round_{name}_normalization")
    {name}_normalized_fp32 = lm.cast({name}_normalized_bf16, to="fp32", id="widen_{name}_normalization")
'''
        if name == 'query':
            body += f'    query_scaled = query_normalized_fp32 * {scale!r}\n'
        result = 'query_scaled' if name == 'query' else 'key_normalized_fp32'
        body += f'    lm.store({out}[batch, head, position, :], {result}, coalesced=False, id="store_{out}")\n'
    author.stage('normalize_qk', ('query','key'), ('q_normalized','k_normalized'), body)


def _prefix(author, batch, heads_v, padded, chunk):
    author.tensor('prefix_g', (batch, heads_v, padded))
    author.stage('prefix_decay', ('g',), ('prefix_g',), f'''batch = lm.program(prefix_g, axis=0, dimension=0, tile=1)
head = lm.program(prefix_g, axis=1, dimension=1, tile=1)
position = lm.program(prefix_g, axis=2, dimension=2, tile=1)
with compute:
    current = lm.coordinate(source="program", name="position", id="current")
    chunk_base = (current // {chunk}) * {chunk}
    in_chunk = current % {chunk}
    lanes = lm.coordinate(source="range", start=0, extent={chunk}, id="lanes")
    positions = chunk_base + lanes
    stored = lm.load(g[batch, head, positions], id="load_g")
    values = lm.cast(stored, to="fp32", id="widen_g")
    previous = lm.compare(lanes, in_chunk, op="le", id="prefix_domain")
    selected = lm.select(previous, values, 0.0, id="masked_prefix")
    total = lm.reduce(selected, op="sum", axis=0, across_loop=False, id="prefix_sum")
    lm.store(prefix_g[batch, head, position], total, coalesced=False, id="store_prefix")
''')


def _transform(author, batch, heads_k, heads_v, width, chunks, chunk):
    author.tensor('u_transposed', (batch, heads_v, chunks * width, chunk))
    author.tensor('w_transposed', (batch, heads_v, chunks * width, chunk))
    body = f'''batch = lm.program(u_transposed, axis=0, dimension=0, tile=1)
head = lm.program(u_transposed, axis=1, dimension=1, tile=1)
column = lm.program(u_transposed, axis=2, dimension=2, tile=1)
with compute:
    head_index = lm.coordinate(source="program", name="head", id="head_index")
    key_head = head_index % {heads_k}
    column_index = lm.coordinate(source="program", name="column", id="column_index")
    chunk_base = (column_index // {width}) * {chunk}
    component = column_index % {width}
    lanes = lm.coordinate(source="range", start=0, extent={chunk}, id="lanes")
    positions = chunk_base + lanes
    keys = lm.load(k_normalized[batch, lm.scalar_index(key_head), positions, :], id="load_keys")
    beta_stored = lm.load(beta[batch, head, positions], id="load_beta")
    betas = lm.cast(beta_stored, to="fp32", id="widen_beta")
    weighted_keys = keys * lm.broadcast(betas, axis=0)
    gram = lm.mma(weighted_keys, keys, instruction={{"contract":"triton.dot.fp32_ieee"}}, tile_shape=({chunk},{chunk},{width}), id="key_gram")
    same = lm.compare(gram, gram, op="eq", id="matrix_shape")
    zero_int = same * 0
    zero_float = lm.cast(zero_int, to="fp32", id="matrix_zero")
    rows = zero_int + lm.broadcast(lanes, axis=0)
    columns = zero_int + lm.broadcast(lanes, axis=1)
    lower = lm.compare(rows, columns, op="gt", id="strict_lower")
    diagonal = lm.compare(rows, columns, op="eq", id="diagonal")
    cumulative = lm.load(prefix_g[batch, head, positions], id="load_prefix")
    cumulative_rows = zero_float + lm.broadcast(cumulative, axis=0)
    cumulative_columns = zero_float + lm.broadcast(cumulative, axis=1)
    decay = lm.exp(cumulative_rows - cumulative_columns, id="chunk_decay_mask")
    negative_gram = gram * -1.0
    decayed_gram = negative_gram * decay
    triangular_0 = lm.select(lower, decayed_gram, 0.0, id="triangular_base")
'''
    for index in range(1, chunk):
        body += f'''    row_mask_{index} = lm.compare(rows, {index}, op="eq", id="row_mask_{index}")
    before_rows_{index} = lm.compare(rows, {index}, op="lt", id="before_rows_{index}")
    before_columns_{index} = lm.compare(columns, {index}, op="lt", id="before_columns_{index}")
    row_selected_{index} = lm.select(row_mask_{index}, triangular_{index-1}, 0.0, id="row_selected_{index}")
    row_{index} = lm.reduce(row_selected_{index}, op="sum", axis=0, across_loop=False, id="extract_row_{index}")
    sub_domain_{index} = before_rows_{index} * before_columns_{index}
    sub_{index} = lm.select(sub_domain_{index}, triangular_{index-1}, 0.0, id="submatrix_{index}")
    products_{index} = sub_{index} * lm.broadcast(row_{index}, axis=0)
    correction_{index} = lm.reduce(products_{index}, op="sum", axis=0, across_loop=False, id="row_correction_{index}")
    row_updated_{index} = row_{index} + correction_{index}
    row_matrix_{index} = zero_float + lm.broadcast(row_updated_{index}, axis=1)
    update_domain_{index} = row_mask_{index} * before_columns_{index}
    triangular_{index} = lm.select(update_domain_{index}, row_matrix_{index}, triangular_{index-1}, id="update_row_{index}")
'''
    body += f'''    identity = lm.cast(diagonal, to="fp32", id="identity")
    transform = triangular_{chunk-1} + identity
    values_stored = lm.load(value[batch, head, positions, lm.scalar_index(component)], id="load_value_column")
    values = lm.cast(values_stored, to="fp32", id="widen_value_column")
    values_beta = values * betas
    u_products = transform * lm.broadcast(values_beta, axis=1)
    u = lm.reduce(u_products, op="sum", axis=1, across_loop=False, id="transformed_value")
    lm.store(u_transposed[batch, head, column, :], u, coalesced=False, id="store_u")
'''
    body += '''    g_stored = lm.load(g[batch, head, positions], id="load_raw_g")
    raw_g = lm.cast(g_stored, to="fp32", id="widen_raw_g")
    raw_decay = lm.exp(raw_g, id="raw_gate_exp")
    key_column = lm.load(k_normalized[batch, lm.scalar_index(key_head), positions, lm.scalar_index(component)], id="load_key_column")
    key_beta = key_column * betas
    decayed_key = key_beta * raw_decay
    w_products = transform * lm.broadcast(decayed_key, axis=1)
    w = lm.reduce(w_products, op="sum", axis=1, across_loop=False, id="transformed_key")
    lm.store(w_transposed[batch, head, column, :], w, coalesced=False, id="store_w")
'''
    inputs = ('k_normalized','beta','prefix_g','value','g')
    outputs = ('u_transposed','w_transposed')
    author.stage('transformed_columns', inputs, outputs, body)


def _columns(output, width, tile):
    if width > tile:
        return (f'for column in lm.range({output}, name="value_columns", dimension=3, tile={tile}, num_stages=1, loop_unroll_factor=1):\n'
                '    with compute:\n', '        ', 'column',
                'lm.coordinate(source="loop_tile", name="column", id="column_indices")')
    return ('with compute:\n', '    ', ':',
            f'lm.coordinate(source="range", start=0, extent={width}, id="column_indices")')


def _correction(author, batch, heads_v, width, chunk, index, column_tile):
    output = f'v_new_{index}'
    author.tensor(output, (batch, heads_v, chunk, width))
    state = f'state_{index}'
    body = f'''batch = lm.program({output}, axis=0, dimension=0, tile=1)
head = lm.program({output}, axis=1, dimension=1, tile=1)
position = lm.program({output}, axis=2, dimension=2, tile=1)
with compute:
    dimensions = lm.coordinate(source="range", start=0, extent={width}, id="key_dimensions")
    w_columns = dimensions + {index * width}
    w = lm.load(w_transposed[batch, head, w_columns, position], id="load_w")
'''
    if index == 0:
        body += '''    initial_products = w * 0.0
    initial_correction = lm.reduce(initial_products, op="sum", axis=0, across_loop=False, id="initial_state_dot")
'''
    prefix, pad, column, coordinate = _columns(output, width, column_tile)
    body += prefix
    lines = [f'column_indices = {coordinate}', f'u_columns = column_indices + {index * width}',
             'u = lm.load(u_transposed[batch, head, u_columns, position], id="load_u")']
    if index:
        lines += [f'state_values = lm.load({state}[batch, head, :, {column}], id="load_state")',
                  'products = state_values * lm.broadcast(w, axis=0)',
                  'correction = lm.reduce(products, op="sum", axis=0, across_loop=False, id="state_correction")',
                  'corrected = u - correction']
    else:
        lines += ['corrected = u - initial_correction']
    lines += [f'lm.store({output}[batch, head, position, {column}], corrected, coalesced=False, id="store_corrected")']
    body += '\n'.join(pad + line for line in lines)
    author.stage(f'correct_chunk_{index}', ('u_transposed','w_transposed') + ((state,) if index else ()),
                 (output,), body)


def _chunk_output(author, batch, heads_k, heads_v, width, chunk, index, column_tile):
    output, values, state = f'out_chunk_{index}', f'v_new_{index}', f'state_{index}'
    author.tensor(output, (batch, heads_v, chunk, width))
    body = f'''batch = lm.program({output}, axis=0, dimension=0, tile=1)
head = lm.program({output}, axis=1, dimension=1, tile=1)
position = lm.program({output}, axis=2, dimension=2, tile=1)
with compute:
    head_index = lm.coordinate(source="program", name="head", id="head_index")
    key_head = head_index % {heads_k}
    row = lm.coordinate(source="program", name="position", id="row")
    global_row = row + {index * chunk}
    lanes = lm.coordinate(source="range", start=0, extent={chunk}, id="lanes")
    positions = lanes + {index * chunk}
    q = lm.load(q_normalized[batch, lm.scalar_index(key_head), lm.scalar_index(global_row), :], id="load_q")
    keys = lm.load(k_normalized[batch, lm.scalar_index(key_head), positions, :], id="load_k")
    products_qk = keys * lm.broadcast(q, axis=1)
    dot = lm.reduce(products_qk, op="sum", axis=1, across_loop=False, id="qk_dot")
    row_prefix = lm.load(prefix_g[batch, head, lm.scalar_index(global_row)], id="load_row_prefix")
    prefixes = lm.load(prefix_g[batch, head, positions], id="load_prefixes")
    causal_decay = lm.exp(row_prefix - prefixes, id="causal_decay")
    weighted = dot * causal_decay
    causal = lm.compare(lanes, row, op="le", id="causal_domain")
    attention = lm.select(causal, weighted, 0.0, id="masked_attention")
    g_stored = lm.load(g[batch, head, lm.scalar_index(global_row)], id="load_raw_gate")
    raw_g = lm.cast(g_stored, to="fp32", id="widen_gate")
    decay = lm.exp(raw_g, id="raw_gate_exp")
    inter_q = q * decay
'''
    if index == 0:
        body += '''    initial_inter_products = inter_q * 0.0
    initial_inter = lm.reduce(initial_inter_products, op="sum", axis=0, across_loop=False, id="initial_inter_attention")
'''
    prefix, pad, column, _ = _columns(output, width, column_tile)
    body += prefix
    lines = [f'chunk_values = lm.load({values}[batch, head, :, {column}], id="load_new_values")',
             'intra_products = chunk_values * lm.broadcast(attention, axis=0)',
             'intra = lm.reduce(intra_products, op="sum", axis=0, across_loop=False, id="intra_attention")']
    if index:
        lines += [f'state_values = lm.load({state}[batch, head, :, {column}], id="load_state")',
                  'inter_products = state_values * lm.broadcast(inter_q, axis=0)',
                  'inter = lm.reduce(inter_products, op="sum", axis=0, across_loop=False, id="inter_attention")',
                  'combined = inter + intra']
    else:
        lines += ['combined = initial_inter + intra']
    lines += [f'lm.store({output}[batch, head, position, {column}], combined, coalesced=False, id="store_chunk_output")']
    body += '\n'.join(pad + line for line in lines)
    author.stage(f'output_chunk_{index}', ('q_normalized','k_normalized','prefix_g','g',values)
                 + ((state,) if index else ()), (output,), body)


def _state_update(author, batch, heads_k, heads_v, width, chunk, index, column_tile):
    output, values, previous = f'state_{index+1}', f'v_new_{index}', f'state_{index}'
    author.tensor(output, (batch, heads_v, width, width))
    body = f'''batch = lm.program({output}, axis=0, dimension=0, tile=1)
head = lm.program({output}, axis=1, dimension=1, tile=1)
key_dimension = lm.program({output}, axis=2, dimension=2, tile=1)
with compute:
    head_index = lm.coordinate(source="program", name="head", id="head_index")
    key_head = head_index % {heads_k}
    lanes = lm.coordinate(source="range", start=0, extent={chunk}, id="lanes")
    positions = lanes + {index * chunk}
    last = lm.coordinate(source="range", start={(index+1)*chunk-1}, extent=1, id="last_position")
    g_stored = lm.load(g[batch, head, positions], id="load_raw_g")
    raw_g = lm.cast(g_stored, to="fp32", id="widen_g")
    last_stored = lm.load(g[batch, head, lm.scalar_index(last)], id="load_last_raw_g")
    last_g = lm.cast(last_stored, to="fp32", id="widen_last_g")
    chunk_decay = lm.exp(last_g, id="state_decay")
    token_decay = lm.exp(last_g - raw_g, id="token_decay")
    keys = lm.load(k_normalized[batch, lm.scalar_index(key_head), positions, key_dimension], id="load_key_column")
    decayed_keys = keys * token_decay
'''
    if index == 0:
        body += '    initial_state = chunk_decay * 0.0\n'
    prefix, pad, column, _ = _columns(output, width, column_tile)
    body += prefix
    lines = [f'new_values = lm.load({values}[batch, head, :, {column}], id="load_new_values")',
             'products = new_values * lm.broadcast(decayed_keys, axis=0)',
             'addition = lm.reduce(products, op="sum", axis=0, across_loop=False, id="state_addition")']
    if index:
        lines += [f'old = lm.load({previous}[batch, head, key_dimension, {column}], id="load_old_state")',
                  'decayed = old * chunk_decay', 'updated = decayed + addition']
    else:
        lines += ['updated = initial_state + addition']
    lines += [f'lm.store({output}[batch, head, key_dimension, {column}], updated, coalesced=False, id="store_state")']
    body += '\n'.join(pad + line for line in lines)
    author.stage(f'update_state_{index+1}', ('k_normalized','g',values) + ((previous,) if index else ()),
                 (output,), body)


def _pack(author, chunks, chunk):
    body = '''batch = lm.program(output, axis=0, dimension=0, tile=1)
position = lm.program(output, axis=1, dimension=1, tile=1)
head = lm.program(output, axis=2, dimension=2, tile=1)
with compute:
    row = lm.coordinate(source="program", name="position", id="row")
'''
    for index in range(chunks):
        body += f'''    local_{index} = row - {index * chunk}
    piece_{index} = lm.load(out_chunk_{index}[batch, head, lm.scalar_index(local_{index}), :], id="load_chunk_{index}")
'''
    if chunks == 1:
        final = 'piece_0'
    else:
        body += '    combined = ' + ' + '.join(f'piece_{index}' for index in range(chunks)) + '\n'
        final = 'combined'
    body += f'''    rounded = lm.cast({final}, to="bf16", id="final_rounding")
    lm.store(output[batch, position, head, :], rounded, coalesced=False, id="store_output")
'''
    author.stage('pack_output', tuple(f'out_chunk_{index}' for index in range(chunks)), ('output',), body)


def _source(batch, sequence, scale, *, heads_k, heads_v, width, chunk, column_tile):
    chunks = (sequence + chunk - 1) // chunk
    author = _Author(batch, sequence, heads_k, heads_v, width)
    _normalization(author, batch, sequence, heads_k, width, scale)
    _prefix(author, batch, heads_v, chunks * chunk, chunk)
    _transform(author, batch, heads_k, heads_v, width, chunks, chunk)
    for index in range(chunks):
        _correction(author, batch, heads_v, width, chunk, index, column_tile)
        _chunk_output(author, batch, heads_k, heads_v, width, chunk, index, column_tile)
        if index + 1 < chunks:
            _state_update(author, batch, heads_k, heads_v, width, chunk, index, column_tile)
    _pack(author, chunks, chunk)
    return author.render()


def source_for(batch_size: int, seq_len: int, *, scale: float = 1 / math.sqrt(WIDTH)) -> str:
    if any(type(value) is not int or value <= 0 for value in (batch_size, seq_len)):
        raise ValueError('Chunk delta requires positive original batch and sequence dimensions')
    if type(scale) is not float or not math.isfinite(scale):
        raise ValueError('Chunk delta requires its checked finite FP32 scale specialization')
    return _source(batch_size, seq_len, scale, heads_k=KEY_HEADS, heads_v=VALUE_HEADS,
                   width=WIDTH, chunk=CHUNK, column_tile=COLUMN_TILE)


def program_for(batch_size: int, seq_len: int, *, scale: float = 1 / math.sqrt(WIDTH)):
    from open_cake_ir.compiler.program_frontend import parse_program
    return parse_program(source_for(batch_size, seq_len, scale=scale)).program


def source_for_workload(workload, case_id: str) -> str:
    abi = workload.tensor_abi(case_id)
    if workload.target != TARGET or len(abi) != 6 or len(abi[0].shape) != 4:
        raise ValueError('Chunk delta target or original public ABI differs')
    batch, _, sequence, _ = abi[0].shape
    qshape = (batch, KEY_HEADS, sequence, WIDTH)
    vshape = (batch, VALUE_HEADS, sequence, WIDTH)
    expected = (('query', qshape, 'bf16', 'input'), ('key', qshape, 'bf16', 'input'),
                ('value', vshape, 'bf16', 'input'), ('g', vshape[:3], 'bf16', 'input'),
                ('beta', vshape[:3], 'bf16', 'input'),
                ('output', (batch, sequence, VALUE_HEADS, WIDTH), 'bf16', 'output'))
    if tuple((arg.name, tuple(arg.shape), arg.dtype, arg.mode) for arg in abi) != expected:
        raise ValueError('Chunk delta original ordered tensor ABI differs')
    scalars = workload.document['semantics'].get('fixed_scalar_inputs')
    if (not isinstance(scalars, dict) or set(scalars) != {'scale'}
            or not isinstance(scalars['scale'], dict)
            or set(scalars['scale']) != {'dtype','value','binding'}
            or scalars['scale']['dtype'] != 'float32'
            or scalars['scale']['binding'] != 'literal_input'):
        raise ValueError('Chunk delta requires the original scalar scale binding')
    return source_for(batch, sequence, scale=scalars['scale']['value'])
