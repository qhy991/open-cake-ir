"""Static CAKE stage builders for the original Decoder backward arithmetic."""
from __future__ import annotations

TARGET = 'xcore1002'


def tensor(name, shape, dtype='bf16', output=False):
    return f'{name}: cake.Tensor({shape!r}, "{dtype}"' + (', mode="output")' if output else ')')


def header(name, arguments):
    return (f'@cake.schedule(name="{name}", target="{TARGET}", backend="triton", entry_point="cake_{name}")\n'
            f'def {name}(lm, ' + ', '.join(arguments) + '):\n'
            '    compute = lm.role(execution_groups=[0, 1, 2, 3])\n')


def round_bf16(value, rounded, indent='        '):
    return f'{indent}{rounded} = lm.cast({value}, to="bf16", id="{rounded}")\n'


def linear_loops(paths):
    """Each (input, weight, suffix) is a complete BF16 output boundary."""
    source = ''
    for x, weight, suffix in paths:
        source += f'''    for k_{suffix} in lm.range({x}, name="k_{suffix}", dimension=2, tile=32, num_stages=1, loop_unroll_factor=1):
        with compute:
            x_{suffix} = lm.load({x}[batch, row, k_{suffix}], id="load_x_{suffix}")
            w_{suffix} = lm.load({weight}[k_{suffix}, column], id="load_w_{suffix}")
            wt_{suffix} = lm.transpose(w_{suffix}, id="transpose_w_{suffix}")
            dot_{suffix} = lm.mma(x_{suffix}, wt_{suffix}, instruction={{"contract":"triton.dot.bf16_fp32"}}, tile_shape=(16,32,32), id="dot_{suffix}")
    with compute:
        rounded_{suffix} = lm.cast(dot_{suffix}, to="bf16", id="rounded_{suffix}")
'''
    return source


def linear_grid(output):
    return f'''    row = lm.program({output}, axis=0, dimension=1, tile=16)
    column = lm.program({output}, axis=1, dimension=2, tile=32)
    batch = lm.program({output}, axis=2, dimension=0, tile=1)
'''


def linear_paths(name, b, s, output_width, paths, output):
    """Add rounded BF16 linear results in the original left-associated order."""
    arguments=[]
    for x,w,k in paths:
        arguments.extend((tensor(x,(b,s,k)),tensor(w,(k,output_width))))
    arguments.append(tensor(output,(b,s,output_width),output=True))
    source=header(name,arguments)+linear_grid(output)
    source+=linear_loops([(x,w,str(i)) for i,(x,w,_) in enumerate(paths)])
    source+='    with compute:\n'
    previous='rounded_0'
    for i in range(1,len(paths)):
        source+=f'''        prior_{i} = lm.cast({previous}, to="fp32", id="prior_{i}")
        next_{i} = lm.cast(rounded_{i}, to="fp32", id="next_{i}")
        sum_{i} = prior_{i} + next_{i}
        rounded_sum_{i} = lm.cast(sum_{i}, to="bf16", id="rounded_sum_{i}")
'''
        previous=f'rounded_sum_{i}'
    source+=f'        lm.store({output}[batch, row, column], {previous}, id="store_output")\n'
    return source


def swiglu_backward(b,s,h,i,unused):
    arguments=[tensor('grad_output',(b,s,h)),tensor('down_weight',(h,i))]
    arguments.extend(tensor(n,(b,s,i)) for n in ('gate','up','silu_up'))
    arguments.extend(tensor(n,shape) for n,shape in unused.items())
    arguments.extend(tensor(n,(b,s,i),output=True) for n in ('grad_gate','grad_up'))
    source=header('swiglu_backward',arguments)+linear_grid('grad_gate')
    source+=linear_loops([('grad_output','down_weight','swiglu')])
    source+='''    with compute:
        grad = lm.cast(rounded_swiglu, to="fp32", id="widen_grad")
        gate_stored = lm.load(gate[batch, row, column], id="load_gate")
        up_stored = lm.load(up[batch, row, column], id="load_up")
        silu_stored = lm.load(silu_up[batch, row, column], id="load_silu")
        gate_value = lm.cast(gate_stored, to="fp32", id="widen_gate")
        up_value = lm.cast(up_stored, to="fp32", id="widen_up")
        silu_value = lm.cast(silu_stored, to="fp32", id="widen_silu")
        negative = up_value * -1.0
        exponential = lm.exp(negative, id="exp")
        sigmoid = lm.reciprocal(exponential + 1.0, id="sigmoid")
        complement = sigmoid * -1.0 + 1.0
        derivative = sigmoid * (up_value * complement + 1.0)
        derivative_bf16 = lm.cast(derivative, to="bf16", id="round_derivative")
        derivative_fp32 = lm.cast(derivative_bf16, to="fp32", id="widen_derivative")
        gate_product = grad * silu_value
        gate_gradient = lm.cast(gate_product, to="bf16", id="round_gate_gradient")
        first_product = grad * gate_value
        first_bf16 = lm.cast(first_product, to="bf16", id="round_first_product")
        first_fp32 = lm.cast(first_bf16, to="fp32", id="widen_first_product")
        second_product = first_fp32 * derivative_fp32
        up_gradient = lm.cast(second_product, to="bf16", id="round_up_gradient")
        lm.store(grad_gate[batch, row, column], gate_gradient, id="store_gate")
        lm.store(grad_up[batch, row, column], up_gradient, id="store_up")
'''
    return source


def rms_input(name,b,s,h,eps,gradient,residual,variance,weight,residual_gradient,output):
    arguments=[tensor(gradient,(b,s,h)),tensor(residual,(b,s,h)),
               tensor(variance,(b,s,1),'fp32'),tensor(weight,(h,))]
    if residual_gradient != gradient:
        arguments.append(tensor(residual_gradient,(b,s,h)))
    arguments.append(tensor(output,(b,s,h),output=True))
    return header(name,arguments)+f'''    row = lm.program({output}, axis=0, dimension=1, tile=1)
    batch = lm.program({output}, axis=1, dimension=0, tile=1)
    column = lm.program({output}, axis=2, dimension=2, tile={1 << (h-1).bit_length()})
    with compute:
        grad_stored = lm.load({gradient}[batch, row, column], id="load_grad")
        residual_stored = lm.load({residual}[batch, row, column], id="load_residual")
        weight_stored = lm.load({weight}[column], id="load_weight")
        var = lm.load({variance}[batch, row, :], id="load_variance")
        grad = lm.cast(grad_stored, to="fp32", id="widen_grad")
        value = lm.cast(residual_stored, to="fp32", id="widen_residual")
        weight_fp32 = lm.cast(weight_stored, to="fp32", id="widen_weight")
        inverse = lm.rsqrt(var + {eps!r}, id="inverse")
        normalized_gradient = grad * weight_fp32
        base = normalized_gradient * inverse
        products = normalized_gradient * value
        total = lm.reduce(products, op="sum", axis=0, scope="cta", across_loop=False, id="total")
        inverse_squared = lm.square(inverse, id="inverse_squared")
        inverse_cubed = inverse_squared * inverse
        grad_variance = total * -0.5 * inverse_cubed
        correction = value * {2.0/h!r} * grad_variance
        complete = base + correction
        hidden_bf16 = lm.cast(complete, to="bf16", id="round_hidden")
        hidden_fp32 = lm.cast(hidden_bf16, to="fp32", id="widen_hidden")
        original_gradient = lm.load({residual_gradient}[batch, row, column], id="load_residual_gradient")
        original_fp32 = lm.cast(original_gradient, to="fp32", id="widen_residual_gradient")
        with_residual = original_fp32 + hidden_fp32
        rounded = lm.cast(with_residual, to="bf16", id="round_output")
        lm.store({output}[batch, row, column], rounded, id="store_output")
'''


def rms_weight(name,b,s,h,gradient,normalized,output):
    source=header(name,[tensor(gradient,(b,s,h)),tensor(normalized,(b,s,h),'fp32'),tensor(output,(h,),'fp32',True)])
    source+=f'''    channel = lm.program({output}, axis=0, dimension=0, tile=32)
    with compute:
        group = lm.coordinate(source="program", name="channel", id="group")
        zero = group * 0
'''
    for batch in range(b):
        source+=f'''    with compute:
        batch_{batch} = zero + {batch}
    for row_{batch} in lm.range({gradient}, name="rows_{batch}", dimension=1, tile=32, num_stages=1, loop_unroll_factor=1):
        with compute:
            stored_{batch} = lm.load({gradient}[lm.scalar_index(batch_{batch}), row_{batch}, channel], id="load_grad_{batch}")
            grad_{batch} = lm.cast(stored_{batch}, to="fp32", id="widen_grad_{batch}")
            value_{batch} = lm.load({normalized}[lm.scalar_index(batch_{batch}), row_{batch}, channel], id="load_normalized_{batch}")
            products_{batch} = grad_{batch} * value_{batch}
            total_{batch} = lm.reduce(products_{batch}, op="sum", axis=0, scope="cta", across_loop=True, id="total_{batch}")
'''
        if batch:
            prior='total_0' if batch==1 else f'accumulated_{batch-1}'
            source+=f'    with compute:\n        accumulated_{batch} = {prior} + total_{batch}\n'
    result='total_0' if b==1 else f'accumulated_{b-1}'
    return source+f'    with compute:\n        lm.store({output}[channel], {result}, id="store_output")\n'


def softmax_backward(b,s,heads,depth):
    return header('softmax_backward',[tensor('grad_attn_weights',(b*heads,s,s)),
        tensor('attn_weights',(b,heads,s,s)),tensor('grad_attn_logits',(b*heads,s,s),output=True)])+f'''    query = lm.program(grad_attn_logits, axis=0, dimension=1, tile=1)
    key = lm.program(grad_attn_logits, axis=1, dimension=2, tile=32)
    group = lm.program(grad_attn_logits, axis=2, dimension=0, tile=1)
    with compute:
        group_index = lm.coordinate(source="program", name="group", id="group_index")
        batch_index = group_index // {heads}
        head_index = group_index % {heads}
        keys = lm.coordinate(source="range", start=0, extent={1 << (s-1).bit_length()}, id="keys")
        grad_stored = lm.load(grad_attn_weights[group, query, keys], id="load_grad")
        weight_stored = lm.load(attn_weights[lm.scalar_index(batch_index), lm.scalar_index(head_index), query, keys], id="load_weights")
        grad = lm.cast(grad_stored, to="fp32", id="widen_grad")
        weights = lm.cast(weight_stored, to="fp32", id="widen_weights")
        products = grad * weights
        total = lm.reduce(products, op="sum", axis=0, scope="cta", across_loop=False, id="total")
        tile_grad_stored = lm.load(grad_attn_weights[group, query, key], id="load_tile_grad")
        tile_weight_stored = lm.load(attn_weights[lm.scalar_index(batch_index), lm.scalar_index(head_index), query, key], id="load_tile_weight")
        tile_grad = lm.cast(tile_grad_stored, to="fp32", id="widen_tile_grad")
        tile_weight = lm.cast(tile_weight_stored, to="fp32", id="widen_tile_weight")
        centered = tile_grad - total
        weighted = tile_weight * centered
        scaled = weighted / {depth ** 0.5!r}
        rounded = lm.cast(scaled, to="bf16", id="round_output")
        lm.store(grad_attn_logits[group, query, key], rounded, id="store_output")
'''


def weight_gradient(name,b,s,m,n,left,right,output):
    """Fixed batches accumulate as FP32 tiles; no per-batch global matrices."""
    source=header(name,[tensor(left,(b,s,m)),tensor(right,(b,s,n)),tensor(output,(m,n),output=True)])
    source+=f'''    row = lm.program({output}, axis=0, dimension=0, tile=16)
    column = lm.program({output}, axis=1, dimension=1, tile=32)
    with compute:
        row_index = lm.coordinate(source="program", name="row", id="row_index")
        zero = row_index * 0
'''
    for batch in range(b):
        source+=f'''    with compute:
        batch_{batch} = zero + {batch}
    for sequence_{batch} in lm.range({left}, name="sequence_{batch}", dimension=1, tile=32, num_stages=1, loop_unroll_factor=1):
        with compute:
            left_{batch} = lm.load({left}[lm.scalar_index(batch_{batch}), sequence_{batch}, row], id="load_left_{batch}")
            right_{batch} = lm.load({right}[lm.scalar_index(batch_{batch}), sequence_{batch}, column], id="load_right_{batch}")
            a_{batch} = lm.transpose(left_{batch}, id="transpose_left_{batch}")
            b_{batch} = lm.transpose(right_{batch}, id="transpose_right_{batch}")
            product_{batch} = lm.mma(a_{batch}, b_{batch}, instruction={{"contract":"triton.dot.bf16_fp32"}}, tile_shape=(16,32,32), id="product_{batch}")
'''
        if batch:
            prior='product_0' if batch==1 else f'accumulated_{batch-1}'
            source+=f'    with compute:\n        accumulated_{batch} = {prior} + product_{batch}\n'
    result='product_0' if b==1 else f'accumulated_{b-1}'
    return source+f'''    with compute:
        rounded = lm.cast({result}, to="bf16", id="round_output")
        lm.store({output}[row, column], rounded, id="store_output")
'''


def attention_output_gradient(b,s,h,heads,depth):
    return header('attention_output_gradient',[tensor('grad_hidden_states_attn',(b,s,h)),
        tensor('o_weight',(h,heads*depth)),tensor('grad_attn_output',(b,s,heads,depth),output=True)])+f'''    row = lm.program(grad_attn_output, axis=0, dimension=1, tile=16)
    head = lm.program(grad_attn_output, axis=1, dimension=2, tile=1)
    batch = lm.program(grad_attn_output, axis=2, dimension=0, tile=1)
    with compute:
        head_index = lm.coordinate(source="program", name="head", id="head_index")
    for feature in lm.range(grad_attn_output, name="features", dimension=3, tile=32, num_stages=1, loop_unroll_factor=1):
        with compute:
            features = lm.coordinate(source="loop_tile", name="feature", id="feature_coordinates")
            columns = head_index * {depth} + features
        for hidden in lm.range(grad_hidden_states_attn, name="hidden", dimension=2, tile=32, num_stages=1, loop_unroll_factor=1):
            with compute:
                a = lm.load(grad_hidden_states_attn[batch, row, hidden], id="load_gradient")
                weight = lm.load(o_weight[hidden, columns], id="load_weight")
                wt = lm.transpose(weight, id="transpose_weight")
                product = lm.mma(a, wt, instruction={{"contract":"triton.dot.bf16_fp32"}}, tile_shape=(16,32,32), id="product")
        with compute:
            rounded = lm.cast(product, to="bf16", id="round_output")
            lm.store(grad_attn_output[batch, row, head, feature], rounded, id="store_output")
'''


def attention_weight_gradient(b,s,heads,depth):
    return header('attention_weight_gradient',[tensor('grad_attn_output',(b,s,heads,depth)),
        tensor('value_states_repeated',(b,heads,s,depth)),tensor('grad_attn_weights',(b*heads,s,s),output=True)])+f'''    row = lm.program(grad_attn_weights, axis=0, dimension=1, tile=16)
    column = lm.program(grad_attn_weights, axis=1, dimension=2, tile=32)
    group = lm.program(grad_attn_weights, axis=2, dimension=0, tile=1)
    with compute:
        group_index = lm.coordinate(source="program", name="group", id="group_index")
        batch_index = group_index // {heads}
        head_index = group_index % {heads}
    for feature in lm.range(grad_attn_output, name="features", dimension=3, tile=32, num_stages=1, loop_unroll_factor=1):
        with compute:
            a = lm.load(grad_attn_output[lm.scalar_index(batch_index), row, lm.scalar_index(head_index), feature], id="load_gradient")
            value = lm.load(value_states_repeated[lm.scalar_index(batch_index), lm.scalar_index(head_index), column, feature], id="load_value")
            product = lm.mma(a, value, instruction={{"contract":"triton.dot.bf16_fp32"}}, tile_shape=(16,32,32), id="product")
    with compute:
        rounded = lm.cast(product, to="bf16", id="round_output")
        lm.store(grad_attn_weights[group, row, column], rounded, id="store_output")
'''


def query_gradient(b,s,heads,depth):
    return header('rotated_query_gradient',[tensor('grad_attn_logits',(b*heads,s,s)),
        tensor('key_states_repeated',(b,heads,s,depth)),tensor('grad_query_rotated',(b,s,heads*depth),output=True)])+f'''    row = lm.program(grad_query_rotated, axis=0, dimension=1, tile=16)
    column = lm.program(grad_query_rotated, axis=1, dimension=2, tile=32)
    batch = lm.program(grad_query_rotated, axis=2, dimension=0, tile=1)
    with compute:
        batch_index = lm.coordinate(source="program", name="batch", id="batch_index")
        column_index = lm.coordinate(source="program", name="column", id="column_index")
        columns = lm.coordinate(source="program_tile", name="column", id="columns")
        head_index = column_index * 32 // {depth}
        features = columns % {depth}
        group_index = batch_index * {heads} + head_index
    for key in lm.range(grad_attn_logits, name="keys", dimension=2, tile=32, num_stages=1, loop_unroll_factor=1):
        with compute:
            a = lm.load(grad_attn_logits[lm.scalar_index(group_index), row, key], id="load_logits")
            values = lm.load(key_states_repeated[batch, lm.scalar_index(head_index), key, features], id="load_keys")
            wt = lm.transpose(values, id="transpose_keys")
            product = lm.mma(a, wt, instruction={{"contract":"triton.dot.bf16_fp32"}}, tile_shape=(16,32,32), id="product")
    with compute:
        rounded = lm.cast(product, to="bf16", id="round_output")
        lm.store(grad_query_rotated[batch, row, column], rounded, id="store_output")
'''


def grouped_gradient(name,b,s,heads,kv,depth,*,key_gradient):
    left='grad_attn_logits' if key_gradient else 'attn_weights'
    right='query_states_rotated' if key_gradient else 'grad_attn_output'
    output='grad_key_rotated' if key_gradient else 'grad_value'
    left_shape=(b*heads,s,s) if key_gradient else (b,heads,s,s)
    source=header(name,[tensor(left,left_shape),tensor(right,(b,s,heads,depth)),tensor(output,(b,s,kv*depth),output=True)])
    source+=f'''    row = lm.program({output}, axis=0, dimension=1, tile=16)
    column = lm.program({output}, axis=1, dimension=2, tile=32)
    batch = lm.program({output}, axis=2, dimension=0, tile=1)
    with compute:
        batch_index = lm.coordinate(source="program", name="batch", id="batch_index")
        column_index = lm.coordinate(source="program", name="column", id="column_index")
        columns = lm.coordinate(source="program_tile", name="column", id="columns")
        kv_index = column_index * 32 // {depth}
        features = columns % {depth}
'''
    groups=heads//kv
    for group in range(groups):
        left_access=(f'{left}[lm.scalar_index(global_head_{group}), query_{group}, row]' if key_gradient else
                     f'{left}[batch, lm.scalar_index(head_{group}), query_{group}, row]')
        dimension=1 if key_gradient else 2
        source+=f'''    with compute:
        head_{group} = kv_index * {groups} + {group}
        global_head_{group} = batch_index * {heads} + head_{group}
    for query_{group} in lm.range({left}, name="queries_{group}", dimension={dimension}, tile=32, num_stages=1, loop_unroll_factor=1):
        with compute:
            left_{group} = lm.load({left_access}, id="load_left_{group}")
            right_{group} = lm.load({right}[batch, query_{group}, lm.scalar_index(head_{group}), features], id="load_right_{group}")
            a_{group} = lm.transpose(left_{group}, id="transpose_left_{group}")
            b_{group} = lm.transpose(right_{group}, id="transpose_right_{group}")
            product_{group} = lm.mma(a_{group}, b_{group}, instruction={{"contract":"triton.dot.bf16_fp32"}}, tile_shape=(16,32,32), id="product_{group}")
    with compute:
        bf16_{group} = lm.cast(product_{group}, to="bf16", id="round_head_{group}")
        fp32_{group} = lm.cast(bf16_{group}, to="fp32", id="widen_head_{group}")
'''
        if group:
            prior='fp32_0' if group==1 else f'accumulated_{group-1}'
            source+=f'        accumulated_{group} = {prior} + fp32_{group}\n'
    result='fp32_0' if groups==1 else f'accumulated_{groups-1}'
    return source+f'''    with compute:
        rounded = lm.cast({result}, to="bf16", id="round_group_sum")
        lm.store({output}[batch, row, column], rounded, id="store_output")
'''


def inverse_rope(name,b,s,heads,depth,gradient,output):
    width=heads*depth
    return header(name,[tensor(gradient,(b,s,width)),tensor('cos',(b,s,depth)),
        tensor('sin',(b,s,depth)),tensor(output,(b,s,width),output=True)])+f'''    row = lm.program({output}, axis=0, dimension=1, tile=1)
    column = lm.program({output}, axis=1, dimension=2, tile=32)
    batch = lm.program({output}, axis=2, dimension=0, tile=1)
    with compute:
        columns = lm.coordinate(source="program_tile", name="column", id="columns")
        feature = columns % {depth}
        first = lm.compare(feature, {depth//2}, op="lt", id="first_half")
        rotated_columns = lm.select(first, columns + {depth//2}, columns - {depth//2}, id="rotated_columns")
        original_stored = lm.load({gradient}[batch, row, column], id="load_grad")
        rotated_stored = lm.load({gradient}[batch, row, rotated_columns], id="load_rotated")
        cosine_stored = lm.load(cos[batch, row, feature], id="load_cos")
        sine_stored = lm.load(sin[batch, row, feature], id="load_sin")
        original = lm.cast(original_stored, to="fp32", id="widen_original")
        rotated = lm.cast(rotated_stored, to="fp32", id="widen_rotated")
        cosine = lm.cast(cosine_stored, to="fp32", id="widen_cos")
        sine = lm.cast(sine_stored, to="fp32", id="widen_sin")
        signed_rotated = lm.select(first, rotated * -1.0, rotated, id="rotation_sign")
        cosine_product = original * cosine
        sine_product = signed_rotated * (sine * -1.0)
        cosine_bf16 = lm.cast(cosine_product, to="bf16", id="round_cos_product")
        sine_bf16 = lm.cast(sine_product, to="bf16", id="round_sin_product")
        cosine_fp32 = lm.cast(cosine_bf16, to="fp32", id="widen_cos_product")
        sine_fp32 = lm.cast(sine_bf16, to="fp32", id="widen_sin_product")
        result = cosine_fp32 + sine_fp32
        rounded = lm.cast(result, to="bf16", id="round_output")
        lm.store({output}[batch, row, column], rounded, id="store_output")
'''
