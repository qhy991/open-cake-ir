"""Pure CAKE ConvNextV2 starter with explicit FP32 stages and GELU approximation."""
from __future__ import annotations

import math

TASK = 'L2/035_convnextv2_block_with_grn'
TARGET = 'xcore1002'


def _tensor(name, shape, output=False):
    return f'{name}: cake.Tensor({shape!r}, "fp32"' + (', mode="output")' if output else ')')


def _header(name, tensors):
    return (f'@cake.schedule(name="{name}", target="{TARGET}", backend="triton", entry_point="cake_{name}")\n'
            f'def {name}(lm, ' + ', '.join(tensors) + '):\n'
            '    compute = lm.role(execution_groups=[0, 1, 2, 3])\n')


def _depthwise(b, c, h, w):
    source = _header('depthwise', [_tensor('x', (b, c, h, w)),
        _tensor('dwconv_weight', (c, 1, 7, 7)), _tensor('dwconv_bias', (c,)),
        _tensor('convolved', (b, h*w, c), True)])
    source += f'''    pixel = lm.program(convolved, axis=0, dimension=1, tile=1)
    channel = lm.program(convolved, axis=1, dimension=2, tile=32)
    batch = lm.program(convolved, axis=2, dimension=0, tile=1)
    with compute:
        position = lm.coordinate(source="program", name="pixel", id="position")
        y = position // {w}
        x_index = position % {w}
'''
    previous = None
    for i in range(7):
        for j in range(7):
            suffix = f'{i}_{j}'
            source += f'''        y_{suffix} = y + {i-3}
        x_{suffix} = x_index + {j-3}
        input_{suffix} = lm.load(x[batch, channel, y_{suffix}, x_{suffix}], id="input_{suffix}")
        weight_{suffix} = lm.load(dwconv_weight[channel, 0:1, {i}:{i+1}, {j}:{j+1}], id="weight_{suffix}")
        product_{suffix} = input_{suffix} * weight_{suffix}
'''
            value = f'product_{suffix}'
            if previous:
                source += f'        sum_{suffix} = {previous} + {value}\n'
                value = f'sum_{suffix}'
            previous = value
    return source + f'''        bias = lm.load(dwconv_bias[channel], id="bias")
        result = {previous} + bias
        lm.store(convolved[batch, pixel, channel], result, id="store_convolved")
'''


def _layernorm(b, c, p, epsilon):
    return _header('layernorm', [_tensor('convolved', (b,p,c)),
        _tensor('layernorm_weight', (c,)), _tensor('layernorm_bias', (c,)),
        _tensor('normalized', (b,p,c), True)]) + f'''    pixel = lm.program(normalized, axis=0, dimension=1, tile=1)
    batch = lm.program(normalized, axis=1, dimension=0, tile=1)
    channel = lm.program(normalized, axis=2, dimension=2, tile={1 << (c-1).bit_length()})
    with compute:
        channels = lm.coordinate(source="program_tile", name="channel", id="channels")
        values = lm.load(convolved[batch, pixel, channels], id="values")
        total = lm.reduce(values, op="sum", axis=0, scope="cta", across_loop=False, id="total")
        mean = total / {float(c)!r}
        centered = values - mean
        valid = lm.compare(channels, {c}, op="lt", id="valid")
        valid_centered = lm.select(valid, centered, 0.0, id="valid_centered")
        squares = lm.square(valid_centered, id="squares")
        variance_sum = lm.reduce(squares, op="sum", axis=0, scope="cta", across_loop=False, id="variance_sum")
        variance = variance_sum / {float(c)!r}
        inverse = lm.rsqrt(variance + {epsilon!r}, id="inverse")
        weight = lm.load(layernorm_weight[channels], id="weight")
        bias = lm.load(layernorm_bias[channels], id="bias")
        result = centered * inverse * weight + bias
        lm.store(normalized[batch, pixel, channel], result, id="store_normalized")
'''


def _gelu_lines(value, result, indent='        '):
    # Evaluate the complementary-error-function polynomial directly on the negative
    # branch to avoid cancellation in 1 + erf(x / sqrt(2)). Device error is unqualified.
    return '\n'.join(indent + line for line in [
        f'negative = lm.compare({value}, 0.0, op="lt", id="negative")',
        f'absolute = lm.select(negative, {value} * -1.0, {value}, id="absolute")',
        f'z = absolute * {1 / math.sqrt(2)!r}',
        't = lm.reciprocal(z * 0.3275911 + 1.0, id="t")',
        'polynomial = (((((t * 1.061405429 - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t',
        'exponential = lm.exp(z * z * -1.0, id="exponential")',
        'erfc = polynomial * exponential',
        f'tail = {value} * 0.5 * erfc',
        f'{result} = lm.select(negative, tail, {value} - tail, id="gelu")',
    ]) + '\n'


def _expansion(b, c, p):
    f = 4*c
    return _header('expansion', [_tensor('normalized',(b,p,c)),
        _tensor('pwconv1_weight',(f,c)), _tensor('pwconv1_bias',(f,)),
        _tensor('activated',(b,p,f), True)]) + f'''    pixel = lm.program(activated, axis=0, dimension=1, tile=16)
    feature = lm.program(activated, axis=1, dimension=2, tile=32)
    batch = lm.program(activated, axis=2, dimension=0, tile=1)
    for channel in lm.range(normalized, name="channels", dimension=2, tile=32, num_stages=1, loop_unroll_factor=1):
        with compute:
            left = lm.load(normalized[batch, pixel, channel], id="left")
            right = lm.load(pwconv1_weight[feature, channel], id="right")
            product = lm.mma(left, right, instruction={{"contract":"triton.dot.fp32_ieee"}}, tile_shape=(16,32,32), id="product")
    with compute:
        bias = lm.load(pwconv1_bias[feature], id="bias")
        expanded = product + lm.broadcast(bias, axis=1)
''' + _gelu_lines('expanded', 'result') + '''        lm.store(activated[batch, pixel, feature], result, id="store_activated")
'''


def _spatial_norm(b, c, p):
    f=4*c
    loop = p > 128
    inner = '        ' if loop else '    '
    source = _header('spatial_norm', [_tensor('activated',(b,p,f)), _tensor('global_norm',(b,f),True)])
    source += '''    feature = lm.program(global_norm, axis=0, dimension=1, tile=1)
    batch = lm.program(global_norm, axis=1, dimension=0, tile=1)
'''
    if loop:
        source += '    for pixel in lm.range(activated, name="pixels", dimension=1, tile=128, num_stages=1, loop_unroll_factor=1):\n'
    index = 'pixel' if loop else ':'
    source += inner + 'with compute:\n' + '\n'.join(inner + '    '+line for line in [
        f'values = lm.load(activated[batch, {index}, feature], id="values")',
        'squares = lm.square(values, id="squares")',
        f'total = lm.reduce(squares, op="sum", axis=0, scope="cta", across_loop={loop}, id="total")',
    ]) + '\n'
    return source + '''    with compute:
        inverse = lm.rsqrt(total, id="inverse")
        norm = lm.reciprocal(inverse, id="norm")
        lm.store(global_norm[batch, feature], norm, coalesced=False, id="store_norm")
'''


def _channel_norm(b, c, eps):
    f=4*c
    return _header('channel_norm', [_tensor('global_norm',(b,f)), _tensor('response_norm',(b,f),True)]) + f'''    batch = lm.program(response_norm, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(global_norm[batch, :], id="values")
        total = lm.reduce(values, op="sum", axis=0, scope="cta", across_loop=False, id="total")
        mean = total / {float(f)!r}
        result = values / (mean + {eps!r})
        lm.store(response_norm[batch, :], result, id="store_response")
'''


def _projection(b, c, p):
    f=4*c
    return _header('projection', [_tensor('activated',(b,p,f)),
        _tensor('response_norm',(b,f)), _tensor('grn_weight',(1,1,1,f)),
        _tensor('grn_bias',(1,1,1,f)), _tensor('pwconv2_weight',(c,f)),
        _tensor('projected',(b,p,c),True)]) + f'''    pixel = lm.program(projected, axis=0, dimension=1, tile=16)
    channel = lm.program(projected, axis=1, dimension=2, tile=32)
    batch = lm.program(projected, axis=2, dimension=0, tile=1)
    for feature in lm.range(activated, name="features", dimension=2, tile=32, num_stages=1, loop_unroll_factor=1):
        with compute:
            values = lm.load(activated[batch, pixel, feature], id="values")
            norm = lm.load(response_norm[batch, feature], id="norm")
            weight = lm.load(grn_weight[0:1, 0:1, 0:1, feature], id="weight")
            bias = lm.load(grn_bias[0:1, 0:1, 0:1, feature], id="bias")
            scaled = values * lm.broadcast(norm, axis=1)
            weighted = scaled * lm.broadcast(weight, axis=1)
            shifted = weighted + lm.broadcast(bias, axis=1)
            left = shifted + values
            right = lm.load(pwconv2_weight[channel, feature], id="right")
            product = lm.mma(left, right, instruction={{"contract":"triton.dot.fp32_ieee"}}, tile_shape=(16,32,32), id="product")
    with compute:
        lm.store(projected[batch, pixel, channel], product, id="store_projected")
'''


def _output(b, c, h, w):
    return _header('residual', [_tensor('x',(b,c,h,w)), _tensor('projected',(b,h*w,c)),
        _tensor('pwconv2_bias',(c,)), _tensor('output',(b,c,h,w),True)]) + f'''    column = lm.program(output, axis=0, dimension=3, tile=1)
    row = lm.program(output, axis=1, dimension=2, tile=1)
    batch = lm.program(output, axis=2, dimension=0, tile=1)
    with compute:
        y = lm.coordinate(source="program", name="row", id="y")
        x_index = lm.coordinate(source="program", name="column", id="x_index")
        pixel = y * {w} + x_index
        values = lm.load(projected[batch, pixel, :], id="values")
        bias = lm.load(pwconv2_bias[:], id="bias")
        with_bias = values + bias
        original = lm.load(x[batch, :, row, column], id="original")
        result = original + with_bias
        lm.store(output[batch, :, row, column], result, coalesced=False, id="store_output")
'''


def source_for(batch, channels, height, width, *, eps, layer_norm_eps):
    """Create the full original ABI; smaller dimensions are for CPU controls only."""
    if any(type(v) is not int or v <= 0 for v in (batch,channels,height,width)):
        raise ValueError('ConvNext dimensions must be positive integers')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in (eps,layer_norm_eps)):
        raise ValueError('ConvNext epsilons must be positive finite scalars')
    b,c,h,w = batch,channels,height,width
    stages = [_depthwise(b,c,h,w), _layernorm(b,c,h*w,float(layer_norm_eps)),
        _expansion(b,c,h*w), _spatial_norm(b,c,h*w), _channel_norm(b,c,float(eps)),
        _projection(b,c,h*w), _output(b,c,h,w)]
    bindings = [
        ('depthwise',('x','dwconv_weight','dwconv_bias','convolved')),
        ('layernorm',('convolved','layernorm_weight','layernorm_bias','normalized')),
        ('expansion',('normalized','pwconv1_weight','pwconv1_bias','activated')),
        ('spatial_norm',('activated','global_norm')),
        ('channel_norm',('global_norm','response_norm')),
        ('projection',('activated','response_norm','grn_weight','grn_bias','pwconv2_weight','projected')),
        ('residual',('x','projected','pwconv2_bias','output')),
    ]
    inputs = ('x','dwconv_weight','dwconv_bias','layernorm_weight','layernorm_bias',
              'pwconv1_weight','pwconv1_bias','grn_weight','grn_bias','pwconv2_weight','pwconv2_bias')
    tail = f'\ncake.program(program_id="c550_bench_convnext_b{b}_c{c}_h{h}_w{w}", inputs={inputs!r}, outputs=("output",), stages=(\n'
    for name,names in bindings:
        tail += f'    cake.stage(name="{name}", schedule={name}, bindings={dict(zip(names,names))!r}),\n'
    return 'from open_cake_ir.compiler import frontend as cake\n\n' + '\n'.join(stages) + tail + '))\n'


def program_for(batch, channels, height, width, *, eps, layer_norm_eps):
    from open_cake_ir.compiler.program_frontend import parse_program
    return parse_program(source_for(batch,channels,height,width,eps=eps,layer_norm_eps=layer_norm_eps)).program


def source_for_workload(workload, case_id='primary'):
    abi = workload.tensor_abi(case_id)
    if workload.target != TARGET or len(abi) != 12 or len(abi[0].shape) != 4:
        raise ValueError('ConvNext Workload target or argument count differs')
    b,c,h,w = abi[0].shape
    f=4*c
    expected = (('x',(b,c,h,w)), ('dwconv_weight',(c,1,7,7)), ('dwconv_bias',(c,)),
        ('layernorm_weight',(c,)), ('layernorm_bias',(c,)), ('pwconv1_weight',(f,c)),
        ('pwconv1_bias',(f,)), ('grn_weight',(1,1,1,f)), ('grn_bias',(1,1,1,f)),
        ('pwconv2_weight',(c,f)), ('pwconv2_bias',(c,)), ('output',(b,c,h,w)))
    if tuple((a.name,tuple(a.shape),a.dtype,a.mode) for a in abi) != tuple(
            (name,shape,'fp32','output' if name=='output' else 'input') for name,shape in expected):
        raise ValueError('ConvNext original ordered tensor ABI differs')
    scalars=workload.document['semantics'].get('fixed_scalar_inputs')
    if not isinstance(scalars,dict) or set(scalars) != {'eps','layer_norm_eps'} or any(
        not isinstance(value,dict) or set(value) != {'dtype','value','binding'} or value['dtype'] != 'float32'
        or value['binding'] not in {'literal_input','original_factory_literal'} for value in scalars.values()):
        raise ValueError('ConvNext requires the original scalar bindings')
    return source_for(b,c,h,w,eps=scalars['eps']['value'],layer_norm_eps=scalars['layer_norm_eps']['value'])
