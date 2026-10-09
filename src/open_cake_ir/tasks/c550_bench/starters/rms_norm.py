"""Residual RMSNorm with both original BF16 rounding boundaries."""


def source(workload) -> str:
    document = workload.document
    if document['semantics']['benchmark']['task'] != 'L1/069_rms_norm':
        raise ValueError('Residual RMSNorm starter requires the original L1/069 task')
    tensors = document['tensors']
    batch, sequence, width = tensors['hidden_states']['shape']
    epsilon = document['semantics']['fixed_scalar_inputs']['eps']['value']
    return f'''from open_cake_ir.compiler import frontend as cake

@cake.schedule(name="bench_residual_rmsnorm", target="xcore1002", backend="triton",
               entry_point="cake_bench_residual_rmsnorm")
def candidate(lm, hidden_states: cake.Tensor(({batch}, {sequence}, {width}), "bf16"),
              residual: cake.Tensor(({batch}, {sequence}, {width}), "bf16"),
              weight: cake.Tensor(({width},), "bf16"),
              output: cake.Tensor(({batch}, {sequence}, {width}), "bf16", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    batch = lm.program(hidden_states, axis=0, dimension=0, tile=1)
    row = lm.program(hidden_states, axis=1, dimension=1, tile=1)
    with compute:
        stored_x = lm.load(hidden_states[batch, row, :], id="load_x")
        x_fp32 = lm.cast(stored_x, to="fp32", id="widen_x")
        stored_residual = lm.load(residual[batch, row, :], id="load_residual")
        residual_fp32 = lm.cast(stored_residual, to="fp32", id="widen_residual")
        sum_fp32 = x_fp32 + residual_fp32
        sum_bf16 = lm.cast(sum_fp32, to="bf16", id="round_residual_sum")
        values = lm.cast(sum_bf16, to="fp32", id="widen_rounded_sum")
        squares = lm.square(values, id="square")
        square_sum = lm.reduce(squares, op="sum", axis=0, scope="cta", across_loop=False, id="sum_square")
        mean_square = square_sum / {float(width)!r}
        inverse = lm.rsqrt(mean_square + {epsilon!r}, id="inverse")
        normalized = values * inverse
        normalized_bf16 = lm.cast(normalized, to="bf16", id="round_before_weight")
        normalized_fp32 = lm.cast(normalized_bf16, to="fp32", id="widen_normalized")
        stored_weight = lm.load(weight[:], id="load_weight")
        weights = lm.cast(stored_weight, to="fp32", id="widen_weight")
        weighted = normalized_fp32 * weights
        narrowed = lm.cast(weighted, to="bf16", id="round_output")
        lm.store(output[batch, row, :], narrowed, coalesced=False, id="store_out")
'''
