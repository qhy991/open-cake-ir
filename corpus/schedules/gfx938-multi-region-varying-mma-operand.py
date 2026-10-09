from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="multi-region-attention", target="gfx938", backend="triton", entry_point="multi_region_attention")
def candidate(lm, q: cake.Tensor((17, 33), "fp32"), k: cake.Tensor((32, 33), "fp32"), v: cake.Tensor((32, 49), "fp32"), out: cake.Tensor((17, 49), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row_block = lm.program(q, axis=0, dimension=0, tile=16)
    for kk in lm.range(q, name="k_loop", dimension=1, tile=16, num_stages=2, loop_unroll_factor=1, disallow_acc_multi_buffer=True):
        with compute:
            query = lm.load(q[row_block, kk], reuse="streamed", id="load_q")
            keys = lm.load(k[:, kk], reuse="streamed", id="load_k")
            scores = lm.mma(query, keys, instruction={"contract": "triton.dot.fp32_ieee"}, tile_shape=[16, 32, 16], id="qk_dot")
    with compute:
        logits = scores * 0.0625
        peak = lm.reduce(logits, op="max", axis=1, scope="cta", across_loop=False, id="max_logit")
        shifted = logits - lm.broadcast(peak, axis=0)
        weights = lm.exp(shifted, id="exp")
        total = lm.reduce(weights, op="sum", axis=1, scope="cta", across_loop=False, id="sum_exp")
    for kv in lm.range(v, name="vo_loop", dimension=1, tile=16, num_stages=2, loop_unroll_factor=1, disallow_acc_multi_buffer=True):
        with compute:
            values = lm.load(v[:, kv], reuse="streamed", id="load_v")
            scaled = values * 2.0
            values_t = lm.transpose(scaled, id="transpose_v")
            chunk = lm.mma(weights, values_t, instruction={"contract": "triton.dot.fp32_ieee"}, tile_shape=[16, 16, 32], id="av_dot")
            lm.store(out[row_block, kv], chunk / lm.broadcast(total, axis=0), coalesced=True, id="store_out")
