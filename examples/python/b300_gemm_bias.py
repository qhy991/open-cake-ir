from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="gemm-bias-b1-smoke-b300", target="sm_103a", backend="triton",
               entry_point="cake_gemm_bias_b1_smoke",
               metadata={"workload_contract_sha256": "b3337c25ac111faf7084910696a4857c9c50c8fcb14826184ed4317bfc16caaf"},
               residency={"ctas_per_multiprocessor": 4, "registers_per_thread": 128})
def gemm_bias(lm, a: cake.Tensor((512, 256), "bf16"),
              b: cake.Tensor((256, 256), "bf16"), bias: cake.Tensor((256,), "fp32"),
              c: cake.Tensor((512, 256), "fp32", mode="output")):
    compute = lm.role(warps=[0, 1, 2, 3])
    m_block = lm.program(a, axis=0, dimension=0, tile=64)
    n_block = lm.program(b, axis=1, dimension=0, tile=64)
    for k in lm.range(a, name="k_loop", dimension=1, tile=64, num_stages=2,
                      loop_unroll_factor=1, disallow_acc_multi_buffer=True):
        with compute:
            a_tile = lm.load(a[m_block, k], reuse="streamed", id="load_a")
            b_tile = lm.load(b[n_block, k], reuse="streamed", id="load_b")
            acc = lm.mma(a_tile, b_tile, instruction={"contract": "triton.dot.bf16_fp32"},
                         tile_shape=(64, 64, 64), id="dot")
    with compute:
        bias_tile = lm.load(bias[n_block], reuse="streamed", id="load_bias")
        c_tile = lm.add(acc, lm.broadcast(bias_tile, axis=1), id="add_bias")
        lm.store(c[m_block, n_block], c_tile, id="store_c")
