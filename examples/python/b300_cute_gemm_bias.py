from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="b300-cute-register-primary", target="sm_103a", backend="cutlass_cute_dsl",
               entry_point="cake_cute_register",
               metadata={"workload_contract_sha256": "b3337c25ac111faf7084910696a4857c9c50c8fcb14826184ed4317bfc16caaf"})
def gemm_bias(lm, a: cake.Tensor((512, 256), "bf16"),
              b: cake.Tensor((256, 256), "bf16"), bias: cake.Tensor((256,), "fp32"),
              c: cake.Tensor((512, 256), "fp32", mode="output")):
    compute = lm.role(warps=[0])
    m_block = lm.program(a, axis=0, dimension=0, tile=16)
    n_block = lm.program(b, axis=1, dimension=0, tile=16)
    for k in lm.range(a, name="k_loop", dimension=1, tile=16, num_stages=1,
                      loop_unroll_factor=1, disallow_acc_multi_buffer=True):
        with compute:
            a_tile = lm.load(a[m_block, k], reuse="streamed", id="load_a")
            b_tile = lm.load(b[n_block, k], reuse="streamed", id="load_b")
            acc = lm.mma(a_tile, b_tile, instruction={"contract": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
                                                   "shape": [16, 8, 16], "cta_group": 1,
                                                   "operand_source": "register", "operand_major": ["k", "k"]},
                         tile_shape=(16, 16, 16), id="dot")
    with compute:
        bias_tile = lm.load(bias[n_block], reuse="streamed", id="load_bias")
        c_tile = lm.add(acc, lm.broadcast(bias_tile, axis=1), id="add_bias")
        lm.store(c[m_block, n_block], c_tile, coalesced=False, id="store_c")
