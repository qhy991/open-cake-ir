from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="rmsnorm-b8-smoke-b300", target="sm_103a", backend="triton",
               entry_point="cake_rmsnorm_b8_smoke",
               metadata={"workload_contract_sha256": "946f8ef156383eddc8834497fb3d3afdfb9a0014624347d7b5f5ab2ca0d67b63"},
               residency={"ctas_per_multiprocessor": 4, "registers_per_thread": 96})
def rmsnorm(lm, x: cake.Tensor((8, 512, 128), "fp32"),
            gamma: cake.Tensor((128,), "fp32"),
            y: cake.Tensor((8, 512, 128), "fp32", mode="output")):
    compute = lm.role(warps=[0, 1, 2, 3])
    row_block = lm.program(x, axis=0, dimension=1, tile=64)
    batch = lm.program(x, axis=1, dimension=0, tile=1)
    with compute:
        x_tile = lm.load(x[batch, row_block, :], reuse="reused", id="load_x")
        sq = lm.square(x_tile, id="square")
        sumsq = lm.reduce(sq, op="sum", axis=1, scope="cta", id="sum_sq")
        meansq = lm.mul(sumsq, 0.0078125, id="mean")
        shifted = lm.add(meansq, 1e-6, id="shift")
        inv_rms = lm.rsqrt(shifted, id="rsqrt")
        gamma_tile = lm.load(gamma[:], reuse="streamed", id="load_gamma")
        normed = lm.mul(x_tile, lm.broadcast(inv_rms, axis=0), id="scale")
        y_tile = lm.mul(normed, lm.broadcast(gamma_tile, axis=1), id="weight")
        lm.store(y[batch, row_block, :], y_tile, id="store_y")
