from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="fma-b8-smoke", target="sm_100a", backend="triton",
               entry_point="cake_fma_b8_smoke",
               residency={"ctas_per_multiprocessor": 4, "registers_per_thread": 64})
def fma(lm, a: cake.Tensor((8, 128), "fp32"), b: cake.Tensor((8, 128), "fp32"),
        c: cake.Tensor((8, 128), "fp32"), y: cake.Tensor((8, 128), "fp32", mode="output")):
    compute = lm.role(warps=[0, 1, 2, 3])
    batch = lm.program(a, axis=0, dimension=0, tile=1)
    with compute:
        a_tile = lm.load(a[batch, :], reuse="streamed", id="load_a")
        b_tile = lm.load(b[batch, :], reuse="streamed", id="load_b")
        c_tile = lm.load(c[batch, :], reuse="streamed", id="load_c")
        y_tile = lm.fma(a_tile, b_tile, c_tile, id="fma")
        lm.store(y[batch, :], y_tile, id="store_y")
