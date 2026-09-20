from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="xcore1002-dot-fp32-64", target="xcore1002", backend="triton",
               entry_point="xcore1002_dot_fp32_64")
def matrix(lm, a: cake.Tensor((64, 64), "fp32"),
           b: cake.Tensor((64, 64), "fp32"),
           out: cake.Tensor((64, 64), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    m = lm.program(a, axis=0, dimension=0, tile=64)
    n = lm.program(b, axis=1, dimension=0, tile=64)
    with compute:
        left = lm.load(a[m, :], id="load_a")
        right = lm.load(b[n, :], id="load_b")
        result = lm.mma(left, right, instruction={"contract": "triton.dot.fp32_ieee"},
                        tile_shape=(64, 64, 64), id="dot")
        lm.store(out[m, n], result, id="store_out")
