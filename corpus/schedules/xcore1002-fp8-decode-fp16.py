from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="xcore1002-fp8-decode-fp16", target="xcore1002", backend="triton",
               entry_point="xcore1002_fp8_decode_fp16")
def candidate(lm, x: cake.Tensor((1, 256), "fp8_e4m3"),
              out: cake.Tensor((1, 256), "fp16", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        value = lm.load(x[row, :], id="load")
        decoded = lm.cast(value, to="fp16", id="cast")
        lm.store(out[row, :], decoded, id="store")
