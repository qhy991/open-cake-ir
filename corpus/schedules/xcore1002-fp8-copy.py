from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="xcore1002-fp8-copy", target="xcore1002", backend="triton",
               entry_point="xcore1002_fp8_copy")
def candidate(lm, x: cake.Tensor((1, 256), "fp8_e4m3"),
              out: cake.Tensor((1, 256), "fp8_e4m3", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        value = lm.load(x[row, :], id="load")
        lm.store(out[row, :], value, id="store")
