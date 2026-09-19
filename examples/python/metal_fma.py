from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="metal-fma-odd", target="apple_gpu_family8", backend="metal",
               entry_point="cake_metal_fma")
def metal_fma(lm, a: cake.Tensor((3, 37), "fp32"),
              b: cake.Tensor((3, 37), "fp32"),
              c: cake.Tensor((3, 37), "fp32"),
              out: cake.Tensor((3, 37), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(a, axis=0, dimension=0, tile=1)
    with compute:
        a_values = lm.load(a[row, :], id="load_a")
        b_values = lm.load(b[row, :], id="load_b")
        c_values = lm.load(c[row, :], id="load_c")
        result = lm.fma(a_values, b_values, c_values,
                        instruction={"contract": "metal.fma.f32"}, id="fma")
        lm.store(out[row, :], result, coalesced=False, id="store_out")
