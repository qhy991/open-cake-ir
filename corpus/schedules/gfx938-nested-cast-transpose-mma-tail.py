from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="nested-cast-kn-gemm", target="gfx938", backend="triton", entry_point="nested_cast_kn_gemm")
def candidate(lm, a: cake.Tensor((35, 33), "bf16"), b: cake.Tensor((33, 49), "bf16"), out: cake.Tensor((35, 49), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    n = lm.program(b, axis=0, dimension=1, tile=32)
    with compute:
        for m in lm.range(a, name="m_loop", dimension=0, tile=16):
            for k in lm.range(a, name="k_loop", dimension=1, tile=16):
                av = lm.load(a[m,k], id="load_a")
                bv = lm.load(b[k,n], id="load_b")
                af = lm.cast(av, to="fp32", id="cast_a")
                bf = lm.cast(bv, to="fp32", id="cast_b")
                bt = lm.transpose(bf, id="transpose_b")
                result = lm.mma(af, bt, instruction={"contract":"triton.dot.fp32_ieee"}, tile_shape=[16,32,16], id="dot")
            lm.store(out[m,n], result, id="store_out")
