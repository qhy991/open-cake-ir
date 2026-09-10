"""First stage: contraction+bias, materialized as a private BF16 intermediate."""
from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="epilogue_producer", target="sm_100a", backend="triton",
               entry_point="epilogue_producer")
def candidate(lm, a: cake.Tensor((2, 8), "bf16"), b: cake.Tensor((8, 8), "bf16"),
              bias: cake.Tensor((8,), "fp32"), mid: cake.Tensor((2, 8), "bf16", mode="output")):
    compute = lm.role(warps=[0, 1, 2, 3])
    row = lm.program(a, axis=0, dimension=0, tile=1)
    with compute:
        ar = lm.load(a[row, :], id="load_a")
        br = lm.load(b[:, :], id="load_b")
        biases = lm.load(bias[:], id="load_bias")
        af = lm.cast(ar, to="fp32", id="a_fp32")
        bf = lm.cast(br, to="fp32", id="b_fp32")
        products = bf * lm.broadcast(af, axis=0)
        total = lm.reduce(products, op="sum", axis=0, scope="cta", across_loop=False, id="sum_k")
        shifted = total + biases
        rounded = lm.cast(shifted, to="bf16", id="round_intermediate")
        lm.store(mid[row, :], rounded, coalesced=False, id="store_mid")
