"""Second stage: unary SiLU over the rounded producer output."""
from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="epilogue_consumer", target="sm_100a", backend="triton",
               entry_point="epilogue_consumer")
def candidate(lm, mid: cake.Tensor((2, 8), "bf16"), out: cake.Tensor((2, 8), "bf16", mode="output")):
    compute = lm.role(warps=[0, 1, 2, 3])
    row = lm.program(mid, axis=0, dimension=0, tile=1)
    with compute:
        rounded = lm.load(mid[row, :], id="load_mid")
        values = lm.cast(rounded, to="fp32", id="mid_fp32")
        negated = values * -1.0
        decayed = lm.exp(negated, id="exp")
        gate = lm.reciprocal(decayed + 1.0, id="reciprocal")
        activated = values * gate
        result = lm.cast(activated, to="bf16", id="round_output")
        lm.store(out[row, :], result, coalesced=False, id="store_out")
