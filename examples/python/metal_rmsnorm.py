"""Weighted FP32 RMSNorm: fixed epsilon after the row mean of squared inputs."""
from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="metal-rmsnorm", target="apple_gpu_family8", backend="metal",
               entry_point="cake_rmsnorm")
def rmsnorm(lm, x: cake.Tensor((128, 1024), "fp32"),
            weight: cake.Tensor((1024,), "fp32"),
            out: cake.Tensor((128, 1024), "fp32", mode="output")):
    compute = lm.role(warps=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[row, :], id="load_x")
        weights = lm.load(weight[:], id="load_weight")
        squared = lm.square(values, id="square")
        total = lm.reduce(squared, op="sum", axis=0, scope="cta", across_loop=False, id="sum")
        mean = total / 1024.0
        variance = mean + 0.00001
        inverse = lm.rsqrt(variance, id="rsqrt")
        normalized = values * inverse
        result = normalized * weights
        lm.store(out[row, :], result, coalesced=False, id="store_out")
