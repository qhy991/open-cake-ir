"""BF16 tile plus FP32 bias uses the same promotion in either operand order."""
from open_cake_ir.compiler import frontend as cake

@cake.schedule(name="mixed-dtype-b8", target="sm_100a", backend="triton",
               entry_point="cake_mixed_dtype")
def mixed(lm, x: cake.Tensor((8, 128), "bf16"), bias: cake.Tensor((128,), "fp32"),
          y: cake.Tensor((8, 128), "fp32", mode="output")):
    compute = lm.role(warps=[0, 1, 2, 3])
    batch = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        x_tile = lm.load(x[batch, :])
        bias_tile = lm.load(bias[:])
        result = x_tile + bias_tile
        lm.store(y[batch, :], result)
