"""FP32 row tiles; one active Metal lane evaluates (x + y) * 2."""
from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="metal-elementwise", target="apple_gpu_family8", backend="metal",
               entry_point="cake_elementwise")
def elementwise(lm, x: cake.Tensor((3, 37), "fp32"),
                y: cake.Tensor((3, 37), "fp32"),
                out: cake.Tensor((3, 37), "fp32", mode="output")):
    compute = lm.role(warps=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        a = lm.load(x[row, :], id="load_x")
        b = lm.load(y[row, :], id="load_y")
        result = (a + b) * 2.0
        lm.store(out[row, :], result, coalesced=False, id="store_out")
