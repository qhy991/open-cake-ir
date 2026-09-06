"""FP32 row sum; odd row extents need no padded input."""
from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="metal-row-sum", target="apple_gpu_family8", backend="metal",
               entry_point="cake_row_sum")
def row_sum(lm, x: cake.Tensor((5, 65), "fp32"),
            out: cake.Tensor((5,), "fp32", mode="output")):
    compute = lm.role(warps=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[row, :], id="load_x")
        total = lm.reduce(values, op="sum", axis=0, scope="cta", across_loop=False, id="row_sum")
        lm.store(out[row], total, coalesced=False, id="store_out")
