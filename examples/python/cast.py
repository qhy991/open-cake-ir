"""BF16-to-FP32 cast using the canonical frontend `to` parameter."""
from open_cake_ir.compiler import frontend as cake

@cake.schedule(name="cast-b8-smoke-v1", target="sm_100a", backend="triton",
               entry_point="cake_cast_b8_smoke",
               metadata={"workload_contract_sha256": "0000000000000000000000000000000000000000000000000000000000000000"},
               residency={"ctas_per_multiprocessor": 4, "registers_per_thread": 64})
def cast(lm, x: cake.Tensor((8, 128), "bf16"), y: cake.Tensor((8, 128), "fp32", mode="output")):
    compute = lm.role(warps=[0, 1, 2, 3])
    batch = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        x_tile = lm.load(x[batch, :], reuse="streamed", id="load_x")
        y_tile = lm.cast(x_tile, to="fp32", id="cast_x")
        lm.store(y[batch, :], y_tile, id="store_y")
