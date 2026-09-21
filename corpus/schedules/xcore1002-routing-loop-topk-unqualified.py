from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="xcore1002-loop-topk", target="xcore1002", backend="triton", entry_point="loop_topk")
def candidate(lm, scores: cake.Tensor((1, 32), "fp32"),
              out: cake.Tensor((1, 8), "int32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    batch = lm.program(out, axis=0, dimension=0, tile=1)
    best = lm.buffer(shape=(8,), dtype="fp32")
    indices = lm.buffer(shape=(8,), dtype="int32")
    for chunk in lm.range(scores, name="score", dimension=1, tile=8):
        with compute:
            values = lm.load(scores[batch, chunk])
            lm.top_k(values, k=8, across_loop=True, tie_break="lowest_index", nan_policy="reject_input", out=[best, indices])
    with compute:
        lm.store(out[batch, :], indices)
