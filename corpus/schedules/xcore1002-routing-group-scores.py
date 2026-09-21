from open_cake_ir.compiler import frontend as cake

@cake.schedule(name="moe_group_scores", target="xcore1002", backend="triton", entry_point="cake_moe_group_scores")
def candidate(lm, routing_logits: cake.Tensor((1, 256), "fp32", mode="input"), routing_bias: cake.Tensor((256,), "bf16", mode="input"), group_scores: cake.Tensor((1, 8), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    token = lm.program(group_scores, axis=0, dimension=0, tile=1)
    group = lm.program(group_scores, axis=1, dimension=1, tile=1)
    with compute:
        group_id = lm.coordinate(source="program", name="group")
        lanes = lm.coordinate(source="range", start=0, extent=32)
        base = group_id * 32
        experts = lanes + base
        logits = lm.load(routing_logits[token, experts])
        bias = lm.load(routing_bias[experts])
        bias32 = lm.cast(bias, to="fp32")
        negative = logits * -1.0
        exp_negative = lm.exp(negative)
        denominator = exp_negative + 1.0
        sigmoid = lm.reciprocal(denominator)
        scores = sigmoid + bias32
        best = lm.buffer(shape=(2,), dtype="fp32")
        chosen = lm.buffer(shape=(2,), dtype="int32")
        lm.top_k(scores, k=2, tie_break="lowest_index", nan_policy="reject_input", out=[best, chosen])
        total = lm.reduce(best, op="sum", axis=0, across_loop=False)
        lm.store(group_scores[token, group], total, coalesced=False)
