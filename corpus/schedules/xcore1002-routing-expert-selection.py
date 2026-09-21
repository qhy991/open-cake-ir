from open_cake_ir.compiler import frontend as cake

@cake.schedule(name="moe_expert_selection", target="xcore1002", backend="triton", entry_point="cake_moe_expert_selection")
def candidate(lm, routing_logits: cake.Tensor((1, 256), "fp32", mode="input"), routing_bias: cake.Tensor((256,), "bf16", mode="input"), selected_groups: cake.Tensor((1, 4), "int32", mode="input"), expert_ids: cake.Tensor((1, 8), "int32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    token = lm.program(expert_ids, axis=0, dimension=0, tile=1)
    with compute:
        ids = lm.coordinate(source="range", start=0, extent=256)
        group_ids = ids // 32
        logits = lm.load(routing_logits[token, :])
        bias = lm.load(routing_bias[:])
        bias32 = lm.cast(bias, to="fp32")
        negative = logits * -1.0
        exp_negative = lm.exp(negative)
        denominator = exp_negative + 1.0
        sigmoid = lm.reciprocal(denominator)
        scores = sigmoid + bias32
        group_0 = lm.load(selected_groups[token, 0:1])
        allowed_0 = lm.compare(group_ids, group_0, op="eq")
        group_1 = lm.load(selected_groups[token, 1:2])
        allowed_1 = lm.compare(group_ids, group_1, op="eq")
        group_2 = lm.load(selected_groups[token, 2:3])
        allowed_2 = lm.compare(group_ids, group_2, op="eq")
        group_3 = lm.load(selected_groups[token, 3:4])
        allowed_3 = lm.compare(group_ids, group_3, op="eq")
        allowed = allowed_0 + allowed_1 + allowed_2 + allowed_3
        pruned = lm.select(allowed, scores, "negative_infinity")
        best = lm.buffer(shape=(8,), dtype="fp32")
        chosen = lm.buffer(shape=(8,), dtype="int32")
        lm.top_k(pruned, k=8, tie_break="lowest_index", nan_policy="reject_input", out=[best, chosen])
        lm.store(expert_ids[token, :], chosen, coalesced=False)
