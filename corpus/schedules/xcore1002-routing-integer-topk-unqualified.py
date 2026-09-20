from open_cake_ir.compiler import frontend as cake

@cake.schedule(name="moe_group_selection", target="xcore1002", backend="triton", entry_point="cake_moe_group_selection")
def candidate(lm, group_scores: cake.Tensor((1, 8), "int32", mode="input"), selected_groups: cake.Tensor((1, 4), "int32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    token = lm.program(selected_groups, axis=0, dimension=0, tile=1)
    with compute:
        scores = lm.load(group_scores[token, :])
        best = lm.buffer(shape=(4,), dtype="int32")
        chosen = lm.buffer(shape=(4,), dtype="int32")
        lm.top_k(scores, k=4, tie_break="lowest_index", nan_policy="reject_input", out=[best, chosen])
        lm.store(selected_groups[token, :], chosen, coalesced=False)
