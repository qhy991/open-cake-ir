from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="indexed-gather-b8-smoke-b300", target="sm_103a", backend="triton",
               entry_point="cake_indexed_gather_b8_smoke",
               metadata={"workload_contract_sha256": "2d031c0821a12ad18178302ba4bfd95263f59381ddacbfa7cfac8382ea95a96c"},
               residency={"ctas_per_multiprocessor": 4, "registers_per_thread": 64})
def indexed_gather(lm, expert_rows: cake.Tensor((4, 8, 16), "bf16"),
                   expert_ids: cake.Tensor((8, 8), "int32"),
                   row_ids: cake.Tensor((8, 8), "int32"),
                   gathered_rows: cake.Tensor((8, 8, 16), "bf16", mode="output")):
    compute = lm.role(warps=[0, 1, 2, 3])
    token = lm.program(expert_ids, axis=0, dimension=0, tile=1)
    with compute:
        expert_id_tile = lm.load(expert_ids[token, :], reuse="streamed", id="load_expert_ids")
        row_id_tile = lm.load(row_ids[token, :], reuse="streamed", id="load_row_ids")
        gathered_tile = lm.load(expert_rows[expert_id_tile, row_id_tile, :],
                                reuse="streamed", id="load_selected_rows")
        lm.store(gathered_rows[token, :, :], gathered_tile, id="store_gathered_rows")
