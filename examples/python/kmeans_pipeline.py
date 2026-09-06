from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="python-kmeans-pipeline", target="sm_100a", backend="cutlass_cute_dsl",
               entry_point="cake_flash_kmeans_assignment_full", grid=(1, 1, 1))
def kmeans(lm, tokens: cake.Tensor((128, 128), "bf16"),
           centroids: cake.Tensor((1024, 128), "bf16"),
           centroid_sq: cake.Tensor((1024,), "fp32")):
    epilogue = lm.role(warps=[0, 1, 2, 3])
    mma = lm.role(warps=[4])
    tma = lm.role(warps=[5])
    reduce = lm.role(warps=[6])
    smem_operands = lm.smem(98304)
    tmem_accumulator = lm.tmem(131072, tensor_columns=256, allocating_role=epilogue)
    main = lm.pipeline(stages=2)
    tiles_ready = lm.barrier(count=2, producers=[tma], consumers=[mma], pipeline=main, mechanism="mbarrier")
    accumulator_ready = lm.barrier(count=1, producers=[mma], consumers=[epilogue], mechanism="mbarrier")
    distance_ready = lm.barrier(count=1, producers=[epilogue], consumers=[reduce], mechanism="barrier.sync")
    distance_scratch = lm.buffer(space="global", dtype="fp32", shape=(128, 1024))
    best_index = lm.buffer(dtype="int32", shape=(128,))
    assignments = lm.buffer(space="global", dtype="int32", shape=(128,), mode="output")
    token_stage = smem_operands.view(dtype="bf16", shape=(128, 64), byte_offset=0,
                                      stages=2, swizzle="swizzle_128b")
    centroid_stage = smem_operands.view(dtype="bf16", shape=(256, 64), byte_offset=32768,
                                         stages=2, swizzle="swizzle_128b")
    accumulator = tmem_accumulator.view(dtype="fp32", shape=(128, 256), byte_offset=0, stages=1)
    for centroid_tile in lm.range(centroids, name="centroid_loop", dimension=0, tile=256,
                                  warp_specialize=True, disallow_acc_multi_buffer=True):
        for k_tile in lm.range(centroids, name="k_loop", dimension=1, tile=64, num_stages=2,
                               warp_specialize=True, disallow_acc_multi_buffer=True):
            with tma:
                lm.load(tokens, out=token_stage, movement="tma", descriptor_box=(128, 64),
                        signals=[tiles_ready], pipeline=main, id="load_tokens")
                lm.load(centroids, out=centroid_stage, movement="tma", descriptor_box=(256, 64),
                        signals=[tiles_ready], pipeline=main, id="load_centroids")
            with mma:
                lm.mma(token_stage, centroid_stage, out=accumulator, id="dot_mma",
                       waits=[tiles_ready], signals=[accumulator_ready], pipeline=main,
                       instruction={"contract": "tcgen05.mma.cta_group::1.kind::f16", "shape": [128, 256, 16],
                                    "cta_group": 1, "operand_source": "shared", "operand_major": ["k", "k"]},
                       tile_shape=(128, 256, 64))
        with epilogue:
            lm.epilogue(accumulator, centroid_sq, out=distance_scratch, id="distance_epilogue",
                        waits=[accumulator_ready], signals=[distance_ready],
                        formula="centroid_sq_minus_two_dot", coalesced=True, subtile=(128, 64),
                        source_atom={"op": "tcgen05.Ld32x32b", "repetition": 64})
    with reduce:
        lm.reduce_argmin(distance_scratch, out=best_index, id="argmin", waits=[distance_ready],
                         tie_break="lowest_index", nan_policy="reject_input")
        lm.store(assignments, best_index, id="store_assignment")
