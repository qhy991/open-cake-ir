from open_cake_ir.compiler import frontend as cake


@cake.schedule(
    name="fib004-native-fp16-m8828-k64-s4",
    target="sm_103a",
    backend="native_cuda",
    entry_point="cake_fib004_native_m8828_s4",
)
def candidate(
    lm,
    a: cake.Tensor((8828, 2048), "fp16"),
    b: cake.Tensor((128, 2048), "fp16"),
    out: cake.Tensor((8828, 128), "fp16", mode="output"),
):
    epilogue = lm.role(execution_groups=[0, 1, 2, 3])
    mma = lm.role(execution_groups=[4])
    copy = lm.role(execution_groups=[5])
    operands = lm.smem(98304)
    tensor = lm.tmem(32768, tensor_columns=64, allocating_role=epilogue)
    main = lm.pipeline(stages=4)
    ready = lm.barrier(
        count=2, producers=[copy], consumers=[mma],
        pipeline=main, mechanism="mbarrier")
    done = lm.barrier(
        count=1, producers=[mma], consumers=[epilogue],
        mechanism="mbarrier")
    a_stage = operands.view(
        dtype="fp16", shape=(128, 64), byte_offset=0,
        stages=4, swizzle="swizzle_128b")
    b_stage = operands.view(
        dtype="fp16", shape=(64, 64), byte_offset=65536,
        stages=4, swizzle="swizzle_128b")
    acc = tensor.view(dtype="fp32", shape=(128, 64), byte_offset=0, stages=1)
    dot = lm.buffer(dtype="fp32", shape=(128, 64))
    rounded = lm.buffer(dtype="fp16", shape=(128, 64))
    row = lm.program(a, axis=0, dimension=0, tile=128)
    column = lm.program(b, axis=1, dimension=0, tile=64)
    for k in lm.range(
        a, name="contraction", dimension=1, tile=64, num_stages=4,
        warp_specialize=True, disallow_acc_multi_buffer=True,
    ):
        with copy:
            lm.load(
                a[row, k], out=a_stage, movement="tma",
                descriptor_box=(128, 64), signals=[ready],
                pipeline=main, id="load_a")
            lm.load(
                b[column, k], out=b_stage, movement="tma",
                descriptor_box=(64, 64), signals=[ready],
                pipeline=main, id="load_b")
        with mma:
            lm.mma(
                a_stage, b_stage, out=acc, id="mma",
                waits=[ready], signals=[done], pipeline=main,
                instruction={
                    "contract": "tcgen05.mma.cta_group::1.kind::f16",
                    "shape": [128, 64, 16], "cta_group": 1,
                    "operand_source": "shared", "operand_major": ["k", "k"],
                },
                tile_shape=(128, 64, 64),
            )
    with epilogue:
        lm.load(
            acc, out=dot, movement="tmem",
            source_atom={"op": "tcgen05.Ld32x32b", "repetition": 16},
            waits=[done], id="read_acc")
        lm.cast(dot, to="fp16", out=rounded, id="round_out")
        lm.store(out[row, column], rounded, coalesced=False, id="store")
