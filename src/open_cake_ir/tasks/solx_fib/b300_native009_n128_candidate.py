from open_cake_ir.compiler import frontend as cake


@cake.schedule(
    name="fib009-native-fp16-m8828-n128-k64",
    target="sm_103a",
    backend="native_cuda",
    entry_point="cake_fib009_native_m8828_n128",
)
def candidate(
    lm,
    a: cake.Tensor((8828, 2048), "fp16"),
    b: cake.Tensor((5120, 2048), "fp16"),
    out: cake.Tensor((8828, 5120), "fp16", mode="output"),
):
    epilogue = lm.role(execution_groups=[0, 1, 2, 3])
    mma = lm.role(execution_groups=[4])
    copy = lm.role(execution_groups=[5])
    operands = lm.smem(65536)
    tensor = lm.tmem(65536, tensor_columns=128, allocating_role=epilogue)
    main = lm.pipeline(stages=2)
    ready = lm.barrier(
        count=2, producers=[copy], consumers=[mma],
        pipeline=main, mechanism="mbarrier")
    done = lm.barrier(
        count=1, producers=[mma], consumers=[epilogue],
        mechanism="mbarrier")
    a_stage = operands.view(
        dtype="fp16", shape=(128, 64), byte_offset=0,
        stages=2, swizzle="swizzle_128b")
    b_stage = operands.view(
        dtype="fp16", shape=(128, 64), byte_offset=32768,
        stages=2, swizzle="swizzle_128b")
    acc = tensor.view(dtype="fp32", shape=(128, 128), byte_offset=0, stages=1)
    dot = lm.buffer(dtype="fp32", shape=(128, 128))
    rounded = lm.buffer(dtype="fp16", shape=(128, 128))
    row = lm.program(a, axis=0, dimension=0, tile=128)
    column = lm.program(b, axis=1, dimension=0, tile=128)
    for k in lm.range(
        a, name="contraction", dimension=1, tile=64, num_stages=2,
        warp_specialize=True, disallow_acc_multi_buffer=True,
    ):
        with copy:
            lm.load(
                a[row, k], out=a_stage, movement="tma",
                descriptor_box=(128, 64), signals=[ready],
                pipeline=main, id="load_a")
            lm.load(
                b[column, k], out=b_stage, movement="tma",
                descriptor_box=(128, 64), signals=[ready],
                pipeline=main, id="load_b")
        with mma:
            lm.mma(
                a_stage, b_stage, out=acc, id="mma",
                waits=[ready], signals=[done], pipeline=main,
                instruction={
                    "contract": "tcgen05.mma.cta_group::1.kind::f16",
                    "shape": [128, 128, 16], "cta_group": 1,
                    "operand_source": "shared", "operand_major": ["k", "k"],
                },
                tile_shape=(128, 128, 64),
            )
    with epilogue:
        lm.load(
            acc, out=dot, movement="tmem",
            source_atom={"op": "tcgen05.Ld32x32b", "repetition": 32},
            waits=[done], id="read_acc")
        lm.cast(dot, to="fp16", out=rounded, id="round_out")
        lm.store(out[row, column], rounded, coalesced=False, id="store")
