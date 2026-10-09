"""Pure CAKE starter for the fixed C550-Bench L1/048 public ABI.

The task name says SwiGLU, but its reference applies GELU-tanh to the gate.
Only shape metadata is consumed here. No reference code, tensors or device math
is executed by this module. Native correctness and timing need separate gates.
"""
from __future__ import annotations

import math

TASK = "L1/048_fused_gate_up_projection_with_swiglu"
TARGET = "xcore1002"
HIDDEN_SIZE = 3072
INTERMEDIATE_SIZE = 24576


def _projection_source(batch, sequence, hidden, intermediate, *, name, weight, output):
    return f'''@cake.schedule(name="{name}", target="{TARGET}", backend="triton", entry_point="cake_{name}")
def {name}(lm, x: cake.Tensor(({batch}, {sequence}, {hidden}), "bf16"),
           {weight}: cake.Tensor(({intermediate}, {hidden}), "bf16"),
           {output}: cake.Tensor(({batch}, {sequence}, {intermediate}), "bf16", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row = lm.program(x, axis=0, dimension=1, tile=16)
    column = lm.program({weight}, axis=1, dimension=0, tile=32)
    batch = lm.program(x, axis=2, dimension=0, tile=1)
    for k in lm.range(x, name="hidden_loop", dimension=2, tile=64,
                      num_stages=1, loop_unroll_factor=1):
        with compute:
            left = lm.load(x[batch, row, k], reuse="streamed", id="load_x")
            right = lm.load({weight}[column, k], reuse="streamed", id="load_weight")
            projection = lm.mma(left, right, instruction={{"contract": "triton.dot.bf16_fp32"}},
                                tile_shape=(16, 32, 64), id="projection")
    with compute:
        rounded_projection = lm.cast(projection, to="bf16", id="round_projection")
        lm.store({output}[batch, row, column], rounded_projection, id="store_projection")
'''


def _activation_source(batch, sequence, intermediate):
    return f'''@cake.schedule(name="gelu_product", target="{TARGET}", backend="triton", entry_point="cake_gelu_product")
def gelu_product(lm, gate_output: cake.Tensor(({batch}, {sequence}, {intermediate}), "bf16"),
                 up_output: cake.Tensor(({batch}, {sequence}, {intermediate}), "bf16"),
                 output: cake.Tensor(({batch}, {sequence}, {intermediate}), "bf16", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row = lm.program(gate_output, axis=0, dimension=1, tile=1)
    column = lm.program(gate_output, axis=1, dimension=2, tile=256)
    batch = lm.program(gate_output, axis=2, dimension=0, tile=1)
    with compute:
        gate_stored = lm.load(gate_output[batch, row, column], id="load_gate")
        gate = lm.cast(gate_stored, to="fp32", id="widen_gate")
        gate_squared = lm.square(gate, id="gate_squared")
        gate_cubed = gate_squared * gate
        inner = (gate + gate_cubed * 0.044715) * {math.sqrt(2.0 / math.pi)!r}
        tangent = lm.tanh(inner, instruction={{"contract": "maca.tanh.f32"}}, id="tanh")
        activated = gate * 0.5 * (tangent + 1.0)
        activated_bf16 = lm.cast(activated, to="bf16", id="round_activation")
        activated_fp32 = lm.cast(activated_bf16, to="fp32", id="widen_activation")
        up_stored = lm.load(up_output[batch, row, column], id="load_up")
        up = lm.cast(up_stored, to="fp32", id="widen_up")
        product = activated_fp32 * up
        rounded_product = lm.cast(product, to="bf16", id="round_product")
        lm.store(output[batch, row, column], rounded_product, id="store_output")
'''


def source_for(batch_size: int, seq_len: int) -> str:
    """Return a complete static Program for one original, unflattened shape."""
    if any(type(value) is not int or value <= 0 for value in (batch_size, seq_len)):
        raise ValueError("Gate/up batch and sequence dimensions must be positive integers")
    gate = _projection_source(batch_size, seq_len, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                              name="gate_projection", weight="gate_proj", output="gate_output")
    up = _projection_source(batch_size, seq_len, HIDDEN_SIZE, INTERMEDIATE_SIZE,
                            name="up_projection", weight="up_proj", output="up_output")
    activation = _activation_source(batch_size, seq_len, INTERMEDIATE_SIZE)
    return ('from open_cake_ir.compiler import frontend as cake\n\n' + gate + '\n' + up
            + '\n' + activation + f'''
cake.program(program_id="c550_bench_l1_048_b{batch_size}_s{seq_len}",
    inputs=("x", "gate_proj", "up_proj"), outputs=("output",), stages=(
        cake.stage(name="gate_projection", schedule=gate_projection,
                   bindings={{"x": "x", "gate_proj": "gate_proj", "gate_output": "gate_output"}}),
        cake.stage(name="up_projection", schedule=up_projection,
                   bindings={{"x": "x", "up_proj": "up_proj", "up_output": "up_output"}}),
        cake.stage(name="gelu_product", schedule=gelu_product,
                   bindings={{"gate_output": "gate_output", "up_output": "up_output", "output": "output"}}),
))
''')


def program_for(batch_size: int, seq_len: int):
    from open_cake_ir.compiler.program_frontend import parse_program
    return parse_program(source_for(batch_size, seq_len)).program


def source_for_workload(workload, case_id: str) -> str:
    """Bind the starter to the exact ordered external Workload ABI."""
    abi = workload.tensor_abi(case_id)
    if workload.target != TARGET or len(abi) != 4 or len(abi[0].shape) != 3:
        raise ValueError("Gate/up Workload target or argument count differs")
    batch, sequence, _ = abi[0].shape
    expected = (
        ("x", (batch, sequence, HIDDEN_SIZE), "bf16", "input"),
        ("gate_proj", (INTERMEDIATE_SIZE, HIDDEN_SIZE), "bf16", "input"),
        ("up_proj", (INTERMEDIATE_SIZE, HIDDEN_SIZE), "bf16", "input"),
        ("output", (batch, sequence, INTERMEDIATE_SIZE), "bf16", "output"),
    )
    if tuple((arg.name, tuple(arg.shape), arg.dtype, arg.mode) for arg in abi) != expected:
        raise ValueError("Gate/up original ordered tensor ABI differs")
    return source_for(batch, sequence)
