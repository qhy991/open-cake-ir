"""RoPE CAKE source for the original C550-Bench L1/011 tensor ABI.

Requires the INT64 storage successor and independently qualified MACA FP32 trig.
The original factory owns frequency scaling. The scalar bridge freezes and checks
the original attention_scaling argument before launching this three-tensor ABI.
"""
from __future__ import annotations

import math

TASK = 'L1/011_rotary_position_embedding'
TARGET = 'xcore1002'


def source_for(batch_size: int, seq_len: int, *, attention_scaling: float = 1.0) -> str:
    if any(type(value) is not int or value <= 0 for value in (batch_size, seq_len)):
        raise ValueError('RoPE batch and sequence dimensions must be positive integers')
    if type(attention_scaling) is not float or not math.isfinite(attention_scaling):
        raise ValueError('RoPE requires its explicit finite FP32 scalar specialization')
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="c550_bench_rope_b{batch_size}_s{seq_len}", target="xcore1002", backend="triton", entry_point="cake_bench_rope")
def candidate(lm, position_ids: cake.Tensor(({batch_size}, {seq_len}), "int64"),
              inv_freq: cake.Tensor((64,), "fp32"),
              cos_sin: cake.Tensor(({batch_size}, {seq_len}, 128, 2), "bf16", mode="output")):
    compute = lm.role(execution_groups=[0])
    batch = lm.program(position_ids, axis=0, dimension=0, tile=1)
    row = lm.program(position_ids, axis=1, dimension=1, tile=1)
    frequency = lm.program(cos_sin, axis=2, dimension=2, tile=1)
    with compute:
        positions_i64 = lm.load(position_ids[batch, row], id="load_positions")
        positions = lm.cast(positions_i64, to="fp32", id="positions_to_fp32")
        frequency_index = lm.coordinate(source="program", name="frequency", id="frequency_index")
        inverse_index = frequency_index % 64
        frequencies = lm.load(inv_freq[lm.scalar_index(inverse_index)], id="load_scaled_inv_freq")
        angles = positions * frequencies
        cosine = lm.cos(angles, instruction={{"contract": "maca.cos.f32"}}, id="cosine")
        sine = lm.sin(angles, instruction={{"contract": "maca.sin.f32"}}, id="sine")
        scaled_cosine = cosine * {attention_scaling!r}
        scaled_sine = sine * {attention_scaling!r}
        components = lm.coordinate(source="range", start=0, extent=2, id="components")
        use_cosine = lm.compare(components, 0, op="eq", id="use_cosine")
        stacked = lm.select(use_cosine, scaled_cosine, scaled_sine, id="stack_cos_sin")
        rounded = lm.cast(stacked, to="bf16", id="round_output")
        lm.store(cos_sin[batch, row, frequency, :], rounded, coalesced=False, id="store_output")
'''


def source_for_workload(workload, case_id: str) -> str:
    abi = workload.tensor_abi(case_id)
    if workload.target != TARGET or len(abi) != 3 or len(abi[0].shape) != 2:
        raise ValueError('RoPE Workload target or argument count differs')
    batch, sequence = abi[0].shape
    expected = (('position_ids', (batch, sequence), 'int64', 'input'),
                ('inv_freq', (64,), 'fp32', 'input'),
                ('cos_sin', (batch, sequence, 128, 2), 'bf16', 'output'))
    if tuple((arg.name, tuple(arg.shape), arg.dtype, arg.mode) for arg in abi) != expected:
        raise ValueError('RoPE original ordered tensor ABI differs')
    scalars = workload.document['semantics'].get('fixed_scalar_inputs')
    if (not isinstance(scalars, dict) or set(scalars) != {'attention_scaling'}
            or not isinstance(scalars['attention_scaling'], dict)
            or set(scalars['attention_scaling']) != {'dtype', 'value', 'binding'}
            or scalars['attention_scaling']['dtype'] != 'float32'
            or scalars['attention_scaling']['binding'] not in {'literal_input', 'original_factory_literal'}):
        raise ValueError('RoPE requires the original checked scalar binding')
    return source_for(batch, sequence, attention_scaling=scalars['attention_scaling']['value'])
