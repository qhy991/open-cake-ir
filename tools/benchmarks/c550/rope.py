"""RoPE CAKE source for the original C550-Bench L1/011 tensor ABI.

Requires the INT64 storage successor and independently qualified MACA FP32 trig.
The original factory owns frequency scaling. The scalar bridge freezes and checks
the original attention_scaling argument before launching this three-tensor ABI.
"""
from __future__ import annotations

import math

from open_cake_ir.tasks.c550_bench.binding import (
    BENCH_COMMIT, validate_document, validate_oracle_numerics,
)

TASK = 'L1/011_rotary_position_embedding'
TARGET = 'xcore1002'


def source_for(batch_size: int, seq_len: int, *, attention_scaling: float = 1.0) -> str:
    """Unquantized IEEE angle helper, retained as the original negative control."""
    return _source(batch_size, seq_len, attention_scaling, rne10=False)


def source_for_rne10(batch_size: int, seq_len: int) -> str:
    """Original-domain helper; the full Bench binding is checked by source_for_workload.

    Frequencies must be zero or within [2^-19, 2047]. Positions must be the
    original sequential integers in [0, 2047]. This is not a general TF32 cast.
    """
    if type(seq_len) is not int or not 1 <= seq_len <= 2048:
        raise ValueError('Bounded RoPE requires original positions in [0, 2047]')
    return _source(batch_size, seq_len, 1.0, rne10=True)


def _source(batch_size, seq_len, attention_scaling, *, rne10):
    if any(type(value) is not int or value <= 0 for value in (batch_size, seq_len)):
        raise ValueError('RoPE batch and sequence dimensions must be positive integers')
    if type(attention_scaling) is not float or not math.isfinite(attention_scaling):
        raise ValueError('RoPE requires its explicit finite FP32 scalar specialization')
    conversion = '''        scaled_frequency = frequencies * 32.0
        frequency_half = lm.cast(scaled_frequency, to="fp16", id="round_scaled_frequency_to_fp16")
        frequency_wide = lm.cast(frequency_half, to="fp32", id="widen_rounded_frequency")
        frequency_rne10 = frequency_wide * 0.03125
''' if rne10 else ''
    angle_frequency = 'frequency_rne10' if rne10 else 'frequencies'
    suffix = '_rne10' if rne10 else ''
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="c550_bench_rope_b{batch_size}_s{seq_len}{suffix}", target="xcore1002", backend="triton", entry_point="cake_bench_rope")
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
{conversion}        angles = positions * {angle_frequency}
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
    document = workload.document
    semantics = document['semantics']
    binding = semantics.get('benchmark', {})
    if (case_id != 'primary' or binding.get('task') != TASK
            or binding.get('commit') != BENCH_COMMIT):
        raise ValueError('Bounded RoPE requires the fixed original Bench task and primary case')
    policy = validate_oracle_numerics(semantics.get('oracle_numerics'))
    if (policy['float32_matmul_precision'] != 'high' or not policy['allow_tf32']
            or policy['initialization']['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] != '1'):
        raise ValueError('Bounded RoPE requires the retained original HIGH numerical policy')
    abi = workload.tensor_abi(case_id)
    if workload.target != TARGET or len(abi) != 3 or len(abi[0].shape) != 2:
        raise ValueError('RoPE Workload target or argument count differs')
    batch, sequence = abi[0].shape
    expected = (('position_ids', (batch, sequence), 'int64', 'input'),
                ('inv_freq', (64,), 'fp32', 'input'),
                ('cos_sin', (batch, sequence, 128, 2), 'bf16', 'output'))
    if tuple((arg.name, tuple(arg.shape), arg.dtype, arg.mode) for arg in abi) != expected:
        raise ValueError('RoPE original ordered tensor ABI differs')
    if sequence > 2048:
        raise ValueError('Bounded RoPE requires original positions in [0, 2047]')
    scalars = semantics.get('fixed_scalar_inputs')
    if (not isinstance(scalars, dict) or set(scalars) != {'attention_scaling'}
            or not isinstance(scalars['attention_scaling'], dict)
            or set(scalars['attention_scaling']) != {'dtype', 'value', 'binding'}
            or scalars['attention_scaling']['dtype'] != 'float32'
            or scalars['attention_scaling']['binding'] != 'original_factory_literal'
            or type(scalars['attention_scaling']['value']) is not float
            or scalars['attention_scaling']['value'] != 1.0):
        raise ValueError('RoPE requires the original checked scalar binding')
    if (document['oracle']['custom_inputs_entrypoint'] != 'get_inputs'
            or semantics.get('original_input_specifications') != {
                name: {'type': 'custom'} for name in ('position_ids', 'inv_freq', 'attention_scaling')}):
        raise ValueError('Bounded RoPE requires its known original input factory')
    # The original Bench owner binds the complete factory/reference text, input
    # specifications and tolerance. An ABI alone cannot prove the value domain.
    validate_document(document)
    return source_for_rne10(batch, sequence)
