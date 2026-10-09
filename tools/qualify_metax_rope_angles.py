#!/usr/bin/env python3
"""Original-domain RoPE angle comparison; production TF32 admission stays closed."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]

TASK = 'L1/011_rotary_position_embedding'
TARGET = 'xcore1002'
SHAPES = ((1, 2048), (16, 256), (2, 131))
PRECISIONS = ('tf32', 'ieee')
TF32 = 'triton.dot.fp32_tf32'


def angle_source(batch, sequence, precision):
    if ((batch, sequence) not in SHAPES or type(batch) is not int
            or type(sequence) is not int or precision not in PRECISIONS):
        raise ValueError('angle probe requires one of its three original shapes and explicit precision')
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="rope_angle_{precision}", target="xcore1002", backend="triton", entry_point="rope_angle_{precision}")
def candidate(lm, position_ids: cake.Tensor(({batch},{sequence},1), "int64"),
              inv_freq: cake.Tensor((64,1), "fp32"), angles: cake.Tensor(({batch},{sequence},64), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0,1,2,3])
    frequency = lm.program(inv_freq,axis=0,dimension=0,tile=16)
    row = lm.program(position_ids,axis=1,dimension=1,tile=32)
    batch = lm.program(position_ids,axis=2,dimension=0,tile=1)
    with compute:
        k = lm.coordinate(source="range",start=0,extent=32)
        frequencies = lm.load(inv_freq[frequency,k],id="load_frequency_kpad")
        positions_i64 = lm.load(position_ids[batch,row,k],id="load_positions_kpad")
        positions = lm.cast(positions_i64,to="fp32",id="positions_to_fp32")
        product = lm.mma(frequencies,positions,instruction={{"contract":"triton.dot.fp32_{precision}"}},tile_shape=(16,32,32),id="angle_product")
        transposed = lm.transpose(product,id="angle_transpose")
        lm.store(angles[batch,row,frequency],transposed,id="store_angles")
'''


def probe_emission(batch, sequence, precision):
    from open_cake_ir.compiler import Target, frontend
    from open_cake_ir.compiler.ir import Schedule
    from open_cake_ir.compiler.backends.triton import emit, preflight
    from open_cake_ir.compiler.verifier import verify
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    if TF32 in target.instruction_contracts:
        raise ValueError('pre-admission angle probe requires production TF32 to remain closed')
    document = frontend.parse(angle_source(batch, sequence, precision)).document
    schedule = Schedule.from_dict(document)
    probe = replace(target, instruction_contracts=target.instruction_contracts | {TF32})
    if any(item.blocks_lowering for item in (*verify(schedule, probe), *preflight(schedule, probe))):
        raise ValueError('angle probe has a blocking software finding')
    return document, emit(schedule, probe), target


def compare_angles(expected, observed, batch, sequence):
    """Retain bit equality and finite error without inventing a Bench tolerance."""
    count = batch * sequence * 64
    if len(expected) != count * 4 or len(observed) != count * 4:
        raise ValueError('angle comparison requires the complete declared FP32 output')
    reference_words = struct.unpack('<' + 'I' * count, expected)
    output_words = struct.unpack('<' + 'I' * count, observed)
    reference = struct.unpack('<' + 'f' * count, expected)
    actual = struct.unpack('<' + 'f' * count, observed)
    if any(not math.isfinite(value) for value in reference):
        raise ValueError('original sampled angle reference must be finite')
    mismatch = sum(left != right for left, right in zip(reference_words, output_words, strict=True))
    finite_errors = [abs(left - right) for left, right in zip(reference, actual, strict=True)
                     if math.isfinite(right)]
    return {'elements': count, 'bitwise_equal': mismatch == 0, 'bit_mismatches': mismatch,
            'nonfinite_outputs': sum(not math.isfinite(value) for value in actual),
            'maximum_finite_absolute_error': max(finite_errors) if finite_errors else None,
            'comparison': 'exact_fp32_bits; finite errors are diagnostic only'}
