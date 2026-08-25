"""Source-ordered one-block llama.cpp Q4_0 x live-Q8_1 conformance oracle."""

from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass
from typing import Mapping, cast

from .workload import WorkloadContract


BLOCK_SIZE = 32
Q4_0_RECORD_BYTES = 18
Q8_1_RECORD_BYTES = 36
Q8_1_PADDED_BLOCKS = 16
Q8_1_WORKSPACE_BYTES = Q8_1_RECORD_BYTES * Q8_1_PADDED_BLOCKS


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _f32(value: float | int) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _f16(value: float | int) -> float:
    return struct.unpack("<e", struct.pack("<e", float(value)))[0]


def _roundf(value: float) -> int:
    """C roundf semantics: nearest integer with halfway cases away from zero."""

    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def _warp_reduce_sum(values: tuple[float, ...]) -> float:
    if len(values) != BLOCK_SIZE:
        raise ValueError("Q8_1 reduction requires exactly 32 lanes")
    lanes = tuple(_f32(value) for value in values)
    for offset in (16, 8, 4, 2, 1):
        previous = lanes
        lanes = tuple(
            _f32(previous[index] + previous[index ^ offset])
            for index in range(BLOCK_SIZE)
        )
    return lanes[0]


@dataclass(frozen=True)
class Q4MmvqMaterial:
    """Public inputs: one FP32 activation row and one raw GGML Q4_0 block."""

    activation: tuple[float, ...]
    q4_0_block: bytes


@dataclass(frozen=True)
class Q4MmvqReference:
    """Named correctness endpoints for the two-kernel conformance boundary."""

    q8_1_workspace: bytes
    q8_quants: tuple[int, ...]
    q8_stored_scale: float
    q8_stored_sum: float
    output: float

    @property
    def consumed_q8_1_block(self) -> bytes:
        return self.q8_1_workspace[:Q8_1_RECORD_BYTES]


def _q4_block(scale: float, packed: bytes) -> bytes:
    if len(packed) != Q4_0_RECORD_BYTES - 2:
        raise ValueError("Q4_0 payload must contain 16 packed bytes")
    return struct.pack("<e", scale) + packed


def _random_material(seed: int) -> Q4MmvqMaterial:
    generator = random.Random(seed)
    activation = tuple(_f32(generator.uniform(-5.0, 5.0)) for _ in range(BLOCK_SIZE))
    packed = bytes(generator.randrange(0, 256) for _ in range(16))
    return Q4MmvqMaterial(activation, _q4_block(-0.3125, packed))


def _stored_sum_stress_material() -> Q4MmvqMaterial:
    # A nonzero-mean, nonuniform row makes half(sum(x)) observably different from
    # half(half(d) * sum(q)); that distinction owns the source-exact correction.
    activation = tuple(
        _f32((0.071 * (index + 1)) + (0.00037 if index % 3 == 0 else -0.00019))
        for index in range(BLOCK_SIZE)
    )
    packed = bytes(((index * 29 + 7) & 0xFF) for index in range(16))
    return Q4MmvqMaterial(activation, _q4_block(0.1875, packed))


def _zero_material() -> Q4MmvqMaterial:
    packed = bytes(((index << 4) | (15 - index)) & 0xFF for index in range(16))
    return Q4MmvqMaterial((0.0,) * BLOCK_SIZE, _q4_block(0.5, packed))


def _rounding_boundary_material() -> Q4MmvqMaterial:
    activation = (127.0, 0.5, -0.5) + (0.0,) * (BLOCK_SIZE - 3)
    packed = bytes((0xF0 if index % 2 == 0 else 0x0F) for index in range(16))
    return Q4MmvqMaterial(activation, _q4_block(-0.25, packed))


def _xor_tree_material() -> Q4MmvqMaterial:
    activation = (
        7.749633368803188e-05,
        -0.0007529330323450267,
        -0.0009441768052056432,
        17.744260787963867,
        -4.191210973658599e-05,
        -0.005857855547219515,
        -0.058281198143959045,
        5.700963973999023,
        0.7684974670410156,
        -0.0005712753627449274,
        -0.00767248822376132,
        -55.02885437011719,
        0.0008660975727252662,
        9.71146011352539,
        7.489255905151367,
        0.029565194621682167,
        -205.21853637695312,
        55.94367218017578,
        0.02320975437760353,
        -0.0001902196672745049,
        -0.5833588242530823,
        -16.543155670166016,
        -0.0009459370048716664,
        785.606689453125,
        -656.573486328125,
        -0.0009622555808164179,
        0.006698894314467907,
        3.223262319806963e-05,
        0.05916828662157059,
        779.940673828125,
        -719.0831298828125,
        -0.00039326149271801114,
    )
    packed = bytes(((index * 17 + 3) & 0xFF) for index in range(16))
    return Q4MmvqMaterial(
        tuple(_f32(value) for value in activation), _q4_block(0.375, packed)
    )


def _q4_zero_scale_material() -> Q4MmvqMaterial:
    activation = tuple(_f32((index - 15.5) / 3.0) for index in range(BLOCK_SIZE))
    packed = bytes(((index * 13 + 11) & 0xFF) for index in range(16))
    return Q4MmvqMaterial(activation, _q4_block(0.0, packed))


def materialize_q4_mmvq_case(
    workload: WorkloadContract, case_id: str
) -> Q4MmvqMaterial:
    """Materialize deterministic raw public inputs without a Torch dependency."""

    case = workload.case(case_id)
    shape = _object(case["shape"], "workload.case.shape")
    if shape != {"N": 1, "M": 1, "K": BLOCK_SIZE}:
        raise ValueError("Q4 MMVQ conformance shape differs")
    mode = case["mode"]
    if mode == "seeded_random_q4_and_activation":
        seed = case["seed"]
        if not isinstance(seed, int):
            raise ValueError("random Q4 MMVQ case requires a seed")
        return _random_material(seed)
    if mode == "constructed_stored_sum_correction_stress":
        return _stored_sum_stress_material()
    if mode == "constructed_zero_scale":
        return _zero_material()
    if mode == "constructed_roundf_half_away_boundary":
        return _rounding_boundary_material()
    if mode == "constructed_xor_tree_sum_discriminator":
        return _xor_tree_material()
    if mode == "seeded_partial_rounding_discriminator":
        return _random_material(1)
    if mode == "constructed_q4_zero_scale":
        return _q4_zero_scale_material()
    raise ValueError(f"unsupported Q4 MMVQ case mode {mode!r}")


def quantize_q8_1_block(
    activation: tuple[float, ...]
) -> tuple[bytes, tuple[int, ...]]:
    if len(activation) != BLOCK_SIZE:
        raise ValueError("Q8_1 producer requires exactly 32 FP32 values")
    values = tuple(_f32(value) for value in activation)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Q8_1 producer rejects non-finite FP32 input")
    amax = max(abs(value) for value in values)
    total = _warp_reduce_sum(values)
    d = _f32(_f32(amax) / _f32(127.0))
    quants = tuple(
        0 if amax == 0.0 else _roundf(_f32(value / d)) for value in values
    )
    if any(value < -128 or value > 127 for value in quants):
        raise ValueError("Q8_1 quantized value is outside int8")
    record = struct.pack("<ee32b", _f16(d), _f16(total), *quants)
    if len(record) != Q8_1_RECORD_BYTES:
        raise AssertionError("Q8_1 record size differs")
    return record, quants


def decode_q4_0_block(record: bytes) -> tuple[float, tuple[int, ...]]:
    if len(record) != Q4_0_RECORD_BYTES:
        raise ValueError("Q4_0 record must contain exactly 18 bytes")
    d4 = struct.unpack("<e", record[:2])[0]
    if not math.isfinite(d4):
        raise ValueError("Q4_0 record rejects a non-finite FP16 scale")
    packed = record[2:]
    low = tuple(value & 0x0F for value in packed)
    high = tuple(value >> 4 for value in packed)
    return d4, low + high


def q4_mmvq_reference(
    workload: WorkloadContract, material: Q4MmvqMaterial
) -> Q4MmvqReference:
    """Evaluate the stored-s Q4_0 x Q8_1 formula for one complete block."""

    semantics = _object(workload.document["semantics"], "workload.semantics")
    if semantics.get("stored_sum") != "fp16(warp_tree_fp32_sum(original_x))":
        raise ValueError("Q4 MMVQ stored-s semantics differ")
    q8_record, quants = quantize_q8_1_block(material.activation)
    q8_workspace = q8_record + bytes(
        Q8_1_RECORD_BYTES * (Q8_1_PADDED_BLOCKS - 1)
    )
    d8, s8 = struct.unpack("<ee", q8_record[:4])
    d4, unsigned_q4 = decode_q4_0_block(material.q4_0_block)
    partial_indices = (
        tuple(range(0, 8)) + tuple(range(16, 24)),
        tuple(range(8, 16)) + tuple(range(24, 32)),
    )
    partials = []
    for indices in partial_indices:
        sumi = sum(unsigned_q4[index] * quants[index] for index in indices)
        scaled_dot = _f32(_f32(sumi) * _f32(d8))
        correction = _f32(_f32(4.0) * _f32(s8))
        centered = _f32(scaled_dot - correction)
        partials.append(_f32(_f32(d4) * centered))
    output = _f32(partials[0] + partials[1])
    return Q4MmvqReference(
        q8_1_workspace=q8_workspace,
        q8_quants=quants,
        q8_stored_scale=d8,
        q8_stored_sum=s8,
        output=output,
    )


def q4_mmvq_metrics(
    workload: WorkloadContract,
    observed_q8_1: bytes,
    observed_output: float,
    reference: Q4MmvqReference,
) -> dict[str, object]:
    """Require byte-exact intermediate custody and the Workload FP32 tolerance."""

    validation = _object(workload.document["validation"], "workload.validation")
    atol = float(validation["atol"])
    rtol = float(validation["rtol"])
    absolute = abs(float(observed_output) - reference.output)
    limit = atol + rtol * abs(reference.output)
    q8_exact = bytes(observed_q8_1) == reference.q8_1_workspace
    output_close = math.isfinite(float(observed_output)) and absolute <= limit
    return {
        "passed": q8_exact and output_close,
        "q8_workspace_byte_exact": q8_exact,
        "q8_workspace_mismatch_count": sum(
            left != right
            for left, right in zip(
                bytes(observed_q8_1), reference.q8_1_workspace, strict=False
            )
        )
        + abs(len(bytes(observed_q8_1)) - len(reference.q8_1_workspace)),
        "output_close": output_close,
        "abs_error": absolute,
        "atol": atol,
        "rtol": rtol,
        "nonfinite_output": not math.isfinite(float(observed_output)),
    }


__all__ = [
    "BLOCK_SIZE",
    "Q4MmvqMaterial",
    "Q4MmvqReference",
    "Q8_1_PADDED_BLOCKS",
    "Q8_1_WORKSPACE_BYTES",
    "decode_q4_0_block",
    "materialize_q4_mmvq_case",
    "quantize_q8_1_block",
    "q4_mmvq_metrics",
    "q4_mmvq_reference",
]
