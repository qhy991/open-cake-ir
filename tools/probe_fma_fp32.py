#!/usr/bin/env python3
"""Prove one finite FP32 input where fused FMA differs from FP32 mul then add."""

from __future__ import annotations

from fractions import Fraction
import json
import struct
from typing import Mapping


SCHEMA = "open-cake.ir-numeric-probe.fma-fp32.v1"
INPUT_BITS = {
    "a": 0xC0CE69DB,
    "b": 0xC022EC9E,
    "c": 0xC07E0807,
}
EXPECTED_FUSED_BITS = 0x41473989
EXPECTED_SEPARATE_BITS = 0x4147398A


def fraction_from_f32_bits(bits: int) -> Fraction:
    """Decode one finite IEEE binary32 payload exactly."""

    sign = -1 if bits >> 31 else 1
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent == 0xFF:
        raise ValueError("the exact probe admits finite FP32 values only")
    if exponent == 0:
        mantissa = fraction
        power = -149
    else:
        mantissa = (1 << 23) | fraction
        power = exponent - 127 - 23
    value = Fraction(mantissa)
    value = value * (1 << power) if power >= 0 else value / (1 << -power)
    return sign * value


def _round_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = remainder * 2
    if twice > denominator or (twice == denominator and quotient & 1):
        quotient += 1
    return quotient


def f32_bits_from_fraction(value: Fraction) -> int:
    """Round one exact rational to IEEE binary32, round-to-nearest ties-to-even."""

    if value == 0:
        return 0
    sign = 1 if value < 0 else 0
    value = abs(value)
    numerator, denominator = value.numerator, value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1

    if exponent > 127:
        return (sign << 31) | 0x7F800000
    if exponent >= -126:
        shift = 23 - exponent
        scaled_numerator = numerator << shift if shift >= 0 else numerator
        scaled_denominator = denominator if shift >= 0 else denominator << -shift
        significand = _round_even(scaled_numerator, scaled_denominator)
        if significand == 1 << 24:
            significand >>= 1
            exponent += 1
            if exponent > 127:
                return (sign << 31) | 0x7F800000
        payload = ((exponent + 127) << 23) | (significand - (1 << 23))
        return (sign << 31) | payload

    significand = _round_even(numerator << 149, denominator)
    if significand == 0:
        return sign << 31
    if significand >= 1 << 23:
        return (sign << 31) | (1 << 23)
    return (sign << 31) | significand


def float_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def probe() -> Mapping[str, object]:
    a = fraction_from_f32_bits(INPUT_BITS["a"])
    b = fraction_from_f32_bits(INPUT_BITS["b"])
    c = fraction_from_f32_bits(INPUT_BITS["c"])
    fused_bits = f32_bits_from_fraction(a * b + c)
    product_bits = f32_bits_from_fraction(a * b)
    separate_bits = f32_bits_from_fraction(
        fraction_from_f32_bits(product_bits) + c
    )
    return {
        "schema": SCHEMA,
        "inputs": {
            name: {"bits": f"0x{bits:08x}", "value": float_from_bits(bits)}
            for name, bits in INPUT_BITS.items()
        },
        "fused": {
            "bits": f"0x{fused_bits:08x}",
            "value": float_from_bits(fused_bits),
        },
        "separate_mul_add": {
            "bits": f"0x{separate_bits:08x}",
            "value": float_from_bits(separate_bits),
        },
        "distinguished": fused_bits != separate_bits,
        "expected": {
            "fused_bits": f"0x{EXPECTED_FUSED_BITS:08x}",
            "separate_bits": f"0x{EXPECTED_SEPARATE_BITS:08x}",
        },
    }


def main() -> int:
    result = probe()
    print(json.dumps(result, indent=2, sort_keys=True))
    return (
        0
        if result["distinguished"]
        and result["fused"]["bits"] == result["expected"]["fused_bits"]
        and result["separate_mul_add"]["bits"]
        == result["expected"]["separate_bits"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
