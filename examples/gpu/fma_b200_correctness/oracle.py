"""Exact integer-arithmetic binary32 oracle and fixed FMA input corpus."""

from __future__ import annotations

import itertools
import random

SHAPE = (8, 128)
ELEMENTS = 1024
FAMILIES = ("cancellation", "signed-zero", "subnormals", "overflow",
            "special-values", "seeded-bits")
MODES = ("ordinary", "nested")
QNAN = 0x7FC00000


def is_nan(bits: int) -> bool:
    return bits & 0x7FFFFFFF > 0x7F800000


def is_inf(bits: int) -> bool:
    return bits & 0x7FFFFFFF == 0x7F800000


def is_zero(bits: int) -> bool:
    return bits & 0x7FFFFFFF == 0


def equivalent(expected: int, actual: int) -> bool:
    # PTX does not promise a NaN payload or sign; a result NaN must be quiet.
    if is_nan(expected):
        return is_nan(actual) and bool(actual & 0x00400000)
    return expected == actual


def _finite(bits: int) -> tuple[int, int]:
    exponent = (bits >> 23) & 255
    significand = bits & 0x7FFFFF
    if exponent:
        significand |= 1 << 23
    return (-significand if bits >> 31 else significand,
            exponent - 150 if exponent else -149)


def _rounded(value: int, exponent: int, zero_sign: int = 0) -> int:
    if value == 0:
        return zero_sign << 31
    sign = int(value < 0)
    value = abs(value)
    top = value.bit_length() - 1 + exponent
    quantum = max(top - 23, -149)
    shift = quantum - exponent
    if shift > 0:
        rounded, remainder = divmod(value, 1 << shift)
        halfway = 1 << (shift - 1)
        rounded += remainder > halfway or (remainder == halfway and rounded & 1)
    else:
        rounded = value << -shift
    if rounded == 0:
        return sign << 31
    if quantum == -149 and rounded < 1 << 23:
        return (sign << 31) | rounded
    top = rounded.bit_length() - 1 + quantum
    if top > 127:
        return (sign << 31) | 0x7F800000
    significand = rounded >> (rounded.bit_length() - 24)
    return (sign << 31) | ((top + 127) << 23) | (significand & 0x7FFFFF)


def mul_bits(a: int, b: int) -> int:
    sign = (a ^ b) >> 31
    if is_nan(a) or is_nan(b) or (is_inf(a) and is_zero(b)) or (is_inf(b) and is_zero(a)):
        return QNAN
    if is_inf(a) or is_inf(b):
        return (sign << 31) | 0x7F800000
    ma, ea = _finite(a)
    mb, eb = _finite(b)
    return _rounded(ma * mb, ea + eb, sign)


def fma_bits(a: int, b: int, c: int) -> int:
    if any(is_nan(x) for x in (a, b, c)):
        return QNAN
    product_sign = (a ^ b) >> 31
    if (is_inf(a) and is_zero(b)) or (is_inf(b) and is_zero(a)):
        return QNAN
    if is_inf(a) or is_inf(b):
        if is_inf(c) and c >> 31 != product_sign:
            return QNAN
        return (product_sign << 31) | 0x7F800000
    if is_inf(c):
        return c
    ma, ea = _finite(a)
    mb, eb = _finite(b)
    mc, ec = _finite(c)
    exponent = min(ea + eb, ec)
    total = ((ma * mb) << (ea + eb - exponent)) + (mc << (ec - exponent))
    # Exact cancellation is +0 in RN. Two negative zero terms instead sum to -0.
    zero_sign = int(ma * mb == 0 and mc == 0 and product_sign == 1 and c >> 31 == 1)
    return _rounded(total, exponent, zero_sign)


def expected_bits(mode: str, a: int, b: int, c: int) -> int:
    inner = fma_bits(a, b, c)
    if mode == "ordinary":
        return inner
    if mode == "nested":
        return fma_bits(inner, c, mul_bits(a, b))
    raise ValueError(f"unknown kernel mode: {mode}")


def inputs(family: str) -> tuple[list[int], list[int], list[int]]:
    one, minus_one = 0x3F800000, 0xBF800000
    if family == "cancellation":
        triples = [(0x3F800001, 0x3F7FFFFE, minus_one),
                   (0x3F800001, 0x3F800001, 0xBF800002),
                   (0xBF800001, 0x3F800001, 0x3F800002),
                   (one, one, minus_one), (minus_one, one, one),
                   (0xC67BED29, 0xBFA776E8, mul_bits(0xC43394DF, 0x41DD0D80))]
    elif family == "signed-zero":
        triples = list(itertools.product((0, 0x80000000), repeat=3))
        triples += [(one, one, minus_one), (minus_one, one, one),
                    (0x80000000, one, 0x80000000), (0, minus_one, 0)]
    elif family == "subnormals":
        triples = [(1, one, 0), (0x80000001, one, 0x80000000),
                   (1, 0x3F000000, 0), (1, 0x3FC00000, 0),
                   (3, 0x3F000000, 0), (0x007FFFFF, one, 1),
                   (0x00800000, 0x3F000000, 0), (0x00800000, one, 0x807FFFFF),
                   (0x00800000, 0x34000000, 0), (0x80800000, one, 0x007FFFFF)]
    elif family == "overflow":
        triples = [(0x7F7FFFFF, 0x40000000, 0xFF7FFFFF),
                   (0x7F7FFFFF, 0x40000000, 0), (0xFF7FFFFF, 0x40000000, 0),
                   (0x7F7FFFFF, 0x3F800001, 0xFF7FFFFF),
                   (0x7F7FFFFF, one, 0x73000000),
                   (0x7F7FFFFF, 0x7F7FFFFF, 0xFF800000)]
    elif family == "special-values":
        triples = list(itertools.product(
            (0, 0x80000000, one, minus_one, 0x7F800000, 0xFF800000,
             0x7FC12345, 0xFFC54321, 0x7F800001, 0xFF800001), repeat=3))
    elif family == "seeded-bits":
        rng = random.Random(20260904)
        triples = [tuple(rng.getrandbits(32) for _ in range(3)) for _ in range(ELEMENTS)]
    else:
        raise ValueError(f"unknown input family: {family}")
    repeated = [triples[i % len(triples)] for i in range(ELEMENTS)]
    return tuple([triple[i] for triple in repeated] for i in range(3))
