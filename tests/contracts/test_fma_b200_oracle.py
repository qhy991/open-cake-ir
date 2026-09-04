"""Protect the exact oracle and its false-pass boundary before GPU submission."""

import ctypes
import ctypes.util
import importlib.util
from pathlib import Path
import random
import struct
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fma_b200_oracle", ROOT / "examples/gpu/fma_b200_correctness/oracle.py")
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)


def as_float(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def as_bits(value):
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        return 0xFF800000 if value < 0 else 0x7F800000


class FmaOracleTests(unittest.TestCase):
    def test_literal_rounding_and_special_value_anchors(self):
        anchors = [
            (0x3F800001, 0x3F7FFFFE, 0xBF800000, 0xA8800000),
            (0x3F800001, 0x3F800001, 0xBF800002, 0x28800000),
            (0x80000000, 0x3F800000, 0x80000000, 0x80000000),
            (0x80000000, 0x3F800000, 0, 0),
            (1, 0x3F000000, 0, 0), (1, 0x3FC00000, 0, 2),
            (3, 0x3F000000, 0, 2),
            (0x007FFFFF, 0x3F800000, 1, 0x00800000),
            (0x7F7FFFFF, 0x40000000, 0xFF7FFFFF, 0x7F7FFFFF),
            (0x7F800000, 0, 0, oracle.QNAN),
            (0x7F800000, 0x3F800000, 0xFF800000, oracle.QNAN),
        ]
        for a, b, c, expected in anchors:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(oracle.fma_bits(a, b, c), expected)

    def test_integer_oracle_against_host_fmaf(self):
        library = ctypes.CDLL(ctypes.util.find_library("m") or None)
        fma = library.fmaf
        fma.argtypes = [ctypes.c_float] * 3
        fma.restype = ctypes.c_float
        triples = []
        for family in oracle.FAMILIES:
            triples.extend(zip(*oracle.inputs(family)))
        rng = random.Random(19)
        triples.extend(tuple(rng.getrandbits(32) for _ in range(3)) for _ in range(20000))
        for a, b, c in triples:
            expected = oracle.fma_bits(a, b, c)
            actual = as_bits(fma(as_float(a), as_float(b), as_float(c)))
            self.assertTrue(oracle.equivalent(expected, actual), (hex(a), hex(b), hex(c)))
        self.assertEqual(len(triples), 26144)

    def test_multiply_and_nested_rounding(self):
        rng = random.Random(41)
        library = ctypes.CDLL(ctypes.util.find_library("m") or None)
        fma = library.fmaf
        fma.argtypes = [ctypes.c_float] * 3
        fma.restype = ctypes.c_float
        for _ in range(5000):
            a, b, c = [rng.getrandbits(32) for _ in range(3)]
            product = as_bits(as_float(a) * as_float(b))
            self.assertTrue(oracle.equivalent(oracle.mul_bits(a, b), product))
            inner = fma(as_float(a), as_float(b), as_float(c))
            actual = as_bits(fma(inner, as_float(c), as_float(product)))
            self.assertTrue(oracle.equivalent(oracle.expected_bits("nested", a, b, c), actual))

    def test_negative_controls_cannot_pass(self):
        self.assertFalse(oracle.equivalent(0xA8800000, 0))  # Unfused cancellation.
        self.assertFalse(oracle.equivalent(1, 0))  # FTZ.
        self.assertFalse(oracle.equivalent(0x80000000, 0))  # Lost zero sign.
        self.assertFalse(oracle.equivalent(oracle.QNAN, 0x7F800000))
        self.assertFalse(oracle.equivalent(oracle.QNAN, 0x7F800001))  # Not quiet.
        self.assertTrue(oracle.equivalent(oracle.QNAN, 0xFFC12345))

    def test_input_shape_and_multiset_are_fixed(self):
        self.assertEqual(oracle.SHAPE, (8, 128))
        self.assertEqual(len(oracle.FAMILIES) * len(oracle.MODES), 12)
        for family in oracle.FAMILIES:
            values = oracle.inputs(family)
            self.assertEqual([len(x) for x in values], [1024] * 3)
            self.assertTrue(all(0 <= x < 2**32 for xs in values for x in xs))
            self.assertEqual(values, oracle.inputs(family))


if __name__ == "__main__":
    unittest.main()
