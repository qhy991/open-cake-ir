"""Exact arithmetic boundary probes; these are not GPU qualification records."""
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import native_cuda_contract as c
import native_cuda_evaluate as e


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(values):
    words = [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]
    assert all(word & 0xffff == 0 for word in words), "test values must be exact BF16"
    return b"".join(struct.pack("<H", word >> 16) for word in words)


class PreparationConsistency(unittest.TestCase):
    def check(self, values, observed):
        e.verify_centroid_preparation(bf16(values), struct.pack("<f", observed), len(values))

    def rejects(self, values, observed, category):
        with self.assertRaises(c.QualificationError) as failure:
            self.check(values, observed)
        self.assertEqual(failure.exception.category, category)

    def test_retained_first_cpu_cuda_discrepancy_is_consistent(self):
        # M3 gpuq-0f42c988d79d, kmeans-tail_nk centroid [0,1]. This small
        # retained regression vector is independent of Torch's CPU sum order.
        values = bytes.fromhex(
            "633f00c0c83e68bfb5bf5bbfb3bfd23e743e2ebd6ac04fbe12bf99bf7b3f72be"
            "bbbc48bee73f7c3fa0bf40bf563ed23dfcbfa43f5cbf99bef93f1f3f1b3fde3e"
            "53bf413e40be3a3fd0be083e02bf4dbeddbe96bd0d4053bfae3d76bf22be023f"
            "393f13bf18be47bf74bf613fe93e85bfa43c8fbe65bc4dbf383fa6bf71bfb73f"
            "9e3e1dbff0be713f0fc0f13e39bfff3ed6bffa3e4ebfe53eab3f9dbfd4be25be"
            "0a3f103e4dbf113fd6bda03fb03e403f4b3fc63e703f29bff73e993ea33e86bf"
            "07401b3e553eddbf4a3f853da2bfb1bf173e2e3f2e3e65bf1b3e183fa3bf323f"
            "973eb03f3f3f823e243bba3f79bf58be8e3f95bf3e3f92bf18be2abfe9be9abf")
        for observed_bits in (0x42e23a9c, 0x42e23a9a):
            with self.subTest(observed_bits=observed_bits):
                e.verify_centroid_preparation(values, struct.pack("<I", observed_bits), 128)

    def test_legal_addition_orders_can_differ_by_sixty_four_ulps(self):
        values = [8.0] + [2.0 ** -9] * 127
        squares = [fp32(value * value) for value in values]
        results = []
        for order in (squares, list(reversed(squares))):
            total = 0.0
            for value in order:
                total = fp32(total + value)
            results.append(total)
            self.check(values, total)
        self.assertEqual(results, [64.0, 64.00048828125])
        words = [struct.unpack("<I", struct.pack("<f", value))[0] for value in results]
        self.assertEqual(words[1] - words[0], 64)

    def test_exact_ones_reject_one_ulp_even_inside_general_error_bound(self):
        self.check([1.0] * 128, 128.0)
        corrupted = struct.unpack("<f", struct.pack("<I", 0x43000001))[0]
        self.rejects([1.0] * 128, corrupted, "preparation_replay_mismatch")

    def test_common_quantum_is_binary_not_gcd_with_odd_factors(self):
        # Squares are 9/4 and 9*2**-24. Their ordinary gcd includes 9,
        # incorrectly suggesting an exact region; the binary quantum does not.
        values = [1.5, 3.0 * 2.0 ** -12]
        observed = fp32(sum(value * value for value in values))
        self.assertNotEqual(observed, sum(value * value for value in values))
        self.check(values, observed)

    def test_general_region_rejects_out_of_bound_corruption(self):
        self.rejects([8.0] + [2.0 ** -9] * 127, 65.0, "preparation_replay_mismatch")

    def test_zero_and_single_exact_normal_product(self):
        # The certificate compares numerical zeros; it does not attest a sign bit.
        self.check([0.0, -0.0], 0.0)
        self.check([0.0, -0.0], -0.0)
        self.rejects([0.0] * 128, 2.0 ** -149, "preparation_replay_mismatch")
        self.check([2.0 ** -63], 2.0 ** -126)
        self.check([-(2.0 ** 63)], 2.0 ** 126)

    def test_nonzero_product_underflow_and_overflow_are_uncovered(self):
        # A normal final sum cannot hide a nonzero subnormal/underflowed square.
        for tiny in (2.0 ** -64, 2.0 ** -133):
            with self.subTest(tiny=tiny):
                self.rejects([1.0, tiny], 1.0, "preparation_model_coverage")
        self.rejects([2.0 ** 64], 1.0, "preparation_model_coverage")
        # Each square is normal and finite; the sum lacks overflow headroom.
        self.rejects([2.0 ** 63] * 4, 2.0 ** 127, "preparation_model_coverage")

    def test_invalid_values_and_byte_shapes_remain_refused(self):
        for observed in (-1.0, float("inf"), float("nan")):
            with self.subTest(observed=observed):
                self.rejects([1.0], observed, "input_contract")
        for bits in (0x7f80, 0xff80, 0x7fc0):
            with self.subTest(bits=bits), self.assertRaises(c.QualificationError) as failure:
                e.verify_centroid_preparation(struct.pack("<H", bits), struct.pack("<f", 1.0), 1)
            self.assertEqual(failure.exception.category, "input_contract")
        for values, norms in ((b"", b""), (b"\0", b"\0" * 4), (bf16([1.0]), b"\0" * 3),
                              (bf16([1.0, 1.0]), b"\0" * 4)):
            with self.subTest(values=values, norms=norms), self.assertRaises(c.QualificationError) as failure:
                e.verify_centroid_preparation(values, norms, 1)
            self.assertEqual(failure.exception.category, "input_contract")

    def test_unsupported_term_count_has_no_silent_default(self):
        for count in (0, -1, True, 1.0, (1 << 24) + 1):
            with self.subTest(count=count), self.assertRaises(c.QualificationError) as failure:
                e.verify_centroid_preparation(b"", b"", count)
            self.assertEqual(failure.exception.category, "preparation_model_coverage")


if __name__ == "__main__":
    unittest.main()
