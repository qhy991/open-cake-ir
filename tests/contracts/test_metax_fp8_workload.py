"""The frozen FP8 task owns its input bytes and high-precision oracle."""

import struct
from pathlib import Path
import unittest

from open_cake_ir.tasks import metax_fp8_gemm
from open_cake_ir.tasks.workloads import load_workload
from open_cake_ir.compiler import frontend


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/workloads/metax-fp8-e4m3-gemm-fp32-xcore1002-m64-n64-k64-v1.json"


class MetaxFP8Workload(unittest.TestCase):
    def test_frozen_contract_and_exact_candidate_abi(self):
        workload = load_workload(CONTRACT)
        self.assertEqual(workload.workload_id, metax_fp8_gemm.WORKLOAD_ID)
        self.assertEqual(len(workload.case_ids), 37)
        self.assertEqual(
            [(arg.name, arg.shape, arg.dtype, arg.mode) for arg in workload.tensor_abi("primary")],
            [("a", (64, 64), "fp8_e4m3", "input"),
             ("b", (64, 64), "fp8_e4m3", "input"),
             ("out", (64, 64), "fp32", "output")],
        )
        schedule = frontend.read_schedule(ROOT / "examples/python/xcore1002_fp8_compensated.py").document
        self.assertEqual(
            [(buffer["name"], tuple(buffer["shape"]), buffer["dtype"], buffer["mode"])
             for buffer in schedule["buffers"] if buffer["mode"] in {"input", "output"}],
            [(arg.name, arg.shape, arg.dtype, arg.mode) for arg in workload.tensor_abi("primary")],
        )

    def test_finite_decoder_and_generator(self):
        decode = metax_fp8_gemm.decode_e4m3fn
        self.assertEqual((decode(0), decode(0x38), decode(0xB8), decode(0x7E)),
                         (0.0, 1.0, -1.0, 448.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            decode(0x7F)
        case = load_workload(CONTRACT).case("heldout_full_finite_05")
        self.assertEqual(metax_fp8_gemm._bytes_for_case(case), metax_fp8_gemm._bytes_for_case(case))
        self.assertTrue(all(code not in (0x7F, 0xFF)
                            for tensor in metax_fp8_gemm._bytes_for_case(case) for code in tensor))

    def test_independent_oracle_identity_zero_and_precision_counterexample(self):
        workload = load_workload(CONTRACT)
        for name in ("zeros", "identity"):
            a, b = metax_fp8_gemm._bytes_for_case(workload.case(name))
            output = metax_fp8_gemm.reference_bytes(a, b)
            expected = ([0.0] * 4096 if name == "zeros" else
                        [metax_fp8_gemm.decode_e4m3fn(code) for code in b])
            self.assertEqual(output, expected)

        a, b = metax_fp8_gemm._bytes_for_case(workload.case("heldout_full_finite_05"))
        exact = metax_fp8_gemm.reference_bytes(a, b)[2 * 64 + 31]
        left = [metax_fp8_gemm.decode_e4m3fn(code) for code in a]
        right = [metax_fp8_gemm.decode_e4m3fn(code) for code in b]
        fp32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
        ordinary_sum = 0.0
        for k in range(64):
            ordinary_sum = fp32(ordinary_sum + fp32(left[2 * 64 + k] * right[k * 64 + 31]))
        self.assertGreater(abs(ordinary_sum - exact), 0.001 + 0.0001 * abs(exact))

    def test_invalid_input_is_rejected_before_reference(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            metax_fp8_gemm.reference_bytes(bytes(3), bytes(4096))
        with self.assertRaisesRegex(ValueError, "finite"):
            metax_fp8_gemm.reference_bytes(bytes([0x7F]) + bytes(4095), bytes(4096))


if __name__ == "__main__":
    unittest.main()
