from __future__ import annotations

import copy
import dataclasses
import json
import struct
import tempfile
import unittest
from pathlib import Path

from open_cake_ir.evaluation import (
    WorkloadContract,
    decode_q4_0_block,
    materialize_q4_mmvq_case,
    q4_mmvq_metrics,
    q4_mmvq_reference,
)


ROOT = Path(__file__).resolve().parents[2]
WORKLOAD = ROOT / "contracts/workloads/llama-q4_0-q8_1-mmvq-f32-v1.json"


class LlamaQ4MmvqWorkloadTests(unittest.TestCase):
    def test_contract_owns_live_quantization_and_never_accepts_q8_as_input(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        document = workload.document

        self.assertEqual(
            workload.workload_id,
            "llama-q4_0-q8_1-mmvq-f32-conformance-v1",
        )
        self.assertEqual(
            workload.case_ids,
            (
                "seeded_random",
                "stored_sum_correction_stress",
                "zero_scale",
                "roundf_boundary",
                "xor_tree_sum",
                "partial_rounding",
                "q4_zero_scale",
            ),
        )
        self.assertEqual(document["operator"], "llama_q4_0_q8_1_mmvq_f32")
        self.assertEqual(document["tensors"]["activation"]["dtype"], "fp32")
        self.assertEqual(document["tensors"]["weight_q4_0"]["layout"], "ggml_q4_0_v1")
        self.assertFalse(document["tensors"]["q8_1_workspace"]["public_input"])
        self.assertEqual(
            document["tensors"]["q8_1_workspace"]["shape"], [16, 36]
        )
        self.assertEqual(document["semantics"]["block_size"], 32)
        self.assertFalse(document["validation"]["performance_measured"])

    def test_contract_retains_its_upstream_revision(self) -> None:
        document = json.loads(WORKLOAD.read_text(encoding="utf-8"))

        self.assertEqual(
            document["provenance"][0]["revision"],
            "0a5ac49bce4c42893368585edf3ffb41a38f0108",
        )
    def test_materialization_and_reference_preserve_exact_block_bytes(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)

        material = materialize_q4_mmvq_case(workload, "seeded_random")
        reference = q4_mmvq_reference(workload, material)

        self.assertEqual(len(material.activation), 32)
        self.assertEqual(len(material.q4_0_block), 18)
        self.assertEqual(len(reference.q8_1_workspace), 576)
        self.assertEqual(len(reference.consumed_q8_1_block), 36)
        self.assertTrue(all(-128 <= value <= 127 for value in reference.q8_quants))
        self.assertEqual(
            reference.consumed_q8_1_block[4:],
            struct.pack("<32b", *reference.q8_quants),
        )
        self.assertEqual(reference.q8_1_workspace[36:], bytes(540))

    def test_q4_low_nibbles_precede_high_half_in_logical_order(self) -> None:
        packed = bytes(((15 - index) << 4) | index for index in range(16))
        scale, values = decode_q4_0_block(struct.pack("<e", -0.5) + packed)

        self.assertEqual(scale, -0.5)
        self.assertEqual(values[:16], tuple(range(16)))
        self.assertEqual(values[16:], tuple(reversed(range(16))))

    def test_q8_stored_sum_is_half_of_original_fp32_sum_not_reconstructed_qsum(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        material = materialize_q4_mmvq_case(
            workload, "stored_sum_correction_stress"
        )

        reference = q4_mmvq_reference(workload, material)
        stored_d, stored_s = struct.unpack(
            "<ee", reference.consumed_q8_1_block[:4]
        )
        reconstructed = struct.unpack(
            "<e", struct.pack("<e", stored_d * sum(reference.q8_quants))
        )[0]
        d4, q4_values = decode_q4_0_block(material.q4_0_block)
        idealized_output = d4 * stored_d * sum(
            (q4 - 8) * q8
            for q4, q8 in zip(q4_values, reference.q8_quants, strict=True)
        )

        self.assertNotEqual(stored_s, reconstructed)
        self.assertEqual(reference.q8_stored_sum, stored_s)
        self.assertGreater(abs(reference.output - idealized_output), 0.001)

    def test_zero_scale_quantizes_to_an_all_zero_q8_block_and_zero_output(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        material = materialize_q4_mmvq_case(workload, "zero_scale")

        reference = q4_mmvq_reference(workload, material)

        self.assertEqual(reference.q8_1_workspace, bytes(576))
        self.assertEqual(reference.output, 0.0)

    def test_nonfinite_activation_is_rejected_before_quantization(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        material = materialize_q4_mmvq_case(workload, "zero_scale")
        invalid = dataclasses.replace(
            material,
            activation=(float("nan"),) + material.activation[1:],
        )

        with self.assertRaisesRegex(ValueError, "non-finite"):
            q4_mmvq_reference(workload, invalid)

    def test_roundf_halfway_cases_are_away_from_zero(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        material = materialize_q4_mmvq_case(workload, "roundf_boundary")

        reference = q4_mmvq_reference(workload, material)

        self.assertEqual(reference.q8_stored_scale, 1.0)
        self.assertEqual(reference.q8_quants[:3], (127, 1, -1))

    def test_xor_tree_sum_differs_from_sequential_fp32_sum(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        material = materialize_q4_mmvq_case(workload, "xor_tree_sum")
        reference = q4_mmvq_reference(workload, material)
        sequential = 0.0
        for value in material.activation:
            sequential = struct.unpack(
                "<f", struct.pack("<f", sequential + value)
            )[0]
        sequential_half = struct.unpack(
            "<e", struct.pack("<e", sequential)
        )[0]

        self.assertNotEqual(reference.q8_stored_sum, sequential_half)
        self.assertEqual(reference.q8_stored_sum, 9.921875)

    def test_two_partial_fp32_placement_differs_from_combined_correction(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        material = materialize_q4_mmvq_case(workload, "partial_rounding")
        reference = q4_mmvq_reference(workload, material)
        d4, q4_values = decode_q4_0_block(material.q4_0_block)
        d8, s8 = struct.unpack("<ee", reference.consumed_q8_1_block[:4])
        sumi = sum(
            q4 * q8
            for q4, q8 in zip(q4_values, reference.q8_quants, strict=True)
        )

        def f32(value: float) -> float:
            return struct.unpack("<f", struct.pack("<f", value))[0]

        combined = f32(
            f32(d4)
            * f32(f32(f32(sumi) * f32(d8)) - f32(f32(8.0) * f32(s8)))
        )
        self.assertNotEqual(
            struct.pack("<f", reference.output),
            struct.pack("<f", combined),
        )

    def test_negative_q4_scale_changes_the_result_sign(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        negative = materialize_q4_mmvq_case(workload, "seeded_random")
        d4 = struct.unpack("<e", negative.q4_0_block[:2])[0]
        self.assertLess(d4, 0.0)
        positive = dataclasses.replace(
            negative,
            q4_0_block=struct.pack("<e", -d4) + negative.q4_0_block[2:],
        )

        negative_output = q4_mmvq_reference(workload, negative).output
        positive_output = q4_mmvq_reference(workload, positive).output

        self.assertEqual(negative_output, -positive_output)

    def test_q4_zero_scale_zeroes_output_but_not_live_q8_workspace(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        material = materialize_q4_mmvq_case(workload, "q4_zero_scale")

        reference = q4_mmvq_reference(workload, material)

        self.assertNotEqual(reference.q8_1_workspace, bytes(576))
        self.assertEqual(reference.output, 0.0)

    def test_nonfinite_q4_scale_is_rejected(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        material = materialize_q4_mmvq_case(workload, "seeded_random")
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                invalid = dataclasses.replace(
                    material,
                    q4_0_block=struct.pack("<e", value) + material.q4_0_block[2:],
                )
                with self.assertRaisesRegex(ValueError, "non-finite FP16 scale"):
                    q4_mmvq_reference(workload, invalid)

    def test_metrics_require_byte_exact_workspace_and_tolerant_fp32_output(self) -> None:
        workload = WorkloadContract.load(WORKLOAD)
        material = materialize_q4_mmvq_case(workload, "seeded_random")
        reference = q4_mmvq_reference(workload, material)

        passed = q4_mmvq_metrics(
            workload,
            reference.q8_1_workspace,
            reference.output,
            reference,
        )
        changed = bytearray(reference.q8_1_workspace)
        changed[-1] ^= 1
        rejected = q4_mmvq_metrics(
            workload,
            bytes(changed),
            reference.output,
            reference,
        )

        self.assertTrue(passed["passed"])
        self.assertTrue(passed["q8_workspace_byte_exact"])
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["q8_workspace_byte_exact"])

    def test_contract_tampering_fails_closed(self) -> None:
        original = json.loads(WORKLOAD.read_text(encoding="utf-8"))
        changes = {
            "caller_q8": lambda value: value["tensors"]["q8_1_workspace"].__setitem__(
                "public_input", True
            ),
            "idealized_sum": lambda value: value["semantics"].__setitem__(
                "stored_sum", "fp16(stored_d * sum(q))"
            ),
            "adjacent_nibbles": lambda value: value["semantics"][
                "q4_0_record"
            ].__setitem__("logical_order", "low -> 2j; high -> 2j+1"),
            "performance_claim": lambda value: value["validation"].__setitem__(
                "performance_measured", True
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, change in changes.items():
                with self.subTest(label=label):
                    document = copy.deepcopy(original)
                    change(document)
                    path = root / f"{label}.json"
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        WorkloadContract.load(path)


if __name__ == "__main__":
    unittest.main()
