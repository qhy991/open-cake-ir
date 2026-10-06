"""Qwen2.5-Omni audio average-pool Workload and independent oracle."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation.core import LaunchObservation  # noqa: E402
from open_cake_ir.evaluation.qwen25_omni_audio_avg_pool import (  # noqa: E402
    classify_qwen25_omni_audio_avg_pool_output,
    evaluate_qwen25_omni_audio_avg_pool,
    generate_qwen25_omni_audio_avg_pool_case,
    load_qwen25_omni_audio_avg_pool_workload,
    qwen25_omni_audio_avg_pool_metrics,
    qwen25_omni_audio_avg_pool_oracle,
)

WORKLOAD_PATH = (
    ROOT
    / "contracts/workloads/qwen25-omni-audio-avg-pool1d-k2-s2-bf16.json"
)
ROUNDING_CASE = "rounding_fixture_frames4_features2"
SMOKE_CASE = "audio_hidden_smoke_frames4_features1280"
SCHEDULE_PATHS = (
    ROOT
    / "corpus/schedules/qwen25-omni-audio-avg-pool-k2-s2-bf16-metal-family9.json",
    ROOT
    / "corpus/schedules/qwen25-omni-audio-avg-pool-k2-s2-bf16-metal-family9-scale-drift.json",
)


def _bits(payload: bytes) -> tuple[int, ...]:
    return struct.unpack(f"<{len(payload) // 2}H", payload)


class Qwen25OmniAudioAvgPoolContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workload = load_qwen25_omni_audio_avg_pool_workload(WORKLOAD_PATH)

    def test_contract_closes_the_even_frame_numeric_domain(self) -> None:
        self.assertEqual(self.workload.case_ids, (ROUNDING_CASE, SMOKE_CASE))
        self.assertEqual(
            self.workload.case(SMOKE_CASE)["shape"],
            {"input_frames": 4, "output_frames": 2, "features": 1280},
        )
        semantics = self.workload.document["semantics"]
        self.assertEqual(semantics["odd_input_frames"], "unsupported")
        self.assertEqual(semantics["accumulator_dtype"], "fp32")
        self.assertEqual(
            semantics["accumulation_order"],
            "positive_zero_then_tap_0_then_tap_1",
        )
        self.assertEqual(semantics["mean_scaling"], "fp32_divide_by_2")
        self.assertEqual(
            semantics["output_conversion"],
            "one_bfloat16_round_to_nearest_ties_to_even",
        )

    def test_compiler_cases_bind_this_exact_workload_contract(self) -> None:
        for path in SCHEDULE_PATHS:
            with self.subTest(path=path.name):
                schedule = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(
                    schedule["metadata"]["workload_contract_sha256"],
                    self.workload.canonical_sha256,
                )

    def test_contract_rejects_semantic_drift_and_an_odd_frame_domain(self) -> None:
        original = json.loads(WORKLOAD_PATH.read_text(encoding="utf-8"))
        mutations = (
            (
                "semantics",
                lambda document: document["semantics"].update(
                    {"mean_scaling": "fp32_multiply_by_0.5"}
                ),
            ),
            (
                "exact even frames",
                lambda document: document["cases"][0]["shape"].update(
                    {"input_frames": 5}
                ),
            ),
        )
        for message, mutate in mutations:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                document = json.loads(json.dumps(original))
                mutate(document)
                path = Path(directory) / "workload.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_qwen25_omni_audio_avg_pool_workload(path)

    def test_rounding_fixture_is_bit_exact_for_both_midpoints_and_cancellation(self) -> None:
        input_bf16 = generate_qwen25_omni_audio_avg_pool_case(
            self.workload, ROUNDING_CASE
        )
        output_bf16 = qwen25_omni_audio_avg_pool_oracle(
            self.workload, input_bf16, case_id=ROUNDING_CASE
        )

        self.assertEqual(
            _bits(input_bf16),
            (0x3F80, 0x3F81, 0x3F81, 0x3F82, 0xC080, 0x4780, 0x4000, 0xC780),
        )
        self.assertEqual(_bits(output_bf16), (0x3F80, 0x3F82, 0xBF80, 0x0000))

        metrics = qwen25_omni_audio_avg_pool_metrics(
            self.workload,
            output_bf16,
            output_bf16,
            case_id=ROUNDING_CASE,
        )
        self.assertTrue(metrics["exact_bf16_bits"])
        self.assertEqual(metrics["total_output_elements"], 4)

    def test_oracle_rejects_nonfinite_input_before_accumulation(self) -> None:
        input_bf16 = bytearray(
            generate_qwen25_omni_audio_avg_pool_case(self.workload, ROUNDING_CASE)
        )
        input_bf16[:2] = struct.pack("<H", 0x7F80)

        with self.assertRaisesRegex(ValueError, "finite BF16"):
            qwen25_omni_audio_avg_pool_oracle(
                self.workload, input_bf16, case_id=ROUNDING_CASE
            )

    def test_audio_width_smoke_generates_four_by_1280_and_two_by_1280_output(self) -> None:
        input_bf16 = generate_qwen25_omni_audio_avg_pool_case(
            self.workload, SMOKE_CASE
        )
        output_bf16 = qwen25_omni_audio_avg_pool_oracle(
            self.workload, input_bf16, case_id=SMOKE_CASE
        )

        self.assertEqual(len(input_bf16), 4 * 1280 * 2)
        self.assertEqual(len(output_bf16), 2 * 1280 * 2)
        self.assertEqual(
            input_bf16,
            generate_qwen25_omni_audio_avg_pool_case(self.workload, SMOKE_CASE),
        )
        self.assertEqual(_bits(output_bf16)[:4], (0xC0EF, 0xC0D5, 0xC0BB, 0xC0A1))

    def test_adapter_preserves_route_receipt_and_classifies_exact_bits(self) -> None:
        expected_input = generate_qwen25_omni_audio_avg_pool_case(
            self.workload, SMOKE_CASE
        )
        expected_output = qwen25_omni_audio_avg_pool_oracle(
            self.workload, expected_input, case_id=SMOKE_CASE
        )

        class ExactLauncher:
            def launch(
                self,
                input_bf16,
                *,
                input_frames,
                output_frames,
                features,
            ):
                self.assertions = (
                    input_bf16 == expected_input,
                    input_frames,
                    output_frames,
                    features,
                )
                return LaunchObservation(expected_output, 1, 0, "a" * 64)

        launcher = ExactLauncher()
        result = evaluate_qwen25_omni_audio_avg_pool(
            self.workload, SMOKE_CASE, launcher
        )

        self.assertEqual(launcher.assertions, (True, 4, 2, 1280))
        self.assertTrue(result.correctness_passed)
        self.assertEqual(result.correctness["mismatch_count"], 0)
        self.assertEqual(result.kernel_calls, 1)
        self.assertEqual(result.fallback_calls, 0)
        self.assertEqual(result.launch_receipt_sha256, "a" * 64)

    def test_candidate_bit_drift_and_malformed_output_are_failed_outcomes(self) -> None:
        input_bf16 = generate_qwen25_omni_audio_avg_pool_case(
            self.workload, ROUNDING_CASE
        )
        oracle = qwen25_omni_audio_avg_pool_oracle(
            self.workload, input_bf16, case_id=ROUNDING_CASE
        )
        drifted = bytearray(oracle)
        drifted[0] ^= 1

        passed, metrics = classify_qwen25_omni_audio_avg_pool_output(
            self.workload, drifted, oracle, case_id=ROUNDING_CASE
        )
        self.assertFalse(passed)
        self.assertEqual(metrics["mismatch_count"], 1)

        passed, rejection = classify_qwen25_omni_audio_avg_pool_output(
            self.workload, b"", oracle, case_id=ROUNDING_CASE
        )
        self.assertFalse(passed)
        self.assertEqual(
            rejection,
            {
                "schema_version": 1,
                "failure_code": "candidate_output_contract_violation",
            },
        )


if __name__ == "__main__":
    unittest.main()
