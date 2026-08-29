"""Exact sorted-half selection for material loop-carried top-k merges."""

from __future__ import annotations

import ast
import json
import math
import random
import struct
import unittest
from pathlib import Path

from open_cake_ir.compiler.analysis import top_k_selection_structure
from open_cake_ir.compiler.emit_triton import emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.profile_model import profile_envelope
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler/targets/sm_100a.json")
QSA = ROOT / "corpus/schedules/qsa-score-topk-t32768.json"
STREAMING = ROOT / "corpus/schedules/top-k-streaming-b8-smoke.json"


def _merge2_qsa() -> Schedule:
    document = json.loads(QSA.read_text(encoding="utf-8"))
    next(
        operation
        for operation in document["operations"]
        if operation["kind"] == "top_k"
    )["parameters"]["source_tiles_per_merge"] = 2
    return Schedule.from_dict(document)


def _wide_small_merge() -> Schedule:
    document = json.loads(STREAMING.read_text(encoding="utf-8"))
    next(buffer for buffer in document["buffers"] if buffer["name"] == "score_tile")[
        "shape"
    ] = [16]
    document["tile_loops"][0]["tile"] = 16
    return Schedule.from_dict(document)


def _fp32_key(value: float, index: int, *, valid: bool = True) -> int:
    if not valid:
        return 0
    if math.isnan(value):
        raise ValueError("top_k reject_input excludes NaN")
    canonical = 0.0 if value == 0.0 else value
    bits = struct.unpack("<I", struct.pack("<f", canonical))[0]
    ordered = bits ^ (0xFFFFFFFF if bits & 0x80000000 else 0x80000000)
    return (ordered << 32) | (0xFFFFFFFF - index)


def _bitonic_merge_descending(values: list[int]) -> list[int]:
    result = list(values)
    stride = len(result) // 2
    while stride:
        for index in range(len(result)):
            partner = index ^ stride
            if partner > index and result[index] < result[partner]:
                result[index], result[partner] = result[partner], result[index]
        stride //= 2
    return result


class HalfSelectionStructureTest(unittest.TestCase):
    def test_k512_removes_exactly_thirty_five_percent_of_comparison_work(self) -> None:
        selection = top_k_selection_structure(512, 1024, 2)

        self.assertEqual(selection.algorithm, "sorted_source_half_bitonic_merge")
        self.assertEqual(selection.comparison_model, "triton_3_7_1_standard_py")
        self.assertEqual(selection.baseline_comparison_lane_work, 51_712)
        self.assertEqual(selection.selected_comparison_lane_work, 33_280)
        self.assertAlmostEqual(
            selection.comparison_lane_reduction_fraction or 0.0,
            0.3564356435643564,
        )

    def test_materiality_gate_keeps_small_or_wider_merges_on_triton_topk(self) -> None:
        boundary = top_k_selection_structure(256, 512, 2)
        small = top_k_selection_structure(128, 256, 2)
        wide = top_k_selection_structure(512, 2048, 2)
        merge1 = top_k_selection_structure(512, 1024, 1)

        self.assertEqual(
            boundary.algorithm,
            "sorted_source_half_bitonic_merge",
        )
        self.assertGreater(boundary.comparison_lane_reduction_fraction or 0.0, 1 / 3)
        self.assertEqual(small.algorithm, "triton_topk")
        self.assertLess(small.comparison_lane_reduction_fraction or 0.0, 1 / 3)
        self.assertEqual(wide.algorithm, "triton_topk")
        self.assertIsNone(wide.comparison_lane_reduction_fraction)
        self.assertEqual(merge1.algorithm, "triton_topk")
        self.assertEqual(merge1.comparison_lane_reduction_fraction, 0.0)

    def test_sorted_halves_merge_to_the_exact_composite_key_prefix(self) -> None:
        k = 512
        state_values = [8.0, 8.0, 3.0, 0.0, -0.0, float("-inf")]
        source_values = [
            7.0,
            8.0,
            -0.0,
            0.0,
            -4.0,
            float("-inf"),
        ]
        state = sorted(
            [_fp32_key(value, index) for index, value in enumerate(state_values)],
            reverse=True,
        )
        source = [
            _fp32_key(value, 100 + index)
            for index, value in enumerate(source_values)
        ]
        state += [0] * (k - len(state))
        source += [0] * (k - len(source))

        bitonic = state + sorted(source)
        observed = _bitonic_merge_descending(bitonic)[:k]
        expected = sorted(state + source, reverse=True)[:k]

        self.assertEqual(observed, expected)
        self.assertGreater(_fp32_key(float("-inf"), 0), 0)
        self.assertGreater(_fp32_key(-0.0, 3), _fp32_key(+0.0, 4))
        self.assertEqual(_fp32_key(1.0, 0, valid=False), 0)
        with self.assertRaisesRegex(ValueError, "reject_input"):
            _fp32_key(float("nan"), 0)

    def test_random_zero_even_odd_and_partial_groups_match_full_reference(self) -> None:
        rng = random.Random(0xCA4E)
        k = 256
        tile = 64
        value_pool = [
            float("-inf"),
            -7.0,
            -0.0,
            0.0,
            1.0,
            1.0,
            9.0,
        ]
        for source_count in (0, 1, 2, 3, 4, 5, 7, 9):
            extent = source_count * tile - (17 if source_count == 9 else 0)
            values = [rng.choice(value_pool) for _ in range(extent)]
            keys = [_fp32_key(value, index) for index, value in enumerate(values)]
            state = [0] * k
            for start in range(0, len(keys), 2 * tile):
                source = keys[start : start + 2 * tile]
                source += [0] * (k - len(source))
                state = _bitonic_merge_descending(state + sorted(source))[:k]
            self.assertEqual(state, sorted(keys, reverse=True)[:k] + [0] * max(0, k - len(keys)))


class HalfSelectionEmissionTest(unittest.TestCase):
    def test_frozen_qsa_merge1_keeps_its_canonical_topk_lowering(self) -> None:
        source = emit(Schedule.load(QSA), TARGET).source

        ast.parse(source)
        self.assertIn("tl.topk(", source)
        self.assertNotIn("tl.bitonic_merge(", source)

    def test_merge2_pair_and_odd_flush_share_the_exact_selector(self) -> None:
        source = emit(_merge2_qsa(), TARGET).source

        ast.parse(source)
        self.assertEqual(source.count("tl.sort("), 2)
        self.assertEqual(source.count("tl.bitonic_merge("), 2)
        self.assertEqual(source.count("tl.split("), 2)
        self.assertNotIn("tl.topk(", source)
        self.assertIn("# CAKE_FLUSH:select_blocks", source)
        self.assertIn("select_blocks_source_valid =", source)

    def test_small_or_wider_merge_preserves_the_existing_topk_path(self) -> None:
        small = emit(Schedule.load(STREAMING), TARGET).source
        wide = emit(_wide_small_merge(), TARGET).source

        for source in (small, wide):
            self.assertIn("tl.topk(", source)
            self.assertNotIn("tl.bitonic_merge(", source)

    def test_profile_exposes_work_reduction_without_declared_allocation_growth(self) -> None:
        schedule = _merge2_qsa()
        lowered = emit(schedule, TARGET).source
        profile = profile_envelope(schedule, TARGET, lowered_source=lowered).as_dict()
        top_k = profile["lowering"]["top_k"][0]

        self.assertEqual(
            top_k["selection_algorithm"],
            "sorted_source_half_bitonic_merge",
        )
        self.assertEqual(top_k["selection_comparison_model"], "triton_3_7_1_standard_py")
        self.assertEqual(
            top_k["selection_comparison_estimate_kind"],
            "toolchain_frontend_model",
        )
        self.assertEqual(
            top_k["selection_comparison_model_source"],
            "Triton v3.7.1 standard.py",
        )
        self.assertEqual(top_k["baseline_comparison_lane_work"], 51_712)
        self.assertEqual(top_k["selected_comparison_lane_work"], 33_280)
        self.assertAlmostEqual(
            top_k["comparison_lane_work_reduction_fraction"],
            0.3564356435643564,
        )
        self.assertEqual(top_k["selection_declared_shared_memory_delta_bytes"], 0)
        self.assertIsNone(top_k["selection_compiled_shared_memory_delta_bytes"])
        self.assertEqual(
            top_k["selection_compiled_resource_missing"],
            ["matched compiled allocation for selected algorithm and control"],
        )
        self.assertEqual(top_k["merge_width"], 1024)
        self.assertEqual(top_k["loop_carried_state_bytes"], 4096)
        self.assertEqual(top_k["pending_source_state_bytes"], 1024)
        self.assertEqual(profile["lowering"]["triton_top_k_count"], 0)
        self.assertEqual(profile["lowering"]["triton_sort_count"], 2)
        self.assertEqual(profile["lowering"]["triton_bitonic_merge_count"], 2)
        self.assertIn(
            "tl.sort and tl.bitonic_merge own internal synchronization/storage plans",
            profile["abstentions"],
        )
        self.assertIn(
            "half-selection compiled resource delta requires matched toolchain artifacts",
            profile["abstentions"],
        )
        self.assertFalse(schedule.allocations)


if __name__ == "__main__":
    unittest.main()
