"""Contracts for source-grounded abstraction extraction."""

from __future__ import annotations

import json
from hashlib import sha256
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "operator_library"
EXTRACTION = ROOT / "abstraction_extraction"
TOOL = ROOT / "tools" / "validate_abstraction_extraction.py"


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _operator_library_digest(root: Path) -> str:
    manifest = _read(root / "manifest.json")
    value = {
        "manifest": manifest,
        "sources": {
            str(relative): _read(root / str(relative))
            for relative in manifest["sources"]
        },
        "operators": {
            str(relative): _read(root / str(relative))
            for relative in manifest["operators"]
        },
    }
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _definition_validator(
    schema: dict[str, object], definition: str
) -> Draft202012Validator:
    return Draft202012Validator(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": schema["$defs"],
            "$ref": f"#/$defs/{definition}",
        },
        format_checker=FormatChecker(),
    )


def _run(
    library: Path = LIBRARY,
    extraction: Path = EXTRACTION,
    output_format: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(TOOL),
        "--library-root",
        str(library),
        "--extraction-root",
        str(extraction),
    ]
    if output_format is not None:
        command.extend(("--format", output_format))
    return subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


class AbstractionExtractionSchemaTests(unittest.TestCase):
    def test_all_three_document_kinds_conform_to_the_public_schema(self) -> None:
        schema = _read(EXTRACTION / "schema.json")
        Draft202012Validator.check_schema(schema)
        manifest = _read(EXTRACTION / "manifest.json")
        documents = {
            "manifest": [manifest],
            "observation": [
                _read(EXTRACTION / relative) for relative in manifest["observations"]
            ],
            "candidate": [
                _read(EXTRACTION / relative) for relative in manifest["candidates"]
            ],
        }
        self.assertTrue(documents["observation"])
        self.assertTrue(documents["candidate"])
        root_validator = Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        )
        for definition, values in documents.items():
            for value in values:
                for mode, validator in (
                    ("definition", _definition_validator(schema, definition)),
                    ("root", root_validator),
                ):
                    with self.subTest(definition=definition, mode=mode):
                        errors = sorted(
                            validator.iter_errors(value),
                            key=lambda error: tuple(
                                str(part) for part in error.absolute_path
                            ),
                        )
                        self.assertEqual(
                            errors,
                            [],
                            "\n".join(
                                f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
                                for error in errors
                            ),
                        )

    def test_every_observation_binds_an_exact_occurrence_symbol(self) -> None:
        library_manifest = _read(LIBRARY / "manifest.json")
        occurrences = {
            occurrence["implementation_id"]: occurrence
            for relative in library_manifest["operators"]
            for occurrence in [_read(LIBRARY / str(relative))]
        }
        extraction_manifest = _read(EXTRACTION / "manifest.json")

        for relative in extraction_manifest["observations"]:
            observation = _read(EXTRACTION / str(relative))
            locator = observation["source_locator"]
            occurrence = occurrences[observation["implementation_id"]]
            with self.subTest(observation=observation["observation_id"]):
                self.assertIn(
                    (
                        locator["path"],
                        locator["revision"],
                        locator["git_object"],
                        locator["symbol"],
                    ),
                    {
                        (
                            item["value"],
                            item["revision"],
                            item["git_object"],
                            item["symbol"],
                        )
                        for item in occurrence["locators"]
                        if item["kind"] == "repository_path"
                    },
                )

    def test_w8_candidate_binds_four_reviewed_projection_spans(self) -> None:
        candidate = _read(
            EXTRACTION / "candidates" / "row-per-simdgroup-groupwise-w8-projection.json"
        )
        observation_paths = (
            "observations/apxinf-full-attention-qgkv-row-projection.json",
            "observations/apxinf-w8-gdn-input-row-projection.json",
            "observations/apxinf-w8-lm-head-row-projection.json",
            "observations/apxinf-w8-mlp-gate-up-row-projection.json",
        )
        observations = [_read(EXTRACTION / path) for path in observation_paths]
        by_id = {
            observation["observation_id"]: observation for observation in observations
        }

        self.assertEqual(
            candidate["observation_ids"],
            [observation["observation_id"] for observation in observations],
        )
        self.assertEqual(
            {observation["pattern_signature"] for observation in observations},
            {candidate["pattern_signature"]},
        )
        common_primitives = set(observations[0]["observed_primitives"]).intersection(
            *(
                set(observation["observed_primitives"])
                for observation in observations[1:]
            )
        )
        self.assertEqual(
            common_primitives,
            {
                "groupwise_int8_scale",
                "lane_zero_row_publish",
                "row_per_simdgroup",
                "simd_sum",
                "vectorized_char4_dot",
            },
        )
        self.assertEqual(
            {
                (
                    observation["source_locator"]["symbol"],
                    observation["source_locator"]["path"],
                    observation["source_span"]["start_line"],
                    observation["source_span"]["end_line"],
                )
                for observation in observations
            },
            {
                (
                    "full_attention_qgkv_v1",
                    "crates/apxinf-metal/src/metal_full_attention_decode_v1.metal",
                    53,
                    83,
                ),
                (
                    "gdn_w8_input_projection",
                    "crates/apxinf-metal/src/metal_w8_gdn.metal",
                    20,
                    47,
                ),
                (
                    "w8_rows_topk4",
                    "crates/apxinf-metal/src/metal_w8.metal",
                    39,
                    73,
                ),
                (
                    "w8_mlp_gate_up",
                    "crates/apxinf-metal/src/metal_w8_mlp.metal",
                    11,
                    42,
                ),
            },
        )
        direct_observations = (
            by_id["apxinf-metal-full-attention.qgkv-row-projection"],
            by_id["apxinf-metal-w8-gdn.input-row-projection"],
            by_id["apxinf-metal-w8-mlp.gate-up-projection"],
        )
        for observation in direct_observations:
            with self.subTest(observation=observation["observation_id"]):
                self.assertIn(
                    "direct_device_row_publish", observation["observed_primitives"]
                )
                self.assertIn("early_row_return", observation["observed_primitives"])
        lm_head = by_id["apxinf-metal-w8-lm-head.row-projection"]
        self.assertNotIn("direct_device_row_publish", lm_head["observed_primitives"])
        self.assertNotIn("early_row_return", lm_head["observed_primitives"])
        for primitive in (
            "invalid_row_sentinel",
            "nan_score_guard",
            "threadgroup_score_token_staging",
        ):
            with self.subTest(primitive=primitive):
                self.assertIn(primitive, lm_head["observed_primitives"])
                self.assertTrue(
                    all(
                        primitive not in observation["observed_primitives"]
                        for observation in direct_observations
                    )
                )
        full_attention = by_id["apxinf-metal-full-attention.qgkv-row-projection"]
        gdn = by_id["apxinf-metal-w8-gdn.input-row-projection"]
        mlp = by_id["apxinf-metal-w8-mlp.gate-up-projection"]
        self.assertIn(
            "layer_slot_weight_scale_offset",
            full_attention["observed_primitives"],
        )
        self.assertIn("runtime_projection_extents", gdn["observed_primitives"])
        self.assertIn("concatenated_gate_up_output", mlp["observed_primitives"])
        variants = " ".join(candidate["observed_variants"])
        for detail in (
            "layer_slot to both packed-weight and scale row bases",
            "output-row count, hidden-column extent, and scale-row stride",
            "one concatenated gate/up device buffer",
            "negative-infinity and maximum-token sentinels",
            "NaN sum",
        ):
            with self.subTest(detail=detail):
                self.assertIn(detail, variants)
        normalized = " ".join(candidate["normalized_steps"])
        for variant_only in (
            "layer_slot",
            "GdnParams",
            "concatenated gate/up",
            "negative-infinity",
            "maximum-token",
            "threadgroup score-token",
        ):
            with self.subTest(variant_only=variant_only):
                self.assertNotIn(variant_only, normalized)

    def test_max_rebased_candidate_preserves_finalize_and_payload_variants(
        self,
    ) -> None:
        candidate = _read(
            EXTRACTION / "candidates" / "max-rebased-exponential-summary.json"
        )
        observation_paths = (
            "observations/mlx-scaled-dot-product-attention-max-rebased-summary.json",
            "observations/mlx-softmax-looped-max-rebased-summary.json",
        )
        observations = [_read(EXTRACTION / path) for path in observation_paths]
        by_id = {
            observation["observation_id"]: observation for observation in observations
        }

        self.assertEqual(
            candidate["observation_ids"],
            [observation["observation_id"] for observation in observations],
        )
        self.assertEqual(
            {observation["pattern_signature"] for observation in observations},
            {candidate["pattern_signature"]},
        )
        self.assertEqual(
            {
                (
                    observation["source_locator"]["path"],
                    observation["source_locator"]["symbol"],
                    observation["source_span"]["start_line"],
                    observation["source_span"]["end_line"],
                )
                for observation in observations
            },
            {
                (
                    "mlx/backend/metal/kernels/sdpa_vector.h",
                    "sdpa_vector",
                    87,
                    169,
                ),
                (
                    "mlx/backend/metal/kernels/softmax.h",
                    "softmax_looped",
                    114,
                    189,
                ),
            },
        )
        attention = by_id[
            "mlx-metal-scaled-dot-product-attention.vector-max-rebased-summary"
        ]
        softmax = by_id["mlx-metal-softmax.looped-max-rebased-summary"]
        self.assertIn("weighted_payload_rebase", attention["observed_primitives"])
        self.assertNotIn("weighted_payload_rebase", softmax["observed_primitives"])
        for primitive in (
            "masked_score_update",
            "sink_seed",
            "zero_denominator_guard",
        ):
            with self.subTest(primitive=primitive):
                self.assertIn(primitive, attention["observed_primitives"])
                self.assertNotIn(primitive, softmax["observed_primitives"])
        self.assertIn("two_pass_finalize", softmax["observed_primitives"])
        self.assertNotIn("two_pass_finalize", attention["observed_primitives"])

    def test_two_pass_rms_preserves_reduction_weight_and_fusion_variants(
        self,
    ) -> None:
        candidate = _read(
            EXTRACTION / "candidates" / "two-pass-row-rms-modulation.json"
        )
        observation_paths = (
            "observations/apxinf-full-attention-input-rms-modulation.json",
            "observations/apxinf-w8-gdn-norm-gate-rms-modulation.json",
        )
        observations = [_read(EXTRACTION / path) for path in observation_paths]
        by_id = {
            observation["observation_id"]: observation for observation in observations
        }

        self.assertEqual(
            candidate["observation_ids"],
            [observation["observation_id"] for observation in observations],
        )
        self.assertEqual(
            {observation["pattern_signature"] for observation in observations},
            {candidate["pattern_signature"]},
        )
        self.assertEqual(
            {
                (
                    observation["source_locator"]["path"],
                    observation["source_locator"]["symbol"],
                    observation["source_span"]["start_line"],
                    observation["source_span"]["end_line"],
                )
                for observation in observations
            },
            {
                (
                    "crates/apxinf-metal/src/metal_full_attention_decode_v1.metal",
                    "full_attention_input_rms_v1",
                    33,
                    51,
                ),
                (
                    "crates/apxinf-metal/src/metal_w8_gdn.metal",
                    "gdn_norm_gate",
                    509,
                    533,
                ),
            },
        )
        common_primitives = set(observations[0]["observed_primitives"]).intersection(
            observations[1]["observed_primitives"]
        )
        self.assertEqual(
            common_primitives,
            {
                "epsilon_stabilized_inverse_rms",
                "fp32_row_sum_of_squares",
                "per_element_norm_weight_modulation",
                "pointwise_inverse_rms_apply",
                "row_extent_mean_square",
                "second_pass_source_reread",
            },
        )
        attention = by_id["apxinf-metal-full-attention.input-rms-modulation"]
        gdn = by_id["apxinf-metal-w8-gdn.norm-gate-rms-modulation"]
        for primitive in (
            "fixed_hidden_extent",
            "lane_strided_row_passes",
            "layer_slot_norm_weight_offset",
            "simd_sum",
            "standalone_normalized_output",
            "zero_centered_norm_weight_factor",
        ):
            with self.subTest(attention_only=primitive):
                self.assertIn(primitive, attention["observed_primitives"])
                self.assertNotIn(primitive, gdn["observed_primitives"])
        for primitive in (
            "direct_norm_weight_factor",
            "early_value_head_return",
            "projected_silu_gate",
            "runtime_value_extent",
            "serial_row_passes",
            "value_head_row_base",
        ):
            with self.subTest(gdn_only=primitive):
                self.assertIn(primitive, gdn["observed_primitives"])
                self.assertNotIn(primitive, attention["observed_primitives"])
        normalized = " ".join(candidate["normalized_steps"])
        for variant_only in (
            "simd_sum",
            "SIMDgroup",
            "threadgroup",
            "barrier",
            "layer_slot",
            "SiLU",
            "weight + 1",
            "value_head",
        ):
            with self.subTest(variant_only=variant_only):
                self.assertNotIn(variant_only, normalized)
        variants = " ".join(candidate["observed_variants"])
        for boundary in (
            "zero-centered weight factor",
            "projected-Z SiLU factor",
            "reduction order",
            "physical memory traffic",
            "does not establish numerical equivalence",
            "a Compiler or Target abstraction",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, variants)
        manifest = _read(EXTRACTION / "manifest.json")
        self.assertEqual(manifest["counts"]["implementation_count"], 15)
        non_claims = " ".join(manifest["selection_policy"]["non_claims"])
        for boundary in (
            "does not establish a common thread or SIMDgroup ownership model",
            "physical row residency",
            "zero-centered versus direct normalization-weight factors",
            "independent of hierarchical-simdgroup-threadgroup-reduction",
            "does not admit a Compiler RMS operation",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, non_claims)

    def test_matrix_k_loop_keeps_only_the_paired_tile_mma_choreography(
        self,
    ) -> None:
        candidate = _read(
            EXTRACTION / "candidates" / "paired-threadgroup-tile-mma-k-loop.json"
        )
        observation_paths = (
            "observations/mlx-convolution-implicit-gemm-threadgroup-mma-k-loop.json",
            "observations/mlx-quantized-qmm-n-threadgroup-mma-k-loop.json",
            "observations/mlx-steel-gemm-paired-threadgroup-mma-k-loop.json",
        )
        observations = [_read(EXTRACTION / path) for path in observation_paths]
        by_id = {
            observation["observation_id"]: observation for observation in observations
        }

        self.assertEqual(
            candidate["observation_ids"],
            [observation["observation_id"] for observation in observations],
        )
        self.assertEqual(
            {observation["pattern_signature"] for observation in observations},
            {candidate["pattern_signature"]},
        )
        self.assertEqual(
            {
                (
                    observation["source_locator"]["path"],
                    observation["source_locator"]["symbol"],
                    observation["source_span"]["start_line"],
                    observation["source_span"]["end_line"],
                )
                for observation in observations
            },
            {
                (
                    "mlx/backend/metal/kernels/quantized.h",
                    "qmm_n_impl",
                    1320,
                    1454,
                ),
                (
                    "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.h",
                    "implicit_gemm_conv_2d",
                    7,
                    176,
                ),
                (
                    "mlx/backend/metal/kernels/steel/gemm/gemm.h",
                    "mlx::steel::GEMMKernel::gemm_loop",
                    48,
                    137,
                ),
            },
        )
        common_primitives = set(observations[0]["observed_primitives"]).intersection(
            *(
                set(observation["observed_primitives"])
                for observation in observations[1:]
            )
        )
        self.assertEqual(
            common_primitives,
            {
                "block_mma_accumulate",
                "k_tile_iteration",
                "paired_loader_advance",
                "paired_operand_block_loaders",
                "paired_threadgroup_operand_tiles",
                "postload_threadgroup_barrier",
                "preload_threadgroup_barrier",
            },
        )

        exclusive_primitives = {
            "mlx-metal-steel-gemm.paired-threadgroup-mma-k-loop": {
                "compile_time_alignment_specialization",
                "mn_edge_load_specialization",
                "parameterized_accumulator_epilogue",
                "safe_k_remainder_load",
                "transpose_aware_edge_tile_shape",
            },
            "mlx-metal-quantized-matmul.qmm-n-threadgroup-mma-k-loop": {
                "fixed_nn_transpose_mode",
                "packed_quantized_weight_addressing",
                "partial_m_load_specialization",
                "quantized_weight_block_loader",
                "runtime_k_remainder_load",
                "safe_or_full_result_store",
                "scale_bias_loader_binding",
            },
            "mlx-metal-convolution.implicit-gemm-threadgroup-mma-k-loop": {
                "conditional_convolution_input_loader",
                "conditional_convolution_weight_loader",
                "fixed_nt_transpose_mode",
                "grid_swizzle_tile_guard",
                "grouped_convolution_addressing",
                "safe_result_store_terminal",
                "spatial_filter_loader_specialization",
            },
        }
        for observation_id, primitives in exclusive_primitives.items():
            observation = by_id[observation_id]
            other_observations = [
                value for key, value in by_id.items() if key != observation_id
            ]
            for primitive in primitives:
                with self.subTest(observation=observation_id, primitive=primitive):
                    self.assertIn(primitive, observation["observed_primitives"])
                    self.assertTrue(
                        all(
                            primitive not in other["observed_primitives"]
                            for other in other_observations
                        )
                    )

        normalized = " ".join(candidate["normalized_steps"]).lower()
        for variant_only in (
            "transpose",
            "layout",
            "quant",
            "dequant",
            "scale",
            "bias",
            "group_size",
            "bits",
            "pack",
            "stride",
            "padding",
            "dilation",
            "filter",
            "window",
            "groups",
            "tail",
            "aligned",
            "k_eff",
            "load_safe",
            "load_unsafe",
            "zero-fill",
            "store",
            "epilogue",
            "uniform",
            "race-free",
            "deadlock",
            "correctness",
            "bitwise",
            "occupancy",
            "throughput",
            "performance",
            "compiler",
            "target",
            "simd_sum",
            "lane_zero",
            "broadcast",
            "final simdgroup reducer",
            "hierarchical",
        ):
            with self.subTest(variant_only=variant_only):
                self.assertNotIn(variant_only, normalized)

        variants = " ".join(candidate["observed_variants"])
        for boundary in (
            "packed quantized-weight loader",
            "convolution stride, padding, dilation",
            "Barrier names and safe or unsafe helper names do not establish",
            "hardware-instruction mapping",
            "Compiler operation, Target capability, Schedule form, lowering route",
            "hierarchical SIMDgroup reduction candidate",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, variants)

        manifest = _read(EXTRACTION / "manifest.json")
        self.assertEqual(
            manifest["counts"],
            {
                "observation_count": 17,
                "candidate_count": 6,
                "implementation_count": 15,
                "source_count": 2,
            },
        )
        non_claims = " ".join(manifest["selection_policy"]["non_claims"])
        for boundary in (
            "records source call order only",
            "does not normalize physical layout",
            "establishes no asynchronous overlap",
            "does not alter the frozen Stage 3 candidate closure",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, non_claims)

    def test_scan_and_sort_handoff_remain_below_candidate_threshold(self) -> None:
        observation_paths = (
            "observations/mlx-scan-contiguous-block-prefix-handoff.json",
            "observations/mlx-sort-block-merge-state-exchange.json",
        )
        observations = [_read(EXTRACTION / path) for path in observation_paths]
        by_id = {
            observation["observation_id"]: observation for observation in observations
        }

        self.assertEqual(
            {
                (
                    observation["source_locator"]["path"],
                    observation["source_locator"]["symbol"],
                    observation["source_span"]["start_line"],
                    observation["source_span"]["end_line"],
                )
                for observation in observations
            },
            {
                (
                    "mlx/backend/metal/kernels/scan.h",
                    "contiguous_scan",
                    247,
                    401,
                ),
                (
                    "mlx/backend/metal/kernels/sort.h",
                    "BlockMergeSort::sort",
                    158,
                    243,
                ),
            },
        )
        self.assertEqual(
            len({observation["pattern_signature"] for observation in observations}),
            2,
        )

        scan = by_id["mlx-metal-scan.contiguous-block-prefix-handoff"]
        sort = by_id["mlx-metal-sort.block-merge-state-exchange"]
        common_primitives = set(scan["observed_primitives"]).intersection(
            sort["observed_primitives"]
        )
        self.assertEqual(
            common_primitives,
            {
                "iteration_carried_thread_state",
                "postpublication_threadgroup_barrier",
                "prepublication_threadgroup_barrier",
                "reused_threadgroup_storage",
                "thread_state_to_threadgroup_publication",
                "threadgroup_state_to_thread_state_update",
            },
        )
        exclusive_primitives = {
            scan["observation_id"]: {
                "block_prefix_carry",
                "contiguous_axis_block_iteration",
                "first_simdgroup_summary_scan",
                "forward_or_reverse_scan",
                "inclusive_or_exclusive_output",
                "last_lane_simdgroup_summary_publish",
                "operator_parameterized_scan",
                "per_thread_serial_prefix",
                "safe_or_unsafe_tail_io",
                "simd_exclusive_prefix_scan",
                "single_thread_block_prefix_publish",
            },
            sort["observation_id"]: {
                "all_thread_multiitem_publication",
                "merge_path_partition",
                "merge_width_doubling",
                "optional_index_carry",
                "partitioned_two_run_merge",
                "terminal_threadgroup_publication",
                "thread_local_initial_sort",
            },
        }
        for observation in observations:
            other = sort if observation is scan else scan
            for primitive in exclusive_primitives[observation["observation_id"]]:
                with self.subTest(
                    observation=observation["observation_id"], primitive=primitive
                ):
                    self.assertIn(primitive, observation["observed_primitives"])
                    self.assertNotIn(primitive, other["observed_primitives"])

        manifest = _read(EXTRACTION / "manifest.json")
        candidates = [_read(EXTRACTION / path) for path in manifest["candidates"]]
        standalone_ids = set(by_id)
        self.assertTrue(
            all(
                standalone_ids.isdisjoint(candidate["observation_ids"])
                for candidate in candidates
            )
        )
        self.assertTrue(
            all(
                candidate["pattern_signature"]
                not in {
                    scan["pattern_signature"],
                    sort["pattern_signature"],
                }
                for candidate in candidates
            )
        )
        self.assertEqual(
            manifest["counts"],
            {
                "observation_count": 17,
                "candidate_count": 6,
                "implementation_count": 15,
                "source_count": 2,
            },
        )
        non_claims = " ".join(manifest["selection_policy"]["non_claims"])
        for boundary in (
            "share only a generic thread-state and threadgroup-scratch handoff",
            "no common state algebra, participant topology, scratch shape",
            "so no candidate is admitted",
            "does not widen hierarchical-simdgroup-threadgroup-reduction",
            "neither observation is evidence for an existing candidate",
            "no barrier safety",
            "or Stage 3 result",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, non_claims)

    def test_runtime_index_fold_preserves_terminal_and_safety_boundaries(
        self,
    ) -> None:
        candidate = _read(
            EXTRACTION / "candidates" / "runtime-index-axis-stride-offset-fold.json"
        )
        observation_paths = (
            "observations/mlx-gather-runtime-index-axis-stride-fold.json",
            "observations/mlx-scatter-runtime-index-axis-stride-fold.json",
        )
        observations = [_read(EXTRACTION / path) for path in observation_paths]
        by_id = {
            observation["observation_id"]: observation for observation in observations
        }

        self.assertEqual(
            candidate["observation_ids"],
            [observation["observation_id"] for observation in observations],
        )
        self.assertEqual(
            {observation["pattern_signature"] for observation in observations},
            {candidate["pattern_signature"]},
        )
        self.assertEqual(
            {
                (
                    observation["source_locator"]["path"],
                    observation["source_locator"]["symbol"],
                    observation["source_span"]["start_line"],
                    observation["source_span"]["end_line"],
                )
                for observation in observations
            },
            {
                (
                    "mlx/backend/metal/kernels/indexing/gather.h",
                    "gather_impl",
                    19,
                    50,
                ),
                (
                    "mlx/backend/metal/kernels/indexing/scatter.h",
                    "scatter_impl",
                    31,
                    57,
                ),
            },
        )
        common_primitives = set(observations[0]["observed_primitives"]).intersection(
            observations[1]["observed_primitives"]
        )
        self.assertEqual(
            common_primitives,
            {
                "axis_stride_fold",
                "index_element_location",
                "negative_index_offset",
                "row_contiguous_index_path",
                "runtime_axis_lookup",
                "strided_index_location",
                "unindexed_slice_base",
            },
        )
        gather = by_id["mlx-metal-gather.runtime-index-axis-stride-fold"]
        scatter = by_id["mlx-metal-scatter.runtime-index-axis-stride-fold"]
        for primitive in (
            "direct_indexed_read_terminal",
            "gather_output_linearization",
            "index_rank_specialization",
        ):
            with self.subTest(primitive=primitive):
                self.assertIn(primitive, gather["observed_primitives"])
                self.assertNotIn(primitive, scatter["observed_primitives"])
        for primitive in (
            "atomic_update_terminal",
            "index_work_bound_guard",
            "multiwork_index_iteration",
            "noncontiguous_update_location",
        ):
            with self.subTest(primitive=primitive):
                self.assertIn(primitive, scatter["observed_primitives"])
                self.assertNotIn(primitive, gather["observed_primitives"])
        normalized = " ".join(candidate["normalized_steps"])
        for variant_only in (
            "IDX_NDIM",
            "three-dimensional",
            "NWORK",
            "atomic_update",
            "conflict policy",
        ):
            with self.subTest(variant_only=variant_only):
                self.assertNotIn(variant_only, normalized)
        variants = " ".join(candidate["observed_variants"])
        for boundary in (
            "does not claim axis validation",
            "index bounds validation",
            "memory safety",
            "destination ownership",
            "race freedom",
            "equivalence to a Compiler access form",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, variants)
        manifest = _read(EXTRACTION / "manifest.json")
        non_claims = " ".join(manifest["selection_policy"]["non_claims"])
        for boundary in (
            "is not a bounds check",
            "no in-bounds or memory-safety result follows",
            "No race freedom, destination ownership, atomic conflict policy",
            "does not establish equivalence to Compiler AccessIndexKind.BUFFER",
            "mask_tiled_axes zero-fill behavior",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, non_claims)


class AbstractionExtractionValidatorTests(unittest.TestCase):
    def test_cli_validates_the_source_grounded_extraction(self) -> None:
        result = _run(output_format="json")

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        manifest = _read(EXTRACTION / "manifest.json")
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["extraction_id"], manifest["extraction_id"])
        self.assertEqual(summary["counts"], manifest["counts"])
        self.assertEqual(summary["verification_scope"], "metadata_consistency")
        self.assertIs(summary["source_bytes_verified"], False)

    def test_cli_supports_text_markdown_and_json_reports(self) -> None:
        expected = {
            None: "abstraction extraction metadata consistent:",
            "text": "abstraction extraction metadata consistent:",
            "markdown": "# Abstraction extraction metadata consistency",
            "json": '"verification_scope": "metadata_consistency"',
        }
        for output_format, marker in expected.items():
            with self.subTest(output_format=output_format or "default"):
                result = _run(output_format=output_format)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(marker, result.stdout)

    def test_schema_semantic_drift_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            _write(extraction / "schema.json", {})

        self._assert_mutation_rejected(
            mutate, "schema differs from the canonical extraction schema"
        )

    def test_nonfinite_json_number_fails_without_a_traceback(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            (extraction / "schema.json").write_text(
                '{"overflow": 1e999}\n', encoding="utf-8"
            )

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            library = fixture / "operator_library"
            extraction = fixture / "abstraction_extraction"
            shutil.copytree(LIBRARY, library)
            shutil.copytree(EXTRACTION, extraction)
            mutate(library, extraction)
            result = _run(library, extraction)

        combined = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("non-finite json number", combined.lower())
        self.assertNotIn("traceback", combined.lower())

    def test_unpaired_surrogate_fails_without_a_traceback(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            (extraction / "schema.json").write_text(
                '{"surrogate": "\\ud800"}\n', encoding="utf-8"
            )

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            library = fixture / "operator_library"
            extraction = fixture / "abstraction_extraction"
            shutil.copytree(LIBRARY, library)
            shutil.copytree(EXTRACTION, extraction)
            mutate(library, extraction)
            result = _run(library, extraction)

        combined = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("cannot be canonicalized", combined.lower())
        self.assertNotIn("traceback", combined.lower())

    def test_unpaired_surrogate_in_candidate_fails_without_a_traceback(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            manifest = _read(extraction / "manifest.json")
            path = extraction / str(manifest["candidates"][0])
            candidate = _read(path)
            candidate["summary"] = "\ud800"
            _write(path, candidate)

        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            library = fixture / "operator_library"
            extraction = fixture / "abstraction_extraction"
            shutil.copytree(LIBRARY, library)
            shutil.copytree(EXTRACTION, extraction)
            mutate(library, extraction)
            result = _run(library, extraction)

        combined = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("cannot be canonicalized as utf-8 json", combined.lower())
        self.assertNotIn("traceback", combined.lower())

    def _assert_mutation_rejected(
        self,
        mutate: Callable[[Path, Path], None],
        expected_diagnostic: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            library = fixture / "operator_library"
            extraction = fixture / "abstraction_extraction"
            shutil.copytree(LIBRARY, library)
            shutil.copytree(EXTRACTION, extraction)
            mutate(library, extraction)

            result = _run(library, extraction)

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(expected_diagnostic, (result.stderr + result.stdout).lower())

    def test_operator_library_input_drift_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del extraction
            manifest = _read(library / "manifest.json")
            source_path = library / str(manifest["sources"][0])
            source = _read(source_path)
            source["title"] = f"{source['title']} drift"
            _write(source_path, source)

        self._assert_mutation_rejected(mutate, "canonical digest differs")

    def test_operator_library_validator_runs_before_extraction_validation(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            library_manifest = _read(library / "manifest.json")
            source_path = library / str(library_manifest["sources"][0])
            source = _read(source_path)
            source["code_copied"] = True
            _write(source_path, source)

            extraction_manifest_path = extraction / "manifest.json"
            extraction_manifest = _read(extraction_manifest_path)
            extraction_manifest["input_library"]["canonical_sha256"] = (
                _operator_library_digest(library)
            )
            _write(extraction_manifest_path, extraction_manifest)

        self._assert_mutation_rejected(mutate, "code_copied")

    def test_paper_source_must_remain_the_library_method_source(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            library_manifest_path = library / "manifest.json"
            library_manifest = _read(library_manifest_path)
            library_manifest["method_source"] = "sources/mlx-main-1f8e74e.json"
            _write(library_manifest_path, library_manifest)

            extraction_manifest_path = extraction / "manifest.json"
            extraction_manifest = _read(extraction_manifest_path)
            extraction_manifest["input_library"]["canonical_sha256"] = (
                _operator_library_digest(library)
            )
            _write(extraction_manifest_path, extraction_manifest)

        self._assert_mutation_rejected(
            mutate,
            "operator_library/manifest.json.method_source",
        )

    def test_observation_locator_drift_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            manifest = _read(extraction / "manifest.json")
            path = extraction / str(manifest["observations"][0])
            observation = _read(path)
            observation["source_locator"]["git_object"] = "0" * 40
            _write(path, observation)

        self._assert_mutation_rejected(
            mutate, "does not match one repository_path locator"
        )

    def test_scan_observation_symbol_drift_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            path = (
                extraction
                / "observations/mlx-scan-contiguous-block-prefix-handoff.json"
            )
            observation = _read(path)
            observation["source_locator"]["symbol"] = "strided_scan"
            _write(path, observation)

        self._assert_mutation_rejected(
            mutate, "does not match one repository_path locator"
        )

    def test_sort_observation_symbol_drift_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            path = (
                extraction
                / "observations/mlx-sort-block-merge-state-exchange.json"
            )
            observation = _read(path)
            observation["source_locator"]["symbol"] = "not_the_bound_symbol"
            _write(path, observation)

        self._assert_mutation_rejected(
            mutate, "does not match one repository_path locator"
        )

    def test_candidate_with_one_observation_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            manifest = _read(extraction / "manifest.json")
            path = extraction / str(manifest["candidates"][0])
            candidate = _read(path)
            candidate["observation_ids"] = candidate["observation_ids"][:1]
            _write(path, candidate)

        self._assert_mutation_rejected(mutate, "requires at least 2 observations")

    def test_in_progress_extraction_may_have_no_candidate_yet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            library = fixture / "operator_library"
            extraction = fixture / "abstraction_extraction"
            shutil.copytree(LIBRARY, library)
            shutil.copytree(EXTRACTION, extraction)
            manifest_path = extraction / "manifest.json"
            manifest = _read(manifest_path)
            for relative in manifest["candidates"]:
                (extraction / str(relative)).unlink()
            manifest["candidates"] = []
            manifest["counts"]["candidate_count"] = 0
            _write(manifest_path, manifest)

            result = _run(library, extraction, output_format="json")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["counts"]["candidate_count"], 0)

    def test_candidate_pattern_must_match_every_observation(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            manifest = _read(extraction / "manifest.json")
            path = extraction / str(manifest["observations"][0])
            observation = _read(path)
            observation["pattern_signature"] = "different.reviewed.pattern"
            _write(path, observation)

        self._assert_mutation_rejected(
            mutate, "differs from referenced observation pattern_signature"
        )

    def test_claim_scope_cannot_expand_to_a_later_stage_result(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            manifest = _read(extraction / "manifest.json")
            path = extraction / str(manifest["candidates"][0])
            candidate = _read(path)
            candidate["claim_scope"] = "performance_result"
            _write(path, candidate)

        self._assert_mutation_rejected(mutate, "must equal 'source_pattern_only'")

    def test_candidate_evidence_breadth_drift_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            manifest = _read(extraction / "manifest.json")
            path = extraction / str(manifest["candidates"][0])
            candidate = _read(path)
            candidate["evidence_breadth"]["observation_count"] += 1
            _write(path, candidate)

        self._assert_mutation_rejected(
            mutate, "does not equal breadth derived from referenced observations"
        )

    def test_candidate_evidence_count_must_remain_an_integer(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            manifest = _read(extraction / "manifest.json")
            path = extraction / str(manifest["candidates"][0])
            candidate = _read(path)
            candidate["evidence_breadth"]["observation_count"] = 2.0
            _write(path, candidate)

        self._assert_mutation_rejected(mutate, "must be an integer >= 2")

    def test_manifest_summary_count_drift_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            path = extraction / "manifest.json"
            manifest = _read(path)
            manifest["counts"]["candidate_count"] += 1
            _write(path, manifest)

        self._assert_mutation_rejected(
            mutate, "does not equal counts derived from listed artifacts"
        )

    def test_extra_root_artifact_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            (extraction / "upstream-kernel.metal").write_text(
                "kernel void vendored() {}\n", encoding="utf-8"
            )

        self._assert_mutation_rejected(
            mutate, "unlisted abstraction_extraction entries"
        )

    def test_extra_observation_directory_artifact_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            (extraction / "observations" / "notes.txt").write_text(
                "not a listed observation\n", encoding="utf-8"
            )

        self._assert_mutation_rejected(mutate, "unlisted directory entries")

    def test_listed_observation_symlink_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            manifest = _read(extraction / "manifest.json")
            path = extraction / str(manifest["observations"][0])
            path.unlink()
            path.symlink_to("../manifest.json")

        self._assert_mutation_rejected(mutate, "regular non-symlink file")

    def test_unreadable_listed_directory_fails_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            library = fixture / "operator_library"
            extraction = fixture / "abstraction_extraction"
            shutil.copytree(LIBRARY, library)
            shutil.copytree(EXTRACTION, extraction)
            observations = extraction / "observations"
            observations.chmod(0)
            try:
                result = _run(library, extraction)
            finally:
                observations.chmod(0o700)

        if result.returncode == 0:
            self.skipTest("current user can enumerate mode-000 directories")
        combined = result.stderr + result.stdout
        self.assertIn("could not enumerate directory", combined.lower())
        self.assertNotIn("traceback", combined.lower())

    def test_reversed_source_span_is_rejected(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            manifest = _read(extraction / "manifest.json")
            path = extraction / str(manifest["observations"][0])
            observation = _read(path)
            observation["source_span"]["end_line"] = (
                observation["source_span"]["start_line"] - 1
            )
            _write(path, observation)

        self._assert_mutation_rejected(mutate, "end_line must be >= start_line")

    def test_candidate_requires_distinct_implementations_and_blobs(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            manifest = _read(extraction / "manifest.json")
            candidate = _read(extraction / str(manifest["candidates"][0]))
            paths_by_id = {
                _read(extraction / str(relative))["observation_id"]: extraction
                / str(relative)
                for relative in manifest["observations"]
            }
            first_path, second_path = (
                paths_by_id[observation_id]
                for observation_id in candidate["observation_ids"][:2]
            )
            first = _read(first_path)
            second = _read(second_path)
            for field in ("implementation_id", "operator_id", "source_id"):
                second[field] = first[field]
            second["source_locator"] = first["source_locator"]
            _write(second_path, second)

        self._assert_mutation_rejected(
            mutate, "requires distinct implementation_id evidence"
        )

    def test_candidate_requires_distinct_git_blobs(self) -> None:
        def mutate(library: Path, extraction: Path) -> None:
            del library
            manifest = _read(extraction / "manifest.json")
            candidate = _read(extraction / str(manifest["candidates"][0]))
            paths_by_id = {
                _read(extraction / str(relative))["observation_id"]: extraction
                / str(relative)
                for relative in manifest["observations"]
            }
            first_path, second_path = (
                paths_by_id[observation_id]
                for observation_id in candidate["observation_ids"][:2]
            )
            first = _read(first_path)
            second = _read(second_path)
            second["source_locator"]["git_object"] = first["source_locator"][
                "git_object"
            ]
            _write(second_path, second)

        self._assert_mutation_rejected(mutate, "requires distinct git_object evidence")

    def test_extraction_artifacts_are_not_clean_start_compiler_sources(self) -> None:
        source_set = _read(ROOT / "compiler" / "source_set.json")
        paths = source_set["paths"]

        self.assertFalse(
            any(
                path == "abstraction_extraction"
                or path.startswith("abstraction_extraction/")
                for path in paths
            )
        )


if __name__ == "__main__":
    unittest.main()
