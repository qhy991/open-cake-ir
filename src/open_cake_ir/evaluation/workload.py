"""Reusable Workload Contract independent of study and provider policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, cast

_FIELDS = {
    "schema_version",
    "workload_id",
    "revision",
    "state",
    "provenance",
    "operator",
    "cases",
    "tensors",
    "semantics",
    "oracle",
    "validation",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _name(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _validate_flash_contract(document: Mapping[str, object]) -> None:
    workload_id = document.get("workload_id")
    revision = document.get("revision")
    if (workload_id, revision) not in {
        ("flash-kmeans-assign-independent-v1", "1"),
        ("flash-kmeans-assign-independent-v2", "2"),
    }:
        raise ValueError("Flash-KMeans workload identity or revision differs")
    expected_tensors = {
        "tokens": {
            "shape": ["B", "N", "D"],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "centroids": {
            "shape": ["B", "K", "D"],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "assignments": {
            "shape": ["B", "N"],
            "dtype": "int32",
            "layout": "contiguous_row_major",
        },
    }
    expected_semantics = {
        "definition": "argmin_k sum_d (tokens[b,n,d] - centroids[b,k,d])^2",
        "accumulator_dtype": "fp32",
        "tie_break": "lowest_index_for_equal_fp32_score",
        "nonfinite": "reject_before_oracle",
    }
    expected_oracle = {
        "kind": "fp32_full_k_chunked",
        "source_sha256": "a74090ca7e5575f5fd6a027600d3330b5fa098078944c089e47c3077af3114d2",
        "chunk_n": 8192,
        "tf32": False,
    }
    expected_validation = {
        "canonical": "exact_int32_ids_with_lowest_index_ties",
        "tie_diagnostic_atol": 0.01,
        "tie_diagnostic_rtol": 0.01,
        "paper_acceptance_rule": "unknown",
    }
    if document.get("tensors") != expected_tensors:
        raise ValueError("Flash-KMeans workload tensors differ")
    if document.get("semantics") != expected_semantics:
        raise ValueError("Flash-KMeans workload semantics differ")
    if document.get("oracle") != expected_oracle:
        raise ValueError("Flash-KMeans workload oracle differs")
    if document.get("validation") != expected_validation:
        raise ValueError("Flash-KMeans workload validation differs")
    expected_cases = (
        ("tail_nk", {"B": 1, "N": 257, "K": 257, "D": 128}, 1001, "random_standard_normal"),
        ("batched_tail", {"B": 2, "N": 129, "K": 257, "D": 128}, 1002, "random_standard_normal"),
        ("b32_smoke", {"B": 32, "N": 512, "K": 1024, "D": 128}, 1003, "random_standard_normal"),
        ("duplicate_tie", {"B": 1, "N": 4, "K": 4, "D": 128}, None, "constructed_duplicate_centroids"),
        ("public_b1", {"B": 1, "N": 65536, "K": 1024, "D": 128}, 0, "random_standard_normal"),
        ("headline_b32", {"B": 32, "N": 65536, "K": 1024, "D": 128}, 0, "random_standard_normal"),
    )
    raw_cases = cast(list[Mapping[str, object]], document["cases"])
    observed = tuple(
        (case.get("case_id"), case.get("shape"), case.get("seed"), case.get("mode"))
        for case in raw_cases
    )
    if observed != expected_cases:
        raise ValueError("Flash-KMeans workload cases differ")
    materialized_by_case = {
        cast(str, case["case_id"]): case.get("materialized") for case in raw_cases
    }
    if materialized_by_case["headline_b32"] is None:
        raise ValueError("Flash-KMeans headline materialized authority is missing")
    if revision == "1" and materialized_by_case["b32_smoke"] is not None:
        raise ValueError("Flash-KMeans v1 unexpectedly owns b32 materialization")
    if revision == "2" and materialized_by_case["b32_smoke"] is None:
        raise ValueError("Flash-KMeans v2 b32 materialized authority is missing")


def _validate_tinygemm_contract(document: Mapping[str, object]) -> None:
    workload_id = document.get("workload_id")
    revision = document.get("revision")
    if (workload_id, revision) not in {
        ("tinygemm2-stage4-independent-v1", "1"),
        ("tinygemm2-stage4-independent-v2", "2"),
    }:
        raise ValueError("TinyGEMM2 workload identity or revision differs")
    expected_tensors = {
        "input": {
            "shape": ["batch", "input_features"],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
        },
        "weight": {
            "shape": ["output_features", "input_features"],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
        },
        "bias": {
            "shape": ["output_features"],
            "dtype": "bf16",
            "layout": "contiguous",
        },
        "output": {
            "shape": ["batch", "output_features"],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
        },
    }
    expected_semantics = {
        "definition": "bf16_round(input_fp32 @ weight_fp32^T + bias_fp32)",
        "accumulator_dtype": "fp32",
        "tie_break": "not_applicable",
        "nonfinite": "equal_nan_false",
    }
    expected_oracle = {
        "kind": "fp32_linear_then_bf16_round",
        "source_sha256": "9cea3117f9202051a5ff9c63221aa81d212aed7fe38b597c7aa540e96c5cb060",
        "atol": 0.01,
        "rtol": 0.01,
    }
    expected_validation = {
        "canonical": "bitwise_equal_to_pinned_stage4_parent_and_tolerance_equal_to_fp32_oracle",
        "parent_revision": "339a8f4cb2644fb85de79f3eed3624a1e91183d1",
        "paper_acceptance_rule": "not_applicable_known_kernel_lane",
    }
    if document.get("tensors") != expected_tensors:
        raise ValueError("TinyGEMM2 workload tensors differ")
    if document.get("semantics") != expected_semantics:
        raise ValueError("TinyGEMM2 workload semantics differ")
    if document.get("oracle") != expected_oracle:
        raise ValueError("TinyGEMM2 workload oracle differs")
    if document.get("validation") != expected_validation:
        raise ValueError("TinyGEMM2 workload validation differs")
    cases = cast(list[Mapping[str, object]], document["cases"])
    if len(cases) != 1 or (
        cases[0].get("case_id"),
        cases[0].get("shape"),
        cases[0].get("seed"),
        cases[0].get("mode"),
    ) != (
        "stage4_n8_m1024_k1024",
        {"batch": 8, "output_features": 1024, "input_features": 1024},
        0,
        "random_normal",
    ):
        raise ValueError("TinyGEMM2 workload cases differ")
    materialized = cases[0].get("materialized")
    if revision == "1" and materialized is not None:
        raise ValueError("TinyGEMM2 v1 unexpectedly owns materialization")
    if revision == "2" and set(
        _object(materialized, "TinyGEMM2 v2 materialized authority")
    ) != {
        "input",
        "weight",
        "bias",
        "fp32_linear_bf16_oracle",
        "parent_output",
    }:
        raise ValueError("TinyGEMM2 v2 materialized authority differs")


def _validate_qsa_contract(document: Mapping[str, object]) -> None:
    """Validate the first task-declared full QSA prefill contract."""

    if (
        document.get("workload_id"),
        document.get("revision"),
    ) != ("qsa-prefill-t32768-task-geometry-v1", "1"):
        raise ValueError("QSA workload identity or revision differs")
    expected_tensors = {
        "q": {
            "shape": ["B", "T", 32, 128],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "k": {
            "shape": ["B", "T", 4, 128],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "v": {
            "shape": ["B", "T", 4, 128],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "index_q": {
            "shape": ["B", "T", 8, 128],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "index_k": {
            "shape": ["B", "T", 1, 128],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "k_norm_weight": {
            "shape": [128],
            "dtype": "fp32",
            "layout": "contiguous",
            "finite_only": True,
        },
        "output": {
            "shape": ["B", "T", 32, 128],
            "dtype": "bf16",
            "layout": "contiguous_row_major",
        },
    }
    expected_semantics = {
        "definition": "complete causal blocks -> mean-pool 4 index keys -> FP32 LayerNorm -> sum_h relu(dot(index_q_h, pooled_key)) / sqrt(128) -> top-512 blocks -> selected-token causal GQA",
        "input_boundary": "post-projection q, k, v, index_q, and index_k; query normalization, RoPE, projection, cache update, and serving integration are excluded",
        "relu_reduction_order": "relu_before_sum_over_index_heads",
        "layernorm_epsilon": 1e-6,
        "task_declared_geometry": {
            "num_heads": 32,
            "num_kv_heads": 4,
            "head_dim": 128,
            "index_n_heads": 8,
            "index_kv_heads": 1,
            "index_head_dim": 128,
            "compress_ratio": 4,
            "token_budget": 2048,
        },
        "checkpoint_configuration_claimed": False,
    }
    expected_oracle = {
        "kind": "fp32_dense_gqa_task_declared",
        "source_sha256": "1e13196f9f72088ac7f4a706c33ad636939019fcb1f9a12229ca9d7da0d40b0b",
        "tf32": False,
    }
    expected_validation = {
        "canonical": "elementwise_tolerance_minimum_match_fraction",
        "atol": 0.05,
        "rtol": 0.05,
        "minimum_match_fraction": 0.999,
        "paper_acceptance_rule": "task_specific_not_qwen_checkpoint_or_serving",
    }
    if document.get("tensors") != expected_tensors:
        raise ValueError("QSA workload tensors differ")
    if document.get("semantics") != expected_semantics:
        raise ValueError("QSA workload semantics differ")
    if document.get("oracle") != expected_oracle:
        raise ValueError("QSA workload oracle differs")
    if document.get("validation") != expected_validation:
        raise ValueError("QSA workload validation differs")
    cases = cast(list[Mapping[str, object]], document["cases"])
    observed = tuple(
        (case.get("case_id"), case.get("shape"), case.get("seed"), case.get("mode"))
        for case in cases
    )
    if observed != (
        ("conformance_t4096", {"B": 1, "T": 4096}, 20260828, "seeded_standard_normal"),
        ("target_t32768", {"B": 1, "T": 32768}, 20260829, "seeded_standard_normal"),
    ):
        raise ValueError("QSA workload cases differ")


def _validate_dsa_contract(document: Mapping[str, object]) -> None:
    """Validate structural and cross-field invariants for sparse MLA decode."""

    if (document.get("workload_id"), document.get("revision")) != (
        "deepseek-v3.2-dsa-sparse-mla-decode-v1",
        "1",
    ):
        raise ValueError("DSA workload identity or revision differs")
    tensors = _object(document.get("tensors"), "DSA tensors")
    semantics = _object(document.get("semantics"), "DSA semantics")
    geometry = _object(semantics.get("geometry"), "DSA geometry")
    matrix = _object(semantics.get("case_matrix"), "DSA case matrix")
    oracle = _object(document.get("oracle"), "DSA oracle")
    baseline = _object(oracle.get("timing_baseline"), "DSA timing baseline")
    validation = _object(document.get("validation"), "DSA validation")
    measurement = _object(validation.get("measurement"), "DSA measurement")
    cases = cast(list[Mapping[str, object]], document["cases"])
    def tensor_shape(name: str) -> list[object]:
        raw = _object(tensors.get(name), f"DSA tensor {name}").get("shape")
        if not isinstance(raw, list):
            raise ValueError(f"DSA tensor {name} shape is invalid")
        return raw

    heads = geometry.get("num_query_heads")
    ckv_dim = geometry.get("compressed_kv_dim")
    rope_dim = geometry.get("rope_dim")
    page_size = geometry.get("page_size")
    top_k = geometry.get("top_k")
    expected_shapes = {
        "q_nope": ["num_tokens", heads, ckv_dim],
        "q_pe": ["num_tokens", heads, rope_dim],
        "ckv_cache": ["num_pages", page_size, ckv_dim],
        "kpe_cache": ["num_pages", page_size, rope_dim],
        "sparse_indices": ["num_tokens", top_k],
        "output": ["num_tokens", heads, ckv_dim],
    }
    if any(tensor_shape(name) != shape for name, shape in expected_shapes.items()):
        raise ValueError("DSA tensor geometry is inconsistent")
    scale = _object(tensors.get("sm_scale"), "DSA sm_scale").get("value")
    scale_authority = _object(
        semantics.get("sm_scale_authority"), "DSA scale authority"
    )
    if scale != scale_authority.get("executed_value"):
        raise ValueError("DSA executed scale is inconsistent")
    if baseline.get("sparse_mla_top_k") != top_k:
        raise ValueError("DSA baseline top-k is inconsistent")
    modes = [case.get("mode") for case in cases]
    if any(not isinstance(mode, str) for mode in modes):
        raise ValueError("DSA case mode is invalid")
    observed_matrix = {
        "total_rows": len(cases),
        "captured_rows": sum(mode.startswith("captured_") for mode in modes),
        "generated_rows": sum(mode.startswith("generated_") for mode in modes),
        "small_rows": sum(mode.endswith("_small") for mode in modes),
        "large_rows": sum(mode.endswith("_large") for mode in modes),
    }
    if set(matrix) != set(observed_matrix) or any(
        matrix.get(name) != count for name, count in observed_matrix.items()
    ):
        raise ValueError("DSA case matrix is inconsistent")
    asset_sources = [
        _object(entry, "DSA provenance entry")
        for entry in cast(list[object], document["provenance"])
        if _object(entry, "DSA provenance entry").get("kind")
        == "task_captured_asset_source"
    ]
    if len(asset_sources) != 1 or asset_sources[0].get("asset_count") != observed_matrix[
        "captured_rows"
    ]:
        raise ValueError("DSA captured asset provenance is inconsistent")
    for name in ("required_cupti_major", "warmup_iterations", "samples_per_trial", "trials"):
        value = measurement.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"DSA measurement {name} is invalid")


def _validate_kda_fused_decode_contract(document: Mapping[str, object]) -> None:
    """Validate structural and cross-field invariants for KDA fused decode."""

    if (document.get("workload_id"), document.get("revision")) != (
        "kimi-k3-kda-fused-decode-v1",
        "1",
    ):
        raise ValueError("KDA fused-decode workload identity or revision differs")
    tensors = _object(document.get("tensors"), "KDA tensors")
    semantics = _object(document.get("semantics"), "KDA semantics")
    geometry = _object(semantics.get("geometry"), "KDA geometry")
    validation = _object(document.get("validation"), "KDA validation")
    tiers = _object(validation.get("cell_tiers"), "KDA cell tiers")
    trajectory = _object(validation.get("trajectory"), "KDA trajectory")
    measurement = _object(validation.get("measurement"), "KDA measurement")
    cases = cast(list[Mapping[str, object]], document["cases"])
    abi = semantics.get("candidate_abi")
    if (
        not isinstance(abi, list)
        or not abi
        or any(not isinstance(name, str) or not name for name in abi)
        or len(abi) != len(set(abi))
        or set(abi) != set(tensors) - {"output"}
    ):
        raise ValueError("KDA candidate ABI and tensors are inconsistent")

    k_dim = geometry.get("K")
    v_dim = geometry.get("V")
    history = geometry.get("conv_history_width")
    width = geometry.get("conv_width")
    slot_pad = geometry.get("state_slot_pad_fp32_elements")
    supported_heads = geometry.get("supported_rank_local_heads")
    if (
        not all(isinstance(value, int) for value in (k_dim, v_dim, slot_pad, width, history))
        or not isinstance(supported_heads, list)
        or history != width - 1
    ):
        raise ValueError("KDA geometry is invalid")
    if semantics.get("state_slots") != "batch_size+4":
        raise ValueError("KDA state-slot formula differs")
    scale = _object(tensors.get("scale"), "KDA scale").get("value")
    if not isinstance(scale, (int, float)) or abs(scale * scale * k_dim - 1.0) > 1e-12:
        raise ValueError("KDA scale and key dimension are inconsistent")

    required_shape_fields = {
        "num_heads",
        "head_dim",
        "batch_size",
        "active_rows",
        "tokens_per_request",
        "state_slot_pad",
    }
    case_ids: list[str] = []
    for case in cases:
        shape = _object(case.get("shape"), "KDA case shape")
        heads = shape.get("num_heads")
        batch = shape.get("batch_size")
        active = shape.get("active_rows")
        case_id = cast(str, case.get("case_id"))
        if (
            set(shape) != required_shape_fields
            or heads not in supported_heads
            or shape.get("head_dim") != k_dim
            or shape.get("tokens_per_request") != semantics.get("tokens_per_graph_row")
            or shape.get("state_slot_pad") != slot_pad
            or not isinstance(batch, int)
            or not isinstance(active, int)
            or active > batch
            or case_id != f"h{heads}-b{batch}-a{active}"
        ):
            raise ValueError("KDA case geometry or identity is inconsistent")
        case_ids.append(case_id)

    tier_ids: list[str] = []
    for tier, raw_ids in tiers.items():
        if (
            not isinstance(tier, str)
            or not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(case_id, str) for case_id in raw_ids)
        ):
            raise ValueError("KDA cell tier is invalid")
        tier_ids.extend(cast(list[str], raw_ids))
    if len(tier_ids) != len(set(tier_ids)) or set(tier_ids) != set(case_ids):
        raise ValueError("KDA cell tiers do not partition the case matrix")

    ssm = _object(tensors.get("ssm_states"), "KDA SSM state")
    if ssm.get("shape") != ["slots", "H", v_dim, k_dim] or ssm.get("strides") != [
        f"H*{v_dim}*{k_dim}+{slot_pad}",
        v_dim * k_dim,
        k_dim,
        1,
    ]:
        raise ValueError("KDA SSM state shape or strides are inconsistent")
    conv = _object(tensors.get("conv_states"), "KDA convolution state")
    if conv.get("shape") != ["slots", history, f"3*H*{k_dim}"]:
        raise ValueError("KDA convolution state shape is inconsistent")

    steps = trajectory.get("steps_per_cell")
    if (
        trajectory.get("coverage") != "all_nine_matrix_cells"
        or not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps <= 0
        or trajectory.get("promotion_required") is not True
    ):
        raise ValueError("KDA trajectory is invalid")
    pair_order = _object(measurement.get("pair_order"), "KDA pair order")
    if (
        set(pair_order) != {"baseline_then_candidate", "candidate_then_baseline"}
        or any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in pair_order.values())
        or len(set(pair_order.values())) != 1
    ):
        raise ValueError("KDA pair order is invalid")
    for name in ("required_cupti_major", "warmup_iterations_per_arm", "samples_per_cohort"):
        value = measurement.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"KDA measurement {name} is invalid")


def _validate_kda_decode_megaop_b200_contract(
    document: Mapping[str, object],
) -> None:
    """Validate the separately named Kimi-K3 B200 rank-local megaop."""

    identity = (document.get("workload_id"), document.get("revision"))
    if identity not in {
        ("kimi-k3-kda-decode-megaop-b200-v1", "1"),
        ("kimi-k3-kda-decode-megaop-b200-v2", "2"),
    }:
        raise ValueError("KDA B200 megaop workload identity or revision differs")
    tensors = _object(document.get("tensors"), "KDA B200 megaop tensors")
    semantics = _object(document.get("semantics"), "KDA B200 megaop semantics")
    geometry = _object(semantics.get("geometry"), "KDA B200 megaop geometry")
    hardware = _object(semantics.get("hardware"), "KDA B200 megaop hardware")
    validation = _object(document.get("validation"), "KDA B200 megaop validation")
    measurement = _object(validation.get("measurement"), "KDA B200 megaop measurement")
    expected_abi = [
        "hidden_states", "qkvg_weight", "bfa_weight", "f_b_weight",
        "conv_states", "w_q_t", "w_k_t", "w_v_t", "conv_bias", "A_log",
        "dt_bias", "onorm_weight", "ssm_states", "cache_indices",
        "o_proj_weight", "scale", "onorm_eps", "lower_bound",
    ]
    if semantics.get("candidate_abi") != expected_abi or set(tensors) != {
        *expected_abi,
        "local_output",
    }:
        raise ValueError("KDA B200 megaop candidate ABI and tensors differ")
    bf16_tensors = {
        "hidden_states", "qkvg_weight", "bfa_weight", "f_b_weight",
        "conv_states", "o_proj_weight", "local_output",
    }
    for name, raw_tensor in tensors.items():
        dtype = "int32" if name == "cache_indices" else "bf16" if name in bf16_tensors else "fp32"
        if _object(raw_tensor, f"KDA B200 tensor {name}").get("dtype") != dtype:
            raise ValueError(f"KDA B200 megaop tensor {name} dtype differs")
    expected_shapes = {
        "hidden_states": [18, 7168], "qkvg_weight": [6144, 7168],
        "bfa_weight": [144, 7168], "f_b_weight": [1536, 128],
        "conv_states": [49, 3, 4608], "w_q_t": [4, 1536],
        "w_k_t": [4, 1536], "w_v_t": [4, 1536], "conv_bias": [4608],
        "A_log": [12], "dt_bias": [1536], "onorm_weight": [128],
        "ssm_states": [49, 12, 128, 128], "cache_indices": [18],
        "o_proj_weight": [7168, 1536], "local_output": [18, 7168],
    }
    for name, shape in expected_shapes.items():
        if _object(tensors.get(name), f"KDA B200 tensor {name}").get("shape") != shape:
            raise ValueError(f"KDA B200 megaop tensor {name} shape differs")
    if _object(tensors.get("conv_states"), "KDA B200 conv state").get(
        "strides"
    ) != [13824, 4608, 1] or _object(
        tensors.get("ssm_states"), "KDA B200 SSM state"
    ).get("strides") != [196608, 16384, 128, 1]:
        raise ValueError("KDA B200 megaop state strides differ")
    scalar_values = tuple(
        _object(tensors.get(name), f"KDA B200 scalar {name}").get("value")
        for name in ("scale", "onorm_eps", "lower_bound")
    )
    if scalar_values != (0.08838834764831843, 1e-5, -5.0):
        raise ValueError("KDA B200 megaop scalar contract differs")
    if semantics.get("state_and_padding") != {
        "negative_row_output": "exact_zero",
        "negative_row_state_effect": "none",
        "selected_slots": "updated_exactly_once_per_call",
        "inactive_and_ghost_slots": "bitwise_unchanged",
        "same_slot_sequential_reuse": "continues_from_prior_conv_and_ssm_state",
    }:
        raise ValueError("KDA B200 megaop state and padding contract differs")
    if semantics.get("output_storage") != {
        "fresh": True, "contiguous": True, "exact_sized": True,
        "zero_offset": True, "input_aliasing": "forbidden",
    }:
        raise ValueError("KDA B200 megaop output storage contract differs")
    if (
        semantics.get("model") != "Kimi-K3"
        or semantics.get("stage") != "one_token_rank_local_decode"
        or semantics.get("parallelism") != {
            "tensor_parallel_size": 8,
            "timed_communication": "none",
            "first_excluded_node": "BF16 TP AllReduce",
        }
    ):
        raise ValueError("KDA B200 megaop rank-local boundary differs")
    oracle = _object(document.get("oracle"), "KDA B200 megaop oracle")
    required_oracle = {
        "kind": "independent_unfused_high_level_oracle_with_black_box_module_baseline",
        "revision": (
            "ff68211690d0d896c2bf9a6800a19424e14b591a"
            if identity[1] == "1"
            else "6f51fa40466ef3c88ee87ddd4160f70667f7eea6"
        ),
        "callable": "bench_kimi_k3_kda_decode_megaop_b200_standalone.reference_megaop",
        "named_endpoints": [
            "qkvg", "bfa", "mixed_qkv", "onorm_gate", "f_a", "beta",
            "forget_gate", "core_output", "local_output",
            "complete_post_call_conv_states", "complete_post_call_ssm_states",
        ],
        "timing_baseline": "source_derived_module_qualified_on_B200_execute_only",
    }
    if any(oracle.get(name) != value for name, value in required_oracle.items()):
        raise ValueError("KDA B200 megaop correctness oracle differs")
    required_validation = {
        "canonical": "all_named_endpoints_complete_pools_storage_and_sequential_reuse_before_timing",
        "atol": 0.02,
        "rtol": 0.02,
        "minimum_match_fraction": 1.0,
        "claim_boundary": "standalone_rank_local_B200_module_only_not_B300_model_forward_or_serving",
    }
    if (
        any(validation.get(name) != value for name, value in required_validation.items())
        or isinstance(validation.get("minimum_match_fraction"), bool)
        or validation.get("all_cases_required") is not True
    ):
        raise ValueError("KDA B200 megaop correctness or claim boundary differs")
    if hardware != {
        "gpu": "NVIDIA B200", "compute_capability": [10, 0],
        "multiprocessor_count": 148, "total_memory_bytes": 191490555904,
    }:
        raise ValueError("KDA B200 megaop hardware differs")
    if geometry != {
        "graph_rows": 18, "rank_local_heads": 12, "global_heads": 96,
        "head_dim": 128, "hidden_size": 7168, "pool_slots": 49,
        "ghost_slot": 0, "conv_width": 4, "conv_history_width": 3,
    }:
        raise ValueError("KDA B200 megaop geometry differs")
    case_shape = {
        "graph_rows": geometry["graph_rows"],
        "num_heads": geometry["rank_local_heads"],
        "head_dim": geometry["head_dim"],
        "hidden_size": geometry["hidden_size"],
        "pool_slots": geometry["pool_slots"],
        "tokens_per_request": 1,
    }
    cases = cast(list[Mapping[str, object]], document["cases"])
    observed_cases = tuple(
        (
            case.get("case_id"),
            _object(case.get("shape"), "KDA B200 megaop case shape"),
            case.get("mode"),
        )
        for case in cases
    )
    if observed_cases != (
        (
            "m18-h12-synthetic-all-active",
            {**case_shape, "active_rows": geometry["graph_rows"]},
            "synthetic_all_active_at_trace_proved_graph_bucket",
        ),
        (
            "m18-h12-one-padded-row",
            {**case_shape, "active_rows": geometry["graph_rows"] - 1},
            "interior_row_7_negative_cuda_graph_padding",
        ),
    ):
        raise ValueError("KDA B200 megaop case shape, identity, or mode differs")
    required_measurement = (
        {
            "correctness_warmup_iterations": 5,
            "correctness_samples_per_trial": 20,
            "correctness_trials": 5,
            "benchmark_warmup_iterations": 5,
            "benchmark_samples_per_trial": 20,
            "benchmark_trials": 25,
            "maximum_median_aba_drift": 0.05,
            "maximum_p90_aba_drift": 0.1,
            "maximum_trial_median_cv": 0.05,
            "minimum_paired_wins": 20,
            "minimum_speedup_each_case": 1.0526315789473684,
        }
        if identity[1] == "1"
        else {
            "correctness_warmup_iterations": 5,
            "correctness_samples_per_trial": 20,
            "correctness_trials": 5,
            "benchmark_warmup_iterations": 5,
            "benchmark_samples_per_trial": 20,
            "benchmark_trials": 25,
            "maximum_order_effect": 0.05,
            "maximum_trial_median_cv": 0.05,
            "minimum_paired_wins": 20,
            "minimum_speedup_each_case": 1.0526315789473684,
            "median_confidence_interval": "exact_nonparametric_at_least_95_percent",
            "baseline_control_acceptance": "interval_contains_1",
            "candidate_promotion_acceptance": "interval_lower_bound_at_least_minimum_speedup",
            "retain_all_raw_samples": True,
            "raw_sample_cv": "report_only",
            "unstable_measurement": "valid negative evidence, not promotion eligible",
        }
    )
    if any(measurement.get(name) != value for name, value in required_measurement.items()):
        raise ValueError("KDA B200 megaop measurement contract differs")
    expected_order = (
        ["baseline_before", "candidate", "baseline_after"]
        if identity[1] == "1"
        else {"baseline_candidate": 13, "candidate_baseline": 12}
    )
    order_field = "trial_order" if identity[1] == "1" else "paired_trial_order"
    if measurement.get(order_field) != expected_order:
        raise ValueError("KDA B200 megaop trial order differs")


@dataclass(frozen=True)
class TensorABI:
    """One ordered argument, resolved only from Workload tensor/case declarations."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    mode: str


def _validate_swiglu_contract(document: Mapping[str, object]) -> None:
    if (
        document.get("workload_id") != "swiglu-fp32-independent-v1"
        or document.get("revision") != "1"
    ):
        raise ValueError("SwiGLU workload identity or revision differs")
    expected_tensors = {
        "up": {
            "shape": ["B", "N", "D"],
            "dtype": "fp32",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "gate": {
            "shape": ["B", "N", "D"],
            "dtype": "fp32",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "y": {
            "shape": ["B", "N", "D"],
            "dtype": "fp32",
            "layout": "contiguous_row_major",
        },
    }
    expected_semantics = {
        "definition": "y = up * gate * 0.5 * (tanh(gate * 0.5) + 1.0)",
        "input_dtype": "fp32",
        "output_dtype": "fp32",
        "nonfinite": "reject_before_oracle",
    }
    expected_oracle = {
        "kind": "cpu_fp64_composition_then_fp32_round",
        "source_sha256": "36161c3cd947d663d9d9e75dd7babb0dff2d799ad021ae80d55508150f217036",
    }
    expected_validation = {
        "canonical": "torch_isclose_against_cpu_fp64_composition",
        "atol": 0.000002,
        "rtol": 0.00002,
        "performance_measured": False,
    }
    if document.get("tensors") != expected_tensors:
        raise ValueError("SwiGLU workload tensors differ")
    if document.get("semantics") != expected_semantics:
        raise ValueError("SwiGLU workload semantics differ")
    if document.get("oracle") != expected_oracle:
        raise ValueError("SwiGLU workload oracle differs")
    if document.get("validation") != expected_validation:
        raise ValueError("SwiGLU workload validation differs")
    expected_cases = (
        (
            "seeded_random",
            {"B": 8, "N": 512, "D": 128},
            20260825,
            "random_standard_normal",
        ),
        (
            "signed_saturation",
            {"B": 8, "N": 512, "D": 128},
            None,
            "constructed_signed_saturation",
        ),
    )
    cases = cast(list[Mapping[str, object]], document["cases"])
    observed = tuple(
        (case.get("case_id"), case.get("shape"), case.get("seed"), case.get("mode"))
        for case in cases
    )
    if observed != expected_cases:
        raise ValueError("SwiGLU workload cases differ")


def _validate_llama_rmsnorm_contract(document: Mapping[str, object]) -> None:
    if (document.get("workload_id"), document.get("revision")) not in {
        ("llama-rmsnorm-mul-fp32-independent-v1", "1"),
        ("llama-rmsnorm-mul-fp32-independent-v2", "2"),
    }:
        raise ValueError("llama RMSNorm workload identity or revision differs")
    expected_tensors = {
        "x": {
            "shape": ["B", "N", "D"],
            "dtype": "fp32",
            "layout": "contiguous_row_major",
            "finite_only": True,
        },
        "gamma": {
            "shape": ["D"],
            "dtype": "fp32",
            "layout": "contiguous",
            "finite_only": True,
        },
        "y": {
            "shape": ["B", "N", "D"],
            "dtype": "fp32",
            "layout": "contiguous_row_major",
        },
    }
    expected_semantics = {
        "definition": "y = x * gamma * rsqrt(mean_D(x * x) + epsilon)",
        "epsilon": 0.000001,
        "reduction_axis": "last",
        "input_dtype": "fp32",
        "output_dtype": "fp32",
        "nonfinite": "reject_before_oracle",
        "llama_slice": "GGML_OP_RMS_NORM + GGML_OP_MUL",
    }
    expected_oracle = {
        "kind": "cpu_fp64_rmsnorm_mul_then_fp32_round",
        "source_sha256": "49da17ec8f983be6fa72dd04ef0018b4dd50b8b79a09895fbf95bd61b7c6b1dd",
    }
    expected_validation = {
        "canonical": "torch_isclose_against_cpu_fp64_rmsnorm_mul",
        "atol": 0.000005,
        "rtol": 0.00005,
        "performance_measured": False,
    }
    if document.get("tensors") != expected_tensors:
        raise ValueError("llama RMSNorm workload tensors differ")
    if document.get("semantics") != expected_semantics:
        raise ValueError("llama RMSNorm workload semantics differ")
    if document.get("oracle") != expected_oracle:
        raise ValueError("llama RMSNorm workload oracle differs")
    if document.get("validation") != expected_validation:
        raise ValueError("llama RMSNorm workload validation differs")
    expected_cases = (
        (
            "seeded_random",
            {"B": 8, "N": 512, "D": 128},
            20260825,
            "random_standard_normal_signed_gamma",
        ),
        (
            "reduction_rsqrt_stress",
            {"B": 8, "N": 512, "D": 128},
            None,
            "constructed_reduction_rsqrt_stress",
        ),
    )
    cases = cast(list[Mapping[str, object]], document["cases"])
    observed = tuple(
        (case.get("case_id"), case.get("shape"), case.get("seed"), case.get("mode"))
        for case in cases
    )
    if observed != expected_cases:
        raise ValueError("llama RMSNorm workload cases differ")


def _validate_llama_q4_mmvq_contract(document: Mapping[str, object]) -> None:
    if (
        document.get("workload_id")
        != "llama-q4_0-q8_1-mmvq-f32-conformance-v1"
        or document.get("revision") != "1"
    ):
        raise ValueError("llama Q4 MMVQ workload identity or revision differs")
    expected_tensors = {
        "activation": {
            "shape": ["K"],
            "dtype": "fp32",
            "layout": "contiguous",
            "finite_only": True,
            "public_input": True,
        },
        "weight_q4_0": {
            "shape": [18],
            "dtype": "uint8",
            "layout": "ggml_q4_0_v1",
            "public_input": True,
        },
        "q8_1_workspace": {
            "shape": [16, 36],
            "dtype": "uint8",
            "layout": "ggml_q8_1_v1",
            "public_input": False,
        },
        "output": {
            "shape": ["M"],
            "dtype": "fp32",
            "layout": "contiguous",
            "public_input": False,
        },
    }
    expected_semantics = {
        "definition": "live_q8_1(activation_fp32) then q4_0_x_q8_1_mmvq to fp32",
        "claim_scope": "one_block_two_kernel_correctness_conformance_only",
        "block_size": 32,
        "q4_0_record": {
            "size_bytes": 18,
            "scale": "fp16 at byte 0",
            "payload": "16 uint8 at byte 2",
            "logical_order": "low nibble -> j; high nibble -> 16+j",
            "zero_point": 8,
            "scale_finite": True,
        },
        "q8_1_record": {
            "size_bytes": 36,
            "scale": "fp16 at byte 0",
            "stored_sum": "fp16 at byte 2",
            "payload": "32 int8 at byte 4",
        },
        "q8_1_workspace": {
            "logical_k": 32,
            "padded_k": 512,
            "record_count": 16,
            "consumer_record_count": 1,
            "padding_records": "15 all-zero Q8_1 records",
        },
        "q8_quantization": {
            "amax": "wave32 max(abs(fp32 x))",
            "sum": "wave32 xor-tree fp32 sum(original x)",
            "quant_scale": "fp32 amax / 127",
            "rounding": "roundf nearest with halfway away from zero",
            "zero_case": "amax == 0 produces d=0,s=0,q=0",
            "stored_scale": "fp16(quant_scale)",
            "stored_sum": "fp16(sum(original x))",
        },
        "stored_sum": "fp16(warp_tree_fp32_sum(original_x))",
        "partial_indices": ["0..7 plus 16..23", "8..15 plus 24..31"],
        "partial_result": (
            "fp32(d4) * (fp32(partial_sumi) * fp32(d8) - 4 * fp32(s8))"
        ),
        "block_result": "fp32(partial_iqs0 + partial_iqs2)",
        "public_q8_input": "forbidden",
        "required_kernel_sequence": ["fp32_to_q8_1", "q4_0_x_q8_1_mmvq"],
        "performance_measured": False,
        "llama_cpp_build_claim": False,
        "llama_cpp_e2e_claim": False,
    }
    expected_oracle = {
        "kind": "stdlib_source_ordered_ggml_q4_0_live_q8_1_block32",
        "source_sha256": "392b388757aa2fb71fd008bdabf4b88c83e6aba50acdcc44e26212e592f1f11d",
    }
    expected_validation = {
        "canonical": "byte_exact_q8_1_workspace_and_tolerant_fp32_block_output",
        "q8_1_workspace": (
            "all 576 bytes exact: consumed block plus 15 zero padding blocks"
        ),
        "atol": 0.000005,
        "rtol": 0.000005,
        "required_kernel_launches": 2,
        "fallback_calls": 0,
        "performance_measured": False,
        "promotion_authorized": False,
    }
    if document.get("tensors") != expected_tensors:
        raise ValueError("llama Q4 MMVQ workload tensors differ")
    if document.get("semantics") != expected_semantics:
        raise ValueError("llama Q4 MMVQ workload semantics differ")
    if document.get("oracle") != expected_oracle:
        raise ValueError("llama Q4 MMVQ workload oracle differs")
    if document.get("validation") != expected_validation:
        raise ValueError("llama Q4 MMVQ workload validation differs")
    expected_cases = (
        (
            "seeded_random",
            {"N": 1, "M": 1, "K": 32},
            20260825,
            "seeded_random_q4_and_activation",
        ),
        (
            "stored_sum_correction_stress",
            {"N": 1, "M": 1, "K": 32},
            None,
            "constructed_stored_sum_correction_stress",
        ),
        (
            "zero_scale",
            {"N": 1, "M": 1, "K": 32},
            None,
            "constructed_zero_scale",
        ),
        (
            "roundf_boundary",
            {"N": 1, "M": 1, "K": 32},
            None,
            "constructed_roundf_half_away_boundary",
        ),
        (
            "xor_tree_sum",
            {"N": 1, "M": 1, "K": 32},
            None,
            "constructed_xor_tree_sum_discriminator",
        ),
        (
            "partial_rounding",
            {"N": 1, "M": 1, "K": 32},
            1,
            "seeded_partial_rounding_discriminator",
        ),
        (
            "q4_zero_scale",
            {"N": 1, "M": 1, "K": 32},
            None,
            "constructed_q4_zero_scale",
        ),
    )
    cases = cast(list[Mapping[str, object]], document["cases"])
    observed = tuple(
        (case.get("case_id"), case.get("shape"), case.get("seed"), case.get("mode"))
        for case in cases
    )
    if observed != expected_cases:
        raise ValueError("llama Q4 MMVQ workload cases differ")


class WorkloadContract:
    """Canonical operator semantics, cases and correctness authority."""

    def __init__(self, document: Mapping[str, object], source_path: Path | None = None) -> None:
        self._document = json.loads(_canonical_json_bytes(document))
        self.source_path = source_path
        self.workload_id = _name(document.get("workload_id"), "workload.workload_id")
        self.canonical_sha256 = sha256(_canonical_json_bytes(document)).hexdigest()
        raw_cases = document.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("workload.cases must be a non-empty list")
        cases: dict[str, Mapping[str, object]] = {}
        for index, value in enumerate(raw_cases):
            case = _object(value, f"workload.cases[{index}]")
            if not {"case_id", "shape", "seed", "mode"} <= set(case) <= {
                "case_id",
                "shape",
                "seed",
                "mode",
                "materialized",
            }:
                raise ValueError(f"workload.cases[{index}] fields differ")
            case_id = _name(case.get("case_id"), f"workload.cases[{index}].case_id")
            if case_id in cases:
                raise ValueError(f"workload case {case_id!r} is duplicated")
            shape = _object(case.get("shape"), f"workload.cases[{index}].shape")
            if not shape or any(
                not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0
                for extent in shape.values()
            ):
                raise ValueError(f"workload.cases[{index}].shape is invalid")
            seed = case.get("seed")
            if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
                raise ValueError(f"workload.cases[{index}].seed is invalid")
            _name(case.get("mode"), f"workload.cases[{index}].mode")
            materialized = case.get("materialized")
            if materialized is not None:
                objects = _object(materialized, f"workload.cases[{index}].materialized")
                for name, raw_receipt in objects.items():
                    receipt = _object(raw_receipt, f"workload.cases[{index}].materialized.{name}")
                    if set(receipt) != {"sha256", "size_bytes"}:
                        raise ValueError(f"workload materialized receipt {name!r} fields differ")
                    digest = receipt.get("sha256")
                    size = receipt.get("size_bytes")
                    if (
                        not isinstance(digest, str)
                        or len(digest) != 64
                        or not isinstance(size, int)
                        or isinstance(size, bool)
                        or size <= 0
                    ):
                        raise ValueError(f"workload materialized receipt {name!r} is invalid")
            cases[case_id] = case
        self._cases = cases

    @classmethod
    def load(cls, path: str | Path) -> "WorkloadContract":
        """Load and validate one frozen Workload Contract."""

        source = Path(path).resolve(strict=True)
        value = json.loads(source.read_text(encoding="utf-8"))
        document = _object(value, "workload")
        if set(document) != _FIELDS or document.get("schema_version") != 1:
            raise ValueError("workload root fields or schema_version differ")
        if document.get("state") != "frozen":
            raise ValueError("workload must be frozen")
        _name(document.get("revision"), "workload.revision")
        _name(document.get("operator"), "workload.operator")
        provenance = document.get("provenance")
        if not isinstance(provenance, list) or not provenance:
            raise ValueError("workload provenance must be a non-empty list")
        for field in ("tensors", "semantics", "oracle", "validation"):
            _object(document.get(field), f"workload.{field}")
        if document.get("operator") == "flash_kmeans_assign":
            _validate_flash_contract(document)
        elif document.get("operator") == "tinygemm2_bf16_linear":
            _validate_tinygemm_contract(document)
        elif document.get("operator") == "qsa_prefill":
            _validate_qsa_contract(document)
        elif document.get("operator") == "dsa_attention":
            _validate_dsa_contract(document)
        elif document.get("operator") == "kda_fused_decode":
            _validate_kda_fused_decode_contract(document)
        elif document.get("operator") == "kimi_k3_kda_decode_megaop_b200":
            _validate_kda_decode_megaop_b200_contract(document)
        elif document.get("operator") in {
            "rmsnorm_fp32", "gemm_bias_bf16_fp32", "indexed_gather_bf16",
        }:
            from .tile_workloads import validate_tile_contract

            validate_tile_contract(document)
        elif document.get("operator") == "swiglu_fp32":
            _validate_swiglu_contract(document)
        elif document.get("operator") == "rmsnorm_mul_fp32":
            _validate_llama_rmsnorm_contract(document)
        elif document.get("operator") == "llama_q4_0_q8_1_mmvq_f32":
            _validate_llama_q4_mmvq_contract(document)
        else:
            raise ValueError("workload operator is unsupported")
        return cls(document, source)

    @property
    def document(self) -> dict[str, object]:
        """Return a detached JSON projection of the frozen contract."""

        return cast(dict[str, object], json.loads(_canonical_json_bytes(self._document)))

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(self._cases)

    def case(self, case_id: str) -> dict[str, object]:
        """Return one detached workload case by canonical ID."""

        try:
            value = self._cases[case_id]
        except KeyError as error:
            raise KeyError(f"unknown workload case {case_id!r}") from error
        return cast(dict[str, object], json.loads(_canonical_json_bytes(value)))

    def tensor_abi(self, case_id: str) -> tuple[TensorABI, ...]:
        """Resolve the explicitly ordered input/output ABI, without operator dispatch.

        A shape component is a positive integer or an exact case-dimension name;
        expressions are not evaluated. Historical contracts without this declaration
        retain their existing admission and do not acquire an inferred ABI.
        """

        shape = _object(self.case(case_id)["shape"], "workload case shape")
        tensors = _object(self._document["tensors"], "workload tensors")
        semantics = _object(self._document["semantics"], "workload semantics")
        abi = _object(semantics.get("candidate_abi"), "explicit workload tensor ABI")
        if set(abi) != {"inputs", "outputs"}:
            raise ValueError("workload tensor ABI must declare inputs and outputs")
        result: list[TensorABI] = []
        for mode in ("input", "output"):
            names = abi[mode + "s"]
            if not isinstance(names, list) or not names:
                raise ValueError(f"workload tensor ABI {mode}s must be non-empty")
            for name in names:
                name = _name(name, "workload tensor ABI name")
                tensor = _object(tensors.get(name), f"workload tensor {name}")
                dimensions = tensor.get("shape")
                if not isinstance(dimensions, list) or not dimensions:
                    raise ValueError(f"workload tensor {name} must have a shape")
                extents = tuple(
                    shape.get(dimension) if isinstance(dimension, str) else dimension
                    for dimension in dimensions
                )
                if any(
                    not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0
                    for extent in extents
                ):
                    raise ValueError(f"workload tensor {name} has an unresolved shape")
                dtype = tensor.get("dtype")
                if not isinstance(dtype, str) or dtype not in {"fp32", "bf16", "int32"}:
                    raise ValueError(f"workload tensor {name} dtype is unsupported")
                if tensor.get("layout") != "contiguous_row_major":
                    raise ValueError(f"workload tensor {name} must be contiguous row major")
                result.append(TensorABI(name, cast(tuple[int, ...], extents), dtype, mode))
        names = [tensor.name for tensor in result]
        if len(names) != len(set(names)) or set(names) != set(tensors):
            raise ValueError("workload tensor ABI must cover each tensor exactly once")
        return tuple(result)
