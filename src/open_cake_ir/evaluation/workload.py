"""Reusable Workload Contract independent of study and provider policy."""

from __future__ import annotations

import json
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
