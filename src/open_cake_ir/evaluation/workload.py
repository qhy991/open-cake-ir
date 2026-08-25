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
            "shape": [576],
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
