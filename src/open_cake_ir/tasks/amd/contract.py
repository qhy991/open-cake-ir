"""Exact frozen AMD operator validation, owned by the task layer."""
from __future__ import annotations

from typing import Mapping, cast


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
