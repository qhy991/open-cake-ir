from __future__ import annotations
from typing import Mapping,cast
from open_cake_ir.evaluation.workload import _name,_object

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

from open_cake_ir.evaluation.workload import WorkloadContract
