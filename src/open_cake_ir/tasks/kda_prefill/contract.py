"""Exact B300 successor boundary for the public recurrent-KDA prefill rows."""

from __future__ import annotations

import math
from typing import Mapping, cast

from open_cake_ir.evaluation.workload import _object


ROWS = (
    ("guard-h64-t65", 64, (65,), 427027),
    ("guard-h96-packed-33-65-17", 96, (33, 65, 17), 427028),
    ("h96_fixed8192", 96, (8192,), 10000),
    ("h96_mixed", 96, (1300, 547, 2048, 963, 271, 3063), 10001),
    ("h96_uniform", 96, (1024,) * 8, 10002),
    ("h64_fixed8192", 64, (8192,), 10003),
    ("h64_mixed", 64, (1300, 547, 2048, 963, 271, 3063), 10004),
    ("h64_uniform", 64, (1024,) * 8, 10005),
)
ABI = (
    "q", "k", "v", "g", "beta", "A_log", "dt_bias", "cu_seqlens",
    "seq_order", "initial_state", "out", "final_state",
)
TENSORS = {
    **{name: (["batch", "total_tokens", "heads", "head_dim"], "bf16")
       for name in ("q", "k", "v", "g", "out")},
    "beta": (["batch", "total_tokens", "heads"], "bf16"),
    "A_log": (["heads"], "fp32"),
    "dt_bias": (["heads", "head_dim"], "fp32"),
    "cu_seqlens": (["num_sequences+1"], "int64"),
    "seq_order": (["num_sequences"], "int32"),
    "initial_state": (["num_sequences", "heads", "head_dim", "head_dim"], "bf16"),
    "final_state": (["num_sequences", "heads", "head_dim", "head_dim"], "bf16"),
}


def validate_contract(document: Mapping[str, object]) -> None:
    """Refuse a silent change to geometry, state effects or comparison quality."""

    if (document.get("workload_id"), document.get("revision"), document.get("state")) != (
        "cake-kda-prefill-b300-v1", "1", "frozen"
    ) or document.get("operator") != "cake_kda_prefill_b300":
        raise ValueError("KDA prefill B300 workload identity differs")
    provenance = document.get("provenance")
    if (not isinstance(provenance, list) or len(provenance) != 2
            or _object(provenance[0], "KDA prefill reference").get("revision") !=
            "e835e0f5565b5b9786c987e00c6b39a26bfecca5"
            or _object(provenance[1], "KDA prefill adaptation").get("kind") !=
            "separate_target_adaptation"):
        raise ValueError("KDA prefill reference provenance differs")
    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != len(ROWS):
        raise ValueError("KDA prefill case count differs")
    for raw_case, (case_id, heads, lengths, seed) in zip(raw_cases, ROWS, strict=True):
        case = _object(raw_case, "KDA prefill case")
        shape = _object(case.get("shape"), "KDA prefill case shape")
        if (case.get("case_id") != case_id or case.get("seed") != seed
                or case.get("mode") != ("fixed" if len(lengths) == 1 else "packed")
                or shape != {
                    "batch": 1, "total_tokens": sum(lengths), "heads": heads,
                    "head_dim": 128, "num_sequences": len(lengths),
                }):
            raise ValueError(f"KDA prefill geometry differs for {case_id}")

    tensors = _object(document.get("tensors"), "KDA prefill tensors")
    if set(tensors) != set(TENSORS):
        raise ValueError("KDA prefill public tensor set differs")
    for name, (shape, dtype) in TENSORS.items():
        tensor = _object(tensors[name], f"KDA prefill tensor {name}")
        expected: dict[str, object] = {
            "shape": shape, "dtype": dtype, "layout": "contiguous_row_major"
        }
        if name == "initial_state":
            expected["side_effect"] = "updated_in_place"
        elif name == "final_state":
            expected["alias_of"] = "initial_state"
        elif name == "out":
            expected["storage"] = "fresh_contiguous_nonaliasing"
        if tensor != expected:
            raise ValueError(f"KDA prefill tensor {name} contract differs")

    semantics = _object(document.get("semantics"), "KDA prefill semantics")
    sequences = _object(semantics.get("sequence_lengths"), "KDA prefill lengths")
    if sequences != {case_id: list(lengths) for case_id, _, lengths, _ in ROWS}:
        raise ValueError("KDA prefill sequence lengths differ")
    constants = _object(semantics.get("constants"), "KDA prefill constants")
    if (semantics.get("target") != "sm_103a"
            or semantics.get("reference_target") != "sm_100a"
            or semantics.get("head_dim") != 128
            or semantics.get("candidate_abi") != list(ABI)
            or semantics.get("definition") !=
            "BF16 Q/K L2 normalization; FP32 gate=lower_bound*sigmoid(exp(A_log)*(g+dt_bias)), beta=sigmoid(beta_logit), V-first recurrent state and BF16-rounded output/state after each token"
            or semantics.get("materialization") !=
            "CPU-seeded BF16 q/k/v/g/beta and initial state; FP32 A_log/dt_bias; six benchmark rows use the frozen public distributions with separately generated B300 bytes; guardrails have their own fixed seeds"
            or semantics.get("variant_rule") !=
            "M64 exactly for fixed one-sequence H64; M128 otherwise"
            or semantics.get("state_effects") != {
                "initial_state_final_state": "exact_full_range_alias_required",
                "partial_overlap": "refused", "other_inputs": "unchanged",
                "out": "fully_overwritten_nonaliasing",
            }
            or not isinstance(constants.get("scale"), (int, float))
            or abs(float(constants["scale"]) * math.sqrt(128) - 1.0) > 1e-14
            or constants.get("lower_bound") != -5.0):
        raise ValueError("KDA prefill target, ABI or state semantics differ")
    if (semantics.get("packed_metadata") !=
            "int64 cumulative offsets partition every token; int32 seq_order is a permutation of sequence ids"):
        raise ValueError("KDA prefill packed metadata contract differs")

    oracle = _object(document.get("oracle"), "KDA prefill oracle")
    if (oracle.get("kind") != "independent_cpu_bf16_token_recurrence"
            or oracle.get("callable") !=
            "open_cake_ir.tasks.kda_prefill.oracle.reference_prefill"
            or oracle.get("reference_access") != "known_kernel_reproduction"
            or oracle.get("external_peer") !=
            "B300 exact-target adaptation of pinned CAKE export; qualification required before performance use"):
        raise ValueError("KDA prefill independent oracle differs")
    validation = _object(document.get("validation"), "KDA prefill validation")
    tiers = _object(validation.get("case_tiers"), "KDA prefill case tiers")
    if (validation.get("atol") != 0.01 or validation.get("rtol") != 0.01
            or validation.get("comparison_scope") != "all_output_and_final_state_elements"
            or validation.get("all_cases_required") is not True
            or tiers != {
                "guardrails": [row[0] for row in ROWS[:2]],
                "benchmark": [row[0] for row in ROWS[2:]],
            }
            or validation.get("state_validation") !=
            "complete_final_state_and_in_place_alias"
            or validation.get("input_preservation") !=
            "q_k_v_g_beta_A_log_dt_bias_offsets_order_unchanged"
            or validation.get("claim_boundary") !=
            "B300 adapted reproduction; not the original B200 paper measurement or model serving"):
        raise ValueError("KDA prefill correctness gates differ")
    measurement = _object(validation.get("measurement"), "KDA prefill measurement")
    if (measurement.get("target") != "sm_103a"
            or measurement.get("timer") != "CUPTI"
            or measurement.get("cache") != "cold_L2_flushed_per_sample"
            or measurement.get("cuda_graph") is not False
            or measurement.get("state_reset") !=
            "fresh_initial_state_per_timed_sample_or_equivalent_rotating_pool"
            or measurement.get("scope") !=
            "kernel_duration_sum_and_full_public_callable_reported_separately"
            or validation.get("performance_criterion") !=
            "unselected_until_exact_B300_external_baseline_and_budget_are_bound"):
        raise ValueError("KDA prefill measurement boundary differs")


def case_lengths(document: Mapping[str, object], case_id: str) -> tuple[int, ...]:
    """Return the contract-owned sequence lengths after full validation."""

    validate_contract(document)
    lengths = _object(_object(document["semantics"], "semantics")["sequence_lengths"],
                      "sequence_lengths")[case_id]
    return tuple(cast(list[int], lengths))
