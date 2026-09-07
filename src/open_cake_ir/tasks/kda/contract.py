from __future__ import annotations
from typing import Mapping,cast
from open_cake_ir.evaluation.workload import _name,_object

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
