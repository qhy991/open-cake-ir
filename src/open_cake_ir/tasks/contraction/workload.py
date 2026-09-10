"""Portable Workloads that contract an axis, so their cost is arithmetic not bandwidth.

Every other family in this Lab is memory-bound. Measured with the Compiler's own work
model at the shapes they freeze, the twenty tasks in `activation`, `rowwise`, `reductions`
and `optimizers` span 0.20 to 1.50 counted FLOPs per compulsory byte: each one reads its
inputs, does a few operations per element, and writes its outputs. No Schedule for any of
them can be arithmetic-bound, so none of them can ask a Schedule to trade arithmetic for
traffic, to keep an accumulator resident, or to tile a contraction.

These tasks contract an axis, so one loaded operand feeds many outputs and the FLOP count
grows as the product of three extents while the traffic grows as their sum. At the frozen
shapes here that is roughly 15 to 21 FLOPs per byte -- an order of magnitude above the
entire rest of the set, and the regime where a scheduling decision changes the answer.

The baselines are deliberately naive: each one keeps the whole contracted operand
register-resident. That is the plain reading of the definition and not a claim that it is
the cheapest arrangement -- it is in fact why the Metal route refuses larger shapes, since
its 1024 live FP32 values per lane is exactly the budget a tiled contraction would free.
Finding that trade is the search this family exists to pose.

All four are migrations of AKA v7 qualified parents. The Ralph engine owns search and
evaluation; these functions neither import candidates nor launch a device.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.tiles.workload import _checked_inputs, _round
from open_cake_ir.tasks.devices import (  # noqa: F401
    BACKENDS, admit_width, backend_for_target, device_name,
)

TASKS = {
    "gemm": ("contraction_gemm_fp32", "1"),
    "gemm_silu": ("contraction_gemm_silu_fp32", "1"),
    "pairwise_sqdist": ("contraction_pairwise_sqdist_fp32", "1"),
    "attention_decode": ("contraction_attention_decode_fp32", "1"),
}
# One (R, K, N) triple serves all four: R is the program extent, K the contracted axis,
# N the second operand's own extent. What differs is what each one contracts *with*.
INPUTS = {
    "gemm": (("a", ("R", "K")), ("b", ("K", "N")), ("bias", ("N",))),
    "gemm_silu": (("a", ("R", "K")), ("b", ("K", "N")), ("bias", ("N",))),
    # The second operand is [N, K] here, not [K, N]: a distance contracts each centroid
    # against the point over the feature axis they share.
    "pairwise_sqdist": (("x", ("R", "K")), ("c", ("N", "K"))),
    "attention_decode": (("q", ("R", "K")), ("k", ("N", "K")), ("v", ("N", "K"))),
}
OUTPUTS = {
    "gemm": ("out", ("R", "N")),
    "gemm_silu": ("out", ("R", "N")),
    "pairwise_sqdist": ("out", ("R", "N")),
    # Attention returns to the head dimension: it contracts K away and then N away.
    "attention_decode": ("out", ("R", "K")),
}
AKA_V7_RECORDS = "AKA/datasets/curated/cuda_kernel_parent_completions_v7/records.jsonl"
AKA_PARENTS = {
    # A row-broadcast bias over an FP32 contraction, which is exactly this task's shape.
    "gemm": "dd_matmul_bias_f32_bt_oc_c_block16_v1",
    "gemm_silu": "dd_matmul_bias_f32_bt_oc_c_block16_v1",
    # The parent computes a condensed upper triangle and takes a square root; this
    # Workload keeps the squared distance over two independent operand sets, so it is a
    # declared variant of that contraction rather than the same output.
    "pairwise_sqdist": "condensed_pairwise_l2_f32_i32_block256_v1",
    "attention_decode": "single_decode_attention_nhd_fp32_i32_hd128_v1",
}
# What each task's contraction folds. Stated per task because the four differ, and the
# difference is what makes them four tasks rather than one shape reused.
_REDUCTION = {
    "gemm": "one_contraction_over_the_shared_axis_of_two_loaded_operands",
    "gemm_silu": "one_contraction_whose_accumulator_a_transcendental_epilogue_consumes",
    "pairwise_sqdist": "one_contraction_over_squares_of_a_difference_not_over_products",
    "attention_decode": "two_chained_contractions_separated_by_a_data_dependent_normalization",
}
CASES = {
    "primary": ("uniform", 7501),
    "zeros": ("zeros", 7502),
    "near_zero": ("near_zero", 7503),
    "alternating": ("alternating", 7504),
    "mixed_magnitude": ("mixed_magnitude", 7505),
}
# Smaller than the elementwise families' bound because every term entering a contraction
# is a product of two of these, and the allowance below carries that square.
MAX_ABS = 2.0
_UNIT_ROUNDOFF = 2.0 ** -24


def _reduction_depth(extent: int) -> int:
    """Roundings one accumulated term passes through: a lane's run, then the lane fold."""
    return -(-extent // 32) + 5


def _allowance(task_name: str, depth: int, columns: int) -> dict[str, float]:
    """Derive this shape's absolute allowance from the contraction it performs.

    A contraction accumulates `depth` products of two bounded operands, and nothing
    downstream divides that error away, so the allowance grows with the contracted extent.
    Every factor is a declared bound of this contract and the result assumes every
    rounding aligns, which no real input achieves.
    """
    fold = _reduction_depth(depth)
    term = MAX_ABS * MAX_ABS
    error = fold * _UNIT_ROUNDOFF * depth * term
    if task_name == "pairwise_sqdist":
        # A squared difference is bounded by (2 * MAX_ABS)^2, and the subtraction that
        # forms it rounds once more before the square.
        error = fold * _UNIT_ROUNDOFF * depth * (2.0 * MAX_ABS) ** 2 * 2.0
    elif task_name == "gemm_silu":
        # The epilogue's gate is bounded by one and its derivative by about 1.1, so the
        # accumulator's error reaches the output roughly unchanged, plus the gate's own.
        error = error * 1.1 + 2e-6
    elif task_name == "attention_decode":
        # The scaled logits carry the contraction's error into an exponential, so what
        # reaches the weights is a *relative* error of twice that (two logits differ),
        # and the output is a convex combination of values bounded by MAX_ABS.
        logit_error = error / math.sqrt(depth)
        error = 2.0 * logit_error * MAX_ABS + fold * _UNIT_ROUNDOFF * columns * MAX_ABS
    return {"atol": float(f"{max(error, 2e-6):.3g}"), "rtol": 2e-5}


def workload_document(task_name: str, *, rows: int = 1024, depth: int = 128,
                      columns: int = 64, backend: str = "metal-m1-pro") -> dict:
    """Create a reusable frozen Workload document; callers persist it outside source."""
    if not isinstance(task_name, str) or task_name not in TASKS or backend not in BACKENDS:
        raise ValueError("unsupported contraction task/backend")
    device = BACKENDS[backend]
    if (any(type(value) is not int for value in (rows, depth, columns))
            or min(rows, depth, columns) <= 0
            or (rows * depth + depth * columns + rows * columns) * 4 > 2**31 - 1):
        raise ValueError("contraction shape must fit the FP32 buffer ABI")
    # Both tiled extents are arange spans on a Triton route, and the contracted one is
    # folded on every route, so each has to be a width the route can actually tile.
    admit_width(backend, depth)
    admit_width(backend, columns)
    operator, revision = TASKS[task_name]
    tensors = {name: {"shape": list(shape), "max_abs": MAX_ABS}
               for name, shape in INPUTS[task_name]}
    output_name, output_shape = OUTPUTS[task_name]
    tensors[output_name] = {"shape": list(output_shape)}
    for tensor in tensors.values():
        tensor.update(dtype="fp32", layout="contiguous_row_major", finite_only=True)
    definitions = {
        "gemm": "out[r,n] = sum_k(a[r,k] * b[k,n]) + bias[n]",
        "gemm_silu": ("t[r,n] = sum_k(a[r,k] * b[k,n]) + bias[n]; "
                      "out[r,n] = t[r,n] / (1 + exp(-t[r,n]))"),
        "pairwise_sqdist": "out[r,n] = sum_k((x[r,k] - c[n,k])^2)",
        "attention_decode": (
            "l[r,n] = sum_k(q[r,k] * k[n,k]) / sqrt(K); m[r] = max_n(l[r,n]); "
            "p[r,n] = exp(l[r,n] - m[r]) / sum_j(exp(l[r,j] - m[r])); "
            "out[r,k] = sum_n(p[r,n] * v[n,k])"),
    }
    arithmetic = {
        "intermediates": "fp32",
        "reduction": _REDUCTION[task_name],
        "reduction_order": "backend_defined_within_tolerance",
        # The whole point of the family, stated so a Schedule cannot claim otherwise.
        "cost_regime": "arithmetic_bound_the_contracted_axis_feeds_many_outputs_per_loaded_byte",
        "output": "fp32",
    }
    if task_name == "attention_decode":
        arithmetic["exp"] = "backend_approximation_within_tolerance"
        arithmetic["shift"] = "subtract_row_maximum_before_exponentiation"
        arithmetic["scale"] = "logits_divided_by_the_square_root_of_the_contracted_extent"
    elif task_name == "gemm_silu":
        arithmetic["exp"] = "backend_approximation_within_tolerance"
        arithmetic["epilogue"] = "applied_to_the_accumulator_before_it_leaves_the_kernel"
    elif task_name == "pairwise_sqdist":
        # Named because it is what forbids a fused multiply-accumulate instruction from
        # serving this contraction: the operand is a difference, not a loaded value.
        arithmetic["operand"] = "a_rounded_difference_formed_before_the_square"
    return {
        "schema_version": 1,
        "workload_id": (f"{operator.replace('_', '-')}-{backend}"
                        f"-r{rows}-k{depth}-n{columns}-v{revision}"),
        "revision": revision, "state": "frozen", "operator": operator,
        "provenance": [
            {"kind": "task_mathematical_specification",
             "path": "src/open_cake_ir/tasks/contraction/workload.py",
             "scope": f"rank2_FP32_{device['provenance_token']}_local_artifact_evaluation"},
            {"kind": "aka_qualified_parent_lineage", "path": AKA_V7_RECORDS,
             "derived_parent_id": AKA_PARENTS[task_name],
             "scope": ("semantic_origin_only_no_B200_correctness_performance_or_shape_claim"
                       "_transfers_to_this_Workload")},
        ],
        "cases": [{"case_id": name, "shape": {"R": rows, "K": depth, "N": columns},
                   "seed": seed, "mode": mode} for name, (mode, seed) in CASES.items()],
        "tensors": tensors,
        "semantics": {
            "definition": definitions[task_name], "target": device["target"],
            "candidate_abi": {"inputs": [name for name, _ in INPUTS[task_name]],
                              "outputs": [output_name]},
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "arithmetic": arithmetic,
            "materialization": {
                "generator": "python_random_Random_per_input_case_seed_plus_10000_times_ABI_index",
                "uniform": "round_fp32_uniform[-2,2]", "zeros": "alternating_signed_zero",
                "near_zero": "round_fp32_uniform[-1e-4,1e-4]",
                "alternating": "round_fp32_alternating_sign_times_1_plus_i_mod_17_over_16",
                "mixed_magnitude": "signed_powers_of_two_exponent[-12,1]",
            },
        },
        "oracle": {
            "kind": f"real_{task_name}_fsum_then_fp32",
            "callable": "open_cake_ir.tasks.contraction.workload.reference_outputs",
            "implementation": "independent_standard_library_math_no_compiler_or_generated_source",
            "output_rounding": "round_to_nearest_ties_to_even",
        },
        "validation": {
            "primary_case": "primary", "all_cases_required": True, "equal_nan": False,
            "comparison": "elementwise_atol_rtol", **_allowance(task_name, depth, columns),
            "qualification": "all_input_cases_at_fixed_shape_no_framework_or_cross_shape_claim",
            "tolerance_rationale": (
                "A contraction accumulates one product of two bounded operands per step of "
                "the contracted extent, and nothing downstream divides that error away, so "
                "the allowance is derived from that extent and from the emitted body's own "
                "fold depth. Every other factor is a declared bound of this contract. It is "
                "a worst case assuming every rounding aligns. Not device calibration."),
        },
    }


def validate_contraction_contract(document: Mapping[str, object]) -> None:
    """Admit only this exact typed task, backend, ABI, numerical domain and case protocol."""
    workload = WorkloadContract(document)
    task_name = next((name for name, (operator, revision) in TASKS.items()
                      if document.get("operator") == operator
                      and document.get("revision") == revision), None)
    semantics = document.get("semantics")
    backend = backend_for_target(semantics.get("target") if isinstance(semantics, Mapping) else None)
    if task_name is None or backend is None or workload.case_ids != tuple(CASES):
        raise ValueError("contraction operator/revision, backend or required input cases differ")
    shape = workload.case("primary")["shape"]
    if set(shape) != {"R", "K", "N"}:
        raise ValueError("contraction case requires R/K/N dimensions")
    expected = workload_document(task_name, rows=shape["R"], depth=shape["K"],
                                 columns=shape["N"], backend=backend)
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("contraction frozen semantic/ABI/input/validation contract differs")
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float]]:
    """Return deterministic FP32 inputs in the contract's explicit ABI order."""
    validate_contraction_contract(workload.document)
    case = workload.case(case_id)
    result = {}
    args = [arg for arg in workload.tensor_abi(case_id) if arg.mode == "input"]
    for position, arg in enumerate(args):
        rng = random.Random(case["seed"] + position * 10000)
        values = []
        for index in range(math.prod(arg.shape)):
            if case["mode"] == "zeros":
                value = -0.0 if index % 2 else 0.0
            elif case["mode"] == "near_zero":
                value = rng.uniform(-1e-4, 1e-4)
            elif case["mode"] == "alternating":
                value = (1 if index % 2 else -1) * (1 + (index % 17) / 16)
            elif case["mode"] == "mixed_magnitude":
                value = (1 if index % 2 else -1) * 2.0 ** rng.randint(-12, 1)
            else:
                value = rng.uniform(-MAX_ABS, MAX_ABS)
            values.append(_round(value, "fp32"))
        result[arg.name] = values
    return result


def reference_outputs(workload: WorkloadContract, case_id: str,
                      inputs: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Independent fsum oracle; each contraction is exact before one FP32 rounding."""
    validate_contraction_contract(workload.document)
    operator = workload.document["operator"]
    shape = workload.case(case_id)["shape"]
    rows, depth, columns = shape["R"], shape["K"], shape["N"]
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    checked = dict(zip((arg.name for arg in args), _checked_inputs(workload, args, inputs)))
    result: list[float] = []
    if operator in {"contraction_gemm_fp32", "contraction_gemm_silu_fp32"}:
        left, right, biases = checked["a"], checked["b"], checked["bias"]
        gated = operator == "contraction_gemm_silu_fp32"
        for row in range(rows):
            base = row * depth
            for column in range(columns):
                total = math.fsum(left[base + k] * right[k * columns + column]
                                  for k in range(depth)) + biases[column]
                if gated:
                    # exp(-total) overflows for valid negative contractions. Both
                    # branches implement the same SiLU without changing its contract.
                    if total >= 0:
                        total = total / (1.0 + math.exp(-total))
                    else:
                        decay = math.exp(total)
                        total = total * decay / (1.0 + decay)
                result.append(_round(total, "fp32"))
        return {"out": result}
    if operator == "contraction_pairwise_sqdist_fp32":
        points, centroids = checked["x"], checked["c"]
        for row in range(rows):
            base = row * depth
            for column in range(columns):
                offset = column * depth
                result.append(_round(math.fsum(
                    _round(points[base + k] - centroids[offset + k], "fp32") ** 2
                    for k in range(depth)), "fp32"))
        return {"out": result}
    queries, keys, values = checked["q"], checked["k"], checked["v"]
    scale = 1.0 / math.sqrt(depth)
    for row in range(rows):
        base = row * depth
        logits = [math.fsum(queries[base + k] * keys[column * depth + k]
                            for k in range(depth)) * scale for column in range(columns)]
        # The row maximum is exact in FP32, so the shift introduces no rounding of its own.
        peak = max(logits)
        weights = [math.exp(logit - peak) for logit in logits]
        total = math.fsum(weights)
        for feature in range(depth):
            result.append(_round(math.fsum(
                weight * values[column * depth + feature]
                for column, weight in enumerate(weights)) / total, "fp32"))
    return {"out": result}
