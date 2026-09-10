"""Portable backward Workloads whose epilogue amplifies a signed row reduction.

This is the family's defining structure and what separates it from every other one here.
The activation family has no reduction at all. RMSNorm and LayerNorm reduce a *square*, so
every term shares a sign, the sum never cancels, and the epilogue then divides by that
same sum -- which normalizes its error away. These tasks reduce signed terms, so the sum
can cancel to far less than the magnitude of its own terms, and the epilogue multiplies
the result by a per-element tensor instead of dividing by it, so the reduction's error is
amplified rather than cancelled.

The two tasks reach that structure differently, and the contract says which: the softmax
backward folds the upstream gradient itself and forms its product in the epilogue, while
the LayerNorm input gradient folds one plain mean and one mean of a product.

That is why this family derives its allowance from the frozen row width instead of naming
a constant. The emitted Metal body accumulates `columns / 32` values per lane and then
folds 32 lanes, so the accumulation depth grows with the shape; the oracle uses `math.fsum`
and is exact. A single predeclared atol would be either wrong at 1024 columns or vacuous
at 8. See `_allowance`.

Both tasks are migrations of AKA v7 qualified parents. The Ralph engine owns search and
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
    "softmax_backward": ("softmax_backward_fp32", "1"),
    "layernorm_backward_input": ("layernorm_backward_input_fp32", "2"),
    "rmsnorm_input_gradient": ("rmsnorm_input_gradient_fp32", "2"),
    # The family's only maximum reduction, and its only per-row scalar output. Both
    # exist so the set asks a Schedule something the sum-shaped tasks never do.
    "absmax_rescale": ("absmax_rescale_fp32", "1"),
    "cosine_similarity": ("cosine_similarity_fp32", "1"),
}
# Declared input tensors in ABI order with their shapes. Three distinct ranks appear in
# one contract here, which no earlier family needed: the row statistics are per-row
# scalars and the affine scale is per-feature.
INPUTS = {
    "softmax_backward": (("p", ("R", "C")), ("dp", ("R", "C"))),
    "layernorm_backward_input": (("dy", ("R", "C")), ("x", ("R", "C")),
                                 ("mean", ("R",)), ("rstd", ("R",)), ("gamma", ("C",))),
    "rmsnorm_input_gradient": (("dy", ("R", "C")), ("x", ("R", "C")),
                               ("gamma", ("C",)), ("rrms", ("R",))),
    "absmax_rescale": (("x", ("R", "C")),),
    "cosine_similarity": (("a", ("R", "C")), ("b", ("R", "C"))),
}
# Every earlier task writes the input shape back. Cosine similarity collapses each row to
# one number, which is a different store schedule and a different comparison surface.
OUTPUTS = {
    "softmax_backward": ("out", ("R", "C")),
    "layernorm_backward_input": ("out", ("R", "C")),
    "rmsnorm_input_gradient": ("out", ("R", "C")),
    "absmax_rescale": ("out", ("R", "C")),
    "cosine_similarity": ("out", ("R",)),
}
# What each task's row reduction actually folds. Stated per task because the two differ,
# and the difference is the reason their allowances are derived separately.
_REDUCTION = {
    "softmax_backward": "one_row_sum_of_the_upstream_gradient_multiplied_in_the_epilogue",
    "layernorm_backward_input": "two_row_means_the_first_over_a_product_of_two_tensors",
    "rmsnorm_input_gradient": "one_row_sum_over_a_product_of_three_tensors",
    "absmax_rescale": "one_row_maximum_of_the_absolute_value_not_a_sum",
    "cosine_similarity": "three_row_sums_collapsed_into_one_value_per_row",
}
AKA_REVIEW = "AKA/datasets/curated/cuda_kernel_parent_completions_v7/records.jsonl"
AKA_PARENTS = {
    "softmax_backward": "softmax_backward_contiguous_fp32_int32_v1",
    "layernorm_backward_input": "layer_norm_backward_input_contiguous_fp32_i32_block256_v1",
    "rmsnorm_input_gradient": "row_rmsnorm_dx_contiguous_f32_i32_b128_v1",
}
# The v7 parent completions carry semantics but no Cake Schedule.
AKA_V7_RECORDS = "AKA/datasets/curated/cuda_kernel_parent_completions_v7/records.jsonl"
AKA_V7_PARENTS = {
    "absmax_rescale": "grouped_absmax_e4m3_fp32_i32_contiguous_16x16_v1",
    "cosine_similarity": "rowwise_cosine_similarity_contiguous_fp32_i32_block256_v1",
}
# Both migrated parents clamp their divisor with a maximum. This contract adds the
# regularizer instead, because an additive epsilon is expressible today and a clamp is a
# compare/select the IR does not have. It is a declared variant, not the same function.
EPSILON = 1e-12
CASES = {
    "primary": ("uniform", 7201),
    "zeros": ("zeros", 7202),
    "near_zero": ("near_zero", 7203),
    "alternating": ("alternating", 7204),
    "mixed_magnitude": ("mixed_magnitude", 7205),
}
# Smaller than the activation family's bound because every quantity here is multiplied by
# two or three others before it reaches the reduction, and the allowance below grows with
# the product of those bounds.
MAX_ABS = 2.0
# Inputs whose sign is fixed by whatever produced them. A reciprocal standard deviation and
# a reciprocal root-mean-square are both strictly positive, so admitting a negative one is
# not a wider test but a different function: it flips the sign of the normalization the
# epilogue applies. Revision 2 narrows both. The same defect in the activation family's
# softplus gradient let its output reach 1.4e8 against an operator range of 16.
NONNEGATIVE = {
    "softmax_backward": (),
    "layernorm_backward_input": ("rstd",),
    "rmsnorm_input_gradient": ("rrms",),
    "absmax_rescale": (),
    "cosine_similarity": (),
}
# One FP32 unit in the last place at the top of the significand.
_UNIT_ROUNDOFF = 2.0 ** -24
# The emitted Metal body gives each of 32 lanes `columns / 32` sequential additions and
# then folds the lanes in a tree, so this is how many roundings a single accumulated term
# can pass through. Triton's tree is shallower; using the deeper of the two routes keeps
# one allowance valid for both.
def _reduction_depth(columns: int) -> int:
    return -(-columns // 32) + 5


def _allowance(task_name: str, columns: int) -> dict[str, float]:
    """Derive this shape's absolute allowance from the accumulation the body performs.

    Not device calibration: every factor is a declared bound of this contract, and the
    result is a worst case that assumes every rounding aligns, which no real input does.
    """
    depth = _reduction_depth(columns)
    if task_name == "softmax_backward":
        # S is a plain sum of `columns` terms bounded by MAX_ABS, so its error is bounded
        # by depth * u * columns * MAX_ABS, and the epilogue multiplies it by p.
        error = depth * _UNIT_ROUNDOFF * columns * MAX_ABS * MAX_ABS
    elif task_name == "absmax_rescale":
        # A maximum is selected, not accumulated, so no rounding survives the reduction
        # itself. Only the reciprocal and the final product round, and the divisor is at
        # least the epsilon, so the quotient is bounded by MAX_ABS / EPSILON.
        return {"atol": 2e-6, "rtol": 2e-5}
    elif task_name == "cosine_similarity":
        # Three sums of `columns` bounded terms feed one reciprocal square root whose
        # argument is at least EPSILON squared, so the divisor's relative error is what
        # reaches an output that is itself bounded by one.
        error = depth * _UNIT_ROUNDOFF * columns * MAX_ABS * MAX_ABS
        return {"atol": float(f"{max(3.0 * error, 2e-6):.3g}"), "rtol": 2e-5}
    elif task_name == "rmsnorm_input_gradient":
        # S folds a product of three bounded tensors, and the epilogue scales it by
        # rrms^3 / columns and then by x, so the sum's growth with `columns` cancels.
        term = MAX_ABS ** 3
        error = depth * _UNIT_ROUNDOFF * term * (MAX_ABS ** 3) * MAX_ABS
    else:
        # g = dy * gamma is bounded by MAX_ABS^2 and xhat = (x - mean) * rstd by
        # 2 * MAX_ABS^2. Both reductions are means, so the sum's growth with `columns`
        # cancels and only the depth survives; the epilogue then scales the larger of the
        # two errors by xhat and by rstd.
        gmax = MAX_ABS * MAX_ABS
        xhat_max = 2.0 * MAX_ABS * MAX_ABS
        error = MAX_ABS * (xhat_max * depth * _UNIT_ROUNDOFF * gmax * xhat_max
                           + depth * _UNIT_ROUNDOFF * gmax)
    # The floor covers the epilogue's own FP32 roundings where the reduction is tiny.
    return {"atol": float(f"{max(error, 2e-6):.3g}"), "rtol": 2e-5}


def workload_document(task_name: str, *, rows: int = 128, columns: int = 1024,
                      backend: str = "metal-m1-pro") -> dict:
    """Create a reusable frozen Workload document; callers persist it outside source."""
    if not isinstance(task_name, str) or task_name not in TASKS or backend not in BACKENDS:
        raise ValueError("unsupported rowwise task/backend")
    device = BACKENDS[backend]
    if (type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0
            or rows * columns * 4 > 2**31 - 1):
        raise ValueError("rowwise shape must fit the FP32 buffer ABI")
    admit_width(backend, columns)
    operator, revision = TASKS[task_name]
    tensors = {name: {"shape": list(shape), "max_abs": MAX_ABS}
               for name, shape in INPUTS[task_name]}
    for name in NONNEGATIVE[task_name]:
        tensors[name]["nonnegative"] = True
    output_name, output_shape = OUTPUTS[task_name]
    tensors[output_name] = {"shape": list(output_shape)}
    for tensor in tensors.values():
        tensor.update(dtype="fp32", layout="contiguous_row_major", finite_only=True)
    definitions = {
        "softmax_backward": ("S[r] = sum_j(dp[r,j]); "
                             "out[r,c] = dp[r,c] - p[r,c] * S[r]"),
        "layernorm_backward_input": (
            "xhat[r,c] = (x[r,c] - mean[r]) * rstd[r]; g[r,c] = dy[r,c] * gamma[c]; "
            "S1[r] = mean_j(g[r,j] * xhat[r,j]); S2[r] = mean_j(g[r,j]); "
            "out[r,c] = rstd[r] * (g[r,c] - S2[r] - xhat[r,c] * S1[r])"),
        "rmsnorm_input_gradient": (
            "S[r] = sum_j(dy[r,j] * x[r,j] * gamma[j]); "
            "c2[r] = -(rrms[r]^3 / C) * S[r]; "
            "out[r,c] = rrms[r] * dy[r,c] * gamma[c] + c2[r] * x[r,c]"),
        "absmax_rescale": (f"a[r] = max_j(abs(x[r,j])) + {EPSILON!r}; "
                           "out[r,c] = x[r,c] / a[r]"),
        "cosine_similarity": (
            "aa[r] = sum_j(a[r,j]^2); bb[r] = sum_j(b[r,j]^2); ab[r] = sum_j(a[r,j]*b[r,j]); "
            f"out[r] = ab[r] / sqrt((aa[r] + {EPSILON!r}) * (bb[r] + {EPSILON!r}))"),
    }
    arithmetic = {
        "intermediates": "fp32",
        "reduction": _REDUCTION[task_name],
        "reduction_order": "backend_defined_within_tolerance",
        "output": "fp32",
        # Both tasks read quantities a forward pass would have produced. This contract
        # freezes the formula over its declared input box and asserts nothing about
        # whether those inputs are a consistent forward state.
        "saved_state": "declared_inputs_not_recomputed_forward_results",
    }
    if task_name == "layernorm_backward_input":
        arithmetic["row_statistics"] = "supplied_per_row_mean_and_nonnegative_reciprocal_standard_deviation"
        arithmetic["normalization"] = "both_reductions_are_means_over_the_feature_extent"
    elif task_name == "rmsnorm_input_gradient":
        arithmetic["row_statistics"] = "supplied_per_row_nonnegative_reciprocal_root_mean_square"
        arithmetic["scratch"] = "c2_is_required_scratch_not_a_public_semantic_output"
    elif task_name == "absmax_rescale":
        arithmetic["absolute_value"] = "relu_of_the_operand_plus_relu_of_its_negation"
        arithmetic["regularizer"] = "added_to_the_row_maximum_not_a_clamp_against_it"
    elif task_name == "cosine_similarity":
        arithmetic["reciprocal_square_root"] = "backend_approximation_within_tolerance"
        arithmetic["regularizer"] = "added_to_each_squared_norm_not_a_clamp_against_it"
    else:
        arithmetic["upstream"] = "dp_is_the_visible_grad_term_no_unstated_upstream_interpretation"
    return {
        "schema_version": 1,
        "workload_id": f"{operator.replace('_', '-')}-{backend}-r{rows}-c{columns}-v{revision}",
        "revision": revision, "state": "frozen", "operator": operator,
        "provenance": [
            {"kind": "task_mathematical_specification",
             "path": "src/open_cake_ir/tasks/rowwise/workload.py",
             "scope": f"rank2_FP32_{device['provenance_token']}_local_artifact_evaluation"},
            {"kind": "aka_qualified_parent_lineage",
             "path": AKA_REVIEW if task_name in AKA_PARENTS else AKA_V7_RECORDS,
             "derived_parent_id": AKA_PARENTS.get(task_name) or AKA_V7_PARENTS[task_name],
             "scope": ("semantic_origin_only_no_B200_correctness_performance_or_shape_claim"
                       "_transfers_to_this_Workload")},
        ],
        "cases": [{"case_id": name, "shape": {"R": rows, "C": columns}, "seed": seed, "mode": mode}
                  for name, (mode, seed) in CASES.items()],
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
            "callable": "open_cake_ir.tasks.rowwise.workload.reference_outputs",
            "implementation": "independent_standard_library_math_no_compiler_or_generated_source",
            "output_rounding": "round_to_nearest_ties_to_even",
        },
        "validation": {
            "primary_case": "primary", "all_cases_required": True, "equal_nan": False,
            "comparison": "elementwise_atol_rtol", **_allowance(task_name, columns),
            "qualification": "all_input_cases_at_fixed_shape_no_framework_or_cross_shape_claim",
            "tolerance_rationale": (
                "The reduction folds signed terms, so it can cancel to far less than the "
                "magnitude of those terms, and the epilogue multiplies the result by a "
                "per-element tensor instead of dividing by it, which amplifies the "
                "reduction's error rather than normalizing it away. The absolute "
                "allowance is therefore derived from "
                "this frozen row width: a lane accumulates ceil(columns/32) terms before a "
                "five-deep lane fold, and every other factor is a declared bound of this "
                "contract. It is a worst case assuming every rounding aligns. Not device "
                "calibration."),
        },
    }


def validate_rowwise_contract(document: Mapping[str, object]) -> None:
    """Admit only this exact typed task, backend, ABI, numerical domain and case protocol."""
    workload = WorkloadContract(document)
    task_name = next((name for name, (operator, revision) in TASKS.items()
                      if document.get("operator") == operator
                      and document.get("revision") == revision), None)
    semantics = document.get("semantics")
    backend = backend_for_target(semantics.get("target") if isinstance(semantics, Mapping) else None)
    if task_name is None or backend is None or workload.case_ids != tuple(CASES):
        raise ValueError("rowwise operator/revision, backend or required input cases differ")
    shape = workload.case("primary")["shape"]
    if set(shape) != {"R", "C"}:
        raise ValueError("rowwise case requires R/C dimensions")
    expected = workload_document(task_name, rows=shape["R"], columns=shape["C"], backend=backend)
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("rowwise frozen semantic/ABI/input/validation contract differs")
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def _task_of(document: Mapping[str, object]) -> str:
    operator = document["operator"]
    return next(name for name, (registered, _) in TASKS.items() if registered == operator)


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float]]:
    """Return deterministic FP32 inputs in the contract's explicit ABI order."""
    validate_rowwise_contract(workload.document)
    task_name = _task_of(workload.document)
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
            if arg.name in NONNEGATIVE[task_name]:
                value = abs(value)
            values.append(_round(value, "fp32"))
        result[arg.name] = values
    return result


def reference_outputs(workload: WorkloadContract, case_id: str,
                      inputs: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Independent fsum oracle; the reductions are exact before one FP32 rounding."""
    validate_rowwise_contract(workload.document)
    operator = workload.document["operator"]
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    checked = dict(zip((arg.name for arg in args), _checked_inputs(workload, args, inputs)))
    for name in NONNEGATIVE[_task_of(workload.document)]:
        if any(value < 0.0 for value in checked[name]):
            raise ValueError(f"oracle input {name} is declared non-negative")
    width = next(arg.shape[-1] for arg in args if len(arg.shape) == 2)
    result: list[float] = []
    if operator == "softmax_backward_fp32":
        probabilities, gradients = checked["p"], checked["dp"]
        for start in range(0, len(gradients), width):
            total = math.fsum(gradients[start:start + width])
            result.extend(_round(gradient - probability * total, "fp32")
                          for probability, gradient
                          in zip(probabilities[start:start + width],
                                 gradients[start:start + width]))
        return {"out": result}
    if operator == "absmax_rescale_fp32":
        values = checked["x"]
        for start in range(0, len(values), width):
            row = values[start:start + width]
            inverse = 1.0 / (max(abs(value) for value in row) + EPSILON)
            result.extend(_round(value * inverse, "fp32") for value in row)
        return {"out": result}
    if operator == "cosine_similarity_fp32":
        left, right = checked["a"], checked["b"]
        for start in range(0, len(left), width):
            first, second = left[start:start + width], right[start:start + width]
            aa = math.fsum(value * value for value in first) + EPSILON
            bb = math.fsum(value * value for value in second) + EPSILON
            ab = math.fsum(x * y for x, y in zip(first, second))
            result.append(_round(ab / math.sqrt(aa * bb), "fp32"))
        return {"out": result}
    if operator == "rmsnorm_input_gradient_fp32":
        upstream, values = checked["dy"], checked["x"]
        scales, roots = checked["gamma"], checked["rrms"]
        for row in range(len(values) // width):
            start = row * width
            total = math.fsum(g * v * s for g, v, s
                              in zip(upstream[start:start + width],
                                     values[start:start + width], scales))
            second = -(roots[row] ** 3 / width) * total
            result.extend(
                _round(roots[row] * g * s + second * v, "fp32")
                for g, v, s in zip(upstream[start:start + width],
                                   values[start:start + width], scales))
        return {"out": result}
    upstream, values = checked["dy"], checked["x"]
    means, inverses, scales = checked["mean"], checked["rstd"], checked["gamma"]
    for row in range(len(values) // width):
        start = row * width
        centered = [(value - means[row]) * inverses[row]
                    for value in values[start:start + width]]
        weighted = [gradient * scale
                    for gradient, scale in zip(upstream[start:start + width], scales)]
        first = math.fsum(g * xhat for g, xhat in zip(weighted, centered)) / width
        second = math.fsum(weighted) / width
        result.extend(_round(inverses[row] * (g - second - xhat * first), "fp32")
                      for g, xhat in zip(weighted, centered))
    return {"out": result}
