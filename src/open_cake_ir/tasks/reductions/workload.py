"""Portable Workloads whose program maps over features and reduces across the rows.

Every other family here maps one program to one row and folds along it. These map one
program to one *column* and fold along the other axis, and their outputs are rank-1 over
the feature extent rather than rank-2. That inverts which extent is parallel and which is
sequential, which is the whole point of the family: a Schedule that is good at folding a
row is not automatically good at folding a column, and nothing in the earlier families
could ask that question.

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
    "layernorm_gamma_beta_backward": ("layernorm_gamma_beta_backward_fp32", "1"),
    "bias_gradient_reduction": ("bias_gradient_reduction_fp32", "1"),
    "per_channel_moments": ("per_channel_moments_fp32", "1"),
    # The only maximum folded down the rows. Quantization scales are computed exactly
    # this way; the FP8 cast that consumes the scale is a dtype the backends cannot name,
    # so it is deliberately outside this contract.
    "channel_absmax_scale": ("channel_absmax_scale_fp32", "1"),
}
INPUTS = {
    "layernorm_gamma_beta_backward": (("dy", ("R", "C")), ("x", ("R", "C")),
                                      ("mean", ("R",)), ("rstd", ("R",))),
    "bias_gradient_reduction": (("dout", ("R", "C")), ("bias", ("C",))),
    "per_channel_moments": (("x", ("R", "C")),),
    "channel_absmax_scale": (("x", ("R", "C")),),
}
# The only family here that writes more than one buffer, and both of its outputs are
# per-feature vectors rather than a copy of the input shape.
OUTPUTS = {
    "layernorm_gamma_beta_backward": (("dgamma", ("C",)), ("dbeta", ("C",))),
    "bias_gradient_reduction": (("dbias", ("C",)),),
    "per_channel_moments": (("mean", ("C",)), ("mean_square", ("C",))),
    "channel_absmax_scale": (("amax", ("C",)), ("scale", ("C",))),
}
AKA_V7_RECORDS = "AKA/datasets/curated/cuda_kernel_parent_completions_v7/records.jsonl"
AKA_PARENTS = {
    "layernorm_gamma_beta_backward": "layernorm_gamma_beta_backward_contiguous_fp32_i32_block256_v1",
    "bias_gradient_reduction": "matmul_backward_bias_accumulate_contiguous_fp32_int32_block256_v1",
    "per_channel_moments": "nhwc_raw_moments_contiguous_f32_i32_block256_v1",
    "channel_absmax_scale": "grouped_absmax_e4m3_fp32_i32_contiguous_16x16_v1",
}
# The migrated absmax parent clamps its divisor with a maximum; this contract adds the
# regularizer instead, because a clamp is a compare/select the IR does not have.
EPSILON = 1e-12
# The E4M3 finite maximum the migrated parent scales against. Only the scale is computed
# here: the cast that consumes it needs a dtype no backend in this Lab can name.
FP8_E4M3_MAX = 448.0
CASES = {
    "primary": ("uniform", 7301),
    "zeros": ("zeros", 7302),
    "near_zero": ("near_zero", 7303),
    "alternating": ("alternating", 7304),
    "mixed_magnitude": ("mixed_magnitude", 7305),
}
MAX_ABS = 2.0
_UNIT_ROUNDOFF = 2.0 ** -24


def _reduction_depth(rows: int) -> int:
    """The fold is over the row extent here, so the depth grows with rows, not columns."""
    return -(-rows // 32) + 5


def _allowance(task_name: str, rows: int) -> dict[str, float]:
    """Derive this shape's absolute allowance from the accumulation the body performs.

    Nothing normalizes these sums: the output *is* the reduction, so its error is the
    reduction's error and grows with the row extent rather than being divided away.
    """
    depth = _reduction_depth(rows)
    if task_name == "channel_absmax_scale":
        # A maximum is selected rather than accumulated, so the fold contributes no
        # rounding of its own and only the reciprocal that forms the scale does.
        return {"atol": 2e-6, "rtol": 2e-5}
    # The layernorm parameter gradient folds a product of three bounded quantities; the
    # bias reduction and the raw moments fold one, and the moments then divide by the row
    # extent, which cancels the sum's growth with it.
    term = MAX_ABS ** 3 if task_name == "layernorm_gamma_beta_backward" else MAX_ABS ** 2
    error = depth * _UNIT_ROUNDOFF * term * (1 if task_name == "per_channel_moments" else rows)
    return {"atol": float(f"{max(error, 2e-6):.3g}"), "rtol": 2e-5}


def workload_document(task_name: str, *, rows: int = 128, columns: int = 1024,
                      backend: str = "metal-m1-pro") -> dict:
    """Create a reusable frozen Workload document; callers persist it outside source."""
    if not isinstance(task_name, str) or task_name not in TASKS or backend not in BACKENDS:
        raise ValueError("unsupported reductions task/backend")
    device = BACKENDS[backend]
    if (type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0
            or rows * columns * 4 > 2**31 - 1):
        raise ValueError("reductions shape must fit the FP32 buffer ABI")
    # The fold runs down the rows, so it is the row extent this route has to tile.
    admit_width(backend, rows)
    operator, revision = TASKS[task_name]
    tensors = {name: {"shape": list(shape), "max_abs": MAX_ABS}
               for name, shape in INPUTS[task_name]}
    for name, shape in OUTPUTS[task_name]:
        tensors[name] = {"shape": list(shape)}
    for tensor in tensors.values():
        tensor.update(dtype="fp32", layout="contiguous_row_major", finite_only=True)
    definitions = {
        "layernorm_gamma_beta_backward": (
            "dgamma[c] = sum_i(dy[i,c] * (x[i,c] - mean[i]) * rstd[i]); "
            "dbeta[c] = sum_i(dy[i,c])"),
        "bias_gradient_reduction": "dbias[c] = bias[c] + sum_i(dout[i,c])",
        "per_channel_moments": ("mean[c] = sum_i(x[i,c]) / R; "
                                "mean_square[c] = sum_i(x[i,c]^2) / R"),
        "channel_absmax_scale": (f"amax[c] = max_i(abs(x[i,c])) + {EPSILON!r}; "
                                 f"scale[c] = amax[c] / {FP8_E4M3_MAX!r}"),
    }
    arithmetic = {
        "intermediates": "fp32",
        "reduction": "one_reduction_per_output_down_the_row_extent_not_along_it",
        "reduction_order": "backend_defined_within_tolerance",
        "program_axis": "one_program_per_feature_the_row_extent_is_the_sequential_one",
        "output": "fp32",
    }
    if task_name == "layernorm_gamma_beta_backward":
        arithmetic["row_statistics"] = "supplied_per_row_mean_and_reciprocal_standard_deviation"
        arithmetic["saved_state"] = "declared_inputs_not_recomputed_forward_results"
    elif task_name == "per_channel_moments":
        arithmetic["moments"] = "raw_second_moment_not_a_centered_or_unbiased_variance"
    elif task_name == "channel_absmax_scale":
        arithmetic["absolute_value"] = "relu_of_the_operand_plus_relu_of_its_negation"
        arithmetic["regularizer"] = "added_to_the_column_maximum_not_a_clamp_against_it"
        arithmetic["quantization"] = "the_scale_only_no_reduced_precision_cast_is_in_this_contract"
    else:
        arithmetic["accumulation"] = "the_prior_bias_gradient_is_added_not_overwritten"
    return {
        "schema_version": 1,
        "workload_id": f"{operator.replace('_', '-')}-{backend}-r{rows}-c{columns}-v{revision}",
        "revision": revision, "state": "frozen", "operator": operator,
        "provenance": [
            {"kind": "task_mathematical_specification",
             "path": "src/open_cake_ir/tasks/reductions/workload.py",
             "scope": f"rank2_FP32_{device['provenance_token']}_local_artifact_evaluation"},
            {"kind": "aka_qualified_parent_lineage", "path": AKA_V7_RECORDS,
             "derived_parent_id": AKA_PARENTS[task_name],
             "scope": ("semantic_origin_only_no_B200_correctness_performance_or_shape_claim"
                       "_transfers_to_this_Workload")},
        ],
        "cases": [{"case_id": name, "shape": {"R": rows, "C": columns}, "seed": seed, "mode": mode}
                  for name, (mode, seed) in CASES.items()],
        "tensors": tensors,
        "semantics": {
            "definition": definitions[task_name], "target": device["target"],
            "candidate_abi": {"inputs": [name for name, _ in INPUTS[task_name]],
                              "outputs": [name for name, _ in OUTPUTS[task_name]]},
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
            "callable": "open_cake_ir.tasks.reductions.workload.reference_outputs",
            "implementation": "independent_standard_library_math_no_compiler_or_generated_source",
            "output_rounding": "round_to_nearest_ties_to_even",
        },
        "validation": {
            "primary_case": "primary", "all_cases_required": True, "equal_nan": False,
            "comparison": "elementwise_atol_rtol", **_allowance(task_name, rows),
            "qualification": "all_input_cases_at_fixed_shape_no_framework_or_cross_shape_claim",
            "tolerance_rationale": (
                "The output is the reduction, so nothing divides its error away and the "
                "allowance grows with the row extent rather than the feature extent. A "
                "lane accumulates ceil(rows/32) terms before a five-deep fold, and every "
                "other factor is a declared bound of this contract. It is a worst case "
                "assuming every rounding aligns. Not device calibration."),
        },
    }


def validate_reductions_contract(document: Mapping[str, object]) -> None:
    """Admit only this exact typed task, backend, ABI, numerical domain and case protocol."""
    workload = WorkloadContract(document)
    task_name = next((name for name, (operator, revision) in TASKS.items()
                      if document.get("operator") == operator
                      and document.get("revision") == revision), None)
    semantics = document.get("semantics")
    backend = backend_for_target(semantics.get("target") if isinstance(semantics, Mapping) else None)
    if task_name is None or backend is None or workload.case_ids != tuple(CASES):
        raise ValueError("reductions operator/revision, backend or required input cases differ")
    shape = workload.case("primary")["shape"]
    if set(shape) != {"R", "C"}:
        raise ValueError("reductions case requires R/C dimensions")
    expected = workload_document(task_name, rows=shape["R"], columns=shape["C"], backend=backend)
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("reductions frozen semantic/ABI/input/validation contract differs")
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float]]:
    """Return deterministic FP32 inputs in the contract's explicit ABI order."""
    validate_reductions_contract(workload.document)
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
    """Independent fsum oracle folding down the rows, one rounding per output element."""
    validate_reductions_contract(workload.document)
    operator = workload.document["operator"]
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    checked = dict(zip((arg.name for arg in args), _checked_inputs(workload, args, inputs)))
    rows, width = next(arg.shape for arg in args if len(arg.shape) == 2)
    if operator == "per_channel_moments_fp32":
        values = checked["x"]
        return {"mean": [_round(math.fsum(values[row * width + column]
                                          for row in range(rows)) / rows, "fp32")
                         for column in range(width)],
                "mean_square": [_round(math.fsum(values[row * width + column] ** 2
                                                 for row in range(rows)) / rows, "fp32")
                                for column in range(width)]}
    if operator == "channel_absmax_scale_fp32":
        values = checked["x"]
        peaks = [max(abs(values[row * width + column]) for row in range(rows)) + EPSILON
                 for column in range(width)]
        return {"amax": [_round(peak, "fp32") for peak in peaks],
                "scale": [_round(peak / FP8_E4M3_MAX, "fp32") for peak in peaks]}
    if operator == "bias_gradient_reduction_fp32":
        upstream, prior = checked["dout"], checked["bias"]
        return {"dbias": [_round(prior[column] + math.fsum(
            upstream[row * width + column] for row in range(rows)), "fp32")
            for column in range(width)]}
    upstream, values = checked["dy"], checked["x"]
    means, inverses = checked["mean"], checked["rstd"]
    dgamma, dbeta = [], []
    for column in range(width):
        dgamma.append(_round(math.fsum(
            upstream[row * width + column] * (values[row * width + column] - means[row])
            * inverses[row] for row in range(rows)), "fp32"))
        dbeta.append(_round(math.fsum(
            upstream[row * width + column] for row in range(rows)), "fp32"))
    return {"dgamma": dgamma, "dbeta": dbeta}
