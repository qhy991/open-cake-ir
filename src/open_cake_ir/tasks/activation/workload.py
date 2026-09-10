"""Portable activation Workloads, deterministic inputs and independent math.

Every task here is element-local: the output at one coordinate reads only the inputs at
that coordinate, so no Schedule needs a cross-lane exchange or a loop-carried reduction.
That is the point of the family. RMSNorm is bandwidth-bound, GEMM is bound by the
cross-lane exchange and softmax by a transcendental behind a row reduction; these isolate
the transcendental with nothing else in the way.

Most tasks here are migrations of AKA qualified parents that the v6 expressibility review
recorded as `current_ir_expressibility: expressible`. That review also recorded the
composition each one needs, which is why none of them asks for a new primitive: absolute
value is `relu(x) + relu(-x)`, and SELU's two arms come from `relu(x)` and `x - relu(x)`
rather than from a conditional. `AKA_PARENTS` carries the derived parent id into each
frozen document's provenance so the lineage stays inspectable.

The mathematics is device-independent. A backend binds one Target, one lowering route and
that route's own instruction contracts, and `open_cake_ir.tasks.devices` owns that table,
so the same task freezes for an Apple device or an NVIDIA one without being rewritten.

The Ralph engine owns search and evaluation; these functions neither import candidates
nor launch a device.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.tiles.workload import _checked_inputs, _round
# The portable registry, not the Apple-only one: these tasks are device-independent
# mathematics and each backend names its own Target, route and instruction contracts.
from open_cake_ir.tasks.devices import (  # noqa: F401
    BACKENDS, admit_width, backend_for_target, device_name,
)

TASKS = {
    "silu": ("silu_fp32", "1"),
    "swiglu": ("swiglu_fp32", "1"),
    # GELU's tanh formulation, which is what serving stacks ship; the erf form is a
    # different function once both are evaluated in FP32, so it is not this task.
    "gelu_tanh": ("gelu_tanh_fp32", "1"),
    "softsign": ("softsign_fp32", "1"),
    "selu": ("selu_fp32", "1"),
    # The gradient tasks read a saved forward output beside the incoming gradient.
    # They are the family's second ABI shape and its only arithmetic that is not a gate.
    # Revision 2 narrows y to the range softplus can actually produce; see NONNEGATIVE.
    "softplus_gradient": ("softplus_gradient_fp32", "2"),
    "gelu_tanh_backward": ("gelu_tanh_backward_fp32", "1"),
    # ELU is SELU with a unit outer scale, and PReLU replaces SELU's exponential arm with
    # a per-feature slope. Both reach their piecewise definition through the same ReLU
    # arms, so neither asks for the conditional the IR does not have.
    "prelu": ("prelu_fp32", "1"),
}
# The declared input tensors in ABI order with their shapes. The first one also carries
# the program axis. PReLU's slope is the family's only per-feature operand.
INPUTS = {
    "silu": (("x", ("R", "C")),),
    "swiglu": (("x", ("R", "C")), ("up", ("R", "C"))),
    "gelu_tanh": (("x", ("R", "C")),),
    "softsign": (("x", ("R", "C")),),
    "selu": (("x", ("R", "C")),),
    "prelu": (("x", ("R", "C")), ("slope", ("C",))),
    "softplus_gradient": (("y", ("R", "C")), ("dy", ("R", "C"))),
    "gelu_tanh_backward": (("x", ("R", "C")), ("dy", ("R", "C"))),
}
# The AKA qualified parent each task was migrated from. Every one of these is a
# `dynamic_valid` row of docs/data/aka-qualified-ir-v6-review-20260904, so the semantics
# were reviewed and B200-checked before this Workload existed. This Workload is a fresh
# frozen instance, not a replay: shape, input distributions and tolerance are declared
# here, and no recorded B200 result transfers to it on any device.
AKA_REVIEW = "docs/data/aka-qualified-ir-v6-review-20260904/lab-terminal-results.jsonl"
AKA_PARENTS = {
    "softsign": "softsign_contiguous_fp32_int32_block256_v1",
    "selu": "selu_contiguous_fp32_int32_block256_v1",
    "softplus_gradient": "softplus_gradient_fp32_contiguous_i32_gridstride_v1",
}
# The v7 parent completions carry semantics but no Cake Schedule, so these four are
# recorded against that dataset rather than against the v6 Lab-terminal export.
AKA_V7_RECORDS = "AKA/datasets/curated/cuda_kernel_parent_completions_v7/records.jsonl"
AKA_V7_PARENTS = {
    "gelu_tanh_backward": "gelu_tanh_backward_f32_vec4_direct_v1",
    "prelu": "prelu_nchw_per_channel_contiguous_fp32_i32_block256_v1",
}
# Inputs whose sign is fixed by whatever produced them. A saved forward output cannot be
# an arbitrary signed number, and admitting one is not a wider test -- it is a different
# function. softplus(x) = log(1 + exp(x)) is strictly positive, so `y` here is too; at the
# y = -16 revision 1 admitted, 1 - exp(-y) reaches -8.9e6 and the output reaches 1.4e8,
# seven orders of magnitude outside the operator's real range, where the relative tolerance
# then admits an absolute error of 2844 and the case tests nothing.
NONNEGATIVE = {
    "silu": (), "swiglu": (), "gelu_tanh": (), "softsign": (), "selu": (), "prelu": (),
    "gelu_tanh_backward": (),
    "softplus_gradient": ("y",),
}
CASES = {
    "primary": ("uniform", 7101),
    "zeros": ("zeros", 7102),
    "near_zero": ("near_zero", 7103),
    "alternating": ("alternating", 7104),
    "mixed_magnitude": ("mixed_magnitude", 7105),
}
# The sigmoid of the most negative admitted input raises e to this bound. 16 keeps
# exp(-x) at 8.9e6, far inside FP32, so no admitted input makes an intermediate
# overflow and no task in this family owes its result to an infinity.
MAX_ABS = 16.0
GELU_INNER_SCALE = 0.7978845608028654  # sqrt(2/pi), rounded to the nearest double.
GELU_CUBIC_SCALE = 0.044715
# The standard SELU constants. The migrated AKA parent froze its own scalars (alpha 1.5,
# lambda 0.75) for one fixed instance; this Workload is its own frozen instance and uses
# the published ones, so it is the same composition over different declared scalars.
SELU_ALPHA = 1.6732632423543772
SELU_LAMBDA = 1.0507009873554805
# tanh has more than one admitted spelling on every Target and they do not cost the same,
# so the Schedule names one rather than letting the backend choose. Which one is a
# property of the device, so it comes from the registry rather than from this module.
# The Metal spelling was admitted by Compiler open-cake-ir-sm100a-v71.


def _tolerance(task_name: str) -> dict[str, object]:
    """Return one task's predeclared allowance and the reasoning that fixed it."""
    if task_name == "softplus_gradient":
        # 1 - exp(-y) is a Sterbenz-exact subtraction, so it adds no error of its own,
        # but it also does not attenuate the exponential's. Near y = 0 the exponential
        # sits next to 1.0 where one FP32 ULP is 1.2e-7, and the surviving difference is
        # then scaled by |dy| <= 16, which admits 1.9e-6 absolute before the final
        # product's own rounding. The relative allowance cannot cover this because the
        # difference itself is what cancels.
        return {"atol": 8e-6, "rtol": 2e-5}
    if task_name == "gelu_tanh_backward":
        # The sech-squared factor is formed as 1 - tanh(u)^2, which cancels near
        # saturation while its coefficient 0.5*x*k0*(1+3*k1*x^2) reaches 225.6 at the
        # input bound. The output's sensitivity to the backend tanh is therefore
        # 0.5 + 2*225.6, two ULP of a value in [-1,1] is 1.19e-7, and |dy| <= 16 scales
        # the result. This is a worst case that assumes the coefficient is largest
        # exactly where the tangent has not yet saturated, which no input achieves.
        coefficient = 0.5 * MAX_ABS * GELU_INNER_SCALE * (
            1.0 + 3.0 * GELU_CUBIC_SCALE * MAX_ABS * MAX_ABS)
        return {"atol": float(f"{MAX_ABS * (0.5 + 2.0 * coefficient) * 2.0 ** -23:.3g}"),
                "rtol": 2e-5}
    return {"atol": 2e-6, "rtol": 2e-5}


def workload_document(task_name: str, *, rows: int = 128, columns: int = 1024,
                      backend: str = "metal-m1-pro") -> dict:
    """Create a reusable frozen Workload document; callers persist it outside source."""
    if not isinstance(task_name, str) or task_name not in TASKS or backend not in BACKENDS:
        raise ValueError("unsupported activation task/backend")
    device = BACKENDS[backend]
    if (type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0
            or rows * columns * 4 > 2**31 - 1):
        raise ValueError("activation shape must fit the FP32 buffer ABI")
    # A width the declared route cannot tile would freeze a Workload with no Schedule.
    admit_width(backend, columns)
    operator, revision = TASKS[task_name]
    inputs = INPUTS[task_name]
    tensors = {name: {"shape": list(shape), "max_abs": MAX_ABS} for name, shape in inputs}
    for name in NONNEGATIVE[task_name]:
        tensors[name]["nonnegative"] = True
    tensors["out"] = {"shape": ["R", "C"]}
    for tensor in tensors.values():
        tensor.update(dtype="fp32", layout="contiguous_row_major", finite_only=True)
    definitions = {
        "silu": "out[r,c] = x[r,c] / (1 + exp(-x[r,c]))",
        "swiglu": "out[r,c] = x[r,c] / (1 + exp(-x[r,c])) * up[r,c]",
        "gelu_tanh": (f"out[r,c] = 0.5 * x[r,c] * (1 + tanh({GELU_INNER_SCALE!r} * "
                      f"(x[r,c] + {GELU_CUBIC_SCALE!r} * x[r,c]^3)))"),
        "softsign": "out[r,c] = x[r,c] / (1 + abs(x[r,c]))",
        "selu": (f"out[r,c] = {SELU_LAMBDA!r} * (relu(x[r,c]) + {SELU_ALPHA!r} * "
                 f"exp(x[r,c] - relu(x[r,c])) - {SELU_ALPHA!r})"),
        "softplus_gradient": "out[r,c] = dy[r,c] * (1 - exp(-y[r,c]))",
        "gelu_tanh_backward": (
            f"u[r,c] = {GELU_INNER_SCALE!r} * (x[r,c] + {GELU_CUBIC_SCALE!r} * x[r,c]^3); "
            f"out[r,c] = dy[r,c] * (0.5 * (1 + tanh(u[r,c])) + 0.5 * x[r,c] * "
            f"(1 - tanh(u[r,c])^2) * {GELU_INNER_SCALE!r} * "
            f"(1 + 3 * {GELU_CUBIC_SCALE!r} * x[r,c]^2))"),
        "prelu": "out[r,c] = relu(x[r,c]) + slope[c] * (x[r,c] - relu(x[r,c]))",
    }
    arithmetic = {
        "intermediates": "fp32",
        # Naming the absence is the family's defining commitment: a Schedule that
        # introduces a cross-lane exchange here is not reading the definition.
        "reduction": "none_every_output_reads_one_input_coordinate",
        "output": "fp32",
    }
    if task_name in {"gelu_tanh", "gelu_tanh_backward"}:
        arithmetic["tanh"] = "backend_approximation_within_tolerance"
        arithmetic["cubic"] = "evaluated_before_the_hyperbolic_tangent_in_fp32"
        if task_name == "gelu_tanh_backward":
            # The parent's own recorded mechanism: reuse the tangent instead of forming
            # a hyperbolic cosine, which the finite frozen domain makes exact enough.
            arithmetic["sech_squared"] = "one_minus_the_squared_tangent_not_a_hyperbolic_cosine"
            arithmetic["saved_state"] = "x_is_the_forward_input_not_a_recomputed_result"
    elif task_name == "prelu":
        arithmetic["branch"] = "relu_arms_no_conditional_the_positive_arm_cancels_exactly"
        arithmetic["slope"] = "one_declared_value_per_feature_broadcast_along_the_row"
    elif task_name == "softsign":
        # Named because it is the composition, not a request for a new primitive.
        arithmetic["absolute_value"] = "relu_of_the_operand_plus_relu_of_its_negation"
    elif task_name == "selu":
        arithmetic["exp"] = "backend_approximation_within_tolerance"
        arithmetic["branch"] = "relu_arms_no_conditional_the_positive_arm_cancels_exactly"
    elif task_name == "softplus_gradient":
        arithmetic["exp"] = "backend_approximation_within_tolerance"
        arithmetic["saved_output"] = "y_is_a_declared_input_not_a_recomputed_forward_result"
    else:
        arithmetic["exp"] = "backend_approximation_within_tolerance"
        arithmetic["sigmoid"] = "reciprocal_of_one_plus_the_negated_exponential"
    provenance = [{"kind": "task_mathematical_specification",
                   "path": "src/open_cake_ir/tasks/activation/workload.py",
                   "scope": f"rank2_FP32_{device['provenance_token']}_local_artifact_evaluation"}]
    if task_name in AKA_PARENTS or task_name in AKA_V7_PARENTS:
        provenance.append({
            "kind": "aka_qualified_parent_lineage",
            "path": AKA_REVIEW if task_name in AKA_PARENTS else AKA_V7_RECORDS,
            "derived_parent_id": AKA_PARENTS.get(task_name) or AKA_V7_PARENTS[task_name],
            "scope": ("semantic_origin_only_no_B200_correctness_performance_or_shape_claim"
                      "_transfers_to_this_Apple_Workload"),
        })
    return {
        "schema_version": 1,
        "workload_id": f"{operator.replace('_', '-')}-{backend}-r{rows}-c{columns}-v{revision}",
        "revision": revision, "state": "frozen", "operator": operator,
        "provenance": provenance,
        "cases": [{"case_id": name, "shape": {"R": rows, "C": columns}, "seed": seed, "mode": mode}
                  for name, (mode, seed) in CASES.items()],
        "tensors": tensors,
        "semantics": {
            "definition": definitions[task_name], "target": device["target"],
            "candidate_abi": {"inputs": [name for name, _ in inputs], "outputs": ["out"]},
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "arithmetic": arithmetic,
            "materialization": {
                "generator": "python_random_Random_per_input_case_seed_plus_10000_times_ABI_index",
                "uniform": "round_fp32_uniform[-2,2]", "zeros": "alternating_signed_zero",
                "near_zero": "round_fp32_uniform[-1e-4,1e-4]",
                "alternating": "round_fp32_alternating_sign_times_1_plus_i_mod_17_over_16",
                # Two exponents below the normalization families' bound, so the
                # exponential of the most negative input stays finite in FP32.
                "mixed_magnitude": "signed_powers_of_two_exponent[-12,4]",
            },
        },
        "oracle": {
            "kind": f"real_{task_name}_double_then_fp32",
            "callable": "open_cake_ir.tasks.activation.workload.reference_outputs",
            "implementation": "independent_standard_library_math_no_compiler_or_generated_source",
            "output_rounding": "round_to_nearest_ties_to_even",
        },
        "validation": {
            "primary_case": "primary", "all_cases_required": True, "equal_nan": False,
            "comparison": "elementwise_atol_rtol", **_tolerance(task_name),
            "qualification": "all_input_cases_at_fixed_shape_no_framework_or_cross_shape_claim",
            "tolerance_rationale": (
                "Element-local arithmetic accumulates no reduction error, so the allowance "
                "covers only the emitted chain's FP32 roundings against the oracle's single "
                "one, a backend transcendental that differs by a few units in the last "
                "place, and for the softplus gradient the near-one exponential whose ULP "
                "the Sterbenz-exact subtraction preserves and |dy| then scales by 16. "
                "Not device calibration."),
        },
    }


def validate_activation_contract(document: Mapping[str, object]) -> None:
    """Admit only this exact typed task, backend, ABI, numerical domain and case protocol."""
    workload = WorkloadContract(document)
    task_name = next((name for name, (operator, revision) in TASKS.items()
                      if document.get("operator") == operator and document.get("revision") == revision), None)
    semantics = document.get("semantics")
    backend = backend_for_target(semantics.get("target") if isinstance(semantics, Mapping) else None)
    if task_name is None or backend is None or workload.case_ids != tuple(CASES):
        raise ValueError("activation operator/revision, backend or required input cases differ")
    shape = workload.case("primary")["shape"]
    if set(shape) != {"R", "C"}:
        raise ValueError("activation case requires R/C dimensions")
    expected = workload_document(task_name, rows=shape["R"], columns=shape["C"], backend=backend)
    # Canonical JSON equality distinguishes bools from integers and forbids extra fields.
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("activation frozen semantic/ABI/input/validation contract differs")
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def _task_of(document: Mapping[str, object]) -> str:
    operator = document["operator"]
    return next(name for name, (registered, _) in TASKS.items() if registered == operator)


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float]]:
    """Return deterministic FP32 inputs in the contract's explicit ABI order."""
    validate_activation_contract(workload.document)
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
                value = (1 if index % 2 else -1) * 2.0 ** rng.randint(-12, 4)
            else:
                value = rng.uniform(-2, 2)
            if arg.name in NONNEGATIVE[task_name]:
                value = abs(value)
            values.append(_round(value, "fp32"))
        result[arg.name] = values
    return result


def reference_outputs(workload: WorkloadContract, case_id: str,
                      inputs: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Independent double-precision oracle rounded once to FP32."""
    validate_activation_contract(workload.document)
    operator = workload.document["operator"]
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    checked = dict(zip((arg.name for arg in args), _checked_inputs(workload, args, inputs)))
    for name in NONNEGATIVE[_task_of(workload.document)]:
        if any(value < 0.0 for value in checked[name]):
            raise ValueError(f"oracle input {name} is declared non-negative")
    if operator == "gelu_tanh_fp32":
        return {"out": [_round(0.5 * value * (1.0 + math.tanh(
            GELU_INNER_SCALE * (value + GELU_CUBIC_SCALE * value ** 3))), "fp32")
            for value in checked["x"]]}
    if operator == "gelu_tanh_backward_fp32":
        result = []
        for value, gradient in zip(checked["x"], checked["dy"]):
            inner = GELU_INNER_SCALE * (value + GELU_CUBIC_SCALE * value ** 3)
            tangent = math.tanh(inner)
            derivative = (0.5 * (1.0 + tangent) + 0.5 * value * (1.0 - tangent * tangent)
                          * GELU_INNER_SCALE * (1.0 + 3.0 * GELU_CUBIC_SCALE * value * value))
            result.append(_round(gradient * derivative, "fp32"))
        return {"out": result}
    if operator == "prelu_fp32":
        slopes = checked["slope"]
        width = len(slopes)
        return {"out": [_round(value if value > 0.0 else value * slopes[index % width], "fp32")
                        for index, value in enumerate(checked["x"])]}
    if operator == "softsign_fp32":
        return {"out": [_round(value / (1.0 + abs(value)), "fp32") for value in checked["x"]]}
    if operator == "selu_fp32":
        # Written as the mathematical piecewise function; the Schedule reaches it through
        # relu arms because the IR admits no conditional, and the two agree exactly.
        return {"out": [_round(SELU_LAMBDA * (value if value > 0.0
                                              else SELU_ALPHA * math.expm1(value)), "fp32")
                        for value in checked["x"]]}
    if operator == "softplus_gradient_fp32":
        return {"out": [_round(gradient * -math.expm1(-value), "fp32")
                        for value, gradient in zip(checked["y"], checked["dy"])]}
    # The bound on x keeps math.exp(-x) finite, so the sigmoid needs no branch.
    gates = [1.0 / (1.0 + math.exp(-value)) for value in checked["x"]]
    gated = [value * gate for value, gate in zip(checked["x"], gates)]
    if operator == "swiglu_fp32":
        gated = [value * projected for value, projected in zip(gated, checked["up"])]
    return {"out": [_round(value, "fp32") for value in gated]}
