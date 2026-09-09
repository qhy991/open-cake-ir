"""Portable optimizer-step Workloads: element-local updates that carry accumulator state.

These are element-local like the activation family, but they differ from it in the two
ways that make an optimizer step its own regime. They write several buffers per element
rather than one, so the Schedule owns a real store schedule instead of a single epilogue.
And some of their inputs are *accumulators* that a previous step produced, which the
contract declares non-negative because a reciprocal square root reads them; a family whose
inputs are all freely signed cannot express that.

Neither property existed anywhere else in this Lab, which is the point of the family.

All three tasks are migrations of AKA v7 qualified parents. The Ralph engine owns search
and evaluation; these functions neither import candidates nor launch a device.
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
    "momentum_sgd": ("momentum_sgd_fp32", "1"),
    "adamw": ("adamw_fp32", "1"),
    "adadelta": ("adadelta_fp32", "1"),
}
INPUTS = {
    "momentum_sgd": ("param", "grad", "moment"),
    "adamw": ("param", "grad", "first", "second"),
    "adadelta": ("param", "grad", "square_accumulator", "update_accumulator"),
}
OUTPUTS = {
    "momentum_sgd": ("moment_out", "param_out"),
    "adamw": ("first_out", "second_out", "param_out"),
    "adadelta": ("square_out", "update_out", "param_out"),
}
# Inputs a reciprocal square root reads. The contract declares them non-negative and the
# oracle refuses a negative value rather than returning a quiet NaN.
NONNEGATIVE = {
    "momentum_sgd": (),
    "adamw": ("second",),
    "adadelta": ("square_accumulator", "update_accumulator"),
}
AKA_V7_RECORDS = "AKA/datasets/curated/cuda_kernel_parent_completions_v7/records.jsonl"
AKA_PARENTS = {
    "momentum_sgd": "momentum_sgd_update_contiguous_fp32_i32_block256_v1",
    "adamw": "adamw_contiguous_fp32_int32_block256_v1",
    "adadelta": "derived_adadelta_update_contiguous_f32_i32_b256_v1",
}
CASES = {
    "primary": ("uniform", 7401),
    "zeros": ("zeros", 7402),
    "near_zero": ("near_zero", 7403),
    "alternating": ("alternating", 7404),
    "mixed_magnitude": ("mixed_magnitude", 7405),
}
MAX_ABS = 2.0
_UNIT_ROUNDOFF = 2.0 ** -24
# One frozen hyperparameter set per task. A step is not a search space; the Schedule may
# reorder the arithmetic but not choose different constants.
MOMENTUM = 0.9
LEARNING_RATE = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_LEARNING_RATE = 0.001
ADAM_WEIGHT_DECAY = 0.01
ADAM_EPSILON = 1e-8
# Bias corrections for a fixed step count of ten, frozen as literals because the step
# index is not part of this contract's ABI.
ADAM_BIAS1 = 1.0 - ADAM_BETA1 ** 10
ADAM_BIAS2 = 1.0 - ADAM_BETA2 ** 10
ADADELTA_DECAY = 0.95
# Larger than the published default. The reciprocal square root of this epsilon is what
# multiplies every rounding when the accumulator is near zero, so the frozen instance
# picks a value whose amplification the declared allowance can still bound.
ADADELTA_EPSILON = 1e-3
ADADELTA_LEARNING_RATE = 1.0


def _allowance(task_name: str) -> dict[str, float]:
    """Derive each step's absolute allowance from the amplification it performs."""
    if task_name == "momentum_sgd":
        # Two multiplies and one subtract; nothing amplifies a rounding.
        return {"atol": 2e-6, "rtol": 2e-5}
    if task_name == "adamw":
        # The parameter update is lr * mhat * rsqrt(vhat + eps); the reciprocal square
        # root is largest when the second moment is zero, and three FP32 roundings of a
        # backend approximation reach the output through that factor.
        ratio = (MAX_ABS / ADAM_BIAS1) * (1.0 / math.sqrt(ADAM_EPSILON))
        return {"atol": float(f"{ADAM_LEARNING_RATE * 3.0 * _UNIT_ROUNDOFF * ratio:.3g}"),
                "rtol": 2e-5}
    # Adadelta's scaled gradient is bounded by sqrt(d + eps) * rsqrt(eps) * |g|, and the
    # update accumulator squares it, which is the largest amplification in the family.
    scaled = math.sqrt(MAX_ABS + ADADELTA_EPSILON) / math.sqrt(ADADELTA_EPSILON) * MAX_ABS
    error = (1.0 - ADADELTA_DECAY) * 2.0 * scaled * (3.0 * _UNIT_ROUNDOFF * scaled)
    return {"atol": float(f"{max(error, 2e-6):.3g}"), "rtol": 2e-5}


def workload_document(task_name: str, *, rows: int = 128, columns: int = 1024,
                      backend: str = "metal-m1-pro") -> dict:
    """Create a reusable frozen Workload document; callers persist it outside source."""
    if not isinstance(task_name, str) or task_name not in TASKS or backend not in BACKENDS:
        raise ValueError("unsupported optimizers task/backend")
    device = BACKENDS[backend]
    if (type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0
            or rows * columns * 4 > 2**31 - 1):
        raise ValueError("optimizers shape must fit the FP32 buffer ABI")
    admit_width(backend, columns)
    operator, revision = TASKS[task_name]
    tensors = {}
    for name in INPUTS[task_name]:
        tensors[name] = {"shape": ["R", "C"], "max_abs": MAX_ABS}
        if name in NONNEGATIVE[task_name]:
            tensors[name]["nonnegative"] = True
    for name in OUTPUTS[task_name]:
        tensors[name] = {"shape": ["R", "C"]}
    for tensor in tensors.values():
        tensor.update(dtype="fp32", layout="contiguous_row_major", finite_only=True)
    definitions = {
        "momentum_sgd": (
            f"moment_out[r,c] = {MOMENTUM!r} * moment[r,c] + {LEARNING_RATE!r} * grad[r,c]; "
            "param_out[r,c] = param[r,c] - moment_out[r,c]"),
        "adamw": (
            f"first_out[r,c] = {ADAM_BETA1!r} * first[r,c] + {1.0 - ADAM_BETA1!r} * grad[r,c]; "
            f"second_out[r,c] = {ADAM_BETA2!r} * second[r,c] + {1.0 - ADAM_BETA2!r} * grad[r,c]^2; "
            f"param_out[r,c] = param[r,c] - {ADAM_LEARNING_RATE!r} * {ADAM_WEIGHT_DECAY!r} * param[r,c] "
            f"- {ADAM_LEARNING_RATE!r} * (first_out[r,c] / {ADAM_BIAS1!r}) "
            f"/ sqrt(second_out[r,c] / {ADAM_BIAS2!r} + {ADAM_EPSILON!r})"),
        "adadelta": (
            f"square_out[r,c] = {ADADELTA_DECAY!r} * square_accumulator[r,c] + "
            f"{1.0 - ADADELTA_DECAY!r} * grad[r,c]^2; "
            f"step[r,c] = sqrt(update_accumulator[r,c] + {ADADELTA_EPSILON!r}) / "
            f"sqrt(square_out[r,c] + {ADADELTA_EPSILON!r}) * grad[r,c]; "
            f"param_out[r,c] = param[r,c] + {ADADELTA_LEARNING_RATE!r} * step[r,c]; "
            f"update_out[r,c] = {ADADELTA_DECAY!r} * update_accumulator[r,c] + "
            f"{1.0 - ADADELTA_DECAY!r} * step[r,c]^2"),
    }
    arithmetic = {
        "intermediates": "fp32",
        "reduction": "none_every_output_reads_one_input_coordinate",
        "state": "every_accumulator_is_read_and_rewritten_as_a_separate_output_buffer",
        "hyperparameters": "frozen_by_this_contract_not_a_search_dimension",
        "output": "fp32",
    }
    if task_name != "momentum_sgd":
        arithmetic["reciprocal_square_root"] = "backend_approximation_within_tolerance"
        arithmetic["epsilon_placement"] = "inside_the_square_root_not_added_to_it"
    if task_name == "adadelta":
        arithmetic["square_root"] = "the_operand_times_its_own_reciprocal_square_root"
    return {
        "schema_version": 1,
        "workload_id": f"{operator.replace('_', '-')}-{backend}-r{rows}-c{columns}-v{revision}",
        "revision": revision, "state": "frozen", "operator": operator,
        "provenance": [
            {"kind": "task_mathematical_specification",
             "path": "src/open_cake_ir/tasks/optimizers/workload.py",
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
            "candidate_abi": {"inputs": list(INPUTS[task_name]),
                              "outputs": list(OUTPUTS[task_name])},
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "arithmetic": arithmetic,
            "materialization": {
                "generator": "python_random_Random_per_input_case_seed_plus_10000_times_ABI_index",
                "uniform": "round_fp32_uniform[-2,2]", "zeros": "alternating_signed_zero",
                "near_zero": "round_fp32_uniform[-1e-4,1e-4]",
                "alternating": "round_fp32_alternating_sign_times_1_plus_i_mod_17_over_16",
                "mixed_magnitude": "signed_powers_of_two_exponent[-12,1]",
                "nonnegative": "the_magnitude_of_the_generated_value_for_declared_accumulators",
            },
        },
        "oracle": {
            "kind": f"real_{task_name}_double_then_fp32",
            "callable": "open_cake_ir.tasks.optimizers.workload.reference_outputs",
            "implementation": "independent_standard_library_math_no_compiler_or_generated_source",
            "output_rounding": "round_to_nearest_ties_to_even",
        },
        "validation": {
            "primary_case": "primary", "all_cases_required": True, "equal_nan": False,
            "comparison": "elementwise_atol_rtol", **_allowance(task_name),
            "qualification": "all_input_cases_at_fixed_shape_no_framework_or_cross_shape_claim",
            "tolerance_rationale": (
                "No reduction runs here, so the allowance covers only the emitted chain's "
                "FP32 roundings and the reciprocal square root the step divides by. That "
                "factor is largest when the accumulator is zero, which is why the epsilon "
                "is part of this contract and the allowance is derived from it. It is a "
                "worst case assuming every rounding aligns. Not device calibration."),
        },
    }


def validate_optimizers_contract(document: Mapping[str, object]) -> None:
    """Admit only this exact typed task, backend, ABI, numerical domain and case protocol."""
    workload = WorkloadContract(document)
    task_name = next((name for name, (operator, revision) in TASKS.items()
                      if document.get("operator") == operator
                      and document.get("revision") == revision), None)
    semantics = document.get("semantics")
    backend = backend_for_target(semantics.get("target") if isinstance(semantics, Mapping) else None)
    if task_name is None or backend is None or workload.case_ids != tuple(CASES):
        raise ValueError("optimizers operator/revision, backend or required input cases differ")
    shape = workload.case("primary")["shape"]
    if set(shape) != {"R", "C"}:
        raise ValueError("optimizers case requires R/C dimensions")
    expected = workload_document(task_name, rows=shape["R"], columns=shape["C"], backend=backend)
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("optimizers frozen semantic/ABI/input/validation contract differs")
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def _task_of(document: Mapping[str, object]) -> str:
    operator = document["operator"]
    return next(name for name, (registered, _) in TASKS.items() if registered == operator)


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float]]:
    """Return deterministic FP32 inputs; declared accumulators keep their magnitude only."""
    validate_optimizers_contract(workload.document)
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
    """Independent double-precision oracle; each output is rounded once to FP32."""
    validate_optimizers_contract(workload.document)
    task_name = _task_of(workload.document)
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    checked = dict(zip((arg.name for arg in args), _checked_inputs(workload, args, inputs)))
    for name in NONNEGATIVE[task_name]:
        if any(value < 0.0 for value in checked[name]):
            raise ValueError(f"oracle input {name} is declared non-negative")
    parameters, gradients = checked["param"], checked["grad"]
    if task_name == "momentum_sgd":
        moments = [MOMENTUM * moment + LEARNING_RATE * gradient
                   for moment, gradient in zip(checked["moment"], gradients)]
        return {"moment_out": [_round(value, "fp32") for value in moments],
                "param_out": [_round(parameter - moment, "fp32")
                              for parameter, moment in zip(parameters, moments)]}
    if task_name == "adamw":
        first, second, updated = [], [], []
        for parameter, gradient, m, v in zip(parameters, gradients,
                                             checked["first"], checked["second"]):
            m_next = ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * gradient
            v_next = ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * gradient * gradient
            step = ((m_next / ADAM_BIAS1)
                    / math.sqrt(v_next / ADAM_BIAS2 + ADAM_EPSILON))
            first.append(_round(m_next, "fp32"))
            second.append(_round(v_next, "fp32"))
            updated.append(_round(parameter - ADAM_LEARNING_RATE * ADAM_WEIGHT_DECAY * parameter
                                  - ADAM_LEARNING_RATE * step, "fp32"))
        return {"first_out": first, "second_out": second, "param_out": updated}
    squares, updates, updated = [], [], []
    for parameter, gradient, h, d in zip(parameters, gradients,
                                         checked["square_accumulator"],
                                         checked["update_accumulator"]):
        h_next = ADADELTA_DECAY * h + (1.0 - ADADELTA_DECAY) * gradient * gradient
        step = (math.sqrt(d + ADADELTA_EPSILON)
                / math.sqrt(h_next + ADADELTA_EPSILON) * gradient)
        squares.append(_round(h_next, "fp32"))
        updated.append(_round(parameter + ADADELTA_LEARNING_RATE * step, "fp32"))
        updates.append(_round(ADADELTA_DECAY * d + (1.0 - ADADELTA_DECAY) * step * step, "fp32"))
    return {"square_out": squares, "update_out": updates, "param_out": updated}
