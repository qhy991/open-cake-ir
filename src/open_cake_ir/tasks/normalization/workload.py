"""Exact M1 Pro normalization Workloads, deterministic inputs and independent math.

Each generated contract binds one shape and five required input distributions. The Ralph engine owns search
and evaluation; these functions neither import candidates nor launch a device.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.tiles.workload import _checked_inputs, _round
from open_cake_ir.tasks.tiles.workload import reference_outputs as tile_reference_outputs

TARGET = "apple_gpu_family7"
TASKS = {
    "rmsnorm": ("rmsnorm_fp32", "3"),
    "layernorm": ("layernorm_fp32", "1"),
    "residual_rmsnorm": ("residual_rmsnorm_fp32", "1"),
}
CASES = {
    "primary": ("uniform", 7001),
    "zeros": ("zeros", 7002),
    "near_zero": ("near_zero", 7003),
    "alternating": ("alternating", 7004),
    "mixed_magnitude": ("mixed_magnitude", 7005),
}
EPSILON = 1e-5


def workload_document(task_name: str, *, rows: int = 128, columns: int = 1024,
                      backend: str = "metal-m1-pro") -> dict:
    """Create a reusable frozen Workload document; callers persist it outside source."""
    if not isinstance(task_name, str) or task_name not in TASKS or backend != "metal-m1-pro":
        raise ValueError("unsupported normalization task/backend")
    if (type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0
            or rows * columns * 4 > 2**31 - 1):
        raise ValueError("normalization shape must fit the FP32 Metal buffer ABI")
    operator, revision = TASKS[task_name]
    tensors = {
        "x": {"shape": ["R", "C"], "max_abs": 256.0},
        "weight": {"shape": ["C"], "max_abs": 1.5},
    }
    if task_name == "layernorm":
        tensors["bias"] = {"shape": ["C"], "max_abs": 1.5}
    elif task_name == "residual_rmsnorm":
        tensors["residual"] = {"shape": ["R", "C"], "max_abs": 256.0}
    inputs = list(tensors)
    tensors["out"] = {"shape": ["R", "C"]}
    for tensor in tensors.values():
        tensor.update(dtype="fp32", layout="contiguous_row_major", finite_only=True)
    definitions = {
        "rmsnorm": "out[r,c] = x[r,c] * weight[c] / sqrt(mean_j(x[r,j]^2) + epsilon)",
        "layernorm": "mu[r] = mean_j(x[r,j]); out[r,c] = (x[r,c]-mu[r]) * weight[c] / sqrt(mean_j((x[r,j]-mu[r])^2) + epsilon) + bias[c]",
        "residual_rmsnorm": "z[r,c] = round_fp32(x[r,c]+residual[r,c]); out[r,c] = z[r,c] * weight[c] / sqrt(mean_j(z[r,j]^2) + epsilon)",
    }
    arithmetic = {"intermediates": "fp32", "reduction_order": "backend_defined_within_tolerance",
                  "rsqrt": "backend_approximation_within_tolerance", "output": "fp32"}
    if task_name == "layernorm":
        arithmetic["variance"] = "centered_population_variance"
    elif task_name == "residual_rmsnorm":
        arithmetic["residual_addition"] = "round_to_nearest_ties_to_even_fp32_before_normalization"
    return {
        "schema_version": 1,
        "workload_id": f"{operator.replace('_', '-')}-metal-m1-pro-r{rows}-c{columns}-v{revision}",
        "revision": revision, "state": "frozen", "operator": operator,
        "provenance": [{"kind": "task_mathematical_specification",
                        "path": "src/open_cake_ir/tasks/normalization/workload.py",
                        "scope": "rank2_FP32_M1_Pro_local_artifact_evaluation"}],
        "cases": [{"case_id": name, "shape": {"R": rows, "C": columns}, "seed": seed, "mode": mode}
                  for name, (mode, seed) in CASES.items()],
        "tensors": tensors,
        "semantics": {
            "definition": definitions[task_name], "target": TARGET,
            "candidate_abi": {"inputs": inputs, "outputs": ["out"]},
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "arithmetic": arithmetic, "epsilon": EPSILON,
            "materialization": {
                "generator": "python_random_Random_per_input_case_seed_plus_10000_times_ABI_index",
                "uniform": "round_fp32_uniform[-2,2]", "zeros": "alternating_signed_zero",
                "near_zero": "round_fp32_uniform[-1e-4,1e-4]",
                "alternating": "round_fp32_alternating_sign_times_1_plus_i_mod_17_over_16",
                "mixed_magnitude": "signed_powers_of_two_exponent[-12,8]",
                "affine": "round_fp32_uniform[-1.5,1.5]_every_17th_zero",
            },
        },
        "oracle": {
            "kind": f"real_{task_name}_fsum_then_fp32",
            "callable": "open_cake_ir.tasks.normalization.workload.reference_outputs",
            "implementation": "independent_standard_library_math_no_compiler_or_generated_source",
            "output_rounding": "round_to_nearest_ties_to_even",
        },
        "validation": {
            "primary_case": "primary", "all_cases_required": True, "equal_nan": False,
            "comparison": "elementwise_atol_rtol", "atol": 2e-5, "rtol": 2e-5,
            "qualification": "all_input_cases_at_fixed_shape_no_framework_or_cross_shape_claim",
            "tolerance_rationale": "Predeclared FP32 reduction and rsqrt allowance against the mathematical oracle; not device calibration.",
        },
    }


def validate_normalization_contract(document: Mapping[str, object]) -> None:
    """Admit only this exact typed task, ABI, numerical domain and case protocol."""
    workload = WorkloadContract(document)
    task_name = next((name for name, (operator, revision) in TASKS.items()
                      if document.get("operator") == operator and document.get("revision") == revision), None)
    if task_name is None or workload.case_ids != tuple(CASES):
        raise ValueError("normalization operator/revision or required input cases differ")
    shape = workload.case("primary")["shape"]
    if set(shape) != {"R", "C"}:
        raise ValueError("normalization case requires R/C dimensions")
    expected = workload_document(task_name, rows=shape["R"], columns=shape["C"])
    # Canonical JSON equality distinguishes bools from integers and forbids extra fields.
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("normalization frozen semantic/ABI/input/validation contract differs")
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float]]:
    """Return deterministic FP32 inputs in the contract's explicit ABI order."""
    validate_normalization_contract(workload.document)
    case = workload.case(case_id)
    result = {}
    args = [arg for arg in workload.tensor_abi(case_id) if arg.mode == "input"]
    for position, arg in enumerate(args):
        rng = random.Random(case["seed"] + position * 10000)
        values = []
        for index in range(math.prod(arg.shape)):
            if arg.name in {"weight", "bias"}:
                value = 0.0 if (index + 1) % 17 == 0 else rng.uniform(-1.5, 1.5)
            elif case["mode"] == "zeros":
                value = -0.0 if index % 2 else 0.0
            elif case["mode"] == "near_zero":
                value = rng.uniform(-1e-4, 1e-4)
            elif case["mode"] == "alternating":
                value = (1 if index % 2 else -1) * (1 + (index % 17) / 16)
            elif case["mode"] == "mixed_magnitude":
                value = (1 if index % 2 else -1) * 2.0 ** rng.randint(-12, 8)
            else:
                value = rng.uniform(-2, 2)
            values.append(_round(value, "fp32"))
        result[arg.name] = values
    return result


def reference_outputs(workload: WorkloadContract, case_id: str,
                      inputs: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Independent fsum oracle; RMSNorm reuses the existing tile mathematics."""
    validate_normalization_contract(workload.document)
    if workload.document["operator"] == "rmsnorm_fp32":
        return tile_reference_outputs(workload, case_id, inputs)
    abi = workload.tensor_abi(case_id)
    args = tuple(arg for arg in abi if arg.mode == "input")
    checked = dict(zip((arg.name for arg in args), _checked_inputs(workload, args, inputs)))
    values, weights = checked["x"], checked["weight"]
    width = args[0].shape[-1]
    if "residual" in checked:
        values = [_round(x + residual, "fp32") for x, residual in zip(values, checked["residual"])]
    result = []
    for start in range(0, len(values), width):
        row = values[start:start + width]
        if "bias" in checked:
            mean = math.fsum(row) / width
            row = [value - mean for value in row]
        inverse = 1.0 / math.sqrt(math.fsum(value * value for value in row) / width + EPSILON)
        for column, value in enumerate(row):
            output = value * inverse * weights[column]
            if "bias" in checked:
                output += checked["bias"][column]
            result.append(_round(output, "fp32"))
    return {"out": result}
