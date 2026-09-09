"""Exact Apple-device FP32 GEMM Workloads, deterministic inputs and independent math.

One contract binds one (M, K, N) shape and five required input distributions. The
output column count is unrolled by the schedule, so N is part of the frozen shape
rather than a runtime extent. The Ralph engine owns search and evaluation; these
functions neither import candidates nor launch a device.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.apple import BACKENDS, backend_for_target
from open_cake_ir.tasks.tiles.workload import _checked_inputs, _round

TASKS = {"gemm_bias": ("gemm_bias_fp32", "1")}
CASES = {
    "primary": ("uniform", 8001),
    "zeros": ("zeros", 8002),
    "near_zero": ("near_zero", 8003),
    "alternating": ("alternating", 8004),
    "mixed_magnitude": ("mixed_magnitude", 8005),
}


def workload_document(task_name: str, *, rows: int = 128, depth: int = 256, columns: int = 16,
                      backend: str = "metal-m2") -> dict:
    """Create a reusable frozen Workload document; callers persist it outside source."""
    if not isinstance(task_name, str) or task_name not in TASKS or backend not in BACKENDS:
        raise ValueError("unsupported GEMM task/backend")
    device = BACKENDS[backend]
    if (any(type(value) is not int for value in (rows, depth, columns))
            or min(rows, depth, columns) <= 0
            or (rows * depth + depth * columns + rows * columns) * 4 > 2**31 - 1):
        raise ValueError("GEMM shape must fit the FP32 Metal buffer ABI")
    operator, revision = TASKS[task_name]
    tensors = {
        "a": {"shape": ["M", "K"], "max_abs": 256.0},
        "b": {"shape": ["K", "N"], "max_abs": 256.0},
        "bias": {"shape": ["N"], "max_abs": 1.5},
    }
    inputs = list(tensors)
    tensors["out"] = {"shape": ["M", "N"]}
    for tensor in tensors.values():
        tensor.update(dtype="fp32", layout="contiguous_row_major", finite_only=True)
    return {
        "schema_version": 1,
        "workload_id": f"{operator.replace('_', '-')}-{backend}-m{rows}-k{depth}-n{columns}-v{revision}",
        "revision": revision, "state": "frozen", "operator": operator,
        "provenance": [{"kind": "task_mathematical_specification",
                        "path": "src/open_cake_ir/tasks/gemm/workload.py",
                        "scope": f"rank2_FP32_{device['provenance_token']}_local_artifact_evaluation"}],
        "cases": [{"case_id": name, "shape": {"M": rows, "K": depth, "N": columns},
                   "seed": seed, "mode": mode} for name, (mode, seed) in CASES.items()],
        "tensors": tensors,
        "semantics": {
            "definition": "out[m,n] = sum_k(a[m,k] * b[k,n]) + bias[n]",
            "target": device["target"],
            "candidate_abi": {"inputs": inputs, "outputs": ["out"]},
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "arithmetic": {"intermediates": "fp32", "accumulation": "backend_defined_within_tolerance",
                           "output": "fp32"},
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
            "kind": "real_gemm_bias_fsum_then_fp32",
            "callable": "open_cake_ir.tasks.gemm.workload.reference_outputs",
            "implementation": "independent_standard_library_math_no_compiler_or_generated_source",
            "output_rounding": "round_to_nearest_ties_to_even",
        },
        "validation": {
            "primary_case": "primary", "all_cases_required": True, "equal_nan": False,
            "comparison": "elementwise_atol_rtol", "atol": 1e-3, "rtol": 1e-4,
            "qualification": "all_input_cases_at_fixed_shape_no_framework_or_cross_shape_claim",
            "tolerance_rationale": "Signed accumulation over K terms cancels, so the allowance is "
                                  "sized from the measured worst sequential-FP32 departure from the "
                                  "fsum oracle over these declared distributions: 2.3e-5 absolute "
                                  "near cancellation and 3.1e-2 absolute at outputs of order 2e5. "
                                  "It is a numerical allowance, not device calibration.",
        },
    }


def validate_gemm_contract(document: Mapping[str, object]) -> None:
    """Admit only this exact typed task, backend, ABI, numerical domain and case protocol."""
    workload = WorkloadContract(document)
    task_name = next((name for name, (operator, revision) in TASKS.items()
                      if document.get("operator") == operator and document.get("revision") == revision), None)
    semantics = document.get("semantics")
    backend = backend_for_target(semantics.get("target") if isinstance(semantics, Mapping) else None)
    if task_name is None or backend is None or workload.case_ids != tuple(CASES):
        raise ValueError("GEMM operator/revision, backend or required input cases differ")
    shape = workload.case("primary")["shape"]
    if set(shape) != {"M", "K", "N"}:
        raise ValueError("GEMM case requires M/K/N dimensions")
    expected = workload_document(task_name, rows=shape["M"], depth=shape["K"],
                                 columns=shape["N"], backend=backend)
    # Canonical JSON equality distinguishes bools from integers and forbids extra fields.
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("GEMM frozen semantic/ABI/input/validation contract differs")
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float]]:
    """Return deterministic FP32 inputs in the contract's explicit ABI order."""
    validate_gemm_contract(workload.document)
    case = workload.case(case_id)
    result = {}
    args = [arg for arg in workload.tensor_abi(case_id) if arg.mode == "input"]
    for position, arg in enumerate(args):
        rng = random.Random(case["seed"] + position * 10000)
        values = []
        for index in range(math.prod(arg.shape)):
            if arg.name == "bias":
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
    """Independent fsum oracle over exact FP32 products; no compiler or generated source."""
    validate_gemm_contract(workload.document)
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    checked = dict(zip((arg.name for arg in args), _checked_inputs(workload, args, inputs)))
    shape = workload.case(case_id)["shape"]
    rows, depth, columns = shape["M"], shape["K"], shape["N"]
    a, b, bias = checked["a"], checked["b"], checked["bias"]
    result = []
    for m in range(rows):
        row = a[m * depth:(m + 1) * depth]
        for n in range(columns):
            # Exact FP32 products summed without intermediate rounding, then rounded once.
            total = math.fsum(row[k] * b[k * columns + n] for k in range(depth)) + bias[n]
            result.append(_round(total, "fp32"))
    return {"out": result}
