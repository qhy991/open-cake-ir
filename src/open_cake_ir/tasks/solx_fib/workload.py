"""SoL-ExecBench tasks whose definitions come from FlashInfer-Bench.

One task here binds one upstream single-operator definition to one fixed shape, its own
BF16 ABI and its own independent CPU oracle. The upstream pack is semantic provenance:
no candidate source, no upstream latency and no upstream score is inherited, and the
upstream's ``matched_ratio`` gate is deliberately replaced by an all-element comparison
(ADR 0065). The ``solx_l1`` sibling owns the SOL-ExecBench L1 subset, which is a
different upstream authority and a separate Executor closure.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.tiles.workload import _checked_inputs, _round
from open_cake_ir.tasks.devices import (
    BACKENDS, admit_dtype, admit_operations, admit_width, backend_for_target,
)

# Upstream pack task id -> (this project's operator, revision). The launcher offers
# whatever this table registers; the `fib_` prefix names the upstream authority so a
# reader of the operator table never has to open a file to learn where a task came from.
TASKS = {
    "fib_rmsnorm_h4096": ("solx_fib_rmsnorm_h4096_bf16", "1"),
}
# The upstream definition's constant axis. It is the task's identity, not a parameter:
# `rmsnorm_h4096` with any other hidden size is a different upstream task.
HIDDEN_SIZE = {
    "fib_rmsnorm_h4096": 4096,
}
UPSTREAM = {
    "fib_rmsnorm_h4096": {
        "task": "flashinfer-bench-tasks/tasks/025_rmsnorm_h4096",
        "definition": "upstream/definitions/rmsnorm/rmsnorm_h4096.json",
        "model": "meta-llama/Llama-3.1-8B",
        # The upstream workload axis, recorded rather than copied. This contract binds one
        # shape; the remaining batch sizes are a later portfolio question, and claiming
        # them here would report a domain this contract never examines.
        "workload_axis": "batch_size in {1,7,15,16,34,63,64,79,170,8804,10827,11832,14418,14509}",
        "workload_count": 14,
    },
}
# The upstream baseline directories hold complete low-level target implementations. They
# are declared here so a restricted arm's refusal is a property of the contract rather
# than a convention someone has to remember (ADR 0062).
RESTRICTED = {
    "fib_rmsnorm_h4096": ("flashinfer-bench-tasks/tasks/025_rmsnorm_h4096/baseline/",
                          "flashinfer-bench-tasks/dcu_sol/025_rmsnorm_h4096/"),
}
CASES = {
    "primary": ("uniform", 2501),
    "zeros": ("zeros", 2502),
    "near_zero": ("near_zero", 2503),
    "alternating": ("alternating", 2504),
    "mixed_magnitude": ("mixed_magnitude", 2505),
}
# The upstream definition fixes this; it is not a tuning parameter.
EPSILON = 1e-5
# BF16 carries an 8-bit significand, so one ulp is 2**-8 relative. The allowance is two
# of them, plus an absolute floor two ulp below the smallest output magnitude these
# distributions produce.
RTOL = 2.0 ** -7
ATOL = 2.0 ** -16
# What a Schedule for this family has to be able to say. Asking the Target rather than
# naming devices keeps the task available on exactly the backends that can express it.
OPERATION_KINDS = ("load", "cast", "elementwise", "reduce", "store")


def workload_document(task_name: str, *, rows: int, columns: int,
                      backend: str = "triton-b300") -> dict:
    """Create one frozen Workload document; callers persist it outside source."""
    if not isinstance(task_name, str) or task_name not in TASKS or backend not in BACKENDS:
        raise ValueError("unsupported SoL-ExecBench FlashInfer-Bench task/backend")
    hidden = HIDDEN_SIZE[task_name]
    if columns != hidden:
        raise ValueError(
            f"{task_name} names hidden size {hidden}; {columns} is a different upstream task")
    if type(rows) is not int or rows <= 0 or rows * columns * 2 > 2**31 - 1:
        raise ValueError("the batch extent must fit the BF16 buffer ABI")
    device = BACKENDS[backend]
    admit_dtype(backend, "bf16")
    admit_operations(backend, OPERATION_KINDS)
    admit_width(backend, columns)
    operator, revision = TASKS[task_name]
    upstream = UPSTREAM[task_name]
    tensors = {
        "x": {"shape": ["R", "C"], "max_abs": 256.0},
        "weight": {"shape": ["C"], "max_abs": 1.5},
        "out": {"shape": ["R", "C"]},
    }
    for tensor in tensors.values():
        tensor.update(dtype="bf16", layout="contiguous_row_major", finite_only=True)
    return {
        "schema_version": 1,
        "workload_id": f"solx-fib-{task_name[4:].replace('_', '-')}-bf16-{backend}-r{rows}-v{revision}",
        "revision": revision, "state": "frozen", "operator": operator,
        "provenance": [
            {"kind": "solx_pack_task", "upstream": "flashinfer-bench",
             "path": upstream["task"], "definition": upstream["definition"],
             "model": upstream["model"], "workload_axis": upstream["workload_axis"],
             "workload_count": upstream["workload_count"],
             "scope": "semantic_source_only; no_upstream_candidate_source_latency_or_score_is_inherited"},
            *({"kind": "restricted_artifact", "path": path,
               "role": "complete_low_level_target_implementation",
               "scope": "black_box_baseline_only; not_readable_by_clean_start_or_direct_low_level_arm"}
              for path in RESTRICTED[task_name]),
            {"kind": "standalone_derived_contract",
             "path": "src/open_cake_ir/tasks/solx_fib/workload.py",
             "scope": f"fixed_shape_BF16_{device['provenance_token']}_semantics_and_CPU_oracle"},
        ],
        "cases": [{"case_id": name, "shape": {"R": rows, "C": columns}, "seed": seed, "mode": mode}
                  for name, (mode, seed) in CASES.items()],
        "tensors": tensors,
        "semantics": {
            "definition": ("out[r,c] = round_bf16(x[r,c] * weight[c] "
                           "/ sqrt(mean_j(x[r,j]^2) + epsilon))"),
            "target": device["target"],
            "candidate_abi": {"inputs": ["x", "weight"], "outputs": ["out"]},
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "arithmetic": {
                "operands": "bf16", "intermediates": "fp32",
                "reduction_order": "backend_defined_within_tolerance",
                "rsqrt": "backend_approximation_within_tolerance",
                "output": "round_to_nearest_ties_to_even_bf16",
            },
            "epsilon": EPSILON,
            "exclusions": (
                "one_batch_extent_only; remaining_upstream_batch_sizes_are_a_portfolio_question_"
                "this_contract_does_not_examine; no_framework_ABI_cross_shape_or_serving_claim"),
            "materialization": {
                "generator": "python_random_Random_per_input_case_seed_plus_10000_times_ABI_index",
                "rounding": "round_bf16_after_generation",
                "uniform": "uniform[-2,2]", "zeros": "alternating_signed_zero",
                "near_zero": "uniform[-1e-4,1e-4]",
                "alternating": "alternating_sign_times_1_plus_i_mod_17_over_16",
                "mixed_magnitude": "signed_powers_of_two_exponent[-12,8]",
                "weight": "uniform[-1.5,1.5]_every_17th_zero",
            },
        },
        "oracle": {
            "kind": "real_rmsnorm_fsum_then_bf16",
            "callable": "open_cake_ir.tasks.solx_fib.workload.reference_outputs",
            "implementation": "independent_standard_library_math_no_compiler_or_generated_source",
            "output_rounding": "round_to_nearest_ties_to_even",
        },
        "validation": {
            "primary_case": "primary", "all_cases_required": True, "equal_nan": False,
            "comparison": "elementwise_atol_rtol", "atol": ATOL, "rtol": RTOL,
            "qualification": "all_input_cases_at_one_batch_extent; B300_device_compile_correctness_timing_profiler_and_framework_evaluation_pending",
            "tolerance_rationale": (
                "BF16 has an 8-bit significand, so rtol 2**-7 is two output ulp: enough for an "
                "FP32 reduction order and an rsqrt approximation that both differ from the fsum "
                "oracle before the final BF16 rounding. atol 2**-16 covers outputs below that "
                "relative floor. The upstream pack gates on matched_ratio 0.99 at atol=rtol=1e-2, "
                "which admits 1 percent wrong elements; every element is compared here, so this "
                "is a stricter gate and not the same gate. Not a device calibration."),
        },
    }


def validate_solx_fib_contract(document: Mapping[str, object]) -> None:
    """Admit only this exact typed task, backend, ABI, numerical domain and case protocol."""
    workload = WorkloadContract(document)
    task_name = next((name for name, (operator, revision) in TASKS.items()
                      if document.get("operator") == operator
                      and document.get("revision") == revision), None)
    semantics = document.get("semantics")
    backend = backend_for_target(semantics.get("target") if isinstance(semantics, Mapping) else None)
    if task_name is None or backend is None or workload.case_ids != tuple(CASES):
        raise ValueError("SoL-ExecBench operator/revision, backend or required input cases differ")
    shape = workload.case("primary")["shape"]
    if set(shape) != {"R", "C"}:
        raise ValueError("SoL-ExecBench normalization case requires R/C dimensions")
    expected = workload_document(task_name, rows=shape["R"], columns=shape["C"], backend=backend)
    # Canonical JSON equality distinguishes bools from integers and forbids extra fields.
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("SoL-ExecBench frozen semantic/ABI/input/validation contract differs")
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float]]:
    """Return deterministic BF16-representable inputs in the contract's ABI order."""
    validate_solx_fib_contract(workload.document)
    case = workload.case(case_id)
    result = {}
    args = [arg for arg in workload.tensor_abi(case_id) if arg.mode == "input"]
    for position, arg in enumerate(args):
        rng = random.Random(case["seed"] + position * 10000)
        values = []
        for index in range(math.prod(arg.shape)):
            if arg.name == "weight":
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
            values.append(_round(value, "bf16"))
        result[arg.name] = values
    return result


def reference_outputs(workload: WorkloadContract, case_id: str,
                      inputs: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Independent fsum oracle over the real-valued definition, rounded once to BF16.

    Every BF16 input is exact in the wider Python float, so the oracle performs no input
    conversion of its own. It does not emulate a particular reduction tree or rsqrt; the
    contract's predeclared tolerance owns that difference.
    """
    validate_solx_fib_contract(workload.document)
    abi = workload.tensor_abi(case_id)
    args = tuple(arg for arg in abi if arg.mode == "input")
    values, weights = _checked_inputs(workload, args, inputs)
    width = args[0].shape[-1]
    epsilon = workload.document["semantics"]["epsilon"]
    result = []
    for start in range(0, len(values), width):
        row = values[start:start + width]
        inverse = 1.0 / math.sqrt(math.fsum(value * value for value in row) / width + epsilon)
        result.extend(_round(value * inverse * weight, "bf16")
                      for value, weight in zip(row, weights))
    return {"out": result}
