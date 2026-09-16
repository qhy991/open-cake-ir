"""SoL-ExecBench tasks whose definitions come from FlashInfer-Bench.

One task here binds one upstream single-operator definition to one batch extent, its own
BF16 ABI and its own independent CPU oracle. The upstream pack is semantic provenance:
no candidate source, no upstream latency and no upstream score is inherited, and the
upstream's ``matched_ratio`` gate is deliberately replaced by an all-element comparison
(ADR 0065). The ``solx_l1`` sibling owns the SOL-ExecBench L1 subset, which is a
different upstream authority and a separate Executor closure.

Every hardware-dependent question is asked of the Target and the route rather than
answered here: which dtype a route can name, which operation kinds a Target admits, and
which row widths a route can tile. A task is therefore available on exactly the devices
that can express it, and unavailable by a refusal that names its own reason.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from functools import lru_cache

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.tiles.workload import _checked_inputs, _round
from open_cake_ir.tasks.devices import (
    BACKENDS, admit_dtype, admit_operations, admit_width, backend_for_target,
)

# Upstream pack task id -> (this project's operator, revision). The `fib_` prefix names
# the upstream authority so a reader of the operator table never has to open a file to
# learn where a task came from.
TASKS = {
    "fib_fused_add_rmsnorm_h2048": ("solx_fib_fused_add_rmsnorm_h2048_bf16", "1"),
    "fib_fused_add_rmsnorm_h4096": ("solx_fib_fused_add_rmsnorm_h4096_bf16", "1"),
    "fib_fused_add_rmsnorm_h7168": ("solx_fib_fused_add_rmsnorm_h7168_bf16", "1"),
    "fib_rmsnorm_h128": ("solx_fib_rmsnorm_h128_bf16", "1"),
    "fib_rmsnorm_h512": ("solx_fib_rmsnorm_h512_bf16", "1"),
    "fib_rmsnorm_h1536": ("solx_fib_rmsnorm_h1536_bf16", "1"),
    "fib_rmsnorm_h2048": ("solx_fib_rmsnorm_h2048_bf16", "1"),
    "fib_rmsnorm_h4096": ("solx_fib_rmsnorm_h4096_bf16", "1"),
    "fib_rmsnorm_h7168": ("solx_fib_rmsnorm_h7168_bf16", "1"),
}
# Everything the upstream definition fixes. `hidden` is the task's identity, not a
# parameter: the same operator at another hidden size is a different upstream task, and
# `epsilon` genuinely differs between them -- the two Llama-3.1-8B captures use 1e-5 and
# every other capture here uses 1e-6, so a single family constant would be wrong for
# seven of these nine. `batches` is the upstream workload axis, recorded because this
# contract binds exactly one of its values.
SPECS = {
    "fib_fused_add_rmsnorm_h2048": {
        "upstream": "tasks/001_fused_add_rmsnorm_h2048",
        "definition": "upstream/definitions/rmsnorm/fused_add_rmsnorm_h2048.json",
        "model": "Qwen/Qwen3-30B-A3B", "hidden": 2048, "epsilon": 1e-6, "residual": True,
        "batches": (1, 6, 34, 64, 79, 12383, 16254), "seed": 2601,
    },
    "fib_fused_add_rmsnorm_h4096": {
        "upstream": "tasks/002_fused_add_rmsnorm_h4096",
        "definition": "upstream/definitions/rmsnorm/fused_add_rmsnorm_h4096.json",
        "model": "meta-llama/Llama-3.1-8B", "hidden": 4096, "epsilon": 1e-5, "residual": True,
        "batches": (1, 7, 15, 16, 34, 63, 64, 79, 170, 8804, 10827, 11832, 14418, 14509),
        "seed": 2611,
    },
    "fib_fused_add_rmsnorm_h7168": {
        "upstream": "tasks/003_fused_add_rmsnorm_h7168",
        "definition": "upstream/definitions/rmsnorm/fused_add_rmsnorm_h7168.json",
        "model": "deepseek-ai/DeepSeek-V3", "hidden": 7168, "epsilon": 1e-6, "residual": True,
        "batches": (1, 7, 18, 32, 64, 539, 11949, 14521), "seed": 2621,
    },
    "fib_rmsnorm_h128": {
        "upstream": "tasks/021_rmsnorm_h128",
        "definition": "upstream/definitions/rmsnorm/rmsnorm_h128.json",
        "model": "Qwen/Qwen3-30B-A3B", "hidden": 128, "epsilon": 1e-6, "residual": False,
        "batches": (4, 24, 32, 136, 192, 256, 316, 1088, 2048, 2528, 49532, 65016, 396256, 520128),
        "seed": 2631,
    },
    "fib_rmsnorm_h512": {
        "upstream": "tasks/022_rmsnorm_h512",
        "definition": "upstream/definitions/rmsnorm/rmsnorm_h512.json",
        "model": "deepseek-ai/DeepSeek-V3", "hidden": 512, "epsilon": 1e-6, "residual": False,
        "batches": (1, 7, 18, 32, 64, 539, 11949, 14521), "seed": 2641,
    },
    "fib_rmsnorm_h1536": {
        "upstream": "tasks/023_rmsnorm_h1536",
        "definition": "upstream/definitions/rmsnorm/rmsnorm_h1536.json",
        "model": "deepseek-ai/DeepSeek-V3", "hidden": 1536, "epsilon": 1e-6, "residual": False,
        "batches": (1, 7, 18, 32, 64, 539, 11949, 14521), "seed": 2651,
    },
    "fib_rmsnorm_h2048": {
        "upstream": "tasks/024_rmsnorm_h2048",
        "definition": "upstream/definitions/rmsnorm/rmsnorm_h2048.json",
        "model": "Qwen/Qwen3-30B-A3B", "hidden": 2048, "epsilon": 1e-6, "residual": False,
        "batches": (1, 6, 34, 64, 79, 12383, 16254), "seed": 2661,
    },
    "fib_rmsnorm_h4096": {
        "upstream": "tasks/025_rmsnorm_h4096",
        "definition": "upstream/definitions/rmsnorm/rmsnorm_h4096.json",
        "model": "meta-llama/Llama-3.1-8B", "hidden": 4096, "epsilon": 1e-5, "residual": False,
        "batches": (1, 7, 15, 16, 34, 63, 64, 79, 170, 8804, 10827, 11832, 14418, 14509),
        "seed": 2501,
    },
    "fib_rmsnorm_h7168": {
        "upstream": "tasks/026_rmsnorm_h7168",
        "definition": "upstream/definitions/rmsnorm/rmsnorm_h7168.json",
        "model": "deepseek-ai/DeepSeek-V3", "hidden": 7168, "epsilon": 1e-6, "residual": False,
        "batches": (1, 7, 18, 32, 64, 539, 11949, 14521), "seed": 2671,
    },
}
PACK = "flashinfer-bench-tasks"
# Directories in the pack that hold complete low-level target implementations. They are
# declared per contract so a restricted arm's refusal is a property of the contract and
# not a convention someone has to remember (ADR 0062).
RESTRICTED_ROOTS = ("{upstream}/baseline/", "dcu_sol/{task}/")
CASES = {
    "primary": ("uniform", 0),
    "zeros": ("zeros", 1),
    "near_zero": ("near_zero", 2),
    "alternating": ("alternating", 3),
    "mixed_magnitude": ("mixed_magnitude", 4),
}
# The default batch extent is the largest upstream value whose CPU oracle stays tractable:
# this contract's cases are correctness cases evaluated element by element in Python, and
# the upstream's prefill-sized extents run to tens of millions of elements. The extent is
# therefore chosen for what the oracle can serve, NOT for saturating memory bandwidth, and
# every contract says so in its own `exclusions`.
SEED_ELEMENT_CAP = 1 << 20
# BF16 carries an 8-bit significand, so one ulp is 2**-8 relative. The allowance is two of
# them, plus an absolute floor for outputs below that relative scale.
RTOL = 2.0 ** -7
ATOL = 2.0 ** -16
# What a Schedule for this family has to be able to say.
OPERATION_KINDS = ("load", "cast", "elementwise", "reduce", "store")


def default_rows(task_name: str) -> int:
    """Return the largest declared upstream batch extent the CPU oracle can serve."""
    spec = SPECS[task_name]
    admitted = [batch for batch in spec["batches"] if batch * spec["hidden"] <= SEED_ELEMENT_CAP]
    if not admitted:
        raise ValueError(f"{task_name} declares no batch extent within the oracle element cap")
    return max(admitted)


@lru_cache(maxsize=None)
def admitting_backends(task_name: str) -> tuple[str, ...]:
    """Name every registered backend that can express this task, possibly none.

    None is a real answer: three of these upstream tasks have a hidden size that is not a
    power of two, which the Triton route cannot tile, and no registered backend lowers
    this family any other way today. Reporting that is the point; a task list filtered to
    hide it would say the pack is smaller than it is.
    """
    admitted = []
    for backend in BACKENDS:
        try:
            workload_document(task_name, rows=default_rows(task_name),
                              columns=SPECS[task_name]["hidden"], backend=backend)
        except ValueError:
            continue
        admitted.append(backend)
    return tuple(admitted)


def launchable_tasks() -> tuple[str, ...]:
    """Return the tasks at least one registered backend admits, in declaration order."""
    return tuple(name for name in TASKS if admitting_backends(name))


def workload_document(task_name: str, *, rows: int, columns: int,
                      backend: str = "triton-b300") -> dict:
    """Create one frozen Workload document; callers persist it outside source."""
    if not isinstance(task_name, str) or task_name not in TASKS or backend not in BACKENDS:
        raise ValueError("unsupported SoL-ExecBench FlashInfer-Bench task/backend")
    spec = SPECS[task_name]
    hidden = spec["hidden"]
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
    residual = spec["residual"]
    tensors = {"x": {"shape": ["R", "C"], "max_abs": 256.0}}
    if residual:
        tensors["residual"] = {"shape": ["R", "C"], "max_abs": 256.0}
    tensors["weight"] = {"shape": ["C"], "max_abs": 1.5}
    inputs = list(tensors)
    tensors["out"] = {"shape": ["R", "C"]}
    for tensor in tensors.values():
        tensor.update(dtype="bf16", layout="contiguous_row_major", finite_only=True)
    arithmetic = {
        "operands": "bf16", "intermediates": "fp32",
        "reduction_order": "backend_defined_within_tolerance",
        "rsqrt": "backend_approximation_within_tolerance",
        "output": "round_to_nearest_ties_to_even_bf16",
    }
    if residual:
        arithmetic["residual_addition"] = "round_to_nearest_ties_to_even_fp32_before_normalization"
    definition = ("z[r,c] = round_fp32(x[r,c] + residual[r,c]); "
                  "out[r,c] = round_bf16(z[r,c] * weight[c] / sqrt(mean_j(z[r,j]^2) + epsilon))"
                  ) if residual else (
                  "out[r,c] = round_bf16(x[r,c] * weight[c] "
                  "/ sqrt(mean_j(x[r,j]^2) + epsilon))")
    materialization = {
        "generator": "python_random_Random_per_input_case_seed_plus_10000_times_ABI_index",
        "rounding": "round_bf16_after_generation",
        "uniform": "uniform[-2,2]", "zeros": "alternating_signed_zero",
        "near_zero": "uniform[-1e-4,1e-4]",
        "alternating": "alternating_sign_times_1_plus_i_mod_17_over_16",
        "mixed_magnitude": "signed_powers_of_two_exponent[-12,8]",
        "weight": "uniform[-1.5,1.5]_every_17th_zero",
    }
    task_id = spec["upstream"].split("/")[-1]
    return {
        "schema_version": 1,
        "workload_id": f"solx-fib-{task_name[4:].replace('_', '-')}-bf16-{backend}-r{rows}-v{revision}",
        "revision": revision, "state": "frozen", "operator": operator,
        "provenance": [
            {"kind": "solx_pack_task", "upstream": "flashinfer-bench",
             "path": f"{PACK}/{spec['upstream']}", "definition": spec["definition"],
             "model": spec["model"], "upstream_batch_axis": list(spec["batches"]),
             "scope": "semantic_source_only; no_upstream_candidate_source_latency_or_score_is_inherited"},
            *({"kind": "restricted_artifact",
               "path": f"{PACK}/{root.format(upstream=spec['upstream'], task=task_id)}",
               "role": "complete_low_level_target_implementation",
               "scope": "black_box_baseline_only; not_readable_by_clean_start_or_direct_low_level_arm"}
              for root in RESTRICTED_ROOTS),
            {"kind": "standalone_derived_contract",
             "path": "src/open_cake_ir/tasks/solx_fib/workload.py",
             "scope": f"fixed_shape_BF16_{device['provenance_token']}_semantics_and_CPU_oracle"},
        ],
        "cases": [{"case_id": name, "shape": {"R": rows, "C": columns},
                   "seed": spec["seed"] + offset, "mode": mode}
                  for name, (mode, offset) in CASES.items()],
        "tensors": tensors,
        "semantics": {
            "definition": definition, "target": device["target"],
            "candidate_abi": {"inputs": inputs, "outputs": ["out"]},
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "arithmetic": arithmetic, "epsilon": spec["epsilon"],
            "exclusions": (
                "one_batch_extent_only; the extent is the largest declared upstream batch "
                "whose element-by-element CPU oracle stays tractable, chosen for oracle "
                "reach and not for memory-bandwidth saturation, so the upstream's larger "
                "extents are neither examined nor claimed; no_framework_ABI_cross_shape_"
                "or_serving_claim"),
            "materialization": materialization,
        },
        "oracle": {
            "kind": ("real_fused_add_rmsnorm_fsum_then_bf16" if residual
                     else "real_rmsnorm_fsum_then_bf16"),
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
    conversion of its own. The fused variant rounds its residual sum to FP32 first, which
    is the one intermediate the upstream definition materializes. Neither emulates a
    particular reduction tree or rsqrt; the contract's predeclared tolerance owns that.
    """
    validate_solx_fib_contract(workload.document)
    abi = workload.tensor_abi(case_id)
    args = tuple(arg for arg in abi if arg.mode == "input")
    checked = dict(zip((arg.name for arg in args), _checked_inputs(workload, args, inputs)))
    values, weights = checked["x"], checked["weight"]
    if "residual" in checked:
        values = [_round(x + residual, "fp32") for x, residual in zip(values, checked["residual"])]
    width = args[0].shape[-1]
    epsilon = workload.document["semantics"]["epsilon"]
    result = []
    for start in range(0, len(values), width):
        row = values[start:start + width]
        inverse = 1.0 / math.sqrt(math.fsum(value * value for value in row) / width + epsilon)
        result.extend(_round(value * inverse * weight, "bf16")
                      for value, weight in zip(row, weights))
    return {"out": result}
