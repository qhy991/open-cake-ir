"""Independent, standard-library mathematics and input data for three tile workloads.

No Compiler, Lab, Torch, generated source, or candidate implementation is imported.
Vectors are flat row-major values already rounded to their declared input dtype.
"""

from __future__ import annotations

import math
import random
import struct
import sys
from collections.abc import Mapping, Sequence

from .workload import TensorABI, WorkloadContract, _name, _object


def source_schedule_path(document: Mapping[str, object]) -> str:
    """Validate the tile Workload's source projection before any consumer uses it."""

    provenance = document.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("tile provenance must be a non-empty list of objects")
    sources = []
    for index, value in enumerate(provenance):
        context = f"tile provenance[{index}]"
        entry = _object(value, context)
        if entry.get("kind") == "source_schedule":
            for field in ("path", "source_commit", "scope"):
                _name(entry.get(field), f"{context}.{field}")
            sources.append(entry["path"])
    if len(sources) != 1:
        raise ValueError("tile provenance requires one visible source Schedule")
    return sources[0]


def validate_tile_contract(document: Mapping[str, object]) -> None:
    """Check supported mathematical/ABI invariants, not a second contract catalogue."""

    workload = WorkloadContract(document)
    source_schedule_path(document)
    operator = document["operator"]
    if document["revision"] != "1" or document["workload_id"] != str(operator).replace("_", "-") + "-v1":
        raise ValueError("tile workload identity or revision differs")
    semantics = _object(document["semantics"], "tile semantics")
    oracle = _object(document["oracle"], "tile oracle")
    validation = _object(document["validation"], "tile validation")
    tensors = _object(document["tensors"], "tile tensors")
    if (
        semantics.get("target") != "sm_100a"
        or semantics.get("input_effects") != "unchanged"
        or semantics.get("output_storage") != "fresh_contiguous_nonaliasing"
        or oracle.get("callable") != "open_cake_ir.evaluation.tile_workloads.reference_outputs"
        or oracle.get("output_rounding") != "round_to_nearest_ties_to_even"
        or validation.get("primary_case") not in workload.case_ids
        or validation.get("all_cases_required") is not True
        or validation.get("equal_nan") is not False
    ):
        raise ValueError("tile workload evaluation boundary differs")
    for name in ("atol", "rtol"):
        value = validation.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= sys.float_info.max:
            raise ValueError(f"tile workload validation.{name} must be finite and nonnegative")
    for case_id in workload.case_ids:
        if workload.case(case_id)["seed"] is None:
            raise ValueError("tile materialization requires an explicit deterministic seed")
        abi = workload.tensor_abi(case_id)
        inputs = tuple(arg for arg in abi if arg.mode == "input")
        outputs = tuple(arg for arg in abi if arg.mode == "output")
        if len(outputs) != 1:
            raise ValueError("tile workload has exactly one output")
        for arg in abi:
            descriptor = _object(tensors[arg.name], "tile tensor")
            if arg.dtype != "int32" and descriptor.get("finite_only") is not True:
                raise ValueError("tile floating tensors must be finite")
            if arg.mode == "input" and arg.dtype != "int32":
                bound = descriptor.get("max_abs")
                if isinstance(bound, bool) or not isinstance(bound, (float, int)) or not 0 < bound <= 16 or _round(bound, arg.dtype) != bound:
                    raise ValueError("tile inputs require a dtype-representable magnitude bound at most 16")
        mode = workload.case(case_id)["mode"]
        arithmetic = semantics.get("arithmetic")
        if operator == "rmsnorm_fp32":
            if (
                tuple(arg.dtype for arg in abi) != ("fp32", "fp32", "fp32")
                or len(inputs[0].shape) != 3
                or inputs[1].shape != inputs[0].shape[-1:]
                or outputs[0].shape != inputs[0].shape
                or mode not in {"uniform", "zeros", "near_zero", "alternating"}
                or oracle.get("kind") != "real_rmsnorm_fsum_then_fp32"
                or arithmetic != {"intermediates": "fp32", "reduction_order": "backend_defined_within_tolerance", "rsqrt": "backend_approximation_within_tolerance", "output": "fp32"}
            ):
                raise ValueError("RMSNorm tensor geometry, arithmetic or oracle differs")
            epsilon = semantics.get("epsilon")
            if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not 0 < epsilon <= 3.4028234663852886e38 or _round(epsilon, "fp32") == 0:
                raise ValueError("RMSNorm epsilon must remain positive and finite in FP32")
        elif operator == "gemm_bias_bf16_fp32":
            if (
                tuple(arg.dtype for arg in abi) != ("bf16", "bf16", "fp32", "fp32")
                or len(inputs[0].shape) != 2 or len(inputs[1].shape) != 2
                or inputs[0].shape[1] != inputs[1].shape[1]
                or inputs[2].shape != inputs[1].shape[:1]
                or outputs[0].shape != (inputs[0].shape[0], inputs[1].shape[0])
                or semantics.get("rhs_storage") != "N_K"
                or semantics.get("padding") != "masked_K_loads_are_zero_and_M_N_tail_stores_are_suppressed"
                or mode not in {"uniform", "zeros", "alternating"}
                or oracle.get("kind") != "real_gemm_bias_fsum_then_fp32"
                or arithmetic != {"operands": "bf16", "accumulator": "fp32", "reduction_order": "backend_defined_within_tolerance", "bias": "fp32_after_contraction", "output": "fp32"}
            ):
                raise ValueError("GEMM tensor geometry, arithmetic or oracle differs")
        elif operator == "indexed_gather_bf16":
            if (
                tuple(arg.dtype for arg in abi) != ("bf16", "int32", "int32", "bf16")
                or len(inputs[0].shape) != 3 or len(inputs[1].shape) != 2
                or inputs[1].shape != inputs[2].shape
                or outputs[0].shape != (*inputs[1].shape, inputs[0].shape[-1])
                or semantics.get("indexing") != {"pairing": "zipped", "valid": "0 <= expert_id < E and 0 <= row_id < R", "invalid": "positive_zero", "negative_indices": "invalid_no_wraparound"}
                or mode not in {"uniform", "index_boundaries", "repeated_indices"}
                or oracle.get("kind") != "paired_index_selection"
                or arithmetic != {"values": "bf16_copy_without_arithmetic", "indices": "signed_int32", "output": "bf16"}
            ):
                raise ValueError("gather tensor geometry, indexing or oracle differs")
        else:
            raise ValueError("unsupported tile workload")
    comparison = "bitwise_bf16" if operator == "indexed_gather_bf16" else "elementwise_atol_rtol"
    if validation.get("comparison") != comparison:
        raise ValueError("tile comparison differs")
    if comparison == "bitwise_bf16" and (validation["atol"] != 0 or validation["rtol"] != 0):
        raise ValueError("gather requires exact comparison")


def _round(value: float, dtype: str) -> float:
    """IEEE FP32 and BF16 round-to-nearest, ties-to-even, without tensor libraries."""

    raw = struct.pack("<f", value)
    if dtype == "bf16":
        bits = struct.unpack("<I", raw)[0]
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        raw = struct.pack("<I", bits)
    return struct.unpack("<f", raw)[0]


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float | int]]:
    """Return deterministic inputs in ABI order as flat, dtype-rounded CPU values.

    Consumers reshape/allocate using ``workload.tensor_abi(case_id)``. This routine
    never imports a device runtime or allocates outputs and is also usable by R2.
    """

    document = workload.document
    operator = document["operator"]
    case = workload.case(case_id)
    rng = random.Random(case["seed"])
    mode = case["mode"]
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    result: dict[str, list[float | int]] = {}
    for position, arg in enumerate(args):
        if arg.dtype == "int32":
            continue
        bound = document["tensors"][arg.name]["max_abs"]
        values = []
        for index in range(math.prod(arg.shape)):
            if mode == "zeros":
                value = 0.0
            elif mode == "near_zero":
                value = (min(bound, 1e-8) * (1 if index % 2 else -1)) if position == 0 else min(bound, 1.0)
            elif mode == "alternating":
                value = (bound if index % 2 else -bound) if position == 0 else min(bound, 1.0 if position == 1 else 0.25)
            else:
                value = (rng.random() * 2.0 - 1.0) * min(1.0, bound)
            values.append(_round(value, arg.dtype))
        result[arg.name] = values
    if operator == "indexed_gather_bf16":
        experts, rows, _ = args[0].shape
        boundary_pairs = (
            (0, 0), (experts - 1, rows - 1), (-1, 0), (experts, 0),
            (0, -1), (0, rows), (-(1 << 31), 0), ((1 << 31) - 1, 0),
            (0, -(1 << 31)), (0, (1 << 31) - 1), (experts - 1, 0), (0, rows - 1),
        )
        pairs = []
        for index in range(math.prod(args[1].shape)):
            if mode == "index_boundaries":
                pair = boundary_pairs[index % len(boundary_pairs)]
            elif mode == "repeated_indices":
                pair = (0, rows - 1)
            else:
                pair = (rng.randrange(experts), rng.randrange(rows))
            pairs.append(pair)
        result[args[1].name] = [pair[0] for pair in pairs]
        result[args[2].name] = [pair[1] for pair in pairs]
    elif operator not in {"rmsnorm_fp32", "gemm_bias_bf16_fp32"}:
        raise ValueError("unsupported tile workload")
    return result


def _checked_inputs(workload: WorkloadContract, args: tuple[TensorABI, ...], values: Mapping[str, Sequence[float | int]]) -> list[list[float | int]]:
    if not isinstance(values, Mapping) or set(values) != {arg.name for arg in args}:
        raise ValueError("oracle inputs must match the complete input ABI")
    tensors = workload.document["tensors"]
    checked = []
    for arg in args:
        vector = values[arg.name]
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or len(vector) != math.prod(arg.shape):
            raise ValueError(f"oracle input {arg.name} must be a complete flat row-major vector")
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"oracle input {arg.name} must be numeric")
            if arg.dtype == "int32":
                if not isinstance(value, int) or not -(1 << 31) <= value < (1 << 31):
                    raise ValueError(f"oracle input {arg.name} must contain signed int32 IDs")
            elif abs(value) > tensors[arg.name]["max_abs"] or not math.isfinite(value) or _round(value, arg.dtype) != value:
                raise ValueError(f"oracle input {arg.name} must contain bounded, finite {arg.dtype} values")
        checked.append(list(vector))
    return checked


def reference_outputs(workload: WorkloadContract, case_id: str, inputs: Mapping[str, Sequence[float | int]]) -> dict[str, list[float]]:
    """Compute the independent mathematical oracle, without consulting a Schedule.

    RMSNorm and GEMM use high-precision ``math.fsum`` and final FP32 rounding;
    they intentionally do not emulate a particular GPU reduction tree. The Workload
    owns their all-element tolerances. Gather is exact selection, including zero sign.
    Tiny cases are suitable for CPU tests; the primary GEMM is not a fast CPU benchmark.
    """

    abi = workload.tensor_abi(case_id)
    args = tuple(arg for arg in abi if arg.mode == "input")
    output, = (arg for arg in abi if arg.mode == "output")
    values = _checked_inputs(workload, args, inputs)
    semantics = workload.document["semantics"]
    operator = workload.document["operator"]
    result: list[float] = []
    if operator == "rmsnorm_fp32":
        x, gamma = values
        width = args[0].shape[-1]
        for start in range(0, len(x), width):
            row = x[start:start + width]
            inverse = 1.0 / math.sqrt(math.fsum(value * value for value in row) / width + semantics["epsilon"])
            result.extend(_round(value * inverse * weight, "fp32") for value, weight in zip(row, gamma))
    elif operator == "gemm_bias_bf16_fp32":
        a, b, bias = values
        m, k = args[0].shape
        n = args[1].shape[0]
        for row in range(m):
            for column in range(n):
                dot = math.fsum(a[row * k + inner] * b[column * k + inner] for inner in range(k))
                result.append(_round(dot + bias[column], "fp32"))
    elif operator == "indexed_gather_bf16":
        source, expert_ids, row_ids = values
        experts, rows, width = args[0].shape
        if semantics["indexing"]["invalid"] != "positive_zero":
            raise ValueError("unsupported invalid-index policy")
        for expert, row in zip(expert_ids, row_ids):
            if 0 <= expert < experts and 0 <= row < rows:
                start = (expert * rows + row) * width
                result.extend(source[start:start + width])
            else:
                result.extend([0.0] * width)
    else:
        raise ValueError("unsupported tile workload")
    return {output.name: result}
