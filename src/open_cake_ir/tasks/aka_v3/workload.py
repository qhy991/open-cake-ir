"""Three source-complete AKA v3 workloads with independent CPU oracles.

The records named here were fixed-B200 qualified in AKA.  This module does not reuse that
node evidence: it gives each derived, fixed-shape Workload its own semantic and oracle
authority, and leaves Open-Cake GPU correctness, timing, profiling and release pending.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence

from open_cake_ir.evaluation.workload import TensorABI, WorkloadContract
from open_cake_ir.tasks.devices import BACKENDS, backend_for_target
from open_cake_ir.tasks.tiles.workload import _round


AKA_V3_RECORDS = "AKA/datasets/curated/cuda_kernel_mechanism_qualified_v3/records.jsonl"
TASKS = {
    "residual_layernorm": (
        "aka_residual_layernorm_fp32", "1",
        "normalization__normalization_layernorm__analysis__l000016_b200_v1__directderived_sol_ultra_v2_b200_claude_opus5_high_aug_v1",
        "fused_residual_layernorm_rowwise_fp32",
    ),
    "gemm_nt_bias": (
        "aka_gemm_nt_bias_fp32", "1",
        "linear_algebra_and_convolution__gemm_matmul__optimization_positive__l000001_b200_v1__directderived_sol_ultra_v2_b200_claude_opus5_high_aug_v1",
        "gemm_nt_bias_tile8x8_fp32_i32_v1",
    ),
    "row_gather": (
        "aka_row_gather_fp32", "1",
        "data_movement_and_layout__gather_scatter__generation__l000007_b200_v1__directderived_sol_ultra_v2_b200_aug_v1",
        "row_gather_contiguous_fp32_i32_v1",
    ),
    "histogram": (
        "aka_histogram_fp32", "1",
        "quantization__quantization_int8__analysis__l000007_b200_v1__directderived_sol_ultra_v2__parent_completion_attempt0001_b200_claude_opus5_high_aug_v1",
        "histogram1d_contiguous_fp32_counts_i32_global_b256_v1",
    ),
    "max_pool1d": (
        "aka_max_pool1d_nwc_fp32", "1",
        "data_movement_and_layout__gather_scatter__analysis__l000139_b200_v1__directderived_sol_ultra_v2__parent_completion_v1_b200_aug_v1",
        "max_pool1d_nwc_fp32_int32_block256",
    ),
    "momentum_sgd": (
        "aka_momentum_sgd_fp32", "1",
        "data_movement_and_layout__gather_scatter__analysis__l000164_b200_v1__directderived_sol_ultra_v2_b200_claude_opus5_high_aug_v1",
        "momentum_sgd_update_contiguous_fp32_i32_block256_v1",
    ),
}
CASES = {
    "residual_layernorm": {
        "primary": ("uniform", 9101), "zeros": ("zeros", 9102),
        "near_zero": ("near_zero", 9103), "alternating": ("alternating", 9104),
        "mixed_magnitude": ("mixed_magnitude", 9105),
    },
    "gemm_nt_bias": {
        "primary": ("uniform", 9101), "zeros": ("zeros", 9102),
        "near_zero": ("near_zero", 9103), "alternating": ("alternating", 9104),
        "mixed_magnitude": ("mixed_magnitude", 9105),
    },
    "row_gather": {
        "primary": ("uniform", 9101), "zeros": ("zeros", 9102),
        "alternating": ("alternating", 9103),
        "boundary_indices": ("boundary_indices", 9104),
        "repeated_indices": ("repeated_indices", 9105),
    },
    "histogram": {
        "primary": ("uniform", 9201), "hot_bin": ("hot_bin", 9202),
        "lower_edge": ("lower_edge", 9203), "upper_edge": ("upper_edge", 9204),
        "out_of_range": ("out_of_range", 9205),
    },
    "max_pool1d": {
        "primary": ("uniform", 9301), "zeros": ("zeros", 9302),
        "alternating": ("alternating", 9303), "negative_only": ("negative_only", 9304),
        "padded_edges": ("padded_edges", 9305),
    },
    "momentum_sgd": {
        "primary": ("nesterov_off", 9401), "nesterov_on": ("nesterov_on", 9402),
        "zeros": ("zeros", 9403), "alternating": ("alternating", 9404),
        "mixed_magnitude": ("mixed_magnitude", 9405),
    },
}
EPSILON = 1e-5


def _task(task_name: str) -> tuple[str, str, str, str]:
    if not isinstance(task_name, str) or task_name not in TASKS:
        raise ValueError("unsupported AKA v3 migration task")
    return TASKS[task_name]


def workload_document(
    task_name: str, *, backend: str = "triton-b200", rows: int = 8,
    columns: int = 256, depth: int = 32, source_rows: int = 32,
    output_rows: int = 24, bins: int = 16, batch: int = 2, input_length: int = 16,
    output_length: int = 8, channels: int = 8, kernel_size: int = 3, stride: int = 2,
    pad: int = 1, elements: int = 1024,
) -> dict:
    """Return a frozen, standalone derived contract for one selected v3 record.

    The shapes are deliberately modest CPU-oracle fixtures.  They are not the former
    campaign shapes, a portability claim, or a benchmark configuration.
    """
    operator, revision, record_id, parent_id = _task(task_name)
    if backend != "triton-b200" or backend not in BACKENDS:
        raise ValueError("AKA v3 derived contracts bind the original B200 target only")
    if any(type(value) is not int or value <= 0 for value in
           (rows, columns, depth, source_rows, output_rows, bins, batch, input_length,
            output_length, channels, kernel_size, stride, elements)) or type(pad) is not int or pad < 0:
        raise ValueError("AKA v3 task dimensions must be positive integers")
    if task_name == "gemm_nt_bias" and (rows * depth + columns * depth + rows * columns) * 4 > 2**31 - 1:
        raise ValueError("GEMM buffers exceed the standalone FP32 ABI")
    if task_name != "gemm_nt_bias" and rows * columns * 4 > 2**31 - 1:
        raise ValueError("rank-two buffers exceed the standalone FP32 ABI")
    if task_name == "row_gather" and (source_rows + output_rows) * columns * 4 > 2**31 - 1:
        raise ValueError("gather buffers exceed the standalone FP32 ABI")

    if task_name == "residual_layernorm":
        tensors = {
            "input_a": {"shape": ["R", "C"], "max_abs": 2.0},
            "input_b": {"shape": ["R", "C"], "max_abs": 2.0},
            "weight": {"shape": ["C"], "max_abs": 1.5},
            "bias": {"shape": ["C"], "max_abs": 1.5},
            "residual": {"shape": ["R", "C"]},
            "normalized": {"shape": ["R", "C"]},
            "mean": {"shape": ["R"]},
            "reciprocal_stddev": {"shape": ["R"]},
        }
        shape = {"R": rows, "C": columns}
        inputs = ["input_a", "input_b", "weight", "bias"]
        outputs = ["residual", "normalized", "mean", "reciprocal_stddev"]
        definition = (
            "residual[r,c] = round_fp32(input_a[r,c] + input_b[r,c]); "
            "mean[r] = mean_j(residual[r,j]); reciprocal_stddev[r] = "
            "1/sqrt(mean_j((residual[r,j]-mean[r])^2)+epsilon); normalized[r,c] = "
            "(residual[r,c]-mean[r])*reciprocal_stddev[r]*weight[c]+bias[c]"
        )
        arithmetic = {"intermediates": "fp32", "variance": "centered_population", "output": "fp32"}
    elif task_name == "gemm_nt_bias":
        if rows % 8 or columns % 8:
            raise ValueError("the derived GEMM contract retains the visible 8x8 tile divisibility")
        tensors = {
            "input": {"shape": ["M", "K"], "max_abs": 2.0},
            "weight_nt": {"shape": ["N", "K"], "max_abs": 2.0},
            "bias": {"shape": ["N"], "max_abs": 1.5},
            "output": {"shape": ["M", "N"]},
        }
        shape = {"M": rows, "K": depth, "N": columns}
        inputs, outputs = ["input", "weight_nt", "bias"], ["output"]
        definition = "output[m,n] = fsum_k(input[m,k] * weight_nt[n,k]) + bias[n]"
        arithmetic = {"operands": "fp32", "accumulator": "fp32", "rhs_storage": "N_K", "output": "fp32"}
    elif task_name == "row_gather":
        tensors = {
            "input": {"shape": ["S", "C"], "max_abs": 2.0},
            "output_row_to_input_row": {"shape": ["O"], "index_upper_bound": source_rows},
            "output": {"shape": ["O", "C"]},
        }
        shape = {"S": source_rows, "O": output_rows, "C": columns}
        inputs, outputs = ["input", "output_row_to_input_row"], ["output"]
        definition = "output[o,c] = input[output_row_to_input_row[o],c]"
        arithmetic = {"values": "fp32_copy_without_arithmetic", "indices": "signed_int32", "output": "fp32"}
    elif task_name == "histogram":
        tensors = {
            "values": {"shape": ["E"], "max_abs": 8.0},
            "counts": {"shape": ["B"]},
        }
        shape = {"E": elements, "B": bins}
        inputs, outputs = ["values"], ["counts"]
        definition = "counts[b] = cardinality({ i | values[i] in [-4,4] and bin(values[i]) = b })"
        arithmetic = {"values": "fp32", "counts": "fp32_exact_integer_increment", "out_of_range": "ignored", "upper_endpoint": "last_bin"}
    elif task_name == "max_pool1d":
        if output_length != (input_length + 2 * pad - kernel_size) // stride + 1:
            raise ValueError("max-pool output length must match the declared window geometry")
        tensors = {
            "input": {"shape": ["N", "X", "C"], "max_abs": 2.0},
            "output": {"shape": ["N", "Y", "C"]},
        }
        shape = {"N": batch, "X": input_length, "Y": output_length, "C": channels}
        inputs, outputs = ["input"], ["output"]
        definition = "output[n,y,c] = max_{x in [y*stride-pad,y*stride-pad+kernel_size) intersect [0,X)} input[n,x,c]"
        arithmetic = {"values": "fp32", "reduction": "maximum", "padding": "excluded_from_window_not_zero_filled", "output": "fp32"}
    else:
        tensors = {
            "mu": {"shape": [1], "max_abs": 1.0}, "lr": {"shape": [1], "max_abs": 1.0},
            "param": {"shape": ["E"], "max_abs": 2.0}, "grad": {"shape": ["E"], "max_abs": 2.0},
            "moment": {"shape": ["E"], "max_abs": 2.0}, "nesterov": {"shape": [1]},
            "param_out": {"shape": ["E"]}, "moment_out": {"shape": ["E"]},
        }
        shape = {"E": elements}
        inputs, outputs = ["mu", "lr", "param", "grad", "moment", "nesterov"], ["param_out", "moment_out"]
        definition = "m[i] = mu*moment[i]+lr*grad[i]; param_out[i] = param[i]-m[i] if nesterov=0 else param[i]-(1+mu)*m[i]+mu*moment[i]; moment_out[i] = m[i]"
        arithmetic = {"values": "fp32", "state": "out_of_place_parameter_and_momentum", "nesterov": "int32_zero_or_one", "output": "fp32"}

    for name, descriptor in tensors.items():
        descriptor.update(dtype="int32" if name in {"output_row_to_input_row", "nesterov"} else "fp32",
                          layout="contiguous_row_major")
        if descriptor["dtype"] == "fp32":
            descriptor["finite_only"] = True
    device = BACKENDS[backend]
    return {
        "schema_version": 1,
        "workload_id": f"{operator.replace('_', '-')}-{backend}-v{revision}",
        "revision": revision,
        "state": "frozen",
        "operator": operator,
        "provenance": [
            {"kind": "aka_qualified_mechanism_v3_record", "path": AKA_V3_RECORDS,
             "case_id": record_id, "derived_parent_id": parent_id,
             "scope": "semantic_source_only; historical_B200_evidence_does_not_transfer"},
            {"kind": "standalone_derived_contract", "path": "src/open_cake_ir/tasks/aka_v3/workload.py",
             "scope": f"fixed_shape_FP32_{device['provenance_token']}_workload_semantics_and_CPU_oracle"},
        ],
        "cases": [{"case_id": name, "shape": shape, "seed": seed, "mode": mode}
                  for name, (mode, seed) in CASES[task_name].items()],
        "tensors": tensors,
        "semantics": {
            "definition": definition, "target": device["target"],
            "candidate_abi": {"inputs": inputs, "outputs": outputs},
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "arithmetic": arithmetic,
            **({"epsilon": EPSILON} if task_name == "residual_layernorm" else {}),
            **({"range": {"minimum": -4.0, "maximum": 4.0}} if task_name == "histogram" else {}),
            **({"window": {"kernel_size": kernel_size, "stride": stride, "pad": pad}} if task_name == "max_pool1d" else {}),
            "materialization": "deterministic_standard_library_FP32_or_int32_per_case_seed",
            "exclusions": "no_framework_ABI_or_cross_shape_equivalence; no_current_GPU_or_performance_claim",
        },
        "oracle": {
            "kind": f"real_{task_name}_standard_library_reference",
            "callable": "open_cake_ir.tasks.aka_v3.workload.reference_outputs",
            "implementation": "independent_standard_library_math_no_compiler_or_generated_source",
            "output_rounding": "round_to_nearest_ties_to_even",
        },
        "validation": {
            "primary_case": "primary", "all_cases_required": True, "equal_nan": False,
            "comparison": "bitwise_fp32" if task_name == "row_gather" else "elementwise_atol_rtol",
            "atol": 0.0 if task_name == "row_gather" else 2e-5,
            "rtol": 0.0 if task_name == "row_gather" else 2e-5,
            "qualification": "CPU_oracle_only; Open-Cake_B200_compile_correctness_sanitizer_timing_profile_pending",
            "performance_domain": "future_B200_protocol_must_use_four_workloads_spanning_at_least_1000x_work",
        },
    }


def _name_for(document: Mapping[str, object]) -> str:
    return next((name for name, (operator, revision, _, _) in TASKS.items()
                 if document.get("operator") == operator and document.get("revision") == revision), "")


def validate_aka_v3_contract(document: Mapping[str, object]) -> None:
    """Require byte-for-byte agreement with one generated standalone contract."""
    workload = WorkloadContract(document)
    task_name = _name_for(document)
    semantics = document.get("semantics")
    backend = backend_for_target(semantics.get("target") if isinstance(semantics, Mapping) else None)
    if not task_name or backend != "triton-b200" or workload.case_ids != tuple(CASES[task_name]):
        raise ValueError("AKA v3 task identity, target, or case protocol differs")
    shape = workload.case("primary")["shape"]
    if task_name == "gemm_nt_bias":
        expected = workload_document(task_name, backend=backend, rows=shape["M"], depth=shape["K"], columns=shape["N"])
    elif task_name == "row_gather":
        expected = workload_document(task_name, backend=backend, source_rows=shape["S"], output_rows=shape["O"], columns=shape["C"])
    elif task_name == "histogram":
        expected = workload_document(task_name, backend=backend, elements=shape["E"], bins=shape["B"])
    elif task_name == "max_pool1d":
        semantics = document["semantics"]
        window = semantics["window"]
        expected = workload_document(task_name, backend=backend, batch=shape["N"], input_length=shape["X"],
                                     output_length=shape["Y"], channels=shape["C"],
                                     kernel_size=window["kernel_size"], stride=window["stride"], pad=window["pad"])
    elif task_name == "momentum_sgd":
        expected = workload_document(task_name, backend=backend, elements=shape["E"])
    else:
        expected = workload_document(task_name, backend=backend, rows=shape["R"], columns=shape["C"])
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("AKA v3 frozen semantic/ABI/input/validation contract differs")
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def _checked_inputs(workload: WorkloadContract, args: tuple[TensorABI, ...], values: Mapping[str, Sequence[float | int]]) -> dict[str, list[float | int]]:
    if not isinstance(values, Mapping) or set(values) != {arg.name for arg in args}:
        raise ValueError("oracle inputs must match the complete input ABI")
    checked: dict[str, list[float | int]] = {}
    tensors = workload.document["tensors"]
    for arg in args:
        vector = values[arg.name]
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or len(vector) != math.prod(arg.shape):
            raise ValueError(f"oracle input {arg.name} must be a complete flat row-major vector")
        if arg.dtype == "int32":
            if any(type(value) is not int or not -(1 << 31) <= value < 1 << 31 for value in vector):
                raise ValueError(f"oracle input {arg.name} must contain signed int32 values")
            if arg.name == "output_row_to_input_row":
                upper = tensors[arg.name]["index_upper_bound"]
                if any(value < 0 or value >= upper for value in vector):
                    raise ValueError("row-gather indices must remain within the declared source rows")
            if arg.name == "nesterov" and set(vector) - {0, 1}:
                raise ValueError("momentum SGD nesterov flag must be zero or one")
        else:
            bound = tensors[arg.name]["max_abs"]
            if any(type(value) not in {int, float} or not math.isfinite(value) or abs(value) > bound or _round(value, "fp32") != value for value in vector):
                raise ValueError(f"oracle input {arg.name} must contain bounded finite FP32 values")
        checked[arg.name] = list(vector)
    return checked


def materialize_case(workload: WorkloadContract, case_id: str) -> dict[str, list[float | int]]:
    """Materialize standalone input vectors; no device or imported CUDA code is used."""
    validate_aka_v3_contract(workload.document)
    case = workload.case(case_id)
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    result: dict[str, list[float | int]] = {}
    for position, arg in enumerate(args):
        rng = random.Random(case["seed"] + position * 10000)
        if arg.dtype == "int32":
            if arg.name == "nesterov":
                result[arg.name] = [1 if case["mode"] == "nesterov_on" else 0]
            else:
                source_rows = workload.case(case_id)["shape"]["S"]
                if case["mode"] == "boundary_indices":
                    result[arg.name] = [0 if i % 2 == 0 else source_rows - 1 for i in range(math.prod(arg.shape))]
                elif case["mode"] == "repeated_indices":
                    result[arg.name] = [source_rows // 2] * math.prod(arg.shape)
                else:
                    result[arg.name] = [rng.randrange(source_rows) for _ in range(math.prod(arg.shape))]
            continue
        bound = workload.document["tensors"][arg.name]["max_abs"]
        values: list[float] = []
        for index in range(math.prod(arg.shape)):
            if _name_for(workload.document) == "histogram":
                if case["mode"] == "hot_bin":
                    value = 0.0 if index % 20 else rng.uniform(-4.0, 4.0)
                elif case["mode"] == "lower_edge":
                    value = -4.0
                elif case["mode"] == "upper_edge":
                    value = 4.0
                elif case["mode"] == "out_of_range":
                    value = -8.0 if index % 2 else 8.0
                else:
                    value = rng.uniform(-4.0, 4.0)
            elif arg.name in {"weight", "bias"}:
                value = 0.0 if (index + 1) % 17 == 0 else rng.uniform(-1.5, 1.5)
            elif case["mode"] == "zeros":
                value = -0.0 if index % 2 else 0.0
            elif case["mode"] == "alternating":
                value = (1 if index % 2 else -1) * min(bound, 1 + (index % 17) / 16)
            elif case["mode"] == "near_zero":
                value = rng.uniform(-1e-4, 1e-4)
            elif case["mode"] == "mixed_magnitude":
                value = (1 if index % 2 else -1) * 2.0 ** rng.randint(-12, 1)
            elif case["mode"] == "negative_only":
                value = -abs(rng.uniform(0.0, bound))
            elif case["mode"] == "padded_edges":
                value = -2.0 if index % 2 else 2.0
            else:
                value = rng.uniform(-bound, bound)
            values.append(_round(value, "fp32"))
        result[arg.name] = values
    if _name_for(workload.document) == "momentum_sgd":
        result["mu"] = [_round(0.9, "fp32")]
        result["lr"] = [_round(0.01, "fp32")]
        result["nesterov"] = [1 if case["mode"] == "nesterov_on" else 0]
    return result


def reference_outputs(workload: WorkloadContract, case_id: str, inputs: Mapping[str, Sequence[float | int]]) -> dict[str, list[float]]:
    """Evaluate each contract directly, deliberately independent from its CUDA baseline."""
    validate_aka_v3_contract(workload.document)
    args = tuple(arg for arg in workload.tensor_abi(case_id) if arg.mode == "input")
    checked = _checked_inputs(workload, args, inputs)
    operator = workload.document["operator"]
    shape = workload.case(case_id)["shape"]
    if operator == "aka_row_gather_fp32":
        width = shape["C"]
        source, indices = checked["input"], checked["output_row_to_input_row"]
        output: list[float] = []
        for index in indices:
            start = index * width
            output.extend(source[start:start + width])
        return {"output": output}
    if operator == "aka_histogram_fp32":
        minimum, maximum = -4.0, 4.0
        bins = shape["B"]
        counts = [0.0] * bins
        for value in checked["values"]:
            if minimum <= value <= maximum:
                index = min(int((value - minimum) / (maximum - minimum) * bins), bins - 1)
                counts[index] += 1.0
        return {"counts": counts}
    if operator == "aka_max_pool1d_nwc_fp32":
        batches, length, output_length, channels = shape["N"], shape["X"], shape["Y"], shape["C"]
        window = workload.document["semantics"]["window"]
        values, output = checked["input"], []
        for batch in range(batches):
            for destination in range(output_length):
                begin = max(destination * window["stride"] - window["pad"], 0)
                end = min(destination * window["stride"] - window["pad"] + window["kernel_size"], length)
                for channel in range(channels):
                    output.append(max(values[(batch * length + position) * channels + channel] for position in range(begin, end)))
        return {"output": output}
    if operator == "aka_momentum_sgd_fp32":
        mu, lr = checked["mu"][0], checked["lr"][0]
        nesterov = checked["nesterov"][0]
        param, grad, moment = checked["param"], checked["grad"], checked["moment"]
        updated = [_round(_round(mu * old, "fp32") + _round(lr * gradient, "fp32"), "fp32")
                   for old, gradient in zip(moment, grad)]
        if nesterov:
            parameter = [_round(value - _round((1.0 + mu) * new, "fp32") + _round(mu * old, "fp32"), "fp32")
                         for value, old, new in zip(param, moment, updated)]
        else:
            parameter = [_round(value - new, "fp32") for value, new in zip(param, updated)]
        return {"param_out": parameter, "moment_out": updated}
    if operator == "aka_gemm_nt_bias_fp32":
        rows, depth, columns = shape["M"], shape["K"], shape["N"]
        left, right, bias = checked["input"], checked["weight_nt"], checked["bias"]
        return {"output": [_round(math.fsum(left[row * depth + inner] * right[column * depth + inner]
                                                for inner in range(depth)) + bias[column], "fp32")
                           for row in range(rows) for column in range(columns)]}
    rows, width = shape["R"], shape["C"]
    left, right, weight, bias = (checked[name] for name in ("input_a", "input_b", "weight", "bias"))
    residual = [_round(a + b, "fp32") for a, b in zip(left, right)]
    means, reciprocal, normalized = [], [], []
    for row in range(rows):
        values = residual[row * width:(row + 1) * width]
        mean = math.fsum(values) / width
        inverse = 1.0 / math.sqrt(math.fsum((value - mean) ** 2 for value in values) / width + EPSILON)
        means.append(_round(mean, "fp32"))
        reciprocal.append(_round(inverse, "fp32"))
        normalized.extend(_round((value - mean) * inverse * weight[column] + bias[column], "fp32")
                          for column, value in enumerate(values))
    return {"residual": residual, "normalized": normalized, "mean": means, "reciprocal_stddev": reciprocal}
