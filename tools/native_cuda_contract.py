"""Native qualification adapters; Workloads and Compiler remain the authorities.

This module is CPU-only. Receipt replay proves internal consistency of retained
observations under the existing cooperating-agent trust boundary, not resistance
to a writer fabricating every raw observation. It never grants framework acceptance.
"""
from __future__ import annotations

import array
import json
import math
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation.core import compare_tile_outputs
from open_cake_ir.evaluation.timing import PairedTimingProtocol, derive_paired_timing
from open_cake_ir.tasks.workloads import load_workload
from open_cake_ir.tasks.tiles.workload import materialize_case, reference_outputs


class QualificationError(ValueError):
    def __init__(self, category, message):
        super().__init__(message)
        self.category = category


def require(condition, message, category="evidence_invalid"):
    if not condition:
        raise QualificationError(category, message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def write_bytes(path, value):
    with Path(path).open("xb") as stream:
        stream.write(value)


def child(root, relative):
    """No receipt can redirect a read outside its retained artifact root."""
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "artifact path must be relative")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "artifact path escapes root")
    root = Path(root).resolve(strict=True)
    current = root
    for part in path.parts:
        current = current / part
        require(not current.is_symlink(), "artifact path traverses a symlink")
    result = current.resolve(strict=True)
    require(root in result.parents and result.is_file(), "artifact is not a retained file")
    return result


def workloads(root=ROOT):
    return {
        "gemm": load_workload(root / "contracts/workloads/gemm-bias-bf16-fp32-v2.json"),
        "kmeans": load_workload(root / "contracts/workloads/flash-kmeans-assign-v2.json"),
    }


def cases(root=ROOT):
    owners = workloads(root)
    rows = []
    for family, workload in owners.items():
        for case_id in workload.case_ids:
            template = "kmeans-partials" if family == "kmeans" else (
                "gemm-bias" if workload.case(case_id)["shape"]["K"] % 8 == 0
                else "gemm-bias-k65-tail")
            rows.append({"id": family + "-" + case_id, "family": family,
                         "case_id": case_id, "template": template})
    rows.append({"id": "two-mma-primary", "family": "two_mma", "case_id": "primary",
                 "template": "two-mma"})
    return rows


def owner(row, root=ROOT):
    return workloads(root)["kmeans" if row["family"] == "kmeans" else "gemm"]


def specialize(row, root=ROOT):
    """Only specialize global ABI dimensions; hardware choices stay in the example."""
    workload = owner(row, root)
    document = read_json(root / "examples/schedules/native" / (row["template"] + ".json"))
    document["target"] = "sm_103a"
    document["schedule_id"] = "qualification-" + row["id"]
    buffers = {value["name"]: value for value in document["buffers"]}
    for tensor in workload.tensor_abi(row["case_id"]):
        require(tensor.name in buffers, "example lacks Workload ABI tensor", "input_contract")
        buffers[tensor.name]["shape"] = list(tensor.shape)
    canonical_single_centroid_tile(document)
    return document


def canonical_single_centroid_tile(document):
    """The constructed K=4 case has one centroid tile, so no TileLoop exists.

    This exact case includes an all-zero centroid at index zero: zero-filled tail
    columns are duplicates of that centroid and cannot beat its lower index.
    No claim is made for arbitrary K<tile inputs. The contraction loop stays intact.
    """
    buffers = {b["name"]: b for b in document["buffers"]}
    loops = document.get("tile_loops", [])
    single = [loop for loop in loops if loop["buffer"] == "centroids"
              and buffers["centroids"]["shape"][loop["dimension"]] <= loop["tile"]]
    for loop in single:
        require(buffers["centroids"]["shape"] == [1, 4, 128], "single centroid tile requires constructed tie case")
        document["program_map"]["axes"].append({"name": "single_centroid_tile", "axis": 2,
            "buffer": "centroids", "dimension": loop["dimension"], "tile": loop["tile"]})
        for access in document["access_maps"]:
            for dimension, component in enumerate(access["indices"]):
                if component.get("source") == "loop_tile" and component.get("name") == loop["iterator"]:
                    component.clear()
                    component.update(source="program_tile", name="single_centroid_tile")
        for operation in document["operations"]:
            if operation["kind"] == "reduce_argmin":
                operation["parameters"]["across_loop"] = False
        loops.remove(loop)


def baseline_schedule(row, root=ROOT):
    """Selected genuine Triton algorithm, specialized through the same Workload ABI."""
    require(row["family"] in {"gemm", "kmeans"}, "structural two-MMA has no performance baseline")
    path = ("examples/schedules/triton/kmeans-partials.json" if row["family"] == "kmeans"
            else "corpus/schedules/gemm-bias-b1-smoke-b300.json")
    document = read_json(root / path)
    document["target"] = "sm_103a"
    document["schedule_id"] = "qualification-triton-" + row["id"]
    buffers = {b["name"]: b for b in document["buffers"]}
    for tensor in owner(row, root).tensor_abi(row["case_id"]):
        buffers[tensor.name]["shape"] = list(tensor.shape)
    canonical_single_centroid_tile(document)
    return document


def validate_metadata(metadata, *, target, abi):
    require(target == "sm_103a", "qualification Workload requires exact sm_103a", "target")
    require(metadata.get("source_language") == "cuda_cpp" and metadata.get("compiler") == "nvcc"
            and metadata.get("target") == target, "native compiler/target metadata differs", "target")
    flags = metadata.get("nvcc_flags")
    require(isinstance(flags, list) and all(isinstance(x, str) for x in flags), "nvcc flags missing")
    # The emitted exact arch/code pair must be the only architecture selector.
    selectors = [x for x in flags if any(s in x for s in ("gpu-architecture", "gpu-code", "-arch", "-code", "gencode", "generate-code"))]
    require(selectors == ["--gpu-architecture=compute_103a", "--gpu-code=sm_103a"],
            "nvcc target selectors differ or include fallback", "target")
    expected = [{"name": a.name, "shape": list(a.shape), "dtype": a.dtype, "mode": a.mode} for a in abi]
    require(metadata.get("arguments") == expected, "emitted global ABI differs from Workload", "input_contract")
    names = [x["name"] for x in expected]
    require(metadata.get("argument_order") == names and metadata.get("signature") == {
        x["name"]: "*" + x["dtype"] for x in expected}, "emitted ABI order/signature differs", "input_contract")
    host = metadata.get("host_abi")
    require(isinstance(host, dict) and set(host) == {"create", "launch", "destroy"}
            and all(isinstance(v, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", v) for v in host.values()),
            "native host ABI missing")
    for name in ("grid", "block"):
        require(isinstance(metadata.get(name), list) and len(metadata[name]) == 3
                and all(type(x) is int and x > 0 for x in metadata[name]), "native launch dimensions invalid")
    require(metadata["block"] == [metadata.get("threads_per_cta"), 1, 1]
            and type(metadata.get("dynamic_shared_bytes")) is int and metadata["dynamic_shared_bytes"] >= 0,
            "native block/shared metadata differs")
    require(metadata.get("link_libraries") == ["cuda", "cudart"], "native link libraries differ")


def compile_commands(nvcc, row_directory, metadata):
    directory = Path(row_directory)
    common = [str(nvcc), *metadata["nvcc_flags"]]
    return {
        "shared": [*common, "--shared", "-Xcompiler=-fPIC", str(directory / "kernel.cu"),
                   "-o", str(directory / "kernel.so"), *["-l" + x for x in metadata["link_libraries"]]],
        "ptx": [*common, "--ptx", str(directory / "kernel.cu"), "-o", str(directory / "kernel.ptx")],
        "cubin": [*common, "--cubin", str(directory / "kernel.cu"), "-o", str(directory / "kernel.cubin")],
    }


CUPTI_DRY_RUN_ITERS = 11


def timing_protocol():
    """Four pairs with the observed FlashInfer 0.6.12 callback contract.

    Even with explicit iteration counts and no graph, the helper calls once to
    exclude initial overhead and five times for event-based estimation before
    its dry runs and CUPTI samples. All callbacks require fresh checked outputs;
    the six setup calls are not part of the 25 reported CUPTI samples.
    """
    samples = 25
    return PairedTimingProtocol(("candidate", "baseline"),
        (("candidate", "baseline"), ("baseline", "candidate")) * 2,
        samples, 1 + 5 + CUPTI_DRY_RUN_ITERS + samples, 0.05, 1.03, 3)


def timing_summary(rows):
    observation = derive_paired_timing(rows, timing_protocol())
    return {"measurement_quality_passed": observation.measurement_quality_passed,
            "classification": observation.classification, "speedup": observation.speedup,
            "pair_wins": dict(observation.pair_wins),
            "pooled_medians_ms": dict(observation.pooled_medians_ms)}


def gemm_reference(row, workload, inputs):
    if row["family"] != "two_mma":
        return reference_outputs(workload, row["case_id"], inputs)
    # Two independently accumulated contractions are added in FP32, then bias.
    # Reuse the external math.fsum contraction with zero bias, not the Schedule DAG.
    zero_bias = dict(inputs, bias=[0.0] * len(inputs["bias"]))
    dot = reference_outputs(workload, row["case_id"], zero_bias)["c"]
    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
    bias = inputs["bias"]
    return {"c": [f32(f32(x + x) + bias[i % len(bias)]) for i, x in enumerate(dot)]}


def compare_raw(row, workload, output, expected):
    """Replay all-element correctness; KMeans diagnostic tolerances never enter."""
    abi = workload.tensor_abi(row["case_id"])
    result, = (a for a in abi if a.mode == "output")
    require(len(output) == len(expected) == math.prod(result.shape), "raw output size differs", "correctness")
    if row["family"] == "kmeans":
        k = workload.case(row["case_id"])["shape"]["K"]
        require(all(type(x) is int and 0 <= x < k for x in expected), "oracle indices invalid", "correctness")
        mismatches = sum(type(a) is not int or not 0 <= a < k or a != b for a, b in zip(output, expected))
        return {"passed": mismatches == 0, "output_mismatches": mismatches,
                "comparison": "exact_int32_ids_with_lowest_index_ties"}
    passed, metrics = compare_tile_outputs(workload, {}, {result.name: list(expected)},
                                          {result.name: list(output)}, {})
    return {"passed": passed, **metrics, "comparison": "elementwise_atol_rtol"}


def vector_bytes(values, dtype):
    code = {"fp32": "f", "int32": "i"}[dtype]
    value = array.array(code, values)
    if sys.byteorder != "little":
        value.byteswap()
    return value.tobytes()


def read_vector(path, dtype):
    payload = Path(path).read_bytes()
    require(len(payload) % 4 == 0, "raw vector contains partial elements", "correctness")
    value = array.array({"fp32": "f", "int32": "i"}[dtype])
    value.frombytes(payload)
    if sys.byteorder != "little":
        value.byteswap()
    return value.tolist()


def finite_bf16(payload, label):
    """Check the frozen finite-input domain without importing a device runtime."""
    require(len(payload) % 2 == 0, f"{label} has partial BF16 elements", "input_contract")
    bits = array.array("H")
    bits.frombytes(payload)
    if sys.byteorder != "little":
        bits.byteswap()
    require(all(value & 0x7f80 != 0x7f80 for value in bits),
            f"{label} contains nonfinite BF16 inputs", "input_contract")
