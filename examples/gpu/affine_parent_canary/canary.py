"""Fixed N2/C2/S2 affine correctness judge and CPU complete-output verifier."""

from __future__ import annotations

import argparse
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

TASK_ID = "open-cake-affine-n2c2s2-v43-b200-correctness-v1"
WORKLOAD = "batch-sensitive-n2-c2-s2"
PARENT = "data_movement_and_layout__gather_scatter__analysis__l000075_b200_v1__directderived_sol_ultra_v2"
N, C, SPATIAL = 2, 2, 2
GUARD_BITS = 0x4B123456


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def number(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


def inputs() -> dict[str, list[int]]:
    """Reproduce the selected parent harness's deterministic finite binary32 inputs."""
    return {
        "x": [bits(((i * 37 + 11) % 257 - 128) / 32) for i in range(N * C * SPATIAL)],
        "scale": [bits(((i * 13 + 3) % 31 - 15) / 16) for i in range(N * C)],
        "bias": [bits(((i * 17 + 5) % 29 - 14) / 32) for i in range(N * C)],
    }


def reference() -> list[int]:
    """Independent coordinate loops and exact rationals for this one fixed sample.

    Every exact product-plus-sum is representable as binary32 here; the assertion
    establishes equality to single-rounded fmaf without approximating arbitrary FMA.
    """
    values = {key: [Fraction.from_float(number(x)) for x in row] for key, row in inputs().items()}
    result = []
    for batch in range(N):
        for channel in range(C):
            plane = batch * C + channel
            for spatial in range(SPATIAL):
                index = plane * SPATIAL + spatial
                exact = values["x"][index] * values["scale"][plane] + values["bias"][plane]
                result_bits = bits(float(exact))
                if Fraction.from_float(number(result_bits)) != exact:
                    raise ValueError("selected reference result is not exactly representable as FP32")
                result.append(result_bits)
    return result


def _bit_row(value, length: int, name: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length or any(
        type(x) is not int or not 0 <= x <= 0xFFFFFFFF for x in value
    ):
        raise ValueError(f"{name} must contain exactly {length} binary32 bit patterns")
    return value


def verify(document: dict) -> dict:
    if (document.get("schema") != "open-cake.affine-complete-output.v1"
            or document.get("task_id") != TASK_ID or document.get("workload") != WORKLOAD
            or document.get("shape") != [N, C, SPATIAL]
            or document.get("device_name") != "NVIDIA B200"
            or document.get("compute_capability") != [10, 0]
            or document.get("performance_measured") is not False):
        raise ValueError("fixed task, target or claim boundary differs")
    original = inputs()
    before, after = document.get("input_bits_before"), document.get("input_bits_after")
    if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != set(original) or set(after) != set(original):
        raise ValueError("complete input fields differ")
    mutations = 0
    for key, row in original.items():
        if _bit_row(before[key], len(row), key) != row:
            raise ValueError("inputs differ from the selected parent sample")
        mutations += sum(a != b for a, b in zip(row, _bit_row(after[key], len(row), key)))
    expected = reference()
    if _bit_row(document.get("expected_bits"), len(expected), "expected_bits") != expected:
        raise ValueError("recorded oracle differs from independent exact arithmetic")
    actual = _bit_row(document.get("actual_bits"), len(expected), "actual_bits")
    mismatches = sum(a != b for a, b in zip(expected, actual))
    guards = _bit_row(document.get("guard_bits"), 2, "guard_bits") == [GUARD_BITS] * 2
    abi = document.get("abi_valid") is True
    return {"status": "passed" if not mismatches and not mutations and guards and abi else "failed",
            "checked_elements": len(expected), "output_mismatch_count": mismatches,
            "input_mutation_count": mutations, "guards_unchanged": guards,
            "recorded_abi_valid": abi, "performance_measured": False,
            "scope": "selected_fixed_instance_only"}


def write_json(path: Path, value: object) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def evaluate(candidate: Path, stage: Path) -> dict:
    if os.environ.get("KERNELINFRA_STAGE_KIND") != "correctness":
        raise ValueError("this task owns one correctness stage only")
    os.environ["TRITON_CACHE_DIR"] = str(stage / "triton-cache")
    os.environ["XDG_CACHE_HOME"] = str(stage / "cache")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable inside broker allocation")
    torch.cuda.set_device(0)
    device = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    if device != "NVIDIA B200" or capability != [10, 0]:
        raise RuntimeError(f"fixed target differs: {device} / {capability}")

    def tensor(row, shape):
        signed = [x if x < 1 << 31 else x - (1 << 32) for x in row]
        return torch.tensor(signed, dtype=torch.int32).view(torch.float32).reshape(shape).to("cuda")

    def read_bits(value):
        return [x & 0xFFFFFFFF for x in value.detach().cpu().contiguous().view(torch.int32).reshape(-1).tolist()]

    before = inputs()
    x, scale, bias = tensor(before["x"], (4, 2)), tensor(before["scale"], (4,)), tensor(before["bias"], (4,))
    # NaNs cannot equal the finite expected outputs; guards expose nearby output writes.
    backing = tensor([GUARD_BITS] + [0x7FC0DEAD] * 8 + [GUARD_BITS], (10,))
    output = backing[1:9].reshape(4, 2)
    pointers = [int(t.data_ptr()) for t in (x, scale, bias, output)]
    spec = importlib.util.spec_from_file_location("affine_parent_kernel", candidate / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = json.loads((candidate / "contract.json").read_text())
    if contract["task_id"] != TASK_ID or contract["parent_case_id"] != PARENT:
        raise ValueError("candidate contract differs")
    returned = getattr(module, contract["entry_point"])(x, scale, bias, out=output)
    torch.cuda.synchronize()
    abi = (isinstance(returned, torch.Tensor) and int(returned.data_ptr()) == pointers[-1]
           and list(returned.shape) == [4, 2] and returned.dtype == torch.float32
           and returned.is_contiguous() and len(set(pointers)) == 4
           and [int(t.data_ptr()) for t in (x, scale, bias, output)] == pointers)
    document = {"schema": "open-cake.affine-complete-output.v1", "task_id": TASK_ID,
                "workload": WORKLOAD, "shape": [N, C, SPATIAL], "device_name": device,
                "compute_capability": capability, "input_bits_before": before,
                "input_bits_after": {key: read_bits(t) for key, t in zip(before, (x, scale, bias))},
                "expected_bits": reference(), "actual_bits": read_bits(output),
                "guard_bits": [read_bits(backing[:1])[0], read_bits(backing[-1:])[0]],
                "abi_valid": abi, "performance_measured": False}
    artifact = stage / "complete-output.json"
    write_json(artifact, document)
    checked = verify(document)
    passed = checked["status"] == "passed"
    return {"schema": "kernelinfra.stage-result.v1", "status": "passed" if passed else "failed",
            "validity": "valid" if passed else "invalid",
            "summary": "N2/C2/S2 complete-output affine, input, guard and wrapper checks finished.",
            "workloads": [{"id": WORKLOAD, "correct": passed}],
            "artifacts": {"complete_output": str(artifact)}, "metrics": checked}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path, help="recompute a complete-output record without GPU")
    arguments = parser.parse_args()
    if arguments.verify is not None:
        result = verify(json.loads(arguments.verify.read_text()))
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "passed" else 1
    try:
        result = evaluate(Path(os.environ["KERNELINFRA_CANDIDATE_DIR"]), Path(os.environ["KERNELINFRA_STAGE_DIR"]))
    except Exception as error:
        result = {"schema": "kernelinfra.stage-result.v1", "status": "failed", "validity": "unknown",
                  "summary": f"judge failure: {type(error).__name__}: {error}", "workloads": [],
                  "metrics": {"performance_measured": False}}
    write_json(Path(os.environ["KERNELINFRA_RESULT"]), result)
    return 0 if result["validity"] == "valid" else 1


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
