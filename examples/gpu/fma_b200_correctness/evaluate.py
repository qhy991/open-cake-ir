"""Task-owned complete-output judge, run only inside a broker allocation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import ELEMENTS, FAMILIES, MODES, SHAPE, equivalent, expected_bits, inputs


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run_cases(candidate: Path, artifacts: Path) -> dict:
    stage = Path(os.environ["KERNELINFRA_STAGE_DIR"])
    os.environ["TRITON_CACHE_DIR"] = str(stage / "triton-cache")
    os.environ["XDG_CACHE_HOME"] = str(stage / "cache")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable inside broker allocation")
    torch.cuda.set_device(0)
    device = torch.cuda.get_device_name(0)
    if "B200" not in device or tuple(torch.cuda.get_device_capability(0)) != (10, 0):
        raise RuntimeError(f"fixed B200 target differs: {device}")
    contract = json.loads((candidate / "contract.json").read_text())
    if contract["shape"] != list(SHAPE) or contract["workloads"] != [
        f"{m}-{f}" for m in MODES for f in FAMILIES
    ]:
        raise RuntimeError("frozen workload contract differs")
    workloads, details, paths = [], [], {}

    def tensor(bits):
        signed = [x if x < (1 << 31) else x - (1 << 32) for x in bits]
        return torch.tensor(signed, dtype=torch.int32).view(torch.float32).reshape(SHAPE)

    def bit_list(value):
        return [x & 0xFFFFFFFF for x in value.detach().cpu().contiguous()
                .view(torch.int32).reshape(-1).tolist()]

    for mode in MODES:
        path = candidate / f"{mode}.py"
        spec = importlib.util.spec_from_file_location(f"fma_candidate_{mode}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        wrapper = getattr(module, contract["entry_points"][mode])
        for family in FAMILIES:
            case_id = f"{mode}-{family}"
            before = inputs(family)
            expected = [expected_bits(mode, *triple) for triple in zip(*before)]
            values = [tensor(bits).to(device="cuda") for bits in before]
            # Every unwritten element must fail, including positions expecting NaN.
            initial = [0x3F123456 if (x & 0x7FFFFFFF) > 0x7F800000 else 0x7FC0DEAD
                       for x in expected]
            output = tensor(initial).to(device="cuda")
            pointers = [int(value.data_ptr()) for value in values] + [int(output.data_ptr())]
            returned = wrapper(*values, out=output)
            torch.cuda.synchronize()
            actual = bit_list(output)
            after = [bit_list(value) for value in values]
            mismatch = sum(not equivalent(e, a) for e, a in zip(expected, actual))
            mutation = sum(x != y for x, y in zip(before, after))
            abi = (isinstance(returned, torch.Tensor)
                   and int(returned.data_ptr()) == pointers[-1]
                   and tuple(returned.shape) == SHAPE and returned.dtype == torch.float32
                   and [int(v.data_ptr()) for v in values] == pointers[:3]
                   and len(set(pointers)) == 4)
            correct = mismatch == 0 and mutation == 0 and abi
            artifact = artifacts / f"{case_id}.complete-output.json"
            write_json(artifact, {
                "case_id": case_id, "mode": mode, "family": family, "shape": list(SHAPE),
                "input_bits_before": before, "input_bits_after": after,
                "expected_bits": expected, "actual_bits": actual,
                "output_mismatch_count": mismatch, "mutated_input_count": mutation,
                "abi_valid": abi, "correct": correct,
            })
            paths[case_id] = str(artifact)
            detail = {"id": case_id, "checked_elements": ELEMENTS,
                      "output_mismatch_count": mismatch, "mutated_input_count": mutation,
                      "abi_valid": abi, "correct": correct}
            details.append(detail)
            workloads.append({"id": case_id, "correct": correct,
                              "notes": f"complete={ELEMENTS} mismatch={mismatch} mutation={mutation} abi={abi}"})
    passed = all(row["correct"] for row in details)
    return {
        "schema": "kernelinfra.stage-result.v1",
        "status": "passed" if passed else "failed",
        "validity": "valid" if passed else "invalid",
        "summary": "Fixed-shape FMA bitwise/non-payload-NaN, input and ABI checks completed.",
        "workloads": workloads, "artifacts": paths,
        "metrics": {"device_name": device, "total_checked_elements": ELEMENTS * len(details),
                    "cases": details, "performance_measured": False},
    }


def failure(error: str, validity="unknown", workloads=None, artifacts=None) -> dict:
    return {"schema": "kernelinfra.stage-result.v1", "status": "failed",
            "validity": validity, "summary": error, "workloads": workloads or [],
            "artifacts": artifacts or {}, "metrics": {"performance_measured": False}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-result", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    candidate = Path(os.environ["KERNELINFRA_CANDIDATE_DIR"])
    result_path = args.worker_result or Path(os.environ["KERNELINFRA_RESULT"])
    stage = Path(os.environ["KERNELINFRA_STAGE_DIR"])
    try:
        if args.worker_result or os.environ["KERNELINFRA_STAGE_KIND"] == "correctness":
            result = run_cases(candidate, args.artifact_dir or stage / "complete-output")
        else:
            stage_id = os.environ["KERNELINFRA_STAGE_ID"]
            tools = {"sanitize-memcheck": "memcheck", "sanitize-racecheck": "racecheck"}
            tool = tools[stage_id]
            worker_path = stage / f"{tool}.worker.json"
            log_path = stage / f"{tool}.log"
            completed = subprocess.run([
                "/usr/local/cuda/bin/compute-sanitizer", "--tool", tool,
                "--error-exitcode", "86", "--target-processes", "all",
                sys.executable, str(Path(__file__).resolve()),
                "--worker-result", str(worker_path),
                "--artifact-dir", str(stage / "complete-output"),
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
            fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            with os.fdopen(fd, "w") as stream:
                stream.write(completed.stdout)
            worker = json.loads(worker_path.read_text()) if worker_path.exists() else None
            clean_marker = ("ERROR SUMMARY: 0 errors" if tool == "memcheck"
                            else "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)")
            if (completed.returncode == 0 and worker and worker["status"] == "passed"
                    and worker["validity"] == "valid" and clean_marker in completed.stdout):
                result = worker
                result["summary"] = f"{tool} and complete-output oracle passed."
                result["artifacts"][tool] = str(log_path)
            else:
                invalid = completed.returncode == 86 or (worker and worker["validity"] == "invalid")
                result = failure(f"{tool} did not pass (exit={completed.returncode})",
                                 "invalid" if invalid else "unknown",
                                 worker.get("workloads", []) if worker else [],
                                 {**(worker.get("artifacts", {}) if worker else {}),
                                  tool: str(log_path)})
    except Exception as error:
        result = failure(f"{type(error).__name__}: {error}")
    write_json(result_path, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
