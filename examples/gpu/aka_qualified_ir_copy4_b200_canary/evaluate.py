#!/usr/bin/env python3
"""Independent B200 judge for the fixed n=1024 Cake copy lowering."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch


ELEMENTS = 1024
CASES = (
    "zeros-n1024",
    "signed-integers-n1024",
    "alternating-binary-fractions-n1024",
    "seeded-scaled-integers-n1024",
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _load_wrapper(candidate_dir: Path) -> Callable[..., torch.Tensor]:
    source = candidate_dir / "kernel.py"
    spec = importlib.util.spec_from_file_location("aka_copy4_candidate", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated candidate: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wrapper = getattr(module, "copy4_contiguous_fp32_n1024", None)
    if not callable(wrapper):
        raise RuntimeError("generated wrapper entry point differs")
    return wrapper


def _input(case_id: str) -> torch.Tensor:
    index = torch.arange(ELEMENTS, dtype=torch.int32)
    if case_id == CASES[0]:
        value = torch.zeros(ELEMENTS, dtype=torch.float32)
    elif case_id == CASES[1]:
        value = ((index % 251) - 125).to(dtype=torch.float32)
    elif case_id == CASES[2]:
        value = torch.where(index % 2 == 0, 0.125, -0.25).to(
            dtype=torch.float32
        )
    elif case_id == CASES[3]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260903)
        value = torch.randint(
            -512, 513, (ELEMENTS,), generator=generator, dtype=torch.int32
        ).to(dtype=torch.float32) * 0.03125
    else:
        raise RuntimeError(f"unknown correctness case: {case_id}")
    if tuple(value.shape) != (ELEMENTS,) or value.dtype != torch.float32:
        raise RuntimeError("independent input generator left the frozen contract")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("independent input generator produced a non-finite value")
    return value.contiguous()


def _run_cases(candidate_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the broker allocation")
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if "B200" not in device_name or capability != (10, 0):
        raise RuntimeError(
            f"fixed B200/sm100 target mismatch: {device_name}, capability={capability}"
        )
    wrapper = _load_wrapper(candidate_dir)
    workloads: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}
    for case_id in CASES:
        source_before = _input(case_id)
        expected = source_before.clone()
        source = source_before.clone().to(device="cuda")
        output = torch.full(
            (ELEMENTS,), -777.25, dtype=torch.float32, device="cuda"
        )
        source_pointer = int(source.data_ptr())
        output_pointer = int(output.data_ptr())
        returned = wrapper(source, out=output)
        torch.cuda.synchronize()
        actual = output.detach().cpu().clone()
        source_after = source.detach().cpu().clone()

        output_mismatch = int(
            torch.count_nonzero(
                expected.view(torch.int32) != actual.view(torch.int32)
            ).item()
        )
        source_mismatch = int(
            torch.count_nonzero(
                source_before.view(torch.int32) != source_after.view(torch.int32)
            ).item()
        )
        returned_output = (
            isinstance(returned, torch.Tensor)
            and int(returned.data_ptr()) == output_pointer
        )
        pointers_disjoint = source_pointer != output_pointer
        correct = (
            output_mismatch == 0
            and source_mismatch == 0
            and returned_output
            and pointers_disjoint
        )
        artifact = artifact_dir / f"{case_id}.complete-output.json"
        _atomic_json(
            artifact,
            {
                "schema": "open-cake.aka-copy4-complete-output.v1",
                "case_id": case_id,
                "source_before": source_before.tolist(),
                "source_after": source_after.tolist(),
                "expected_output": expected.tolist(),
                "actual_output": actual.tolist(),
                "checked_elements": ELEMENTS,
                "output_mismatch_count": output_mismatch,
                "source_mismatch_count": source_mismatch,
                "returned_supplied_output": returned_output,
                "input_output_pointers_disjoint": pointers_disjoint,
                "bitwise_required": True,
                "correct": correct,
            },
        )
        artifacts[case_id] = str(artifact)
        detail = {
            "id": case_id,
            "correct": correct,
            "checked_elements": ELEMENTS,
            "output_mismatch_count": output_mismatch,
            "source_mismatch_count": source_mismatch,
            "returned_supplied_output": returned_output,
            "input_output_pointers_disjoint": pointers_disjoint,
        }
        details.append(detail)
        workloads.append(
            {
                "id": case_id,
                "correct": correct,
                "notes": (
                    f"checked={ELEMENTS} output_mismatch={output_mismatch} "
                    f"source_mismatch={source_mismatch} returned_out={returned_output} "
                    f"disjoint={pointers_disjoint} bitwise=true"
                ),
            }
        )
    valid = all(bool(detail["correct"]) for detail in details)
    return {
        "schema": "kernelinfra.stage-result.v1",
        "status": "passed" if valid else "failed",
        "validity": "valid" if valid else "invalid",
        "summary": (
            "Four fixed n=1024 complete-output distributions passed bitwise identity, "
            "read-only input, supplied-output ABI, and disjoint-pointer checks on B200."
            if valid
            else "At least one fixed n=1024 complete-output or ABI check failed."
        ),
        "workloads": workloads,
        "artifacts": artifacts,
        "metrics": {
            "device_name": device_name,
            "compute_capability": list(capability),
            "total_checked_elements": ELEMENTS * len(CASES),
            "performance_measured": False,
            "cases": details,
        },
    }


def _sanitize(
    *,
    tool: str,
    candidate_dir: Path,
    stage_dir: Path,
    result_path: Path,
) -> int:
    worker_result = stage_dir / f"{tool}.worker-result.json"
    artifact_dir = stage_dir / f"{tool}-complete-output"
    log_path = stage_dir / f"compute-sanitizer-{tool}.log"
    command = [
        "/usr/local/cuda/bin/compute-sanitizer",
        "--tool",
        tool,
        "--error-exitcode",
        "86",
        "--target-processes",
        "all",
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-result",
        str(worker_result),
        "--artifact-dir",
        str(artifact_dir),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    worker: dict[str, Any] | None = None
    try:
        loaded = json.loads(worker_result.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            worker = loaded
    except (OSError, json.JSONDecodeError):
        pass
    if completed.returncode == 0 and worker is not None and worker.get("status") == "passed":
        result = dict(worker)
        result["summary"] = f"{tool} and complete-output correctness passed on B200."
        result["artifacts"] = {
            **dict(worker.get("artifacts", {})),
            f"compute_sanitizer_{tool}": str(log_path),
        }
        _atomic_json(result_path, result)
        return 0
    if completed.returncode == 86:
        validity = "invalid"
        summary = f"compute-sanitizer {tool} reported a memory-safety failure"
    else:
        validity = "unknown"
        summary = (
            f"compute-sanitizer {tool} did not produce a valid pass "
            f"(exit={completed.returncode})"
        )
    _atomic_json(
        result_path,
        {
            "schema": "kernelinfra.stage-result.v1",
            "status": "failed",
            "validity": validity,
            "summary": summary,
            "workloads": [] if worker is None else worker.get("workloads", []),
            "artifacts": {f"compute_sanitizer_{tool}": str(log_path)},
            "metrics": {"performance_measured": False},
        },
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-result", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    arguments = parser.parse_args()
    candidate_dir = Path(os.environ["KERNELINFRA_CANDIDATE_DIR"])
    if arguments.worker_result is not None:
        if arguments.artifact_dir is None:
            raise RuntimeError("worker artifact directory is required")
        try:
            result = _run_cases(candidate_dir, arguments.artifact_dir)
        except Exception as error:
            result = {
                "schema": "kernelinfra.stage-result.v1",
                "status": "failed",
                "validity": "unknown",
                "summary": f"worker failure: {type(error).__name__}: {error}",
                "workloads": [],
                "metrics": {"performance_measured": False},
            }
        _atomic_json(arguments.worker_result, result)
        return 0 if result["status"] == "passed" else 1

    result_path = Path(os.environ["KERNELINFRA_RESULT"])
    stage_dir = Path(os.environ["KERNELINFRA_STAGE_DIR"])
    stage_kind = os.environ["KERNELINFRA_STAGE_KIND"]
    stage_id = os.environ["KERNELINFRA_STAGE_ID"]
    try:
        if stage_kind == "correctness":
            result = _run_cases(candidate_dir, stage_dir / "complete-output")
            _atomic_json(result_path, result)
            return 0 if result["status"] == "passed" else 1
        if stage_kind == "sanitize" and stage_id == "sanitize-memcheck":
            return _sanitize(
                tool="memcheck",
                candidate_dir=candidate_dir,
                stage_dir=stage_dir,
                result_path=result_path,
            )
        if stage_kind == "sanitize" and stage_id == "sanitize-racecheck":
            return _sanitize(
                tool="racecheck",
                candidate_dir=candidate_dir,
                stage_dir=stage_dir,
                result_path=result_path,
            )
        raise RuntimeError(f"unsupported stage: {stage_id}/{stage_kind}")
    except Exception as error:
        _atomic_json(
            result_path,
            {
                "schema": "kernelinfra.stage-result.v1",
                "status": "failed",
                "validity": "unknown",
                "summary": f"judge failure: {type(error).__name__}: {error}",
                "workloads": [],
                "metrics": {"performance_measured": False},
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
