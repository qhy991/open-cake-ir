#!/usr/bin/env python3
"""Independent complete-output judge for the generated state-store wrapper."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import torch


SHAPE = (8, 128)
ELEMENTS = 1024
CASES = (
    "zeros-b8x128",
    "signed-integers-b8x128",
    "alternating-binary-fractions-b8x128",
    "seeded-scaled-integers-b8x128",
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _load_wrapper(candidate_dir: Path) -> Callable[..., tuple[()]]:
    path = candidate_dir / "kernel.py"
    spec = importlib.util.spec_from_file_location("released_state_store_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import generated candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wrapper = getattr(module, "cake_state_store_b8_smoke", None)
    if not callable(wrapper):
        raise RuntimeError("generated wrapper entry point is missing")
    return wrapper


def _inputs(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(ELEMENTS, dtype=np.int32)
    if case_id == CASES[0]:
        state = np.zeros(ELEMENTS, dtype=np.float32)
        update = np.zeros(ELEMENTS, dtype=np.float32)
    elif case_id == CASES[1]:
        state = ((index % 17) - 8).astype(np.float32)
        update = (((index * 7) % 11) - 5).astype(np.float32)
    elif case_id == CASES[2]:
        state = np.where(index % 2 == 0, 0.125, -0.25).astype(np.float32)
        update = np.where(index % 4 < 2, 0.0625, -0.125).astype(np.float32)
    elif case_id == CASES[3]:
        generator = np.random.default_rng(20260902)
        state = (generator.integers(-64, 65, size=ELEMENTS) * 0.25).astype(np.float32)
        update = (generator.integers(-48, 49, size=ELEMENTS) * 0.125).astype(np.float32)
    else:
        raise RuntimeError(f"unknown frozen case: {case_id}")
    state = np.ascontiguousarray(state.reshape(SHAPE))
    update = np.ascontiguousarray(update.reshape(SHAPE))
    for name, value in (("state", state), ("update", update)):
        normal_or_zero = (value == 0) | (np.abs(value) >= np.finfo(np.float32).tiny)
        if value.dtype != np.float32 or not np.isfinite(value).all() or not normal_or_zero.all():
            raise RuntimeError(f"{case_id} {name} leaves the frozen finite normal-or-zero domain")
    return state, update


def _case(
    case_id: str,
    wrapper: Callable[..., tuple[()]],
    stage_dir: Path,
) -> tuple[dict[str, object], Path]:
    state_before, update_before = _inputs(case_id)
    expected = np.add(state_before, update_before, dtype=np.float32)
    state = torch.from_numpy(state_before.copy()).to(device="cuda")
    update = torch.from_numpy(update_before.copy()).to(device="cuda")
    state_pointer_before = int(state.data_ptr())
    returned = wrapper(state, update)
    torch.cuda.synchronize()
    state_pointer_after = int(state.data_ptr())
    actual = state.detach().cpu().numpy().copy()
    update_after = update.detach().cpu().numpy().copy()

    expected_bits = expected.view(np.uint32)
    actual_bits = actual.view(np.uint32)
    update_before_bits = update_before.view(np.uint32)
    update_after_bits = update_after.view(np.uint32)
    state_mismatch_count = int(np.count_nonzero(expected_bits != actual_bits))
    update_mismatch_count = int(
        np.count_nonzero(update_before_bits != update_after_bits)
    )
    max_error = float(
        np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
    )
    empty_tuple = isinstance(returned, tuple) and len(returned) == 0
    pointer_unchanged = state_pointer_before == state_pointer_after
    correct = (
        empty_tuple
        and pointer_unchanged
        and state_mismatch_count == 0
        and update_mismatch_count == 0
        and max_error == 0.0
    )
    artifact = stage_dir / f"{case_id}.complete-output.json"
    _atomic_json(
        artifact,
        {
            "schema": "open-cake.state-store-complete-output.v1",
            "case_id": case_id,
            "shape": list(SHAPE),
            "state_before": state_before.reshape(-1).tolist(),
            "update_before": update_before.reshape(-1).tolist(),
            "expected_state_after": expected.reshape(-1).tolist(),
            "actual_state_after": actual.reshape(-1).tolist(),
            "actual_update_after": update_after.reshape(-1).tolist(),
            "wrapper_returned_empty_tuple": empty_tuple,
            "state_data_ptr_unchanged": pointer_unchanged,
            "checked_state_elements": ELEMENTS,
            "state_mismatch_count": state_mismatch_count,
            "update_mismatch_count": update_mismatch_count,
            "max_error": max_error,
            "bitwise_required": True,
            "correct": correct,
        },
    )
    return (
        {
            "case_id": case_id,
            "correct": correct,
            "wrapper_returned_empty_tuple": empty_tuple,
            "state_data_ptr_unchanged": pointer_unchanged,
            "checked_state_elements": ELEMENTS,
            "state_mismatch_count": state_mismatch_count,
            "update_mismatch_count": update_mismatch_count,
            "max_error": max_error,
            "bitwise": state_mismatch_count == 0 and update_mismatch_count == 0,
        },
        artifact,
    )


def main() -> int:
    result_path = Path(os.environ["KERNELINFRA_RESULT"])
    stage_dir = Path(os.environ["KERNELINFRA_STAGE_DIR"])
    candidate_dir = Path(os.environ["KERNELINFRA_CANDIDATE_DIR"])
    try:
        if os.environ.get("KERNELINFRA_STAGE_KIND") != "correctness":
            raise RuntimeError("this frozen judge accepts only a correctness stage")
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
        details: list[dict[str, object]] = []
        artifacts: dict[str, str] = {}
        workloads: list[dict[str, object]] = []
        for case_id in CASES:
            detail, artifact = _case(case_id, wrapper, stage_dir)
            details.append(detail)
            artifacts[case_id] = str(artifact)
            workloads.append(
                {
                    "id": case_id,
                    "correct": bool(detail["correct"]),
                    "notes": (
                        f"checked=1024 state_mismatch_count={detail['state_mismatch_count']} "
                        f"update_mismatch_count={detail['update_mismatch_count']} "
                        f"max_error={detail['max_error']} empty_tuple={detail['wrapper_returned_empty_tuple']} "
                        f"state_ptr_unchanged={detail['state_data_ptr_unchanged']} bitwise={detail['bitwise']}"
                    ),
                }
            )
        valid = all(bool(detail["correct"]) for detail in details)
        result = {
            "schema": "kernelinfra.stage-result.v1",
            "status": "passed" if valid else "failed",
            "validity": "valid" if valid else "invalid",
            "summary": (
                "Four fixed [8,128] complete-output cases passed independent CPU float32, "
                "empty-tuple ABI, in-place pointer, and read-only update checks on B200."
                if valid
                else "At least one fixed [8,128] GPU correctness or ABI/in-place check failed."
            ),
            "workloads": workloads,
            "artifacts": artifacts,
            "metrics": {
                "device_name": device_name,
                "compute_capability": list(capability),
                "shape": list(SHAPE),
                "total_checked_state_elements": ELEMENTS * len(CASES),
                "performance_measured": False,
                "cases": details,
            },
        }
        _atomic_json(result_path, result)
        return 0 if valid else 1
    except Exception as error:
        _atomic_json(
            result_path,
            {
                "schema": "kernelinfra.stage-result.v1",
                "status": "failed",
                "validity": "unknown",
                "summary": f"correctness judge failure: {type(error).__name__}: {error}",
                "workloads": [],
                "metrics": {"performance_measured": False},
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
