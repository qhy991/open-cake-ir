"""Exact B300 native-CUDA Cake authoring seed for FlashInfer GEMM task 004."""
from __future__ import annotations

from pathlib import Path

from open_cake_ir.evaluation.workload import WorkloadContract

from .gemm import TASKS, validate_contract

_CANDIDATE_PATH = Path(__file__).with_name("b300_native004_candidate.py")
_S4_PATH = Path(__file__).with_name("b300_native004_s4_candidate.py")


def _admit_exact_m8828(workload: WorkloadContract, case_id: str) -> None:
    validate_contract(workload.document)
    if (workload.target != "sm_103a"
            or workload.document["operator"] != TASKS["fib_gemm_n128_k2048"][0]):
        raise ValueError("native FIB 004 seed requires the exact B300 task")
    a, b, out = workload.tensor_abi(case_id)
    if (a.shape != (8828, 2048) or b.shape != (128, 2048)
            or out.shape != (8828, 128)
            or (a.dtype, b.dtype, out.dtype) != ("fp16", "fp16", "fp16")):
        raise ValueError("native FIB 004 seed is bounded to official M8828")


def native_m8828_source(workload: WorkloadContract,
                        case_id: str = "primary") -> str:
    """Return the K64 two-stage Cake Schedule retained as a slow control."""
    _admit_exact_m8828(workload, case_id)
    return _CANDIDATE_PATH.read_text(encoding="utf-8")


def native_m8828_s4_source(workload: WorkloadContract,
                            case_id: str = "primary") -> str:
    """Return the legal K64 mapping with a four-stage TMA pipeline.

    A 128-byte swizzle row is retained; deeper staging is a separate bounded
    performance hypothesis, pending device correctness and timing.
    """
    _admit_exact_m8828(workload, case_id)
    return _S4_PATH.read_text(encoding="utf-8")
