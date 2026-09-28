"""Exact B300 native TMA/tcgen05 Cake seed for FIB 009 M8828."""
from __future__ import annotations

from pathlib import Path

from open_cake_ir.evaluation.workload import WorkloadContract

from .gemm import TASKS, validate_contract


_CANDIDATE_PATH = Path(__file__).with_name("b300_native009_candidate.py")
_N128_PATH = Path(__file__).with_name("b300_native009_n128_candidate.py")


def _admit_exact_m8828(workload: WorkloadContract, case_id: str) -> None:
    validate_contract(workload.document)
    if (workload.target != "sm_103a"
            or workload.document["operator"] != TASKS["fib_gemm_n5120_k2048"][0]):
        raise ValueError("native FIB 009 seed requires the exact B300 task")
    a, b, out = workload.tensor_abi(case_id)
    if (a.shape != (8828, 2048) or b.shape != (5120, 2048)
            or out.shape != (8828, 5120)
            or (a.dtype, b.dtype, out.dtype) != ("fp16", "fp16", "fp16")):
        raise ValueError("native FIB 009 seed is bounded to official M8828")


def native_m8828_source(workload: WorkloadContract,
                        case_id: str = "primary") -> str:
    """Read the complete, exact N64 TMA/tcgen05 Schedule."""
    _admit_exact_m8828(workload, case_id)
    return _CANDIDATE_PATH.read_text(encoding="utf-8")


def native_m8828_n128_source(workload: WorkloadContract,
                             case_id: str = "primary") -> str:
    """Double N ownership with a complete N128 TMA/tcgen05 Schedule."""
    _admit_exact_m8828(workload, case_id)
    return _N128_PATH.read_text(encoding="utf-8")
