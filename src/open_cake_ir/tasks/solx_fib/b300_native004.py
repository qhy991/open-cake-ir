"""Exact B300 native-CUDA Cake authoring seed for FlashInfer GEMM task 004."""
from __future__ import annotations

from pathlib import Path

from open_cake_ir.evaluation.workload import WorkloadContract

from .gemm import TASKS, validate_contract

_CANDIDATE_PATH = Path(__file__).with_name("b300_native004_candidate.py")
_K256_PATH = Path(__file__).with_name("b300_native004_k256_candidate.py")


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
    """Return the complete K64 Cake Schedule for exact M8828.

    This authoring seed passed correctness but lost the paired timing assay.
    Retaining it makes the K256 successor's one changed mapping inspectable.
    """
    _admit_exact_m8828(workload, case_id)
    return _CANDIDATE_PATH.read_text(encoding="utf-8")


def native_m8828_k256_source(workload: WorkloadContract,
                             case_id: str = "primary") -> str:
    """Return the bounded K256 TMA/tcgen05 successor for exact M8828.

    This is a candidate for device qualification, not a task-level selection.
    """
    _admit_exact_m8828(workload, case_id)
    return _K256_PATH.read_text(encoding="utf-8")
