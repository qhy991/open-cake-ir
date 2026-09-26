"""Fixed C550 E4M3FN GEMM semantics, independent of its Cake lowering."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping

from open_cake_ir.evaluation.workload import WorkloadContract

OPERATOR = "metax_fp8_e4m3_gemm_fp32"
WORKLOAD_ID = "metax-fp8-e4m3-gemm-fp32-xcore1002-m64-n64-k64-v1"
SIZE = 64
_FINITE_CODES = tuple(code for code in range(256) if code not in (0x7F, 0xFF))


def decode_e4m3fn(code: int) -> float:
    """Decode one finite E4M3FN byte without a framework or compiler implementation."""
    if type(code) is not int or not 0 <= code <= 255 or code in (0x7F, 0xFF):
        raise ValueError("FP8 input must have a finite E4M3FN encoding")
    sign = -1 if code & 0x80 else 1
    exponent = (code >> 3) & 15
    fraction = code & 7
    if exponent == 0:
        return sign * fraction * 2.0 ** -9
    return sign * (8 + fraction) * 2.0 ** (exponent - 10)


_BOUNDED_CODES = tuple(code for code in _FINITE_CODES if abs(decode_e4m3fn(code)) <= 2)


def workload_document() -> dict:
    cases = [
        {"case_id": mode, "shape": {"M": SIZE, "N": SIZE, "K": SIZE},
         "seed": 20260926 + index, "mode": mode}
        for index, mode in enumerate(("primary", "zeros", "identity", "alternating", "mixed_magnitude"))
    ]
    cases += [
        {"case_id": f"heldout_{domain}_{index:02d}",
         "shape": {"M": SIZE, "N": SIZE, "K": SIZE},
         "seed": 20261000 + 100 * (domain == "full_finite") + index,
         "mode": domain}
        for domain in ("full_finite", "bounded") for index in range(16)
    ]
    return {
        "schema_version": 1, "workload_id": WORKLOAD_ID, "revision": "1",
        "state": "frozen", "operator": OPERATOR,
        "provenance": [
            {"kind": "metax_fp8_precision_investigation",
             "path": "findings/2026-09-26-002-metax-fp64-reduction-capacity.json",
             "scope": "fixed_shape_semantics_only; old_device_and_timing_receipts_do_not_qualify_this_successor"},
        ],
        "cases": cases,
        "tensors": {
            "a": {"shape": ["M", "K"], "dtype": "fp8_e4m3", "layout": "contiguous_row_major", "finite_only": True},
            "b": {"shape": ["K", "N"], "dtype": "fp8_e4m3", "layout": "contiguous_row_major", "finite_only": True},
            "out": {"shape": ["M", "N"], "dtype": "fp32", "layout": "contiguous_row_major", "finite_only": True},
        },
        "semantics": {
            "target": "xcore1002", "candidate_abi": {"inputs": ["a", "b"], "outputs": ["out"]},
            "definition": "out[m,n] = round_fp32(sum_k(decode_e4m3fn(a[m,k]) * decode_e4m3fn(b[k,n])))",
            "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
            "materialization": "task_owned_xorshift32_finite_E4M3FN_bytes; no candidate or host RNG",
            "exclusions": ["fixed_M64_N64_K64_only", "not_native_FP8_MMA_evidence", "no_framework_or_serving_claim"],
        },
        "oracle": {
            "kind": "independent_cpu_high_precision_fp8_gemm",
            "callable": "open_cake_ir.tasks.metax_fp8_gemm.reference_tensors",
            "implementation": "standalone_E4M3FN_byte_decode; exact_dyadic_binary64_products; math.fsum_K; one_FP32_RNE_round",
        },
        "validation": {
            "primary_case": "primary", "all_cases_required": True,
            "comparison": "elementwise_atol_rtol", "atol": 0.001, "rtol": 0.0001,
            "equal_nan": False,
            "qualification": "all_37_cases_all_4096_outputs; C550_device_and_timing_pending",
            "tolerance_rationale": "Preserves the preceding five-case precision bound while adding 32 finite heldout distributions that can reject ordinary FP32 summation.",
        },
    }


def validate_contract(document: Mapping) -> None:
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(workload_document(), sort_keys=True):
        raise ValueError("MetaX FP8 GEMM frozen contract differs")
    workload = WorkloadContract(document)
    for case_id in workload.case_ids:
        workload.tensor_abi(case_id)


def _bytes_for_case(case: Mapping) -> tuple[bytes, bytes]:
    state = case["seed"]

    def next_code(domain: tuple[int, ...]) -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        return domain[state % len(domain)]

    mode = case["mode"]
    domain = _FINITE_CODES if mode in {"full_finite", "mixed_magnitude"} else _BOUNDED_CODES
    a = bytearray(next_code(domain) for _ in range(SIZE * SIZE))
    b = bytearray(next_code(domain) for _ in range(SIZE * SIZE))
    if mode == "zeros":
        a[:] = bytes(len(a))
        b[:] = bytes(len(b))
    elif mode == "identity":
        a[:] = bytes(len(a))
        for index in range(SIZE):
            a[index * SIZE + index] = 0x38  # +1.0
    elif mode == "alternating":
        for row in range(SIZE):
            for k in range(SIZE):
                a[row * SIZE + k] = 0x38 if (row + k) % 2 else 0xB8
    return bytes(a), bytes(b)


def reference_bytes(a: bytes, b: bytes) -> list[float]:
    """Return the complete FP32 oracle; useful without optional torch installed."""
    if len(a) != SIZE * SIZE or len(b) != SIZE * SIZE:
        raise ValueError("FP8 GEMM input shape differs")
    left = [decode_e4m3fn(code) for code in a]
    right = [decode_e4m3fn(code) for code in b]
    return [
        struct.unpack("<f", struct.pack("<f", math.fsum(
            left[m * SIZE + k] * right[k * SIZE + n] for k in range(SIZE))))[0]
        for m in range(SIZE) for n in range(SIZE)
    ]


def materialize_tensors(workload: WorkloadContract, case_id: str):
    import torch

    validate_contract(workload.document)
    a, b = _bytes_for_case(workload.case(case_id))
    return {
        name: torch.tensor(list(value), dtype=torch.uint8).reshape(SIZE, SIZE).view(torch.float8_e4m3fn)
        for name, value in (("a", a), ("b", b))
    }


def reference_tensors(workload: WorkloadContract, case_id: str, inputs: Mapping):
    import torch

    validate_contract(workload.document)
    workload.case(case_id)
    if set(inputs) != {"a", "b"}:
        raise ValueError("FP8 GEMM input ABI differs")
    for name in ("a", "b"):
        value = inputs[name]
        if (tuple(value.shape) != (SIZE, SIZE) or value.dtype != torch.float8_e4m3fn
                or value.device.type != "cpu" or not value.is_contiguous()):
            raise ValueError(f"FP8 GEMM input {name} shape/dtype/device differs")
    a = bytes(inputs["a"].view(torch.uint8).reshape(-1).tolist())
    b = bytes(inputs["b"].view(torch.uint8).reshape(-1).tolist())
    return {"out": torch.tensor(reference_bytes(a, b), dtype=torch.float32).reshape(SIZE, SIZE)}
