"""Independent Evaluation oracle for Qwen2.5-Omni audio average pooling."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, cast

from .core import LaunchObservation
from .workload import WorkloadContract

_WORKLOAD_ID = "qwen25-omni-audio-avg-pool1d-k2-s2-bf16-independent-v1"
_OPERATOR = "qwen25_omni_audio_avg_pool1d_bf16"
_ROOT_FIELDS = {
    "schema_version",
    "workload_id",
    "revision",
    "state",
    "provenance",
    "operator",
    "cases",
    "tensors",
    "semantics",
    "oracle",
    "validation",
}
_EXPECTED_PROVENANCE = [
    {
        "kind": "upstream_git_callsite",
        "repository": "https://github.com/qhy991/ApxInf.git",
        "revision": "32ba4d47b961b495dd878ccb99dfe6bd21d7970f",
        "path": "crates/apxinf-model/src/qwen25_omni/audio.rs",
        "call": "avg_pool1d(&hidden, 2, 2)",
    },
    {
        "kind": "upstream_git_kernel",
        "repository": "https://github.com/qhy991/ApxInf.git",
        "revision": "32ba4d47b961b495dd878ccb99dfe6bd21d7970f",
        "path": "crates/apxinf-cuda/kernels/custom/preprocess.cuh",
        "symbol": "avg_pool1d_bf16_kernel",
    },
    {
        "kind": "upstream_git_configuration",
        "repository": "https://github.com/qhy991/ApxInf.git",
        "revision": "32ba4d47b961b495dd878ccb99dfe6bd21d7970f",
        "path": "crates/apxinf-model/src/qwen25_omni/config.rs",
        "assertion": "audio.hidden_size == 1280",
    },
]
_EXPECTED_CASES = [
    {
        "case_id": "rounding_fixture_frames4_features2",
        "shape": {"input_frames": 4, "output_frames": 2, "features": 2},
        "seed": None,
        "mode": "constructed_bf16_rounding_fixture",
    },
    {
        "case_id": "audio_hidden_smoke_frames4_features1280",
        "shape": {"input_frames": 4, "output_frames": 2, "features": 1280},
        "seed": None,
        "mode": "constructed_finite_bf16_smoke",
    },
]
_EXPECTED_TENSORS = {
    "input": {
        "shape": ["input_frames", "features"],
        "dtype": "bf16",
        "layout": "contiguous_row_major",
        "finite_only": True,
    },
    "output": {
        "shape": ["output_frames", "features"],
        "dtype": "bf16",
        "layout": "contiguous_row_major",
    },
}
_EXPECTED_SEMANTICS = {
    "definition": (
        "output[o,c] = bf16_rne((fp32(input[2*o,c]) + "
        "fp32(input[2*o+1,c])) / fp32(2))"
    ),
    "kernel_size": 2,
    "stride": 2,
    "padding": 0,
    "frame_domain": (
        "input_frames_is_even_and_output_frames_equals_input_frames_div_2"
    ),
    "odd_input_frames": "unsupported",
    "accumulator_dtype": "fp32",
    "accumulation_order": "positive_zero_then_tap_0_then_tap_1",
    "mean_scaling": "fp32_divide_by_2",
    "output_conversion": "one_bfloat16_round_to_nearest_ties_to_even",
    "nonfinite": "reject_before_oracle",
}
_ROUNDING_INPUT_BITS = (
    0x3F80,
    0x3F81,
    0x3F81,
    0x3F82,
    0xC080,
    0x4780,
    0x4000,
    0xC780,
)
_ROUNDING_OUTPUT_BITS = (0x3F80, 0x3F82, 0xBF80, 0x0000)
_EXPECTED_ORACLE = {
    "kind": "packed_little_endian_bf16_fp32_tap_order",
    "rounding_fixture_input_bf16_bits": list(_ROUNDING_INPUT_BITS),
    "rounding_fixture_output_bf16_bits": list(_ROUNDING_OUTPUT_BITS),
    "smoke_input_formula": (
        "bf16_rne(fp32((((frame * 17 + feature * 13) % 257) - 128) / 16))"
    ),
}
_EXPECTED_VALIDATION = {
    "canonical": "exact_bf16_output_bits",
    "atol": 0.0,
    "rtol": 0.0,
    "paper_acceptance_rule": "not_applicable_operator_representation_probe",
}


class Qwen25OmniAudioAvgPoolLauncher(Protocol):
    """Launch one implementation using the contract's packed BF16 ABI."""

    def launch(
        self,
        input_bf16: bytes,
        *,
        input_frames: int,
        output_frames: int,
        features: int,
    ) -> LaunchObservation:
        """Return one output and its already-validated route observation."""


@dataclass(frozen=True)
class Qwen25OmniAudioAvgPoolEvaluation:
    """One exact-bit Evaluation result, independent of a backend driver."""

    workload_sha256: str
    case_id: str
    correctness_passed: bool
    correctness: Mapping[str, object]
    kernel_calls: int
    fallback_calls: int
    launch_receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "correctness", MappingProxyType(dict(self.correctness)))


def _validate_document(document: Mapping[str, object]) -> None:
    if (
        set(document) != _ROOT_FIELDS
        or document.get("schema_version") != 1
        or document.get("workload_id") != _WORKLOAD_ID
        or document.get("revision") != "1"
        or document.get("state") != "frozen"
        or document.get("operator") != _OPERATOR
    ):
        raise ValueError("Qwen2.5-Omni average-pool workload identity differs")
    if document.get("provenance") != _EXPECTED_PROVENANCE:
        raise ValueError("Qwen2.5-Omni average-pool workload provenance differs")
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Qwen2.5-Omni average-pool workload cases differ")
    for raw_case in cases:
        if not isinstance(raw_case, Mapping) or not isinstance(
            raw_case.get("shape"), Mapping
        ):
            raise ValueError("Qwen2.5-Omni average-pool workload cases differ")
        shape = cast(Mapping[str, object], raw_case["shape"])
        input_frames = shape.get("input_frames")
        output_frames = shape.get("output_frames")
        if (
            not isinstance(input_frames, int)
            or isinstance(input_frames, bool)
            or not isinstance(output_frames, int)
            or isinstance(output_frames, bool)
            or input_frames % 2 != 0
            or output_frames * 2 != input_frames
        ):
            raise ValueError("Qwen2.5-Omni average-pool requires exact even frames")
    if cases != _EXPECTED_CASES:
        raise ValueError("Qwen2.5-Omni average-pool workload cases differ")
    if document.get("tensors") != _EXPECTED_TENSORS:
        raise ValueError("Qwen2.5-Omni average-pool workload tensors differ")
    if document.get("semantics") != _EXPECTED_SEMANTICS:
        raise ValueError("Qwen2.5-Omni average-pool workload semantics differ")
    if document.get("oracle") != _EXPECTED_ORACLE:
        raise ValueError("Qwen2.5-Omni average-pool workload oracle differs")
    if document.get("validation") != _EXPECTED_VALIDATION:
        raise ValueError("Qwen2.5-Omni average-pool workload validation differs")


def load_qwen25_omni_audio_avg_pool_workload(
    path: str | Path,
) -> WorkloadContract:
    """Load the operator-specific closed contract without widening the common registry."""

    source = Path(path).resolve(strict=True)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Qwen2.5-Omni average-pool workload must be an object")
    document = cast(Mapping[str, object], raw)
    _validate_document(document)
    return WorkloadContract(document, source)


def _require_workload(workload: WorkloadContract) -> None:
    _validate_document(workload.document)


def _shape(workload: WorkloadContract, case_id: str) -> tuple[int, int, int]:
    case = workload.case(case_id)
    shape = cast(Mapping[str, object], case["shape"])
    input_frames = cast(int, shape["input_frames"])
    output_frames = cast(int, shape["output_frames"])
    features = cast(int, shape["features"])
    if input_frames % 2 != 0 or output_frames * 2 != input_frames:
        raise ValueError("Qwen2.5-Omni average-pool requires exact even frames")
    return input_frames, output_frames, features


def _as_fp32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _bf16_to_fp32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _fp32_to_bf16_rne(value: float) -> int:
    value = _as_fp32(value)
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if (bits & 0x7F800000) == 0x7F800000:
        return bits >> 16
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + rounding_bias) >> 16) & 0xFFFF


def _pack_bf16(bits: tuple[int, ...] | list[int]) -> bytes:
    return struct.pack(f"<{len(bits)}H", *bits)


def _unpack_bf16(payload: object, count: int, label: str) -> tuple[int, ...]:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError(f"Qwen2.5-Omni average-pool {label} must be packed BF16 bytes")
    raw = bytes(payload)
    if len(raw) != count * 2:
        raise ValueError(f"Qwen2.5-Omni average-pool {label} shape differs")
    return cast(tuple[int, ...], struct.unpack(f"<{count}H", raw))


def generate_qwen25_omni_audio_avg_pool_case(
    workload: WorkloadContract,
    case_id: str,
) -> bytes:
    """Generate one deterministic, contiguous row-major packed BF16 input."""

    _require_workload(workload)
    input_frames, _, features = _shape(workload, case_id)
    mode = workload.case(case_id)["mode"]
    if mode == "constructed_bf16_rounding_fixture":
        if (input_frames, features) != (4, 2):
            raise ValueError("Qwen2.5-Omni rounding fixture shape differs")
        return _pack_bf16(list(_ROUNDING_INPUT_BITS))
    if mode == "constructed_finite_bf16_smoke":
        bits = [
            _fp32_to_bf16_rne(
                _as_fp32((((frame * 17 + feature * 13) % 257) - 128) / 16.0)
            )
            for frame in range(input_frames)
            for feature in range(features)
        ]
        return _pack_bf16(bits)
    raise ValueError("Qwen2.5-Omni average-pool case mode is unsupported")


def qwen25_omni_audio_avg_pool_oracle(
    workload: WorkloadContract,
    input_bf16: object,
    *,
    case_id: str,
) -> bytes:
    """Accumulate tap 0 then tap 1 in FP32, divide by two, and round once."""

    _require_workload(workload)
    input_frames, output_frames, features = _shape(workload, case_id)
    input_bits = _unpack_bf16(input_bf16, input_frames * features, "input")
    if any((bits & 0x7F80) == 0x7F80 for bits in input_bits):
        raise ValueError("Qwen2.5-Omni average-pool input must be finite BF16")

    output_bits: list[int] = []
    for output_frame in range(output_frames):
        tap_0_base = output_frame * 2 * features
        tap_1_base = tap_0_base + features
        for feature in range(features):
            pool_sum = _as_fp32(0.0)
            pool_sum = _as_fp32(
                pool_sum + _bf16_to_fp32(input_bits[tap_0_base + feature])
            )
            pool_sum = _as_fp32(
                pool_sum + _bf16_to_fp32(input_bits[tap_1_base + feature])
            )
            pool_average = _as_fp32(pool_sum / 2.0)
            output_bits.append(_fp32_to_bf16_rne(pool_average))
    return _pack_bf16(output_bits)


def qwen25_omni_audio_avg_pool_metrics(
    workload: WorkloadContract,
    candidate_output_bf16: object,
    oracle_output_bf16: object,
    *,
    case_id: str,
) -> dict[str, object]:
    """Compare candidate and oracle as exact BF16 bit patterns."""

    _require_workload(workload)
    _, output_frames, features = _shape(workload, case_id)
    count = output_frames * features
    candidate = _unpack_bf16(candidate_output_bf16, count, "candidate output")
    oracle = _unpack_bf16(oracle_output_bf16, count, "oracle output")
    mismatch_count = sum(
        observed != expected for observed, expected in zip(candidate, oracle)
    )
    return {
        "schema_version": 1,
        "total_output_elements": count,
        "exact_bf16_bits": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "accumulator_dtype": "fp32",
        "accumulation_order": "positive_zero_then_tap_0_then_tap_1",
        "mean_scaling": "fp32_divide_by_2",
        "output_conversion": "one_bfloat16_round_to_nearest_ties_to_even",
    }


def classify_qwen25_omni_audio_avg_pool_output(
    workload: WorkloadContract,
    candidate_output_bf16: object,
    oracle_output_bf16: object,
    *,
    case_id: str,
) -> tuple[bool, dict[str, object]]:
    """Keep a malformed candidate output as a failed Evaluation outcome."""

    try:
        metrics = qwen25_omni_audio_avg_pool_metrics(
            workload,
            candidate_output_bf16,
            oracle_output_bf16,
            case_id=case_id,
        )
    except ValueError as error:
        if "candidate output" not in str(error):
            raise
        return False, {
            "schema_version": 1,
            "failure_code": "candidate_output_contract_violation",
        }
    return bool(metrics["exact_bf16_bits"]), metrics


def evaluate_qwen25_omni_audio_avg_pool(
    workload: WorkloadContract,
    case_id: str,
    launcher: Qwen25OmniAudioAvgPoolLauncher,
) -> Qwen25OmniAudioAvgPoolEvaluation:
    """Generate, oracle, launch once, and classify an implementation exactly."""

    input_bf16 = generate_qwen25_omni_audio_avg_pool_case(workload, case_id)
    oracle = qwen25_omni_audio_avg_pool_oracle(
        workload, input_bf16, case_id=case_id
    )
    input_frames, output_frames, features = _shape(workload, case_id)
    launch = launcher.launch(
        input_bf16,
        input_frames=input_frames,
        output_frames=output_frames,
        features=features,
    )
    passed, metrics = classify_qwen25_omni_audio_avg_pool_output(
        workload,
        launch.output,
        oracle,
        case_id=case_id,
    )
    return Qwen25OmniAudioAvgPoolEvaluation(
        workload_sha256=workload.canonical_sha256,
        case_id=case_id,
        correctness_passed=passed,
        correctness=metrics,
        kernel_calls=launch.kernel_calls,
        fallback_calls=launch.fallback_calls,
        launch_receipt_sha256=launch.launch_receipt_sha256,
    )
