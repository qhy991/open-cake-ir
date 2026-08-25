"""TinyGEMM2 Workload adapter behind the common Evaluation receipt."""

from __future__ import annotations

from hashlib import sha256
from types import MappingProxyType
from typing import Mapping, Protocol, cast

from .core import EvaluationProtocol, EvaluationReceipt, LaunchableCandidate, LaunchObservation
from .workload import WorkloadContract


class TinyGemmLauncher(Protocol):
    """Launch one sealed TinyGEMM candidate."""

    def launch(
        self,
        candidate: LaunchableCandidate,
        input_tensor: object,
        weight: object,
        bias: object,
    ) -> LaunchObservation:
        """Launch exactly once and return the output plus route receipt."""


def _materialized_receipt(tensor: object) -> tuple[str, int]:
    torch = __import__("torch")
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
    ):
        raise ValueError("TinyGEMM materialization must be contiguous BF16")
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes(order="C")
    return sha256(raw).hexdigest(), len(raw)


def _require_materialized_match(
    workload: WorkloadContract,
    case_id: str,
    name: str,
    tensor: object,
) -> None:
    materialized = workload.case(case_id).get("materialized")
    if not isinstance(materialized, Mapping) or name not in materialized:
        raise ValueError("TinyGEMM materialized authority is missing")
    expected = cast(Mapping[str, object], materialized[name])
    observed_sha256, observed_size = _materialized_receipt(tensor)
    if (
        observed_sha256 != expected.get("sha256")
        or observed_size != expected.get("size_bytes")
    ):
        raise ValueError(f"TinyGEMM {name} materialization differs")


def generate_tinygemm_case(
    workload: WorkloadContract,
    case_id: str,
    *,
    device: str,
) -> tuple[object, object, object]:
    """Materialize and verify the contract's sole deterministic CUDA case."""

    if workload.document.get("operator") != "tinygemm2_bf16_linear":
        raise ValueError("TinyGEMM adapter received another Workload")
    case = workload.case(case_id)
    shape = case["shape"]
    if not isinstance(shape, dict) or case.get("mode") != "random_normal":
        raise ValueError("TinyGEMM case shape or mode differs")
    if not device.startswith("cuda"):
        raise ValueError("TinyGEMM exact materialization requires CUDA")
    batch = int(shape["batch"])
    output_features = int(shape["output_features"])
    input_features = int(shape["input_features"])
    torch = __import__("torch")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(case["seed"]))
    input_tensor = torch.randn(
        (batch, input_features), generator=generator, device=device, dtype=torch.float32
    ).div(8).to(torch.bfloat16)
    weight = torch.randn(
        (output_features, input_features),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).div(8).to(torch.bfloat16)
    bias = torch.randn(
        (output_features,), generator=generator, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    tensors = input_tensor.contiguous(), weight.contiguous(), bias.contiguous()
    for name, tensor in zip(("input", "weight", "bias"), tensors, strict=True):
        _require_materialized_match(workload, case_id, name, tensor)
    return tensors


def tinygemm_oracle(
    workload: WorkloadContract,
    input_tensor: object,
    weight: object,
    bias: object,
    *,
    case_id: str,
) -> object:
    """Apply the Workload-owned CPU FP32 linear semantics then BF16 rounding."""

    if workload.document["oracle"]["kind"] != "fp32_linear_then_bf16_round":
        raise ValueError("TinyGEMM oracle authority differs")
    torch = __import__("torch")
    oracle = torch.nn.functional.linear(
        input_tensor.detach().cpu().float(),
        weight.detach().cpu().float(),
        bias.detach().cpu().float(),
    ).to(torch.bfloat16)
    _require_materialized_match(
        workload, case_id, "fp32_linear_bf16_oracle", oracle
    )
    return oracle


def tinygemm_metrics(
    workload: WorkloadContract,
    output: object,
    oracle: object,
    *,
    case_id: str,
) -> dict[str, object]:
    """Derive parent-exact and oracle-tolerance dispositions."""

    torch = __import__("torch")
    if tuple(output.shape) != tuple(oracle.shape) or output.dtype != torch.bfloat16:
        raise ValueError("TinyGEMM output contract differs")
    tolerance = workload.document["oracle"]
    materialized = workload.case(case_id).get("materialized")
    if not isinstance(materialized, Mapping):
        raise ValueError("TinyGEMM materialized authority is missing")
    parent = cast(Mapping[str, object], materialized["parent_output"])
    output_sha256, output_size = _materialized_receipt(output)
    parent_exact = bool(
        output_sha256 == parent.get("sha256")
        and output_size == parent.get("size_bytes")
    )
    output_fp32 = output.detach().cpu().float()
    oracle_fp32 = oracle.detach().cpu().float()
    close = bool(
        torch.allclose(
            output_fp32,
            oracle_fp32,
            atol=float(tolerance["atol"]),
            rtol=float(tolerance["rtol"]),
            equal_nan=False,
        )
    )
    maximum = float((output_fp32 - oracle_fp32).abs().max().item())
    return {
        "bitwise_parent_equal": parent_exact,
        "tolerance_equal": close,
        "maximum_absolute_error": maximum,
    }


def evaluate_tinygemm(
    candidate: LaunchableCandidate,
    workload: WorkloadContract,
    protocol: EvaluationProtocol,
    launcher: TinyGemmLauncher,
    *,
    device: str,
) -> EvaluationReceipt:
    """Materialize, externally oracle, launch once, and issue a common receipt."""

    if protocol.workload_sha256 != workload.canonical_sha256 or protocol.timing != "none":
        raise ValueError("TinyGEMM Evaluation Protocol differs")
    input_tensor, weight, bias = generate_tinygemm_case(
        workload, protocol.case_id, device=device
    )
    oracle = tinygemm_oracle(
        workload, input_tensor, weight, bias, case_id=protocol.case_id
    )
    launch = launcher.launch(candidate, input_tensor, weight, bias)
    metrics = tinygemm_metrics(
        workload, launch.output, oracle, case_id=protocol.case_id
    )
    return EvaluationReceipt(
        candidate_sha256=candidate.candidate_sha256,
        workload_sha256=workload.canonical_sha256,
        evaluation_protocol_sha256=protocol.canonical_sha256,
        purpose=protocol.purpose,
        case_id=protocol.case_id,
        correctness_passed=bool(
            metrics["bitwise_parent_equal"] and metrics["tolerance_equal"]
        ),
        correctness=MappingProxyType(metrics),
        kernel_calls=launch.kernel_calls,
        fallback_calls=launch.fallback_calls,
        launch_receipt_sha256=launch.launch_receipt_sha256,
        timing=None,
    )
