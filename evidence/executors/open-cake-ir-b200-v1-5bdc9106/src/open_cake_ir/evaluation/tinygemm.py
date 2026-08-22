"""TinyGEMM2 Workload adapter behind the common Evaluation receipt."""

from __future__ import annotations

from types import MappingProxyType
from typing import Protocol

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


def generate_tinygemm_case(
    workload: WorkloadContract,
    case_id: str,
    *,
    device: str,
) -> tuple[object, object, object]:
    """Materialize the contract's sole deterministic random-normal case."""

    if workload.document.get("operator") != "tinygemm2_bf16_linear":
        raise ValueError("TinyGEMM adapter received another Workload")
    case = workload.case(case_id)
    shape = case["shape"]
    if not isinstance(shape, dict) or case.get("mode") != "random_normal":
        raise ValueError("TinyGEMM case shape or mode differs")
    batch = int(shape["batch"])
    output_features = int(shape["output_features"])
    input_features = int(shape["input_features"])
    torch = __import__("torch")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(case["seed"]))
    input_tensor = torch.randn(
        (batch, input_features), generator=generator, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    weight = torch.randn(
        (output_features, input_features),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    bias = torch.randn(
        (output_features,), generator=generator, device=device, dtype=torch.float32
    ).to(torch.bfloat16)
    return input_tensor.contiguous(), weight.contiguous(), bias.contiguous()


def tinygemm_oracle(
    workload: WorkloadContract,
    input_tensor: object,
    weight: object,
    bias: object,
) -> object:
    """Apply the Workload-owned FP32 linear semantics then BF16 rounding."""

    if workload.document["oracle"]["kind"] != "fp32_linear_then_bf16_round":
        raise ValueError("TinyGEMM oracle authority differs")
    torch = __import__("torch")
    return (
        torch.matmul(input_tensor.to(torch.float32), weight.to(torch.float32).transpose(0, 1))
        + bias.to(torch.float32)
    ).to(torch.bfloat16)


def tinygemm_metrics(
    workload: WorkloadContract,
    output: object,
    oracle: object,
) -> dict[str, object]:
    """Derive exact and tolerance dispositions without trusting candidate flags."""

    torch = __import__("torch")
    if tuple(output.shape) != tuple(oracle.shape) or output.dtype != torch.bfloat16:
        raise ValueError("TinyGEMM output contract differs")
    tolerance = workload.document["oracle"]
    exact = bool(torch.equal(output, oracle))
    close = bool(
        torch.allclose(
            output.to(torch.float32),
            oracle.to(torch.float32),
            atol=float(tolerance["atol"]),
            rtol=float(tolerance["rtol"]),
            equal_nan=False,
        )
    )
    maximum = float(
        (output.to(torch.float32) - oracle.to(torch.float32)).abs().max().item()
    )
    return {"bitwise_equal": exact, "tolerance_equal": close, "maximum_absolute_error": maximum}


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
    oracle = tinygemm_oracle(workload, input_tensor, weight, bias)
    launch = launcher.launch(candidate, input_tensor, weight, bias)
    metrics = tinygemm_metrics(workload, launch.output, oracle)
    return EvaluationReceipt(
        candidate_sha256=candidate.candidate_sha256,
        workload_sha256=workload.canonical_sha256,
        evaluation_protocol_sha256=protocol.canonical_sha256,
        purpose=protocol.purpose,
        case_id=protocol.case_id,
        correctness_passed=bool(metrics["bitwise_equal"] and metrics["tolerance_equal"]),
        correctness=MappingProxyType(metrics),
        kernel_calls=launch.kernel_calls,
        fallback_calls=launch.fallback_calls,
        launch_receipt_sha256=launch.launch_receipt_sha256,
        timing=None,
    )
