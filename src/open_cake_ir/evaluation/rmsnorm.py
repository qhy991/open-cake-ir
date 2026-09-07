"""Frozen llama.cpp RMSNorm+Mul materialization, oracle and correctness metrics."""

from __future__ import annotations

from typing import Mapping, cast

from .workload import WorkloadContract


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _shape(workload: WorkloadContract, case_id: str) -> tuple[int, int, int]:
    shape = _object(workload.case(case_id)["shape"], "workload.case.shape")
    if set(shape) != {"B", "N", "D"}:
        raise ValueError("RMSNorm case shape differs")
    return int(shape["B"]), int(shape["N"]), int(shape["D"])


def generate_rmsnorm_case(
    workload: WorkloadContract,
    case_id: str,
    *,
    device: str,
) -> tuple[object, object]:
    """Materialize one Workload-owned contiguous x/gamma input pair."""

    torch = __import__("torch")
    case = workload.case(case_id)
    shape = _shape(workload, case_id)
    mode = case["mode"]
    if mode == "random_standard_normal_signed_gamma":
        seed = case["seed"]
        if not isinstance(seed, int):
            raise ValueError("random RMSNorm case requires a seed")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        x = torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
        gamma = 1.0 + 0.25 * torch.randn(
            (shape[2],), generator=generator, dtype=torch.float32, device=device
        )
        gamma[::17] *= -1.0
        return x.contiguous(), gamma.contiguous()
    if mode == "constructed_reduction_rsqrt_stress":
        rows = shape[0] * shape[1]
        columns = torch.arange(shape[2], dtype=torch.int64, device=device)
        row_ids = torch.arange(rows, dtype=torch.int64, device=device)
        selector = row_ids % 5
        alternating = torch.where(columns % 2 == 0, 3.0, -3.0)
        ramp = (columns.to(torch.float32) - shape[2] / 2) / shape[2]
        x = torch.empty((rows, shape[2]), dtype=torch.float32, device=device)
        x[selector == 0] = 0.0
        x[selector == 1] = torch.where(columns % 2 == 0, 1.0e-12, -1.0e-12)
        x[selector == 2] = torch.where(columns % 2 == 0, 1.0e4, -1.0e4)
        x[selector == 3] = alternating
        x[selector == 4] = ramp
        gamma_values = torch.tensor(
            [-1.5, -0.5, 0.0, 0.5, 1.5, 2.0],
            dtype=torch.float32,
            device=device,
        )
        gamma = gamma_values[columns % gamma_values.numel()]
        return x.reshape(shape).contiguous(), gamma.contiguous()
    raise ValueError(f"unsupported RMSNorm case mode {mode!r}")


def rmsnorm_oracle(
    workload: WorkloadContract,
    x: object,
    gamma: object,
) -> object:
    """Compose llama.cpp RMS_NORM+MUL in CPU FP64, then round once to FP32."""

    torch = __import__("torch")
    document = workload.document
    oracle = _object(document["oracle"], "workload.oracle")
    semantics = _object(document["semantics"], "workload.semantics")
    if oracle.get("kind") != "cpu_fp64_rmsnorm_mul_then_fp32_round":
        raise ValueError("RMSNorm oracle kind differs")
    epsilon = float(semantics["epsilon"])
    x64 = x.detach().cpu().to(torch.float64)
    gamma64 = gamma.detach().cpu().to(torch.float64)
    inverse_rms = torch.rsqrt(torch.mean(x64 * x64, dim=-1, keepdim=True) + epsilon)
    return (x64 * inverse_rms * gamma64).to(torch.float32).to(x.device)


def rmsnorm_metrics(
    workload: WorkloadContract,
    output: object,
    reference: object,
) -> dict[str, object]:
    """Apply the Workload-owned tolerance and expose complete diagnostics."""

    torch = __import__("torch")
    validation = _object(workload.document["validation"], "workload.validation")
    rtol = float(validation["rtol"])
    atol = float(validation["atol"])
    absolute = (output - reference).abs()
    relative = absolute / torch.maximum(
        reference.abs(), torch.tensor(1.0e-12, device=reference.device)
    )
    close = torch.isclose(output, reference, rtol=rtol, atol=atol)
    return {
        "passed": bool(close.all().item()),
        "mismatch_count": int((~close).sum().item()),
        "element_count": int(output.numel()),
        "max_abs_error": float(absolute.max().item()),
        "max_rel_error": float(relative.max().item()),
        "rtol": rtol,
        "atol": atol,
        "nonfinite_output_count": int((~torch.isfinite(output)).sum().item()),
    }
