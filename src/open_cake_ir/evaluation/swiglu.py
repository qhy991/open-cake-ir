"""Frozen SwiGLU materialization, independent oracle and correctness metrics."""

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
        raise ValueError("SwiGLU case shape differs")
    return int(shape["B"]), int(shape["N"]), int(shape["D"])


def generate_swiglu_case(
    workload: WorkloadContract,
    case_id: str,
    *,
    device: str,
) -> tuple[object, object]:
    """Materialize one declared deterministic input pair."""

    torch = __import__("torch")
    case = workload.case(case_id)
    shape = _shape(workload, case_id)
    mode = case["mode"]
    if mode == "random_standard_normal":
        seed = case["seed"]
        if not isinstance(seed, int):
            raise ValueError("random SwiGLU case requires a seed")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32, device=device),
            torch.randn(shape, generator=generator, dtype=torch.float32, device=device),
        )
    if mode == "constructed_signed_saturation":
        count = shape[0] * shape[1] * shape[2]
        index = torch.arange(count, dtype=torch.int64, device=device)
        up_values = torch.tensor(
            [-3.0, -1.0, -0.0, 0.0, 0.25, 1.0, 3.0],
            dtype=torch.float32,
            device=device,
        )
        gate_values = torch.tensor(
            [-40.0, -20.0, -4.0, -1.0, -0.0, 0.0, 1.0, 4.0, 20.0, 40.0],
            dtype=torch.float32,
            device=device,
        )
        up = up_values[index % up_values.numel()].reshape(shape).contiguous()
        gate = gate_values[index % gate_values.numel()].reshape(shape).contiguous()
        return up, gate
    raise ValueError(f"unsupported SwiGLU case mode {mode!r}")


def swiglu_oracle(
    workload: WorkloadContract,
    up: object,
    gate: object,
) -> object:
    """Compose the frozen formula in CPU FP64 and round once to declared FP32."""

    torch = __import__("torch")
    document = workload.document
    oracle = _object(document["oracle"], "workload.oracle")
    if oracle.get("kind") != "cpu_fp64_composition_then_fp32_round":
        raise ValueError("SwiGLU oracle kind differs")
    up64 = up.detach().cpu().to(torch.float64)
    gate64 = gate.detach().cpu().to(torch.float64)
    reference = up64 * gate64 * (0.5 * (torch.tanh(gate64 * 0.5) + 1.0))
    return reference.to(torch.float32).to(up.device)


def swiglu_metrics(
    workload: WorkloadContract,
    output: object,
    reference: object,
) -> dict[str, object]:
    """Apply the Workload-owned tolerance and expose complete mismatch diagnostics."""

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
