"""Deterministic Flash-KMeans materialization and external FP32 oracle."""

from __future__ import annotations

import math
from hashlib import sha256
from typing import Mapping, cast

from .workload import WorkloadContract


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch is required for Flash-KMeans Evaluation") from error
    return torch


def _shape(case: Mapping[str, object]) -> tuple[int, int, int, int]:
    raw = case.get("shape")
    if not isinstance(raw, Mapping):
        raise ValueError("Flash-KMeans case shape is missing")
    values: list[int] = []
    for axis in ("B", "N", "K", "D"):
        value = raw.get(axis)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Flash-KMeans case {axis} is invalid")
        values.append(value)
    return cast(tuple[int, int, int, int], tuple(values))


def generate_flash_kmeans_case(
    workload: WorkloadContract,
    case_id: str,
    *,
    device: str,
) -> tuple[object, object]:
    """Materialize one declared case without reading a baseline implementation."""

    if workload.document.get("operator") != "flash_kmeans_assign":
        raise ValueError("workload is not Flash-KMeans assignment")
    torch = _torch()
    case = workload.case(case_id)
    batch, token_count, centroid_count, feature_size = _shape(case)
    mode = case.get("mode")
    if mode == "random_standard_normal":
        seed = case.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("random Flash-KMeans case requires a non-negative seed")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        tokens = torch.randn(
            (batch, token_count, feature_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).contiguous()
        centroids = torch.randn(
            (batch, centroid_count, feature_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).contiguous()
    elif mode == "constructed_duplicate_centroids":
        if (batch, token_count, centroid_count, feature_size) != (1, 4, 4, 128):
            raise ValueError("constructed Flash-KMeans tie case shape differs")
        tokens = torch.zeros((1, 4, 128), dtype=torch.bfloat16, device=device)
        centroids = torch.zeros((1, 4, 128), dtype=torch.bfloat16, device=device)
        tokens[:, 1, :].fill_(1.0)
        tokens[:, 2, :].fill_(-1.0)
        tokens[:, 3, :].fill_(0.5)
        centroids[:, 1, :].fill_(1.0)
        centroids[:, 2, :].fill_(-1.0)
        centroids[:, 3, :].fill_(1.0)
    else:
        raise ValueError(f"unsupported Flash-KMeans generation mode {mode!r}")
    return tokens, centroids


def tensor_raw_sha256(tensor: object, *, chunk_elements: int = 1024) -> tuple[str, int]:
    """Hash a contiguous BF16 rank-three tensor in deterministic batch/chunk order."""

    torch = _torch()
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 3
        or tensor.dtype != torch.bfloat16
        or not tensor.is_contiguous()
        or not isinstance(chunk_elements, int)
        or isinstance(chunk_elements, bool)
        or chunk_elements <= 0
    ):
        raise ValueError("tensor hash requires contiguous rank-three BF16 and positive chunks")
    digest = sha256()
    batch, axis_size, _ = map(int, tensor.shape)
    for batch_index in range(batch):
        for start in range(0, axis_size, chunk_elements):
            stop = min(start + chunk_elements, axis_size)
            raw = (
                tensor[batch_index : batch_index + 1, start:stop, :]
                .contiguous()
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes(order="C")
            )
            digest.update(raw)
    return digest.hexdigest(), tensor.numel() * tensor.element_size()


def assignment_raw_sha256(
    tensor: object, *, chunk_elements: int = 8192
) -> tuple[str, int]:
    """Hash one contiguous rank-two INT32 assignment tensor in row/chunk order."""

    torch = _torch()
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim != 2
        or tensor.dtype != torch.int32
        or not tensor.is_contiguous()
        or not isinstance(chunk_elements, int)
        or isinstance(chunk_elements, bool)
        or chunk_elements <= 0
    ):
        raise ValueError("assignment hash requires contiguous rank-two INT32 and positive chunks")
    digest = sha256()
    batch, axis_size = map(int, tensor.shape)
    for batch_index in range(batch):
        for start in range(0, axis_size, chunk_elements):
            stop = min(start + chunk_elements, axis_size)
            raw = (
                tensor[batch_index : batch_index + 1, start:stop]
                .contiguous()
                .cpu()
                .numpy()
                .tobytes(order="C")
            )
            digest.update(raw)
    return digest.hexdigest(), tensor.numel() * tensor.element_size()


def _validate_inputs(
    tokens: object,
    centroids: object,
    expected_shape: tuple[int, int, int, int] | None,
) -> tuple[object, int, int, int, int]:
    torch = _torch()
    if not isinstance(tokens, torch.Tensor) or not isinstance(centroids, torch.Tensor):
        raise ValueError("Flash-KMeans inputs must be torch tensors")
    if tokens.dtype != torch.bfloat16 or centroids.dtype != torch.bfloat16:
        raise ValueError("Flash-KMeans inputs must be BF16")
    if tokens.ndim != 3 or centroids.ndim != 3:
        raise ValueError("Flash-KMeans inputs must be rank three")
    batch, token_count, feature_size = map(int, tokens.shape)
    centroid_batch, centroid_count, centroid_features = map(int, centroids.shape)
    if (
        centroid_batch != batch
        or centroid_features != feature_size
        or tokens.device != centroids.device
        or not tokens.is_contiguous()
        or not centroids.is_contiguous()
    ):
        raise ValueError("Flash-KMeans input shape, device or layout differs")
    if expected_shape is not None and (
        batch,
        token_count,
        centroid_count,
        feature_size,
    ) != expected_shape:
        raise ValueError("Flash-KMeans input shape differs from expected cell")
    if not bool(torch.isfinite(tokens).all().item()) or not bool(
        torch.isfinite(centroids).all().item()
    ):
        raise ValueError("Flash-KMeans inputs must be finite")
    return torch, batch, token_count, centroid_count, feature_size


def flash_kmeans_oracle(
    workload: WorkloadContract,
    tokens: object,
    centroids: object,
    *,
    case_id: str,
) -> object:
    """Compute the external full-K FP32 oracle with N-only chunking."""

    if workload.document.get("operator") != "flash_kmeans_assign":
        raise ValueError("workload is not Flash-KMeans assignment")
    oracle = cast(Mapping[str, object], workload.document["oracle"])
    chunk_n = oracle["chunk_n"]
    if (
        oracle.get("kind") != "fp32_full_k_chunked"
        or oracle.get("tf32") is not False
        or not isinstance(chunk_n, int)
        or isinstance(chunk_n, bool)
        or chunk_n <= 0
    ):
        raise ValueError("Workload Contract oracle is unsupported")
    expected_shape = _shape(workload.case(case_id))
    torch, batch, token_count, _, _ = _validate_inputs(tokens, centroids, expected_shape)
    output = torch.empty((batch, token_count), dtype=torch.int32, device=tokens.device)
    allow_tf32 = None
    if tokens.device.type == "cuda":
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        centroids_fp32 = centroids.to(torch.float32)
        centroid_sq = (centroids_fp32 * centroids_fp32).sum(dim=-1, dtype=torch.float32)
        centroid_transposed = centroids_fp32.transpose(1, 2)
        for start in range(0, token_count, chunk_n):
            stop = min(start + chunk_n, token_count)
            scores = torch.bmm(tokens[:, start:stop, :].to(torch.float32), centroid_transposed)
            scores.add_(centroid_sq[:, None, :], alpha=-0.5)
            if not bool(torch.isfinite(scores).all().item()):
                raise ValueError("Flash-KMeans oracle score is non-finite")
            output[:, start:stop].copy_(torch.argmax(scores, dim=-1).to(torch.int32))
    finally:
        if allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    return output.contiguous()


def flash_kmeans_metrics(
    workload: WorkloadContract,
    tokens: object,
    centroids: object,
    candidate_ids: object,
    reference_ids: object,
    *,
    case_id: str,
) -> dict[str, object]:
    """Return exact and tie-aware diagnostics without changing the canonical gate."""

    if workload.document.get("operator") != "flash_kmeans_assign":
        raise ValueError("workload is not Flash-KMeans assignment")
    expected_shape = _shape(workload.case(case_id))
    torch, batch, token_count, centroid_count, _ = _validate_inputs(
        tokens, centroids, expected_shape
    )
    oracle = cast(Mapping[str, object], workload.document["oracle"])
    validation = cast(Mapping[str, object], workload.document["validation"])
    chunk_n = oracle["chunk_n"]
    tie_atol = validation["tie_diagnostic_atol"]
    tie_rtol = validation["tie_diagnostic_rtol"]
    if not isinstance(chunk_n, int) or not isinstance(tie_atol, (int, float)) or not isinstance(
        tie_rtol, (int, float)
    ):
        raise ValueError("Workload Contract metric parameters are invalid")
    tie_atol = float(tie_atol)
    tie_rtol = float(tie_rtol)
    if chunk_n <= 0 or tie_atol < 0 or tie_rtol < 0:
        raise ValueError("Flash-KMeans metric parameters are invalid")
    for label, value in (("candidate", candidate_ids), ("reference", reference_ids)):
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.int32
            or tuple(value.shape) != (batch, token_count)
            or value.device != tokens.device
            or not value.is_contiguous()
            or int(value.min().item()) < 0
            or int(value.max().item()) >= centroid_count
        ):
            raise ValueError(f"Flash-KMeans {label} IDs violate the output contract")
    exact_count = 0
    tie_aware_count = 0
    near_tie_mismatch_count = 0
    max_excess = 0.0
    centroids_fp32 = centroids.to(torch.float32)
    batch_indices = torch.arange(batch, device=tokens.device)[:, None]
    for start in range(0, token_count, chunk_n):
        stop = min(start + chunk_n, token_count)
        token_chunk = tokens[:, start:stop, :].to(torch.float32)
        candidate_chunk = candidate_ids[:, start:stop]
        reference_chunk = reference_ids[:, start:stop]
        exact = candidate_chunk == reference_chunk
        exact_count += int(exact.sum().item())
        candidate_centroids = centroids_fp32[batch_indices, candidate_chunk.to(torch.int64)]
        reference_centroids = centroids_fp32[batch_indices, reference_chunk.to(torch.int64)]
        candidate_distance = ((token_chunk - candidate_centroids) ** 2).sum(dim=-1, dtype=torch.float32)
        reference_distance = ((token_chunk - reference_centroids) ** 2).sum(dim=-1, dtype=torch.float32)
        allowance = tie_atol + tie_rtol * reference_distance.abs()
        tie_aware = candidate_distance <= reference_distance + allowance
        tie_aware_count += int(tie_aware.sum().item())
        near_tie_mismatch_count += int(((~exact) & tie_aware).sum().item())
        excess = (candidate_distance - reference_distance).clamp_min(0)
        max_excess = max(max_excess, float(excess.max().item()))
    total = batch * token_count
    mismatch_count = total - exact_count
    if not math.isfinite(max_excess):
        raise ValueError("Flash-KMeans chosen-distance excess is non-finite")
    return {
        "schema_version": 1,
        "total_assignments": total,
        "exact_match": mismatch_count == 0,
        "exact_match_fraction": exact_count / total,
        "mismatch_count": mismatch_count,
        "tie_aware_distance_match": tie_aware_count == total,
        "tie_aware_distance_match_fraction": tie_aware_count / total,
        "near_tie_mismatch_count": near_tie_mismatch_count,
        "max_chosen_distance_excess": max_excess,
        "tie_atol": float(tie_atol),
        "tie_rtol": float(tie_rtol),
    }


def classify_flash_kmeans_output(
    workload: WorkloadContract,
    tokens: object,
    centroids: object,
    candidate_ids: object,
    reference_ids: object,
    *,
    case_id: str,
) -> tuple[bool, dict[str, object]]:
    """Classify candidate output while keeping reachable output violations as outcomes."""

    try:
        metrics = flash_kmeans_metrics(
            workload,
            tokens,
            centroids,
            candidate_ids,
            reference_ids,
            case_id=case_id,
        )
    except ValueError as error:
        if str(error) != "Flash-KMeans candidate IDs violate the output contract":
            raise
        return False, {
            "schema_version": 1,
            "failure_code": "candidate_output_contract_violation",
        }
    passed = bool(metrics["tie_aware_distance_match"]) and (
        metrics["max_chosen_distance_excess"] == 0.0
    )
    return passed, metrics
