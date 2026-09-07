from __future__ import annotations
from typing import Protocol
from types import MappingProxyType
from open_cake_ir.evaluation.core import EvaluationProtocol,EvaluationReceipt,LaunchableCandidate,LaunchObservation
from open_cake_ir.evaluation.workload import WorkloadContract
from .workload import generate_flash_kmeans_case,flash_kmeans_oracle,classify_flash_kmeans_output

class CandidateLauncher(Protocol):
    """Adapter that launches one already-sealed candidate."""

    def launch(
        self,
        candidate: LaunchableCandidate,
        tokens: object,
        centroids: object,
        centroid_sq: object,
    ) -> LaunchObservation:
        """Launch exactly once and return output plus route evidence."""


def evaluate_flash_kmeans(
    candidate: LaunchableCandidate,
    workload: WorkloadContract,
    protocol: EvaluationProtocol,
    launcher: CandidateLauncher,
    *,
    device: str,
) -> EvaluationReceipt:
    """Run materialization, external oracle, one launch, then correctness."""

    if protocol.workload_sha256 != workload.canonical_sha256:
        raise ValueError("EvaluationProtocol Workload bytes differ")
    if protocol.timing != "none":
        raise ValueError("timing protocol requires a separately retained timing assay")
    tokens, centroids = generate_flash_kmeans_case(
        workload, protocol.case_id, device=device
    )
    oracle = flash_kmeans_oracle(
        workload, tokens, centroids, case_id=protocol.case_id
    )
    torch = __import__("torch")
    centroids_fp32 = centroids.to(torch.float32)
    centroid_sq = (centroids_fp32 * centroids_fp32).sum(
        dim=-1, dtype=torch.float32
    ).contiguous()
    launch = launcher.launch(candidate, tokens, centroids, centroid_sq)
    passed, metrics = classify_flash_kmeans_output(
        workload,
        tokens,
        centroids,
        launch.output,
        oracle,
        case_id=protocol.case_id,
    )
    return EvaluationReceipt(
        candidate_sha256=candidate.candidate_sha256,
        workload_sha256=workload.canonical_sha256,
        evaluation_protocol_sha256=protocol.canonical_sha256,
        purpose=protocol.purpose,
        case_id=protocol.case_id,
        correctness_passed=passed,
        correctness=MappingProxyType(dict(metrics)),
        kernel_calls=launch.kernel_calls,
        fallback_calls=launch.fallback_calls,
        launch_receipt_sha256=launch.launch_receipt_sha256,
        timing=None,
    )
