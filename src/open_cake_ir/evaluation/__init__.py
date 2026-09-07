"""Public common Evaluation Interface."""

from .admission import observe_exclusive_b200
from .attempts import (
    BrokerAttempt,
    LogicalEvaluationAttempt,
    evaluate_with_admission_recovery,
    is_resubmittable_admission_failure,
)
from .core import (
    CandidateLauncher,
    EvaluationProtocol,
    EvaluationReceipt,
    LaunchableCandidate,
    LaunchObservation,
    evaluate_flash_kmeans,
)
from .correctness import CorrectnessObservation, audit_flash_kmeans_assignment
from .cuda_driver import (
    CudaDeviceAdmission,
    CudaDriverLaunchReceipt,
    CudaLifecycleError,
    CudaTensorContract,
    LoadedCudaCandidate,
    launch_candidate_once,
    launch_cubin_once,
)
from .cuda_manifest import CudaLaunchManifest, parse_cuda_launch_manifest
from .flash_kmeans import (
    assignment_raw_sha256,
    classify_flash_kmeans_output,
    flash_kmeans_metrics,
    flash_kmeans_oracle,
    generate_flash_kmeans_case,
    tensor_raw_sha256,
)
from .legacy_r45 import replay_legacy_r45_result
from .portfolio import (
    DispatchReceipt,
    ExactShapeDispatcher,
    PortfolioArtifact,
    PortfolioCaseObservation,
    PortfolioEntry,
    PortfolioEvaluationReceipt,
    SemanticKey,
    evaluate_portfolio_observations,
    replay_portfolio_receipt,
)
from .portfolio_runtime import CuptiBenchmark, CuptiPortfolioAssay, StrictCuptiBenchmark
from .profiler import (
    NCU_ATTRIBUTION_METRICS,
    build_ncu_attribution_profile,
    load_ncu_attribution_profile,
    ncu_attribution_feedback,
)
from .program import ProgramContract, ProgramNode
from .qsa import (
    QsaCorrectnessObservation,
    audit_qsa_output,
    materialize_qsa_case,
    qsa_block_scores,
    qsa_selection_mask,
    reference_qsa_output,
)
from .timing import (
    PairedTimingObservation,
    PairedTimingProtocol,
    derive_paired_timing,
    summarize_cohort,
)
from .swiglu import generate_swiglu_case, swiglu_metrics, swiglu_oracle
from .rmsnorm import generate_rmsnorm_case, rmsnorm_metrics, rmsnorm_oracle
from .llama_q4_mmvq import (
    Q4MmvqMaterial,
    Q4MmvqReference,
    decode_q4_0_block,
    materialize_q4_mmvq_case,
    quantize_q8_1_block,
    q4_mmvq_metrics,
    q4_mmvq_reference,
)
from .tinygemm import (
    TinyGemmLauncher,
    evaluate_tinygemm,
    generate_tinygemm_case,
    tinygemm_metrics,
    tinygemm_oracle,
)
from .workload import WorkloadContract

__all__ = [
    "PairedTimingObservation",
    "PairedTimingProtocol",
    "CorrectnessObservation",
    "CandidateLauncher",
    "CudaLaunchManifest",
    "EvaluationProtocol",
    "EvaluationReceipt",
    "LaunchableCandidate",
    "LaunchObservation",
    "CudaDriverLaunchReceipt",
    "CudaLifecycleError",
    "CudaDeviceAdmission",
    "CudaTensorContract",
    "LoadedCudaCandidate",
    "WorkloadContract",
    "ProgramContract",
    "ProgramNode",
    "QsaCorrectnessObservation",
    "audit_qsa_output",
    "materialize_qsa_case",
    "qsa_block_scores",
    "qsa_selection_mask",
    "reference_qsa_output",
    "TinyGemmLauncher",
    "evaluate_tinygemm",
    "generate_tinygemm_case",
    "tinygemm_metrics",
    "tinygemm_oracle",
    "CuptiBenchmark",
    "CuptiPortfolioAssay",
    "StrictCuptiBenchmark",
    "NCU_ATTRIBUTION_METRICS",
    "build_ncu_attribution_profile",
    "load_ncu_attribution_profile",
    "ncu_attribution_feedback",
    "replay_legacy_r45_result",
    "observe_exclusive_b200",
    "DispatchReceipt",
    "ExactShapeDispatcher",
    "PortfolioArtifact",
    "PortfolioEntry",
    "PortfolioCaseObservation",
    "PortfolioEvaluationReceipt",
    "SemanticKey",
    "evaluate_portfolio_observations",
    "replay_portfolio_receipt",
    "BrokerAttempt",
    "LogicalEvaluationAttempt",
    "evaluate_with_admission_recovery",
    "is_resubmittable_admission_failure",
    "audit_flash_kmeans_assignment",
    "assignment_raw_sha256",
    "classify_flash_kmeans_output",
    "flash_kmeans_metrics",
    "flash_kmeans_oracle",
    "generate_flash_kmeans_case",
    "evaluate_flash_kmeans",
    "tensor_raw_sha256",
    "parse_cuda_launch_manifest",
    "launch_cubin_once",
    "launch_candidate_once",
    "derive_paired_timing",
    "summarize_cohort",
    "generate_swiglu_case",
    "swiglu_metrics",
    "swiglu_oracle",
    "generate_rmsnorm_case",
    "rmsnorm_metrics",
    "rmsnorm_oracle",
    "Q4MmvqMaterial",
    "Q4MmvqReference",
    "decode_q4_0_block",
    "materialize_q4_mmvq_case",
    "quantize_q8_1_block",
    "q4_mmvq_metrics",
    "q4_mmvq_reference",
]
