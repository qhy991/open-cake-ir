"""Public common Evaluation Interface."""

from .admission import observe_exclusive_b200, observe_exclusive_cuda
from .attempts import (
    BrokerAttempt,
    LogicalEvaluationAttempt,
    evaluate_with_admission_recovery,
    is_resubmittable_admission_failure,
)
from open_cake_ir.evaluation.core import EvaluationProtocol, EvaluationReceipt, LaunchableCandidate, LaunchObservation
from open_cake_ir.evaluation.cuda_driver import CudaDeviceAdmission, CudaDriverLaunchReceipt, CudaLifecycleError, LoadedCudaCandidate, launch_candidate_once, launch_cubin_once
from open_cake_ir.evaluation.benchmark import CuptiBenchmark, StrictCuptiBenchmark
from .profiler import (
    NCU_ATTRIBUTION_METRICS,
    build_ncu_attribution_profile,
    load_ncu_attribution_profile,
    ncu_attribution_feedback,
)
from .timing import (
    PairedTimingObservation,
    PairedTimingProtocol,
    derive_paired_timing,
    summarize_cohort,
)
from .workload import WorkloadContract

__all__ = [
    "PairedTimingObservation",
    "PairedTimingProtocol",
    "EvaluationProtocol",
    "EvaluationReceipt",
    "LaunchableCandidate",
    "LaunchObservation",
    "CudaDriverLaunchReceipt",
    "CudaLifecycleError",
    "CudaDeviceAdmission",
    "LoadedCudaCandidate",
    "WorkloadContract",
    "CuptiBenchmark",
    "StrictCuptiBenchmark",
    "NCU_ATTRIBUTION_METRICS",
    "build_ncu_attribution_profile",
    "load_ncu_attribution_profile",
    "ncu_attribution_feedback",
    "observe_exclusive_b200",
    "observe_exclusive_cuda",
    "BrokerAttempt",
    "LogicalEvaluationAttempt",
    "evaluate_with_admission_recovery",
    "is_resubmittable_admission_failure",
    "launch_cubin_once",
    "launch_candidate_once",
    "derive_paired_timing",
    "summarize_cohort",
]
