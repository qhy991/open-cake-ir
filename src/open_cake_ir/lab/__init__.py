"""Research Lab public Interface."""

from .checkpoints import CheckpointObservation, TurnObservation, project_checkpoints
from open_cake_ir.lab.core import AnalysisInclusion, CampaignLock, CampaignRef, Lab, RunEvaluator, RunProvider, StudyContract, StudyReport, TurnRequest, scientific_matched_analysis_plan_v2
from open_cake_ir.lab.environments import AuthoringEnvironment, BuildRequest, CandidateSubmission, NativeTritonEnvironment, EnvironmentResult, OpenCakeEnvironment, ToolchainBuilder, TritonToolchainBuilder
from .executor import ExecutorRevision
from .faults import CandidateCompileRejected, RunProtocolFault
from .providers import (
    CANDIDATE_SET_ENVELOPE_V1,
    CODEX_DISABLED_FEATURES,
    CodexInvocationBuilder,
    CodexProviderAdapter,
    CodexRunProvider,
    ProviderAuxiliaryActivity,
    ProviderInvocation,
    ProviderQualificationReceipt,
    ProviderTurn,
    normalize_codex_turn,
    required_live_provider_qualification_scope,
)
from .runtime import BoundedBrokerEvaluator, BrokerSubmitter, CommandBrokerSubmitter
from .ralph import RalphBudget, RalphController, derive_ralph_stop_reason
from .task_package import (
    TASK_AGENTS_RALPH_V1,
    TaskPackage,
    materialize_task_package,
    render_task_package,
    verify_task_package,
)

__all__ = [
    "AnalysisInclusion",
    "CampaignLock",
    "CampaignRef",
    "CheckpointObservation",
    "Lab",
    "RunEvaluator",
    "RunProvider",
    "StudyContract",
    "StudyReport",
    "TurnRequest",
    "TurnObservation",
    "project_checkpoints",
    "scientific_matched_analysis_plan_v2",
    "AuthoringEnvironment",
    "BuildRequest",
    "CandidateSubmission",
    "NativeTritonEnvironment",
    "EnvironmentResult",
    "ExecutorRevision",
    "OpenCakeEnvironment",
    "TritonToolchainBuilder",
    "ToolchainBuilder",
    "CodexInvocationBuilder",
    "CANDIDATE_SET_ENVELOPE_V1",
    "CODEX_DISABLED_FEATURES",
    "CodexProviderAdapter",
    "CodexRunProvider",
    "ProviderAuxiliaryActivity",
    "ProviderInvocation",
    "ProviderQualificationReceipt",
    "ProviderTurn",
    "normalize_codex_turn",
    "required_live_provider_qualification_scope",
    "BoundedBrokerEvaluator",
    "BrokerSubmitter",
    "CommandBrokerSubmitter",
    "RunProtocolFault",
    "CandidateCompileRejected",
    "RalphBudget",
    "RalphController",
    "derive_ralph_stop_reason",
    "TASK_AGENTS_RALPH_V1",
    "TaskPackage",
    "render_task_package",
    "materialize_task_package",
    "verify_task_package",
]
