"""Public Lab facade over explicit lifecycle and evidence owners."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

from . import contracts, execution, preflight, replay, reporting

if TYPE_CHECKING:
    from open_cake_ir.evidence import EvidenceStore, RunAudit

    from .environments import AuthoringEnvironment
    from .task_package import TaskPackage


class Lab:
    """Resolve, execute and audit preregistered studies over frozen dependencies."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        workload_loader: Callable,
        prepare_schedule: Callable,
        validate_authoring: Callable,
        manifest_parser: Callable,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._root = Path(project_root).resolve(strict=True)
        self._clock = clock
        self._load_workload = workload_loader
        self._prepare_schedule = prepare_schedule
        self._validate_authoring = validate_authoring
        self._parse_manifest = manifest_parser

    def task_package(self, lock: contracts.CampaignLock, run_id: str) -> TaskPackage:
        return preflight.task_package(
            lock,
            run_id,
            project_root=self._root,
            workload_loader=self._load_workload,
            prepare_schedule=self._prepare_schedule,
        )

    def preflight(
        self,
        study_path: str | Path,
        *,
        empirical_cost_model_path: str | Path | None = None,
        execution_bindings_path: str | Path | None = None,
    ) -> contracts.CampaignLock:
        return preflight.preflight(
            study_path,
            project_root=self._root,
            workload_loader=self._load_workload,
            validate_authoring=self._validate_authoring,
            manifest_parser=self._parse_manifest,
            empirical_cost_model_path=empirical_cost_model_path,
            execution_bindings_path=execution_bindings_path,
        )

    def reference_campaign(
        self,
        lock: contracts.CampaignLock,
        evidence_root: str | Path,
    ) -> contracts.CampaignRef:
        """Create a read-only Campaign reference after execution or fixture construction."""
        root = Path(evidence_root).resolve(strict=True)
        if not (root / "runs").is_dir() or not (root / "objects/sha256").is_dir():
            raise ValueError("Campaign evidence root is incomplete")
        return contracts.CampaignRef(lock=lock, evidence_root=root)

    def execute(
        self,
        lock: contracts.CampaignLock,
        evidence_root: str | Path,
        *,
        provider: contracts.RunProvider,
        environments: Mapping[str, AuthoringEnvironment],
        evaluator: contracts.RunEvaluator,
    ) -> contracts.CampaignRef:
        return execution.execute_campaign(
            lock,
            evidence_root,
            project_root=self._root,
            workload_loader=self._load_workload,
            validate_authoring=self._validate_authoring,
            clock=self._clock,
            provider=provider,
            environments=environments,
            evaluator=evaluator,
        )

    def _replay_matched_run(
        self,
        evidence: EvidenceStore,
        audit: RunAudit,
        lock: contracts.CampaignLock,
    ) -> bool:
        return replay.replay_matched_run(
            evidence,
            audit,
            lock,
            project_root=self._root,
            manifest_parser=self._parse_manifest,
            task_package=self.task_package,
        )

    def threshold_view(
        self,
        campaign: contracts.CampaignRef,
        latency_threshold_ms: float,
    ) -> Mapping[str, object]:
        return reporting.threshold_view(
            campaign,
            latency_threshold_ms,
            audit_campaign=self.audit,
        )

    def audit(self, campaign: contracts.CampaignRef) -> contracts.StudyReport:
        return reporting.audit_campaign(
            campaign,
            replay_run=self._replay_matched_run,
        )
