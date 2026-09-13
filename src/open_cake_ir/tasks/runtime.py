"""Composition of built-in task contracts with the common Ralph engine."""
from __future__ import annotations

from dataclasses import replace

from open_cake_ir.lab.core import Lab
from open_cake_ir.lab.efficiency_policy import performance_reporting_policy
from open_cake_ir.lab.contracts import StudyContract
from .workloads import load_workload
from .authoring import prepare_schedule, validate_authoring
from .launch import parse_launch_manifest
from .flash_kmeans.study import PortfolioStudyMixin


class TaskLab(PortfolioStudyMixin, Lab):
    def __init__(self, project_root, **kwargs):
        super().__init__(
            project_root,
            workload_loader=load_workload,
            prepare_schedule=prepare_schedule,
            validate_authoring=validate_authoring,
            manifest_parser=parse_launch_manifest,
            **kwargs,
        )

    def preflight(self, study_path, *, empirical_cost_model_path=None,
                  execution_bindings_path=None):
        study = StudyContract.load(study_path)
        if study.document["kind"] == "portfolio":
            if empirical_cost_model_path is not None or execution_bindings_path is not None:
                raise ValueError("empirical selection requires artifact_optimization_only matched search")
            return self._preflight_portfolio(study)
        return super().preflight(study_path, empirical_cost_model_path=empirical_cost_model_path,
                                 execution_bindings_path=execution_bindings_path)

    def audit(self, campaign):
        policy = performance_reporting_policy(campaign.lock.analysis_plan, campaign.lock.claim_scope)
        report = super().audit(campaign)
        if policy is None:
            return report
        from .efficiency import campaign_performance
        performance = campaign_performance(self._root, campaign, report)
        return replace(report, descriptive={**report.descriptive, "performance": performance})
