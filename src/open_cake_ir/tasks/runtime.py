"""Composition of built-in task contracts with the common Ralph engine."""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from open_cake_ir.lab.core import Lab
from open_cake_ir.lab.efficiency_policy import performance_reporting_policy
from open_cake_ir.lab.contracts import StudyContract
from .workloads import load_workload
from .authoring import prepare_schedule, validate_authoring
from .launch import parse_launch_manifest
from .flash_kmeans.study import PortfolioStudyMixin


def _admit_measurement_coverage(study) -> None:
    """Check a Study's measurement-coverage claim against the registry that owns it.

    A Study asking for the weaker untimed gate is asking on the strength of its own
    declaration. Whether a timed assay exists is not the Study's fact: `devices.BACKENDS`
    owns which backend admits an exact target and what measurement source that backend may
    name, and quoting it back is what keeps a Study for a timed target from declaring the
    limitation and being admitted with no timing feedback at all.

    This lives here rather than in `lab.preflight` because `lab` does not import the task
    layer -- `tests/contracts/test_task_boundaries.py` enforces that -- and this layer owns
    both the registry and the Study construction that reads it.
    """

    from open_cake_ir.lab._policies import untimed
    from .devices import backend_for_target, timing_source

    document = study.document
    evaluation = document.get("evaluation_protocol")
    if not isinstance(evaluation, Mapping) or not untimed(evaluation):
        return
    execution = document.get("execution")
    target = execution.get("target") if isinstance(execution, Mapping) else None
    backend = backend_for_target(target)
    if backend is None:
        raise ValueError(
            f"Study reports no timed assay for target {target!r}, which no registered "
            "backend admits; the limitation cannot be checked against its owner")
    declared = timing_source(backend)
    if declared is not None:
        raise ValueError(
            f"Study reports no timed assay, but {backend} declares timing source "
            f"{declared!r} for {target!r}; a measurement-coverage limitation states what "
            "the device cannot do, not what this Study chose not to do")


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
        _admit_measurement_coverage(study)
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
