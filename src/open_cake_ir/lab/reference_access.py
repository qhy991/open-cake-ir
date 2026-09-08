"""Prospective reference handoff policy owned by the Authoring Environment.

This checks controlled artifact slots against reviewed incomplete assets. It does
not prove that arbitrary text/programs cannot reveal a target implementation.
Workload mathematics/oracle and public API contracts retain their existing owners.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .bindings import source_reference_path
from .python_reference import read_skeleton

REFERENCE_ACCESS = frozenset({"clean_start", "known_kernel_reproduction", "direct_low_level"})
VETTED_REFERENCE_ASSETS = (
    "contracts/scaffolds/open-cake-clean-start-v1.json",
    "contracts/scaffolds/direct-cuda-clean-start-v1.cu",
    "contracts/scaffolds/matched-search-v1.md",
)


def reference_access(arm: Mapping[str, object], context: str) -> str:
    value = arm.get("reference_access")
    if not isinstance(value, str) or value not in REFERENCE_ACCESS:
        raise ValueError(f"{context}.reference_access must declare clean_start, known_kernel_reproduction or direct_low_level")
    return value


def validate_declarations(arms: Mapping[str, object]) -> None:
    for name, arm in arms.items():
        if not isinstance(arm, Mapping):
            raise ValueError(f"arm {name} must be an Authoring Environment")
        reference_access(arm, f"arms.{name}")


def validate_reference_handoff(root: Path, arms: Mapping[str, object]) -> None:
    """Apply the same policy to local, external and inherited reference slots.

    The vetted bytes are part of the successor Executor closure. JSON stubs are
    rendered from their parsed document; raw authored Python is not granted a
    trusted role merely because its elaborated body resembles an empty Schedule.
    """
    validate_declarations(arms)
    for name, arm in arms.items():
        access = reference_access(arm, f"arms.{name}")
        if access == "known_kernel_reproduction":
            continue
        prefix = f"arms.{name}.reference_access={access}"
        kind = arm.get("environment_kind")
        if kind not in {"open_cake", "direct_cuda"}:
            raise ValueError(f"{prefix}: inherited target_implementation requires known_kernel_reproduction")
        scaffold = arm.get("scaffold")
        if not isinstance(scaffold, Mapping):
            raise ValueError(f"{prefix}: missing authoring_instructions reference")
        _, path = source_reference_path(root, scaffold.get("path"), "scaffold")
        if path.read_bytes() != (root / VETTED_REFERENCE_ASSETS[2]).read_bytes():
            raise ValueError(f"{prefix}: authoring_instructions are not a vetted restricted scaffold")
        if kind == "open_cake":
            reference = arm.get("schedule_skeleton")
            if not isinstance(reference, Mapping):
                raise ValueError(f"{prefix}: missing target reference")
            _, path = source_reference_path(root, reference.get("path"), "schedule_skeleton")
            if path.suffix == ".py" or read_skeleton(path) != read_skeleton(root / VETTED_REFERENCE_ASSETS[0]):
                raise ValueError(f"{prefix}: target_implementation or unreviewed target reference is forbidden; require vetted incomplete_target_stub")
        elif kind == "direct_cuda":
            reference = arm.get("candidate_skeleton")
            if not isinstance(reference, Mapping):
                raise ValueError(f"{prefix}: missing target reference")
            _, path = source_reference_path(root, reference.get("path"), "candidate_skeleton")
            if path.read_bytes() != (root / VETTED_REFERENCE_ASSETS[1]).read_bytes():
                raise ValueError(f"{prefix}: target_implementation or unreviewed target reference is forbidden; require vetted incomplete_target_stub")


def document_role(name: str, access: str) -> str:
    """Describe controlled package slots; the caller cannot assign arbitrary roles."""
    if access not in REFERENCE_ACCESS:
        raise ValueError("task reference access category differs")
    if name == "workload.json":
        return "mathematical_specification_and_oracle"
    if name == "target.json":
        return "hardware_contract"
    if name == "run-authority.json":
        return "treatment_contract"
    if name == "scaffold.md":
        return "authoring_instructions"
    if name in {"schedule-skeleton.json", "schedule-starter.py", "candidate-skeleton.cu"}:
        return "target_reference" if access == "known_kernel_reproduction" else "incomplete_target_stub"
    if name in {"python-example.py", "candidate-baseline.triton.json", "candidate-baseline.cute.json", "paired-triton-authoring.md", "paired-cute-authoring.md"}:
        return "target_implementation"
    if name in {"schedule.schema.json", "schedule-authoring.md", "python-frontend.md",
                "cuda-launch-abi.json", "candidate.schema.json"}:
        return "authoring_api"
    raise ValueError(f"unclassified task reference slot {name!r}")
