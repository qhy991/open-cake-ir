"""Prospective reference handoff policy owned by the Authoring Environment.

This checks controlled artifact slots against reviewed incomplete assets. It does
not prove that arbitrary text/programs cannot reveal a target implementation.
Workload mathematics/oracle and public API contracts retain their existing owners.
"""
from __future__ import annotations

import json
import keyword
from pathlib import Path
from typing import Mapping

from .bindings import source_reference_path
from .python_reference import read_skeleton
from .pairing import native_backend

REFERENCE_ACCESS = frozenset({"clean_start", "known_kernel_reproduction", "direct_low_level"})
VETTED_REFERENCE_ASSETS = (
    "contracts/scaffolds/open-cake-clean-start-v1.json",
    "contracts/scaffolds/direct-cuda-clean-start-v1.cu",
    "contracts/scaffolds/matched-search-v1.md",
)
PYTHON_CLEAN_START_SCAFFOLD = "contracts/scaffolds/matched-search-python-v1.md"


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


def incomplete_schedule(workload, case_id: str, lowering_route: Mapping[str, object]) -> dict:
    """A task's ABI and route, with no implementation or scheduling decisions."""
    return {
        'schema_version': 2, 'schedule_id': 'open-cake-clean-start-v1', 'target': workload.target,
        'roles': [], 'allocations': [], 'pipelines': [], 'barriers': [], 'operations': [],
        'buffers': [{'name': arg.name, 'space': 'global', 'dtype': arg.dtype,
                     'shape': list(arg.shape), 'mode': arg.mode} for arg in workload.tensor_abi(case_id)],
        'outputs': [arg.name for arg in workload.tensor_abi(case_id) if arg.mode in {'output', 'inout'}],
        'metadata': {'workload_contract_sha256': workload.canonical_sha256},
        'lowering': dict(lowering_route),
    }


def render_incomplete_python_starter(workload, case_id: str,
                                     lowering_route: Mapping[str, object]) -> bytes:
    """Render only the public ABI and route; the placeholder is not a valid Schedule.

    Exact bytes are the clean-start reference policy. No arbitrary Python source is
    admitted by comparing only its parsed operations or ignoring its comments.
    """
    if (not isinstance(lowering_route, Mapping)
        or set(lowering_route) != {'backend', 'entry_point'}
        or not isinstance(lowering_route['backend'], str)
        or not isinstance(lowering_route['entry_point'], str)
        or not lowering_route['entry_point'].isidentifier()
        or keyword.iskeyword(lowering_route['entry_point'])):
        raise ValueError('Python clean-start lowering route differs')
    arguments = []
    names = set()
    for tensor in workload.tensor_abi(case_id):
        if (not tensor.name.isidentifier() or keyword.iskeyword(tensor.name)
            or tensor.name == 'lm' or tensor.name in names
            or tensor.mode not in {'input', 'output'}):
            raise ValueError('Python clean-start tensor ABI cannot be expressed')
        names.add(tensor.name)
        arguments.append(
            f'{tensor.name}: cake.Tensor({tuple(tensor.shape)!r}, '
            f'{json.dumps(tensor.dtype)}, mode={json.dumps(tensor.mode)})'
        )
    route = lowering_route
    source = (
        'from open_cake_ir.compiler import frontend as cake\n\n'
        '@cake.schedule(name="candidate", '
        f'target={json.dumps(workload.target)}, '
        f'backend={json.dumps(route["backend"])}, '
        f'entry_point={json.dumps(route["entry_point"])})\n'
        f'def candidate(lm, {", ".join(arguments)}):\n'
        '    ...\n'
    )
    return source.encode('utf-8')


def validate_reference_handoff(root: Path, arms: Mapping[str, object], *, workload=None, case_id=None) -> None:
    """Apply the same policy to local, external and inherited reference slots.

    The vetted bytes are part of the successor Executor closure. JSON stubs are
    rendered from their parsed document; raw authored Python is not granted a
    trusted role merely because its elaborated body resembles an empty Schedule.
    """
    validate_declarations(arms)
    for name, arm in arms.items():
        access = reference_access(arm, f"arms.{name}")
        prefix = f"arms.{name}.reference_access={access}"
        kind = arm.get("environment_kind")
        native = False
        if kind not in ("open_cake", "direct_cuda"):
            try:
                native = isinstance(kind, str) and native_backend(kind) is not None
            except ValueError:
                native = False
            if not native:
                raise ValueError(f"{prefix}: unsupported Authoring Environment kind {kind!r}")
        if access == "known_kernel_reproduction":
            continue
        if native:
            raise ValueError(f"{prefix}: inherited target_implementation requires known_kernel_reproduction")
        scaffold = arm.get("scaffold")
        if not isinstance(scaffold, Mapping):
            raise ValueError(f"{prefix}: missing authoring_instructions reference")
        _, path = source_reference_path(root, scaffold.get("path"), "scaffold")
        python_clean_start = (kind == 'open_cake' and access == 'clean_start'
                              and arm.get('input_format') == 'python_source_v1')
        vetted_scaffold = (PYTHON_CLEAN_START_SCAFFOLD if python_clean_start
                           else 'contracts/scaffolds/message-author/AGENTS.md'
                           if arm.get('provider', {}).get('harness') == 'responses'
                           else VETTED_REFERENCE_ASSETS[2])
        if path.read_bytes() != (root / vetted_scaffold).read_bytes():
            raise ValueError(f"{prefix}: authoring_instructions are not a vetted restricted scaffold")
        if kind == "open_cake":
            if python_clean_start:
                reference = arm.get('python_starter')
                if not isinstance(reference, Mapping) or set(reference) != {'path'}:
                    raise ValueError(f'{prefix}: Python target reference fields differ')
                _, path = source_reference_path(root, reference['path'], 'python_starter')
                expected = render_incomplete_python_starter(workload, case_id, arm['lowering_route'])
                if path.suffix != '.py' or path.read_bytes() != expected:
                    raise ValueError(f'{prefix}: target_implementation or unreviewed target reference is forbidden')
                continue
            reference = arm.get("schedule_skeleton")
            if not isinstance(reference, Mapping):
                raise ValueError(f"{prefix}: missing target reference")
            _, path = source_reference_path(root, reference.get("path"), "schedule_skeleton")
            expected = (incomplete_schedule(workload, case_id, arm['lowering_route'])
                        if workload is not None and arm.get('input_format') == 'schedule_or_python_v1'
                        else read_skeleton(root / VETTED_REFERENCE_ASSETS[0]))
            if path.suffix == ".py" or read_skeleton(path) != expected:
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
    if name == 'optimization-knowledge.json':
        return 'frozen_optimization_explanations_and_evidence_references'
    if name == 'transformation-api.json':
        return 'granted_Compiler_transformation_API'
    if name == 'authorized-programs.json':
        if access != 'known_kernel_reproduction':
            raise ValueError('complete baseline Program reference is not authorized')
        return 'target_implementation'
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
