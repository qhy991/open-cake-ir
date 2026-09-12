"""Two-file Agent interface derived from one resolved Campaign authority."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Protocol, cast

from open_cake_ir.compiler import Compiler
from .rubrics import derive_rubric
from .pairing import bind_baseline, native_baseline, backend_policy, native_backend
from .python_reference import bind_python_reference
from .reference_access import document_role, reference_access, validate_reference_handoff
from open_cake_ir.compiler import frontend
from open_cake_ir.compiler.schema import schedule_schema_bytes
from open_cake_ir.evaluation import WorkloadContract


TASK_AGENTS_RALPH_V1 = "task_agents_ralph_v1"


class CampaignLockLike(Protocol):
    document: Mapping[str, object]
    study_id: str
    study_kind: str
    claim_scope: str
    run_order: tuple[str, ...]


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _pretty_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _read_relative(root: Path, value: object, context: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path differs")
    if Path(value).is_absolute() and context in {"workload", "schedule_skeleton", "candidate_skeleton", "scaffold"}:
        from .bindings import source_reference_path
        _, path = source_reference_path(root, value, context)
        return path.read_bytes()
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"{context} path is unsafe")
    unresolved = root / value
    if unresolved.is_symlink():
        raise ValueError(f"{context} custody differs")
    path = unresolved.resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{context} custody differs")
    return path.read_bytes()


def build_run_reference_documents(
    project_root: str | Path,
    lock: CampaignLockLike,
    arm: Mapping[str, object],
    *, workload_contract: WorkloadContract, prepare_schedule: Callable,
) -> Mapping[str, bytes]:
    """Build the sole run-authority projection used by the Ralph task package."""

    root = Path(project_root).resolve(strict=True)
    workload = _object(lock.document["workload"], "campaign_lock.workload")
    if workload_contract.canonical_sha256 != workload["canonical_sha256"]:
        raise ValueError("task package Workload differs from CampaignLock")
    compiler = _object(lock.document["compiler_revision"], "campaign_lock.compiler")
    scaffold = _object(arm["scaffold"], "arm.scaffold")
    revision_document = _object(
        json.loads(_read_relative(root, compiler["path"], "compiler_revision")),
        "compiler_revision",
    )
    targets = _object(revision_document["target_definitions"], "target_definitions")
    execution = _object(lock.document['execution'], 'campaign_lock.execution')
    target_id = str(execution['target'])
    target = _object(targets[target_id], f"target_definitions.{target_id}")
    resolved = _object(lock.document["resolved_inputs"], "resolved_inputs")
    assigned_arms = _object(resolved["arm_environments"], "arm_environments")
    if not any(arm == value for value in assigned_arms.values()):
        raise ValueError("task package Authoring Environment differs from the frozen arm")
    validate_reference_handoff(root, assigned_arms)
    access = reference_access(arm, "arm")
    authoring_environment = arm
    if "candidate_selection" in arm:
        selection = _object(arm["candidate_selection"], "arm.candidate_selection")
        model = _object(selection["model"], "arm.candidate_selection.model")
        # The Ralph task package consumes this projection. The complete supplier model
        # belongs to the frozen CampaignLock, not TASK.md.
        authoring_environment = {
            **arm,
            "candidate_selection": {
                "kind": selection["kind"],
                "model": {key: model[key] for key in (
                    "model_id", "compiler_revision_id", "compiler_revision_sha256", "target",
                )},
            },
        }
    run_authority = {
        "schema_version": 1,
        "study": lock.document["study"],
        "workload": lock.document["workload"],
        "authoring_environment": authoring_environment,
        "budget": resolved["budget"],
        "run_protocol": resolved["run_protocol"],
        "evaluation_protocol": lock.document["evaluation_protocol"],
        "execution": lock.document["execution"],
    }
    documents: dict[str, bytes] = {
        "run-authority.json": _canonical_json(run_authority).encode(),
        "workload.json": _read_relative(root, workload["path"], "workload"),
        "target.json": _read_relative(root, target["path"], "target"),
        "scaffold.md": _read_relative(root, scaffold["path"], "scaffold"),
    }
    environment_kind = arm.get("environment_kind")
    if environment_kind == "open_cake":
        skeleton_ref = _object(arm["schedule_skeleton"], "arm.schedule_skeleton")
        skeleton_bytes = _read_relative(root, skeleton_ref["path"], "schedule_skeleton")
        python_starter = str(skeleton_ref["path"]).endswith(".py")
        skeleton = (
            frontend.parse(skeleton_bytes.decode("utf-8"), filename=str(skeleton_ref["path"])).document
            if python_starter else json.loads(skeleton_bytes)
        )
        case_id = str(_object(lock.document["evaluation_protocol"], "protocol")["case_id"])
        skeleton = prepare_schedule(skeleton, workload_contract, case_id, arm)
        if python_starter:
            if arm.get("input_format") != "schedule_or_python_v1":
                raise ValueError("Python starter requires the existing Python-enabled Authoring Environment")
            documents["schedule-starter.py"] = bind_python_reference(
                skeleton_bytes.decode("utf-8"), skeleton, filename=str(skeleton_ref["path"]))
        else:
            documents.update({"schedule.schema.json": schedule_schema_bytes(),
                "schedule-authoring.md": (root / "compiler/AUTHORING_CONTRACT.md").read_bytes(),
                "schedule-skeleton.json": _canonical_json(skeleton).encode()})
        if arm.get("input_format") == "schedule_or_python_v1":
            documents["python-frontend.md"] = (root / "docs/PYTHON_FRONTEND.md").read_bytes()
            if access == "known_kernel_reproduction":
                documents["python-example.py"] = (root / "examples/python/fma.py").read_bytes()
        if access == "known_kernel_reproduction" and arm.get("lowering_route", {}).get("backend") != "metal" and arm.get("input_format") == "schedule_or_python_v1":
            policy = backend_policy(arm["lowering_route"]["backend"])
            documents[policy.authoring_file] = (root / "docs/en" / policy.document).read_bytes()
    elif environment_kind != "direct_cuda" and (policy := native_backend(environment_kind)) is not None:
        open_arm = _object(_object(resolved["arm_environments"], "arm_environments")["open_cake"], "open_cake")
        skeleton_ref = _object(open_arm["schedule_skeleton"], "schedule_skeleton")
        case_id = str(_object(lock.document["evaluation_protocol"], "protocol")["case_id"])
        baseline = bind_baseline(json.loads(_read_relative(root, skeleton_ref["path"], "schedule_skeleton")), workload_contract, case_id)
        compiler_instance = Compiler.load(root, root / str(compiler["path"]))
        lowering = compiler_instance.lower(compiler_instance.assess(baseline))
        documents[policy.baseline_file] = _canonical_json(native_baseline(lowering)).encode()
        documents[policy.authoring_file] = (root / "docs/en" / policy.document).read_bytes()
        if policy.candidate_schema is not None:
            documents["candidate.schema.json"] = _read_relative(root, policy.candidate_schema, "native candidate schema")
    elif environment_kind == "direct_cuda":
        launch = _object(arm["launch_contract"], "arm.launch_contract")
        candidate = _object(arm["candidate_skeleton"], "arm.candidate_skeleton")
        documents.update(
            {
                "cuda-launch-abi.json": _read_relative(
                    root, launch["path"], "launch_contract"
                ),
                "candidate-skeleton.cu": _read_relative(
                    root, candidate["path"], "candidate_skeleton"
                ),
            }
        )
    else:
        raise ValueError("Authoring Environment kind differs")
    return documents


def _document_sections(documents: Mapping[str, bytes], *, access: str) -> str:
    sections: list[str] = []
    for name, payload in documents.items():
        try:
            text = payload.decode("utf-8")
        except UnicodeError as error:
            raise ValueError(f"task reference {name!r} is not UTF-8") from error
        language = (
            "json"
            if name.endswith(".json")
            else "python"
            if name.endswith(".py")
            else "cuda"
            if name.endswith((".cu", ".cuh"))
            else "markdown"
        )
        scope = (
            "This is a Python syntax example for its own declared target and workload. "
            "Use this Run's target.json and schedule starter for the task's target and shapes.\n\n"
            if name == "python-example.py" else ""
        )
        sections.append(f"## Frozen reference: `{name}`\n\nReference role: {document_role(name, access)}.\n\n{scope}```{language}\n{text}\n```")
    return "\n\n".join(sections)


@dataclass(frozen=True)
class TaskPackage:
    """The complete two-file Agent surface for one Run."""

    run_id: str
    arm: str
    task_markdown: str
    agents_markdown: str

    @property
    def task_sha256(self) -> str:
        return sha256(self.task_markdown.encode()).hexdigest()

    @property
    def agents_sha256(self) -> str:
        return sha256(self.agents_markdown.encode()).hexdigest()

    @property
    def canonical_sha256(self) -> str:
        return sha256(
            _canonical_json(
                {
                    "run_id": self.run_id,
                    "arm": self.arm,
                    "task_sha256": self.task_sha256,
                    "agents_sha256": self.agents_sha256,
                }
            ).encode()
        ).hexdigest()

    def evidence_bundle(self, state_card: Mapping[str, object]) -> bytes:
        """Retain exactly what the Agent received for one Ralph iteration."""

        # Derive from the same detached JSON-shaped snapshot that is retained.
        # Nested immutable mappings/tuples must not change rubric state on replay.
        state = cast(dict[str, object], _plain(state_card))
        return _canonical_json(
            {
                "schema_version": 1,
                "kind": TASK_AGENTS_RALPH_V1,
                "run_id": self.run_id,
                "arm": self.arm,
                "task_markdown": self.task_markdown,
                "agents_markdown": self.agents_markdown,
                "state_card": state,
                "rubric": (
                    derive_rubric(state["previous_feedback"])
                    if "previous_feedback" in state else derive_rubric()
                ),
            }
        ).encode()


def render_task_request(
    package: TaskPackage, state_card: Mapping[str, object]
) -> tuple[str, bytes]:
    """Deliver the verified two-file authority and retain that exact projection.

    The caller verifies the materialized files before invocation. JSON quoting is
    reversible: each document string is the original UTF-8 text, not a second
    template. This proves delivered content, not whether the model read a file.
    """

    _object(state_card, 'Ralph controller StateCard')
    bundle = package.evidence_bundle(state_card)
    prompt = (
        "Read the complete TASK.md and AGENTS.md content in the following canonical "
        "task projection. Continue the same Ralph Run under those immutable rules. "
        "The StateCard is the external controller's derived state for this iteration. "
        "The rubric explains only its previous_feedback; it is guidance, not a score "
        "or permission to change acceptance rules or the frozen Compiler.\n\n"
        + bundle.decode('utf-8')
    )
    return prompt, bundle


def render_task_package(
    project_root: str | Path,
    lock: CampaignLockLike,
    run_id: str,
    *, workload_contract: WorkloadContract, prepare_schedule: Callable,
) -> TaskPackage:
    """Render TASK.md and AGENTS.md from canonical owners, never from run history."""

    if lock.study_kind != "matched_search" or run_id not in lock.run_order:
        raise ValueError("task package requires one matched Campaign Run")
    resolved = _object(lock.document["resolved_inputs"], "resolved_inputs")
    interface = _object(resolved.get("agent_interface"), "resolved_inputs.agent_interface")
    if interface != {"schema_version": 1, "kind": TASK_AGENTS_RALPH_V1}:
        raise ValueError("Campaign agent interface differs")
    arm = run_id.rsplit("-", 1)[0]
    arms = _object(resolved["arm_environments"], "resolved_inputs.arm_environments")
    authority = _object(arms[arm], f"arm_environments.{arm}")
    documents = build_run_reference_documents(project_root, lock, authority,
        workload_contract=workload_contract, prepare_schedule=prepare_schedule)
    budget = _object(resolved["budget"], "resolved_inputs.budget")
    evaluation = _object(lock.document["evaluation_protocol"], "evaluation_protocol")
    execution = _object(lock.document["execution"], "execution")
    fixed = execution.get("fixed_baseline")
    selection = (
        _object(fixed, "fixed baseline").get("selection")
        if isinstance(fixed, Mapping)
        else None
    )
    baseline_source = (
        _object(selection, "fixed baseline selection").get("source")
        if isinstance(selection, Mapping)
        else "campaign_fixed_baseline"
    )
    promotion_run = (
        selection.get("promotion_run_id")
        if isinstance(selection, Mapping)
        else None
    )
    output_contract = f'`{{"arm":"{arm}","candidates":[...],"schema_version":1}}`'
    task = f"""# TASK.md — {lock.study_id} / {run_id}

## Objective

Produce structurally distinct `{arm}` Candidates for the frozen Workload case
`{evaluation['case_id']}` and improve the confirmed absolute latency without violating
correctness, artifact custody, or the declared Claim Scope `{lock.claim_scope}`.

The comparison baseline is the frozen black-box `{baseline_source}` artifact
(`promotion_run_id={promotion_run}`). Its identity is in `run-authority.json`; its
implementation is not additional reference access. Improve against its measured latency,
and never call or inspect it from a Candidate.

## Candidate output

Write exactly one valid UTF-8 JSON `candidate-set.json` envelope:

{output_contract}

Whitespace, object-key order, and equivalent JSON escapes are accepted. Object keys
must be unique at every level; numbers must be finite. `schema_version` must be the
integer `1`. Candidate identity uses the existing canonical member projection; direct
CUDA source strings keep their decoded UTF-8 bytes exactly.

The envelope contains between one and {budget['maximum_candidates_per_turn']} Candidates
in provider order. The first Ralph iteration adds it; later iterations update the same
file. Renaming or reformatting is not a structurally distinct Candidate.

## Evaluation and budget

```json
{_pretty_json({'budget': budget, 'evaluation_protocol': evaluation})}
```

Search Evaluation applies the external oracle before timing. Profiler attribution is a
separate correctness-qualified launch with no timing. Only a fresh confirmatory Receipt
can promote a Candidate or support a checkpoint.

## Claim boundary

This Run is governed by `{lock.claim_scope}`. An operator or fixed-Program result is not a
model-forward or serving result. Infrastructure `unknown` is not a Candidate failure.

## Complete frozen authority

The following sections are the complete read-only task authority. They are projections of
the machine Contracts bound by the CampaignLock; do not edit them or infer newer state.

Declared reference access: `{reference_access(authority, "arm")}`.\n\n{_document_sections(documents, access=reference_access(authority, "arm"))}
"""
    if "schedule-starter.py" in documents:
        task = task.replace("Write exactly one valid UTF-8 JSON `candidate-set.json` envelope:",
            "Author each Candidate in Python through the supplied frontend. Put its source text "
            "in a `python_source` member; do not describe a Schedule with JSON fields. "
            "The existing candidate envelope is transport only.\n\n"
            "Write exactly one valid UTF-8 JSON `candidate-set.json` envelope:")
    arm_rule = (
        "Author only Cake IR Schedules or restricted Python through the supplied frontend; preserve the supplied lowering route. Do not invoke CUDA, a GPU, the network, or another compiler."
        if arm == "open_cake" and authority.get("input_format") == "schedule_or_python_v1"
        else "Author only Cake IR Schedules; preserve the supplied lowering route. Do not invoke CUDA, a GPU, the network, or another compiler."
        if arm == "open_cake"
        else f"Author only the supplied kernel-only {native_backend(arm).label} baseline and declared compile/launch metadata. Host Python is forbidden."
        if native_backend(arm) is not None
        else "Author only direct CUDA/PTX source. Do not access the Open Cake Compiler or a target implementation."
    )
    agents = f"""# AGENTS.md — Ralph optimization rules

## Ownership

- Read `TASK.md` completely before changing the Candidate.
- Write only `candidate-set.json`; `TASK.md` and `AGENTS.md` are immutable.
- The primary thread is the sole Candidate writer. Auxiliary work is read-only.
- {arm_rule}

## Ralph loop

- Continue in the same provider thread until the external controller terminates the Run.
- Advance one explicit, falsifiable optimization hypothesis at a time.
- Treat the controller StateCard as a derived view of retained Evidence, not as task policy.
- Do not spend GPU work directly. The external Lab owns filtering, Evaluation, and budgets.
- A rejected, duplicate, incorrect, unstable, or null Candidate is evidence; do not hide it.

## Evidence boundaries

- Local reasoning and probes are advisory; the external Evaluation Receipt is authoritative.
- `lowerable` is not compiled or GPU-correct; correct is not timed; profiled is not serving.
- Never change the oracle, tolerance, Workload, target, evaluator, baseline, or stopping rule.
- Never cache outputs, detect benchmark inputs, call a baseline implementation, or bypass
  the declared artifact interface.

## Completion

- Report only the Candidate and hypothesis you produced. Do not declare a scientific or
  serving result; the external Audit decides what the retained Evidence supports.
"""
    return TaskPackage(run_id, arm, task, agents)


def materialize_task_package(workspace: str | Path, package: TaskPackage) -> None:
    """Create the only two read-only task files in one new Agent workspace."""

    root = Path(workspace).resolve(strict=True)
    if any(root.iterdir()):
        raise ValueError("Ralph task workspace must be empty before materialization")
    for name, payload in (
        ("TASK.md", package.task_markdown),
        ("AGENTS.md", package.agents_markdown),
    ):
        path = root / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            encoded = payload.encode("utf-8")
            if os.write(descriptor, encoded) != len(encoded):
                raise OSError(f"short write for {name}")
        finally:
            os.close(descriptor)


def verify_task_package(workspace: str | Path, package: TaskPackage) -> None:
    """Fail when the Agent changed either frozen task file."""

    root = Path(workspace).resolve(strict=True)
    expected = {
        "TASK.md": package.task_markdown.encode(),
        "AGENTS.md": package.agents_markdown.encode(),
    }
    for name, payload in expected.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"Ralph task file {name} custody differs")
