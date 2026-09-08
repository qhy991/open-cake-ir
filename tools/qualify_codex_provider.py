#!/usr/bin/env python3
"""Qualify an exact Codex or Claude Code harness through the existing zero-GPU two-Turn protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evidence import EvidenceObject, EvidenceStore  # noqa: E402
from open_cake_ir.lab.pairing import comparison_arm
from open_cake_ir.lab.faults import RunProtocolFault  # noqa: E402
from open_cake_ir.lab.providers import (  # noqa: E402
    CANDIDATE_SET_ENVELOPE_V1,
    CODEX_DISABLED_FEATURES,
    resolve_codex_code_mode_host,
    CodexInvocationBuilder,
    CodexProviderAdapter,
    ProviderInvocation,
    ProviderQualificationReceipt,
)
from open_cake_ir.lab.claude import (
    CLAUDE_AUTHORING_TOOLS, CLAUDE_EVENT_CONTRACT, ClaudeInvocationBuilder,
    ClaudeProviderAdapter, parse_claude_turn_events,
)
from open_cake_ir.lab.task_package import (  # noqa: E402
    TASK_AGENTS_RALPH_V1,
    TaskPackage,
    materialize_task_package,
    render_task_request,
    verify_task_package,
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _terminal_message(turn: int, event_contract: str, *, arm: str = "open_cake") -> str:
    document: dict[str, object] = {
        "arm": arm,
        "candidate_written": True,
        "kind": "open_cake_ir_turn",
        "turn": turn,
    }
    if event_contract == "closed_file_change_v1":
        document["tool_calls"] = 1
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    )


def _expected_submission(
    arm: str,
    turn: int,
    reference_nonce: str,
    maximum_candidates_per_turn: int,
    python_source: str | None = None,
) -> dict[str, object]:
    if arm == "open_cake" and python_source is not None:
        members = [{"python_source": python_source +
                    f"\n# qualification turn {turn}; candidate {index}; reference {reference_nonce}\n"}
                   for index in range(maximum_candidates_per_turn)]
    elif arm == "open_cake":
        members: list[object] = [
            {
                "candidate_index": index,
                "qualification_turn": turn,
                "reference_nonce": reference_nonce,
            }
            for index in range(maximum_candidates_per_turn)
        ]
    elif arm == "native_triton":
        members = [
            {
                "kernel_source": (
                    "import triton\nimport triton.language as tl\n\n@triton.jit\n"
                    f"def qualification_{turn}_{index}(x, y):\n"
                    "    value = tl.load(x)\n    tl.store(y, value)\n"
                    f"# frozen reference: {reference_nonce}\n"
                ),
                "compile_constants": {},
                "compile_options": {"num_warps": 4},
                "grid": [1, 1, 1],
            }
            for index in range(maximum_candidates_per_turn)
        ]
    elif arm == "native_cute_dsl":
        members = [
            {
                "kernel_source": (
                    "import cutlass\nimport cutlass.cute as cute\nfrom cutlass.cute.nvgpu import warp\n\n"
                    "@cute.kernel\n"
                    f"def qualification_{turn}_{index}(a: cute.Pointer, b: cute.Pointer, bias: cute.Pointer, c: cute.Pointer):\n"
                    "    lane, _, _ = cute.arch.thread_idx()\n"
                    f"# frozen reference: {reference_nonce}\n"
                ),
                "grid": [1, 1, 1], "block": [32, 1, 1], "dynamic_shared_memory_bytes": 0,
            }
            for index in range(maximum_candidates_per_turn)
        ]
    else:
        members = [
            (
                f"// qualification candidate {index}; turn {turn}; "
                f"reference {reference_nonce}\n"
            )
            for index in range(maximum_candidates_per_turn)
        ]
    return {
        "schema_version": 1,
        "arm": arm,
        "candidates": members,
    }


def _qualification_package(
    run_id: str,
    arm: str,
    candidate: Path,
    reference_nonce: str,
    maximum_candidates_per_turn: int,
    event_contract: str,
    tool_instruction: str,
    python_source: str | None = None,
) -> TaskPackage:
    plan = {
        "candidate_path": str(candidate.absolute()),
        "turns": [
            {
                "turn": turn,
                "change": "add" if turn == 1 else "update",
                "submission": _expected_submission(
                    arm, turn, reference_nonce, maximum_candidates_per_turn, python_source
                ),
                "terminal_message": json.loads(
                    _terminal_message(turn, event_contract, arm=arm)
                ),
            }
            for turn in (1, 2)
        ],
    }
    task = (
        "# TASK.md — provider qualification\n\n"
        f"Prove two-Turn `{arm}` candidate-set add/update behavior.\n\n"
        "Use only the plan entry whose turn equals the controller StateCard turn. "
        "Write that entry's complete submission to candidate_path with its declared "
        "file change, then return its terminal_message as JSON. Do not execute "
        "either candidate. The plan and both task files stay unchanged between turns.\n\n"
        "The submission must be valid UTF-8 JSON. Whitespace, object-key order, and "
        "equivalent JSON escapes are accepted. Keys must be unique at every level, "
        "numbers must be finite, and schema_version must be the integer 1. Preserve "
        "Candidate array order and direct CUDA source strings exactly.\n\n"
        "QUALIFICATION_PLAN_JSON=" + _canonical_json_bytes(plan).decode() + "\n"
    )
    agents = (
        "# AGENTS.md — provider qualification\n\n"
        "Follow the complete TASK.md plan. Write only candidate-set.json. Keep "
        "TASK.md and AGENTS.md unchanged. Do not use a GPU or network.\n"
        + tool_instruction + "\n"
    )
    return TaskPackage(run_id, arm, task, agents)


def _planned_turn(package: TaskPackage, turn: int) -> dict[str, object]:
    """Read expected behavior from the same immutable TASK the provider receives."""

    prefix = "QUALIFICATION_PLAN_JSON="
    line = next(line for line in package.task_markdown.splitlines() if line.startswith(prefix))
    plan = json.loads(line[len(prefix):])
    return next(item for item in plan["turns"] if item["turn"] == turn)


def _planned_candidates(arm: str, plan: dict[str, object]) -> tuple[bytes, ...]:
    members = plan["submission"]["candidates"]
    return tuple(
        str(member).encode("utf-8") if arm == "direct_cuda"
        else _canonical_json_bytes(member)
        for member in members
    )


def _validate_invocation(
    invocation: ProviderInvocation,
    *,
    executable: Path,
    workspace: Path,
    harness: str = "codex",
) -> None:
    if harness == "claude-code":
        tools = ",".join(CLAUDE_AUTHORING_TOOLS)
        expected_options = {"--permission-mode": "acceptEdits", "--tools": tools, "--allowedTools": tools,
                            "--output-format": "stream-json"}
        if (invocation.cwd != workspace or invocation.sandbox != "none"
                or invocation.argv[:2] != (str(executable), "-p")
                or invocation.argv.count("--safe-mode") != 1
                or any(invocation.argv.count(flag) != 1 or
                       invocation.argv[invocation.argv.index(flag) + 1] != value
                       for flag, value in expected_options.items())):
            raise ValueError("Claude qualification tools, permissions or cwd differ")
        return
    if (
        invocation.cwd != workspace
        or invocation.sandbox != "workspace-write"
        or invocation.argv[0] != str(executable)
        or invocation.argv.count('sandbox_mode="workspace-write"') != 1
        or invocation.argv.count('approval_policy="never"') != 1
    ):
        raise ValueError("Codex qualification sandbox or cwd differs")


def _validate_invocation_pair(
    initial: ProviderInvocation,
    resumed: ProviderInvocation,
    *,
    thread_id: str,
    harness: str = "codex",
) -> None:
    if harness == "claude-code":
        if (initial.argv[:-2] != resumed.argv[:-4] or initial.argv[-2] != "--"
                or resumed.argv[-4:-1] != ("--resume", thread_id, "--")
                or initial.cwd != resumed.cwd or initial.sandbox != resumed.sandbox
                or initial.provider_revision != resumed.provider_revision
                or initial.removed_environment != resumed.removed_environment
                or initial.thread_id is not None or resumed.thread_id != thread_id):
            raise ValueError("Claude initial and resume environments differ")
        return
    if (
        initial.argv[:2] != (initial.argv[0], "exec")
        or resumed.argv[:3] != (resumed.argv[0], "exec", "resume")
        or initial.argv[2:-1] != resumed.argv[3:-2]
        or resumed.argv[-2] != thread_id
        or initial.cwd != resumed.cwd
        or initial.sandbox != resumed.sandbox
        or initial.provider_revision != resumed.provider_revision
        or initial.removed_environment != resumed.removed_environment
        or initial.thread_id is not None
        or resumed.thread_id != thread_id
    ):
        raise ValueError("Codex initial and resume environments differ")


def _reported_models(turn, *, harness: str, requested_model: str) -> list[str]:
    if harness != "claude-code":
        return []
    parsed = parse_claude_turn_events(turn.raw_events, expected_terminal_message=turn.terminal_message)
    if set(parsed.reported_models) != {requested_model}:
        raise RunProtocolFault("provider_fault", "Claude reported model differs from the exact requested model",
                               artifact_payloads={"provider_stdout": turn.raw_events})
    return list(parsed.reported_models)


def _validate_workspace(
    workspace: Path,
    candidate: Path,
    *,
    task_files: bool,
) -> None:
    entries = list(workspace.iterdir())
    expected = {candidate}
    if task_files:
        expected.update({workspace / "TASK.md", workspace / "AGENTS.md"})
    if set(entries) != expected or candidate.is_symlink() or not candidate.is_file():
        raise ValueError("Provider qualification workspace custody differs")
    if task_files and any(
        path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222
        for path in (workspace / "TASK.md", workspace / "AGENTS.md")
    ):
        raise ValueError("Provider qualification task-file custody differs")


def _invocation_document(invocation: ProviderInvocation) -> dict[str, object]:
    return {
        "argv": list(invocation.argv),
        "cwd": str(invocation.cwd),
        "sandbox": invocation.sandbox,
        "provider_revision": invocation.provider_revision,
        "removed_environment": list(invocation.removed_environment),
        "thread_id": invocation.thread_id,
    }


def _put_json(evidence: EvidenceStore, value: object) -> EvidenceObject:
    return evidence.put(_canonical_json_bytes(value), media_type="application/json")


def _new_path(value: Path) -> Path:
    """Keep qualification outputs outside every enclosing or linked checkout."""
    if not value.is_absolute():
        raise ValueError("qualification output paths must be absolute")
    parent = value.parent.resolve(strict=True)
    for directory in {value.parent, *value.parent.parents, parent, *parent.parents}:
        marker = directory / ".git"
        if marker.exists() or marker.is_symlink():
            raise ValueError("qualification outputs must be outside Git checkouts")
    return parent / value.name


def _write_anchor(
    path: Path,
    *,
    evidence: EvidenceStore,
    audit: object,
    authority_sha256: str,
    qualification_receipt_sha256: str | None,
    harness: str = "codex",
) -> dict[str, object]:
    terminal_seal = getattr(audit, "terminal_seal_sha256", None)
    if (
        getattr(audit, "archive_integrity", False) is not True
        or getattr(audit, "filesystem_custody_verified", False) is not True
        or terminal_seal is None
    ):
        raise ValueError("Provider qualification Evidence audit failed")
    anchor = {
        "schema_version": 1,
        "kind": "codex_provider_qualification_evidence_anchor" if harness == "codex" else "provider_qualification_evidence_anchor",
        "run_id": getattr(audit, "run_id"),
        "evidence_root": str(evidence.root),
        "authority_sha256": authority_sha256,
        "qualification_receipt_sha256": qualification_receipt_sha256,
        "immediate_audit_integrity": True,
        "terminal_seal_sha256": terminal_seal,
    }
    with path.open("xb") as stream:
        stream.write(_canonical_json_bytes(anchor) + b"\n")
    path.chmod(0o644)
    return anchor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=("codex", "claude-code"), required=True)
    parser.add_argument("--fixture-only", action="store_true", help="never issue a live qualification for executable test doubles")
    parser.add_argument("--python-source", type=Path, help="Workload Python starter required for single-arm artifact qualification")
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--provider-revision", required=True)
    parser.add_argument("--output-schema", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--anchor-output", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reasoning-effort",
        required=True,
        help="exact provider reasoning effort to qualify as a treatment factor",
    )
    parser.add_argument("--service-tier", default="default")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--maximum-candidates-per-turn",
        type=int,
        default=3,
        help="maximum candidates in each qualified arm's Ralph envelope",
    )
    parser.add_argument(
        "--feature-policy",
        choices=("closed_research", "provider_defaults_optimization"),
        default="closed_research",
    )
    parser.add_argument(
        "--remove-env",
        action="append",
        dest="removed_environment",
        default=None,
    )
    args = parser.parse_args()

    executable = args.executable.resolve(strict=True)
    output_schema = args.output_schema.resolve(strict=True)
    workspace = _new_path(args.workspace)
    receipt_output = _new_path(args.receipt_output)
    evidence_root = _new_path(args.evidence_root)
    anchor_output = _new_path(args.anchor_output)
    removed_environment = tuple(args.removed_environment or ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"))
    maximum_candidates_per_turn = args.maximum_candidates_per_turn or 1
    submission_contract = CANDIDATE_SET_ENVELOPE_V1
    schema = json.loads(output_schema.read_text(encoding="utf-8"))
    arm_schema = schema.get("properties", {}).get("arm", {})
    arms = arm_schema.get("enum")
    single_arm = arms == ["open_cake"]
    if not isinstance(arms, list) or len(arms) not in {1, 2} or arms[0] != "open_cake":
        raise ValueError("qualification output schema must declare one supported arm pair or single Open Cake arm")
    try:
        comparison_arm(dict.fromkeys(arms))
    except ValueError as error:
        raise ValueError("qualification output schema must declare one supported arm pair or single Open Cake arm") from error
    if single_arm and (args.feature_policy != "provider_defaults_optimization" or args.python_source is None):
        raise ValueError("single-arm artifact qualification requires provider defaults and --python-source")
    if args.harness == "claude-code" and (not single_arm or args.service_tier != "default"):
        raise ValueError("Claude qualification requires a single artifact-only arm and no service-tier override")
    if not single_arm and args.python_source is not None:
        raise ValueError("Python source qualification requires the single artifact-only arm")
    python_source = None
    if args.python_source is not None:
        python_source = args.python_source.resolve(strict=True).read_text(encoding="utf-8")
        if not python_source.strip():
            raise ValueError("qualification Python source is empty")
        compile(python_source, str(args.python_source), "exec")
    qualification_arms = tuple(arms)
    if args.harness == "claude-code":
        disabled_features = ()
        event_contract = CLAUDE_EVENT_CONTRACT
        tool_instruction = "Use Read for the task files and Write/Edit for candidate-set.json; only Read, Write, Edit, Glob and Grep are permitted."
        receipt_scope = "live_two_turn_tool_rich_provider"
    elif args.feature_policy == "closed_research":
        disabled_features = CODEX_DISABLED_FEATURES
        event_contract = "closed_file_change_v1"
        tool_instruction = "Do not invoke auxiliary tools."
        receipt_scope = "live_two_turn_current_provider"
    else:
        disabled_features = ()
        event_contract = "tool_rich_candidate_v1"
        tool_instruction = (
            "First use the shell tool to run `pwd` without writing a file or "
            "invoking a network/GPU operation."
        )
        receipt_scope = "live_two_turn_tool_rich_provider"
    if args.fixture_only:
        receipt_scope = "zero_gpu_contract_fixture_only"
    if (
        not executable.is_file()
        or not os.access(executable, os.X_OK)
        or not output_schema.is_file()
        or not args.provider_revision
        or workspace.exists()
        or workspace.is_symlink()
        or receipt_output.exists()
        or receipt_output.is_symlink()
        or anchor_output.exists()
        or anchor_output.is_symlink()
        or anchor_output == receipt_output
        or anchor_output == evidence_root
        or evidence_root in anchor_output.parents
        or (
            args.maximum_candidates_per_turn is not None
            and args.maximum_candidates_per_turn <= 0
        )
    ):
        raise ValueError("Provider qualification input custody differs")
    workspace.mkdir(mode=0o750)
    workspaces = {}
    for arm in qualification_arms:
        arm_workspace = workspace / arm
        arm_workspace.mkdir(mode=0o750)
        workspaces[arm] = arm_workspace
    executable_sha256 = sha256(executable.read_bytes()).hexdigest()
    code_mode_host = (resolve_codex_code_mode_host(executable, removed_environment=removed_environment)
                      if args.harness == "codex" else None)
    output_schema_sha256 = sha256(output_schema.read_bytes()).hexdigest()
    reference_nonce = sha256(
        _canonical_json_bytes(
            {
                "provider_revision": args.provider_revision,
                "executable_sha256": executable_sha256,
                "output_schema_sha256": output_schema_sha256,
                **(
                    {"submission_contract": submission_contract}
                ),
            }
        )
    ).hexdigest()
    task_packages: dict[str, TaskPackage] = {}
    for arm, arm_workspace in workspaces.items():
        package = _qualification_package(
            f"{args.run_id}-{arm}", arm, arm_workspace / "candidate-set.json",
            reference_nonce, maximum_candidates_per_turn, event_contract,
            tool_instruction, python_source,
        )
        materialize_task_package(arm_workspace, package)
        task_packages[arm] = package
    task_bundle = {
        arm: {
            "task_markdown": package.task_markdown,
            "agents_markdown": package.agents_markdown,
        }
        for arm, package in task_packages.items()
    }
    reference_bundle = json.dumps(
        task_bundle,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    reference_bundle_sha256 = sha256(reference_bundle.encode()).hexdigest()
    authority = {
        "schema_version": 1,
        "kind": "codex_provider_two_turn_qualification" if args.harness == "codex" else "provider_two_turn_qualification",
        "provider_revision": args.provider_revision,
        "harness": args.harness,
        "qualification_scope": receipt_scope,
        "executable_sha256": executable_sha256,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "output_schema_sha256": output_schema_sha256,
        "removed_environment": list(removed_environment),
        "reference_bundle_sha256": reference_bundle_sha256,
        "disabled_features": list(disabled_features),
        "event_contract": event_contract,
        "feature_policy": args.feature_policy,
        "sandbox": "workspace-write" if args.harness == "codex" else "none",
        "cwd_policy": "same_new_task_workspace",
        "reference_visibility": "workspace_task_files",
        "agent_interface": TASK_AGENTS_RALPH_V1,
        "turns": ["initial_add", "same_thread_resume_update"],
        "gpu_execution_authorized": False,
    }
    if args.harness == "codex":
        authority.update(code_mode_host=code_mode_host, service_tier=args.service_tier)
    authority["submission_contract"] = submission_contract
    if event_contract == "closed_file_change_v1":
        authority["web_search"] = "disabled"
    authority["maximum_candidates_per_turn"] = maximum_candidates_per_turn
    authority["arms"] = list(qualification_arms)
    authority_sha256 = sha256(_canonical_json_bytes(authority)).hexdigest()
    evidence = (
        EvidenceStore.writer(evidence_root)
        if evidence_root.exists()
        else EvidenceStore.create(evidence_root)
    )
    ledger = evidence.start_run(
        args.run_id,
        authority_sha256=authority_sha256,
        authority=authority,
    )
    try:
        adapter_type = CodexProviderAdapter if args.harness == "codex" else ClaudeProviderAdapter
        adapter = adapter_type(timeout_seconds=args.timeout_seconds)
        observations: dict[str, dict[str, object]] = {}
        configuration_sha256s: set[str] = set()
        for arm in qualification_arms:
            arm_workspace = workspaces[arm]
            candidate = arm_workspace / "candidate-set.json"
            package = task_packages[arm]
            common_builder_args = dict(executable=executable, provider_revision=args.provider_revision,
                model=args.model, reasoning_effort=args.reasoning_effort, workspace=arm_workspace,
                removed_environment=removed_environment)
            if args.harness == "claude-code":
                builder = ClaudeInvocationBuilder(**common_builder_args)
            else:
                builder = CodexInvocationBuilder(**common_builder_args,
                    code_mode_host=code_mode_host, service_tier=args.service_tier,
                    output_schema=output_schema, disabled_features=disabled_features,
                    event_contract=event_contract, submission_contract=submission_contract,
                    cwd_policy="independent_task_workspace", reference_visibility="workspace_task_files")
            configuration_sha256s.add(sha256(json.dumps(
                builder.configuration, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest())
            initial_plan = _planned_turn(package, 1)
            verify_task_package(arm_workspace, package)
            initial_prompt, initial_projection = render_task_request(package, {"turn": 1})
            initial_invocation = builder.build(
                initial_prompt,
                thread_id=None,
            )
            _validate_invocation(
                initial_invocation,
                executable=executable,
                workspace=arm_workspace, harness=args.harness,
            )
            initial = adapter.execute(
                initial_invocation,
                candidate_path=candidate,
                expected_change=initial_plan["change"],
                expected_terminal_message=_canonical_json_bytes(initial_plan["terminal_message"]).decode(),
                event_contract=event_contract,
                submission_contract=submission_contract,
                arm=arm,
                maximum_candidates_per_turn=maximum_candidates_per_turn,
            )
            initial_models = _reported_models(initial, harness=args.harness, requested_model=args.model)
            _validate_workspace(
                arm_workspace, candidate, task_files=True
            )
            verify_task_package(arm_workspace, package)
            if (
                initial.candidates != _planned_candidates(arm, initial_plan)
            ):
                raise ValueError("Provider initial candidate bytes differ")

            resumed_plan = _planned_turn(package, 2)
            resumed_prompt, resumed_projection = render_task_request(package, {"turn": 2})
            resumed_invocation = builder.build(
                resumed_prompt,
                thread_id=initial.thread_id,
            )
            _validate_invocation(
                resumed_invocation,
                executable=executable,
                workspace=arm_workspace, harness=args.harness,
            )
            _validate_invocation_pair(
                initial_invocation,
                resumed_invocation,
                thread_id=initial.thread_id, harness=args.harness,
            )
            resumed = adapter.execute(
                resumed_invocation,
                candidate_path=candidate,
                expected_change=resumed_plan["change"],
                expected_terminal_message=_canonical_json_bytes(resumed_plan["terminal_message"]).decode(),
                event_contract=event_contract,
                submission_contract=submission_contract,
                arm=arm,
                maximum_candidates_per_turn=maximum_candidates_per_turn,
            )
            resumed_models = _reported_models(resumed, harness=args.harness, requested_model=args.model)
            _validate_workspace(
                arm_workspace, candidate, task_files=True
            )
            verify_task_package(arm_workspace, package)
            if (
                (
                    resumed.candidates != _planned_candidates(arm, resumed_plan)
                )
                or resumed.thread_id != initial.thread_id
                or initial.provider_tokens <= 0
                or resumed.provider_tokens <= 0
                or initial.candidate_sha256s == resumed.candidate_sha256s
                or (
                    event_contract == "tool_rich_candidate_v1"
                    and (
                        not any(
                            activity.item_type == "command_execution"
                            for activity in initial.tool_activity
                        )
                        or not any(
                            activity.item_type == "command_execution"
                            for activity in resumed.tool_activity
                        )
                    )
                )
            ):
                raise ValueError(
                    "Provider two-Turn identity, usage, or candidate lifecycle differs"
                )
            observations[arm] = {
                "reported_models": [initial_models, resumed_models],
                "builder": builder,
                "candidate": candidate,
                "initial": initial,
                "initial_invocation": initial_invocation,
                "initial_projection": initial_projection,
                "resumed": resumed,
                "resumed_invocation": resumed_invocation,
                "resumed_projection": resumed_projection,
            }

        if (
            len(configuration_sha256s) != 1
            or (
                submission_contract == CANDIDATE_SET_ENVELOPE_V1
                and set(workspace.iterdir()) != set(workspaces.values())
            )
            or len(
                {
                    str(observation["initial"].thread_id)
                    for observation in observations.values()
                }
            )
            != len(qualification_arms)
            or sha256(executable.read_bytes()).hexdigest() != executable_sha256
            or sha256(output_schema.read_bytes()).hexdigest() != output_schema_sha256
            or (
                any(
                    (workspaces[arm] / "TASK.md").read_bytes() != package.task_markdown.encode()
                    or (workspaces[arm] / "AGENTS.md").read_bytes() != package.agents_markdown.encode()
                    for arm, package in task_packages.items()
                )
            )
        ):
            raise ValueError("Provider qualification authority changed")

        if args.harness == "codex":
            resolve_codex_code_mode_host(
                executable, expected=code_mode_host, removed_environment=removed_environment,
            )
        receipt = ProviderQualificationReceipt(
            provider_revision=args.provider_revision,
            executable_sha256=executable_sha256,
            configuration_sha256=next(iter(configuration_sha256s)),
            initial_and_resume_equivalent=True,
            file_lifecycle_observed=True,
            usage_observed=True,
            qualified=True,
            scope=receipt_scope,
        )
        objects = []
        arm_payloads: dict[str, object] = {}
        for arm, observation in observations.items():
            initial = observation["initial"]
            resumed = observation["resumed"]
            initial_invocation = observation["initial_invocation"]
            resumed_invocation = observation["resumed_invocation"]
            prefix = f"{arm}_"
            objects.extend(
                [
                    evidence.put(
                        initial.raw_events,
                        media_type="application/x-ndjson",
                    ).reference(f"{prefix}initial_provider_events"),
                    _put_json(
                        evidence, _invocation_document(initial_invocation)
                    ).reference(f"{prefix}initial_invocation"),
                    evidence.put(
                        resumed.raw_events,
                        media_type="application/x-ndjson",
                    ).reference(f"{prefix}resumed_provider_events"),
                    _put_json(
                        evidence, _invocation_document(resumed_invocation)
                    ).reference(f"{prefix}resumed_invocation"),
                    evidence.put(
                        observation["initial_projection"], media_type="application/json"
                    ).reference(f"{prefix}initial_task_projection"),
                    evidence.put(
                        observation["resumed_projection"], media_type="application/json"
                    ).reference(f"{prefix}resumed_task_projection"),
                ]
            )
            objects.extend(
                [
                    evidence.put(
                        initial.raw_submission,
                        media_type="application/json",
                    ).reference(f"{arm}_initial_submission_envelope"),
                    evidence.put(
                        resumed.raw_submission,
                        media_type="application/json",
                    ).reference(f"{arm}_resumed_submission_envelope"),
                ]
            )
            candidate_media_type = (
                "text/x-cuda" if arm == "direct_cuda" else "application/json"
            )
            for phase, turn in (("initial", initial), ("resumed", resumed)):
                objects.extend(
                    evidence.put(candidate, media_type=candidate_media_type).reference(
                        f"{arm}_{phase}_candidate_{index:04d}"
                    )
                    for index, candidate in enumerate(turn.candidates)
                )
            arm_payloads[arm] = {
                "reported_models": observation["reported_models"],
                "thread_id": initial.thread_id,
                "initial_provider_tokens": initial.provider_tokens,
                "resumed_provider_tokens": resumed.provider_tokens,
                "initial_normalization": initial.normalization,
                "resumed_normalization": resumed.normalization,
                "initial_candidate_sha256s": list(initial.candidate_sha256s),
                "resumed_candidate_sha256s": list(resumed.candidate_sha256s),
                "initial_auxiliary_activity": [
                    dict(activity.document) for activity in initial.tool_activity
                ],
                "resumed_auxiliary_activity": [
                    dict(activity.document) for activity in resumed.tool_activity
                ],
            }
        objects.extend(
            [
                _put_json(evidence, receipt.document).reference(
                    "qualification_receipt"
                ),
                evidence.put(
                    reference_bundle.encode(), media_type="application/json"
                ).reference("qualification_reference"),
            ]
        )
        observed_payload = {
            "submission_contract": submission_contract,
            "maximum_candidates_per_turn": maximum_candidates_per_turn,
            "arms": arm_payloads,
            "objects": objects,
            "reference_bundle_sha256": reference_bundle_sha256,
        }
        ledger.append(
            "provider_qualification_observed",
            observed_payload,
        )
        endpoint = {
            "qualification_receipt_sha256": receipt.canonical_sha256,
            "add_observed": True,
            "update_observed": True,
            "thread_continuity_observed": True,
            "usage_observed": True,
            "sandbox_observed": True,
            "cwd_observed": True,
            "candidate_changed": True,
            "reference_visibility_observed": True,
            "gpu_execution_authorized": False,
            "feature_policy": args.feature_policy,
            "qualification_scope": receipt_scope,
            "harness": args.harness,
            "event_contract": event_contract,
        }
        endpoint["submission_contract"] = submission_contract
        endpoint["arms_qualified"] = list(qualification_arms)
        endpoint["maximum_candidates_per_turn"] = maximum_candidates_per_turn
        ledger.seal(
            protocol_adherence="adhered",
            endpoint_observation="qualified",
            endpoint=endpoint,
        )
    except Exception as error:
        failure_payload: dict[str, object] = {
            "exception_type": type(error).__name__
        }
        if isinstance(error, RunProtocolFault) and error.artifact_payloads:
            references = []
            rejected_roles = []
            for role, payload in sorted(error.artifact_payloads.items()):
                try:
                    references.append(
                        evidence.put(payload, media_type="text/plain").reference(role)
                    )
                except (OSError, ValueError):
                    rejected_roles.append(role)
            if references:
                failure_payload["objects"] = references
            if rejected_roles:
                failure_payload["artifact_rejections"] = rejected_roles
        ledger.append(
            "provider_qualification_failed",
            failure_payload,
        )
        ledger.seal(
            protocol_adherence="provider_fault",
            endpoint_observation="missing",
        )
        _write_anchor(
            anchor_output,
            evidence=evidence,
            audit=evidence.audit_run(args.run_id),
            authority_sha256=authority_sha256,
            qualification_receipt_sha256=None, harness=args.harness,
        )
        raise

    audit = evidence.audit_run(args.run_id)
    if (
        not audit.archive_integrity
        or not audit.filesystem_custody_verified
        or audit.protocol_adherence != "adhered"
        or audit.terminal_seal_sha256 is None
    ):
        raise ValueError("Provider qualification Evidence audit failed")
    with receipt_output.open("xb") as stream:
        stream.write(_canonical_json_bytes(receipt.document) + b"\n")
    receipt_output.chmod(0o644)
    _write_anchor(
        anchor_output,
        evidence=evidence,
        audit=audit,
        authority_sha256=authority_sha256,
        qualification_receipt_sha256=receipt.canonical_sha256, harness=args.harness,
    )
    sys.stdout.write(
        json.dumps(
            {
                "run_id": audit.run_id,
                "authority_sha256": authority_sha256,
                "qualification_receipt_sha256": receipt.canonical_sha256,
                "receipt_output": str(receipt_output),
                "anchor_output": str(anchor_output),
                "terminal_seal_sha256": audit.terminal_seal_sha256,
                "evidence_root": str(evidence.root),
                "integrity": audit.archive_integrity,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
