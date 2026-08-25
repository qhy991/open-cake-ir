#!/usr/bin/env python3
"""Qualify one Codex provider executable through a real zero-GPU two-Turn run."""

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
from open_cake_ir.lab.faults import RunProtocolFault  # noqa: E402
from open_cake_ir.lab.providers import (  # noqa: E402
    CANDIDATE_SET_ENVELOPE_V1,
    CODEX_DISABLED_FEATURES,
    SINGLE_CANDIDATE_V1,
    CodexInvocationBuilder,
    CodexProviderAdapter,
    ProviderInvocation,
    ProviderQualificationReceipt,
    read_frozen_reference_bundle,
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


def _turn_prompt(
    candidate: Path,
    turn: int,
    *,
    arm: str,
    expected_submission: object,
    maximum_candidates_per_turn: int,
    submission_contract: str,
    tool_instruction: str,
    reference_bundle_sha256: str,
    reference_bundle: str,
    prompt_template: Path,
) -> str:
    change = "add" if turn == 1 else "update"
    template = prompt_template.read_text(encoding="utf-8")
    replacements = {
        "{{CANDIDATE_PATH_JSON}}": json.dumps(str(candidate.absolute())),
        "{{EXPECTED_CHANGE}}": change,
        "{{REFERENCE_BUNDLE_SHA256}}": reference_bundle_sha256,
        "{{REFERENCE_BUNDLE}}": reference_bundle,
    }
    if submission_contract == CANDIDATE_SET_ENVELOPE_V1:
        replacements.update(
            {
                "{{ARM}}": arm,
                "{{MAXIMUM_CANDIDATES_PER_TURN}}": str(
                    maximum_candidates_per_turn
                ),
                "{{EXPECTED_CANDIDATE_SET_JSON}}": json.dumps(
                    expected_submission,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                "{{TOOL_INSTRUCTION}}": tool_instruction,
            }
        )
    else:
        replacements["{{EXPECTED_CANDIDATE_JSON}}"] = json.dumps(
            expected_submission,
            sort_keys=True,
            separators=(",", ":"),
        )
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"qualification prompt marker {marker!r} differs")
        template = template.replace(marker, value)
    return template


def _expected_submission(
    arm: str,
    turn: int,
    reference_nonce: str,
    maximum_candidates_per_turn: int,
    submission_contract: str,
) -> tuple[object, tuple[bytes, ...]]:
    if submission_contract == SINGLE_CANDIDATE_V1:
        document = {
            "qualification_turn": turn,
            "reference_nonce": reference_nonce,
        }
        return document, (_canonical_json_bytes(document),)
    if arm == "open_cake":
        members: list[object] = [
            {
                "candidate_index": index,
                "qualification_turn": turn,
                "reference_nonce": reference_nonce,
            }
            for index in range(maximum_candidates_per_turn)
        ]
        projected = tuple(_canonical_json_bytes(member) for member in members)
    else:
        members = [
            (
                f"// qualification candidate {index}; turn {turn}; "
                f"reference {reference_nonce}\n"
            )
            for index in range(maximum_candidates_per_turn)
        ]
        projected = tuple(str(member).encode("utf-8") for member in members)
    return {
        "schema_version": 1,
        "arm": arm,
        "candidates": members,
    }, projected


def _validate_invocation(
    invocation: ProviderInvocation,
    *,
    executable: Path,
    workspace: Path,
) -> None:
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
) -> None:
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


def _validate_workspace(workspace: Path, candidate: Path) -> None:
    entries = list(workspace.iterdir())
    if entries != [candidate] or candidate.is_symlink() or not candidate.is_file():
        raise ValueError("Codex qualification workspace custody differs")


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
    return value.parent.resolve(strict=True) / value.name


def _write_anchor(
    path: Path,
    *,
    evidence: EvidenceStore,
    audit: object,
    authority_sha256: str,
    qualification_receipt_sha256: str | None,
) -> dict[str, object]:
    terminal_seal = getattr(audit, "terminal_seal_sha256", None)
    if (
        getattr(audit, "archive_integrity", False) is not True
        or getattr(audit, "filesystem_custody_verified", False) is not True
        or terminal_seal is None
    ):
        raise ValueError("Codex provider qualification Evidence audit failed")
    anchor = {
        "schema_version": 1,
        "kind": "codex_provider_qualification_evidence_anchor",
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
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--provider-revision", required=True)
    parser.add_argument("--output-schema", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--anchor-output", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
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
        default=None,
        help="qualify the canonical candidate-set envelope for both arms",
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
    references = workspace.parent / f"{workspace.name}-references"
    receipt_output = _new_path(args.receipt_output)
    evidence_root = _new_path(args.evidence_root)
    anchor_output = _new_path(args.anchor_output)
    removed_environment = tuple(args.removed_environment or ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"))
    maximum_candidates_per_turn = args.maximum_candidates_per_turn or 1
    submission_contract = (
        CANDIDATE_SET_ENVELOPE_V1
        if args.maximum_candidates_per_turn is not None
        else SINGLE_CANDIDATE_V1
    )
    qualification_arms = (
        ("open_cake", "direct_cuda")
        if submission_contract == CANDIDATE_SET_ENVELOPE_V1
        else ("open_cake",)
    )
    if args.feature_policy == "closed_research":
        disabled_features = CODEX_DISABLED_FEATURES
        event_contract = "closed_file_change_v1"
        prompt_template = ROOT / "contracts/providers/codex-qualification-prompt-v1.md"
        tool_instruction = "Do not invoke auxiliary tools."
        receipt_scope = "live_two_turn_current_provider"
    else:
        disabled_features = ()
        event_contract = "tool_rich_candidate_v1"
        prompt_template = ROOT / "contracts/providers/codex-qualification-prompt-v2.md"
        tool_instruction = (
            "First use the shell tool to run `pwd` without writing a file or "
            "invoking a network/GPU operation."
        )
        receipt_scope = "live_two_turn_tool_rich_provider"
    if (
        not executable.is_file()
        or not os.access(executable, os.X_OK)
        or not output_schema.is_file()
        or not args.provider_revision
        or workspace.exists()
        or workspace.is_symlink()
        or references.exists()
        or references.is_symlink()
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
        raise ValueError("Codex qualification input custody differs")
    workspace.mkdir(mode=0o750)
    workspaces = {"open_cake": workspace}
    if submission_contract == CANDIDATE_SET_ENVELOPE_V1:
        workspaces = {}
        for arm in qualification_arms:
            arm_workspace = workspace / arm
            arm_workspace.mkdir(mode=0o750)
            workspaces[arm] = arm_workspace
        prompt_template = (
            ROOT
            / "contracts/providers/codex-qualification-candidate-set-prompt-v1.md"
        )
    references.mkdir(mode=0o755)
    executable_sha256 = sha256(executable.read_bytes()).hexdigest()
    output_schema_sha256 = sha256(output_schema.read_bytes()).hexdigest()
    qualification_prompt_sha256 = sha256(
        prompt_template.read_bytes()
    ).hexdigest()
    reference_nonce = sha256(
        _canonical_json_bytes(
            {
                "provider_revision": args.provider_revision,
                "executable_sha256": executable_sha256,
                "output_schema_sha256": output_schema_sha256,
                **(
                    {"submission_contract": submission_contract}
                    if submission_contract == CANDIDATE_SET_ENVELOPE_V1
                    else {}
                ),
            }
        )
    ).hexdigest()
    reference_path = references / "qualification-authority.json"
    with reference_path.open("xb") as stream:
        stream.write(
            _canonical_json_bytes(
                {"schema_version": 1, "qualification_nonce": reference_nonce}
            )
        )
    reference_path.chmod(0o444)
    references.chmod(0o555)
    reference_bundle_sha256, reference_bundle = read_frozen_reference_bundle(references)
    authority = {
        "schema_version": 1,
        "kind": "codex_provider_two_turn_qualification",
        "provider_revision": args.provider_revision,
        "executable_sha256": executable_sha256,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "service_tier": args.service_tier,
        "output_schema_sha256": output_schema_sha256,
        "qualification_prompt_sha256": qualification_prompt_sha256,
        "removed_environment": list(removed_environment),
        "reference_bundle_sha256": reference_bundle_sha256,
        "disabled_features": list(disabled_features),
        "event_contract": event_contract,
        "feature_policy": args.feature_policy,
        "sandbox": "workspace-write",
        "cwd_policy": "same_new_empty_workspace",
        "turns": ["initial_add", "same_thread_resume_update"],
        "gpu_execution_authorized": False,
    }
    if submission_contract == CANDIDATE_SET_ENVELOPE_V1:
        authority["submission_contract"] = submission_contract
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
        adapter = CodexProviderAdapter(timeout_seconds=args.timeout_seconds)
        observations: dict[str, dict[str, object]] = {}
        configuration_sha256s: set[str] = set()
        for arm in qualification_arms:
            arm_workspace = workspaces[arm]
            candidate = arm_workspace / (
                "candidate-set.json"
                if submission_contract == CANDIDATE_SET_ENVELOPE_V1
                else "candidate.json"
            )
            builder = CodexInvocationBuilder(
                executable=executable,
                provider_revision=args.provider_revision,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                service_tier=args.service_tier,
                workspace=arm_workspace,
                output_schema=output_schema,
                removed_environment=removed_environment,
                disabled_features=disabled_features,
                event_contract=event_contract,
                submission_contract=submission_contract,
            )
            configuration_sha256s.add(builder.configuration_sha256)
            expected_initial, initial_candidates = _expected_submission(
                arm,
                1,
                reference_nonce,
                maximum_candidates_per_turn,
                submission_contract,
            )
            initial_invocation = builder.build(
                _turn_prompt(
                    candidate,
                    1,
                    arm=arm,
                    expected_submission=expected_initial,
                    maximum_candidates_per_turn=maximum_candidates_per_turn,
                    submission_contract=submission_contract,
                    tool_instruction=tool_instruction,
                    reference_bundle_sha256=reference_bundle_sha256,
                    reference_bundle=reference_bundle,
                    prompt_template=prompt_template,
                ),
                thread_id=None,
            )
            _validate_invocation(
                initial_invocation,
                executable=executable,
                workspace=arm_workspace,
            )
            initial = adapter.execute(
                initial_invocation,
                candidate_path=candidate,
                expected_change="add",
                expected_terminal_message=_terminal_message(
                    1, event_contract, arm=arm
                ),
                event_contract=event_contract,
                submission_contract=submission_contract,
                arm=(
                    arm
                    if submission_contract == CANDIDATE_SET_ENVELOPE_V1
                    else None
                ),
                maximum_candidates_per_turn=maximum_candidates_per_turn,
            )
            _validate_workspace(arm_workspace, candidate)
            initial_submission = candidate.read_bytes()
            if (
                initial.candidates != initial_candidates
                if submission_contract == CANDIDATE_SET_ENVELOPE_V1
                else json.loads(initial.candidates[0]) != expected_initial
            ):
                raise ValueError("Codex initial candidate bytes differ")

            expected_resumed, resumed_candidates = _expected_submission(
                arm,
                2,
                reference_nonce,
                maximum_candidates_per_turn,
                submission_contract,
            )
            resumed_invocation = builder.build(
                _turn_prompt(
                    candidate,
                    2,
                    arm=arm,
                    expected_submission=expected_resumed,
                    maximum_candidates_per_turn=maximum_candidates_per_turn,
                    submission_contract=submission_contract,
                    tool_instruction=tool_instruction,
                    reference_bundle_sha256=reference_bundle_sha256,
                    reference_bundle=reference_bundle,
                    prompt_template=prompt_template,
                ),
                thread_id=initial.thread_id,
            )
            _validate_invocation(
                resumed_invocation,
                executable=executable,
                workspace=arm_workspace,
            )
            _validate_invocation_pair(
                initial_invocation,
                resumed_invocation,
                thread_id=initial.thread_id,
            )
            resumed = adapter.execute(
                resumed_invocation,
                candidate_path=candidate,
                expected_change="update",
                expected_terminal_message=_terminal_message(
                    2, event_contract, arm=arm
                ),
                event_contract=event_contract,
                submission_contract=submission_contract,
                arm=(
                    arm
                    if submission_contract == CANDIDATE_SET_ENVELOPE_V1
                    else None
                ),
                maximum_candidates_per_turn=maximum_candidates_per_turn,
            )
            _validate_workspace(arm_workspace, candidate)
            resumed_submission = candidate.read_bytes()
            if (
                (
                    resumed.candidates != resumed_candidates
                    if submission_contract == CANDIDATE_SET_ENVELOPE_V1
                    else json.loads(resumed.candidates[0]) != expected_resumed
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
                    "Codex two-Turn identity, usage, or candidate lifecycle differs"
                )
            observations[arm] = {
                "builder": builder,
                "candidate": candidate,
                "initial": initial,
                "initial_invocation": initial_invocation,
                "initial_submission": initial_submission,
                "resumed": resumed,
                "resumed_invocation": resumed_invocation,
                "resumed_submission": resumed_submission,
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
            or read_frozen_reference_bundle(references)[0] != reference_bundle_sha256
        ):
            raise ValueError("Codex provider qualification authority changed")

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
            prefix = "" if submission_contract == SINGLE_CANDIDATE_V1 else f"{arm}_"
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
                ]
            )
            if submission_contract == SINGLE_CANDIDATE_V1:
                objects.extend(
                    [
                        evidence.put(
                            initial.candidates[0], media_type="application/json"
                        ).reference("initial_candidate"),
                        evidence.put(
                            resumed.candidates[0], media_type="application/json"
                        ).reference("resumed_candidate"),
                    ]
                )
            else:
                objects.extend(
                    [
                        evidence.put(
                            observation["initial_submission"],
                            media_type="application/json",
                        ).reference(f"{arm}_initial_submission_envelope"),
                        evidence.put(
                            observation["resumed_submission"],
                            media_type="application/json",
                        ).reference(f"{arm}_resumed_submission_envelope"),
                    ]
                )
                candidate_media_type = (
                    "application/json" if arm == "open_cake" else "text/x-cuda"
                )
                for phase, turn in (("initial", initial), ("resumed", resumed)):
                    objects.extend(
                        evidence.put(candidate, media_type=candidate_media_type).reference(
                            f"{arm}_{phase}_candidate_{index:04d}"
                        )
                        for index, candidate in enumerate(turn.candidates)
                    )
            arm_payloads[arm] = {
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
                    reference_path.read_bytes(), media_type="application/json"
                ).reference("qualification_reference"),
            ]
        )
        if submission_contract == SINGLE_CANDIDATE_V1:
            legacy = arm_payloads["open_cake"]
            observed_payload = {
                "thread_id": legacy["thread_id"],
                "initial_provider_tokens": legacy["initial_provider_tokens"],
                "resumed_provider_tokens": legacy["resumed_provider_tokens"],
                "initial_normalization": legacy["initial_normalization"],
                "resumed_normalization": legacy["resumed_normalization"],
                "initial_candidate_sha256": legacy["initial_candidate_sha256s"][0],
                "resumed_candidate_sha256": legacy["resumed_candidate_sha256s"][0],
                "objects": objects,
                "reference_bundle_sha256": reference_bundle_sha256,
                "initial_auxiliary_activity": legacy["initial_auxiliary_activity"],
                "resumed_auxiliary_activity": legacy["resumed_auxiliary_activity"],
            }
        else:
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
            "event_contract": event_contract,
        }
        if submission_contract == CANDIDATE_SET_ENVELOPE_V1:
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
            qualification_receipt_sha256=None,
        )
        raise

    audit = evidence.audit_run(args.run_id)
    if (
        not audit.archive_integrity
        or not audit.filesystem_custody_verified
        or audit.protocol_adherence != "adhered"
        or audit.terminal_seal_sha256 is None
    ):
        raise ValueError("Codex provider qualification Evidence audit failed")
    with receipt_output.open("xb") as stream:
        stream.write(_canonical_json_bytes(receipt.document) + b"\n")
    receipt_output.chmod(0o644)
    _write_anchor(
        anchor_output,
        evidence=evidence,
        audit=audit,
        authority_sha256=authority_sha256,
        qualification_receipt_sha256=receipt.canonical_sha256,
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
