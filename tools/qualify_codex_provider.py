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
    CODEX_DISABLED_FEATURES,
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


def _terminal_message(turn: int, event_contract: str) -> str:
    document: dict[str, object] = {
        "arm": "open_cake",
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
    reference_bundle_sha256: str,
    reference_bundle: str,
    reference_nonce: str,
    prompt_template: Path,
) -> str:
    change = "add" if turn == 1 else "update"
    template = prompt_template.read_text(encoding="utf-8")
    replacements = {
        "{{CANDIDATE_PATH_JSON}}": json.dumps(str(candidate.absolute())),
        "{{EXPECTED_CHANGE}}": change,
        "{{EXPECTED_CANDIDATE_JSON}}": json.dumps(
            {"qualification_turn": turn, "reference_nonce": reference_nonce},
            sort_keys=True,
            separators=(",", ":"),
        ),
        "{{REFERENCE_BUNDLE_SHA256}}": reference_bundle_sha256,
        "{{REFERENCE_BUNDLE}}": reference_bundle,
    }
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"qualification prompt marker {marker!r} differs")
        template = template.replace(marker, value)
    return template


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
    if getattr(audit, "integrity", False) is not True or terminal_seal is None:
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
    parser.add_argument("--reasoning-effort", default="max")
    parser.add_argument("--service-tier", default="default")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
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
    if args.feature_policy == "closed_research":
        disabled_features = CODEX_DISABLED_FEATURES
        event_contract = "closed_file_change_v1"
        prompt_template = ROOT / "contracts/providers/codex-qualification-prompt-v1.md"
        receipt_scope = "live_two_turn_current_provider"
    else:
        disabled_features = ()
        event_contract = "tool_rich_candidate_v1"
        prompt_template = ROOT / "contracts/providers/codex-qualification-prompt-v2.md"
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
    ):
        raise ValueError("Codex qualification input custody differs")
    workspace.mkdir(mode=0o750)
    references.mkdir(mode=0o755)
    candidate = workspace / "candidate.json"
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
        builder = CodexInvocationBuilder(
            executable=executable,
            provider_revision=args.provider_revision,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            service_tier=args.service_tier,
            workspace=workspace,
            output_schema=output_schema,
            removed_environment=removed_environment,
            disabled_features=disabled_features,
            event_contract=event_contract,
        )
        adapter = CodexProviderAdapter(timeout_seconds=args.timeout_seconds)
        initial_invocation = builder.build(
            _turn_prompt(
                candidate,
                1,
                reference_bundle_sha256=reference_bundle_sha256,
                reference_bundle=reference_bundle,
                reference_nonce=reference_nonce,
                prompt_template=prompt_template,
            ),
            thread_id=None,
        )
        _validate_invocation(
            initial_invocation,
            executable=executable,
            workspace=workspace,
        )
        initial = adapter.execute(
            initial_invocation,
            candidate_path=candidate,
            expected_change="add",
            expected_terminal_message=_terminal_message(1, event_contract),
            event_contract=event_contract,
        )
        _validate_workspace(workspace, candidate)
        if json.loads(initial.candidates[0]) != {
            "qualification_turn": 1,
            "reference_nonce": reference_nonce,
        }:
            raise ValueError("Codex initial candidate bytes differ")

        resumed_invocation = builder.build(
            _turn_prompt(
                candidate,
                2,
                reference_bundle_sha256=reference_bundle_sha256,
                reference_bundle=reference_bundle,
                reference_nonce=reference_nonce,
                prompt_template=prompt_template,
            ),
            thread_id=initial.thread_id,
        )
        _validate_invocation(
            resumed_invocation,
            executable=executable,
            workspace=workspace,
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
            expected_terminal_message=_terminal_message(2, event_contract),
            event_contract=event_contract,
        )
        _validate_workspace(workspace, candidate)
        if (
            json.loads(resumed.candidates[0])
            != {"qualification_turn": 2, "reference_nonce": reference_nonce}
            or resumed.thread_id != initial.thread_id
            or initial.provider_tokens <= 0
            or resumed.provider_tokens <= 0
            or initial.candidate_sha256s == resumed.candidate_sha256s
            or sha256(executable.read_bytes()).hexdigest() != executable_sha256
            or sha256(output_schema.read_bytes()).hexdigest() != output_schema_sha256
            or read_frozen_reference_bundle(references)[0] != reference_bundle_sha256
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
            raise ValueError("Codex two-Turn identity, usage, or candidate lifecycle differs")

        receipt = ProviderQualificationReceipt(
            provider_revision=args.provider_revision,
            executable_sha256=executable_sha256,
            configuration_sha256=builder.configuration_sha256,
            initial_and_resume_equivalent=True,
            file_lifecycle_observed=True,
            usage_observed=True,
            qualified=True,
            scope=receipt_scope,
        )
        objects = [
            evidence.put(initial.raw_events, media_type="application/x-ndjson").reference(
                "initial_provider_events"
            ),
            evidence.put(initial.candidates[0], media_type="application/json").reference(
                "initial_candidate"
            ),
            _put_json(evidence, _invocation_document(initial_invocation)).reference(
                "initial_invocation"
            ),
            evidence.put(resumed.raw_events, media_type="application/x-ndjson").reference(
                "resumed_provider_events"
            ),
            evidence.put(resumed.candidates[0], media_type="application/json").reference(
                "resumed_candidate"
            ),
            _put_json(evidence, _invocation_document(resumed_invocation)).reference(
                "resumed_invocation"
            ),
            _put_json(evidence, receipt.document).reference("qualification_receipt"),
            evidence.put(reference_path.read_bytes(), media_type="application/json").reference(
                "qualification_reference"
            ),
        ]
        ledger.append(
            "provider_qualification_observed",
            {
                "thread_id": initial.thread_id,
                "initial_provider_tokens": initial.provider_tokens,
                "resumed_provider_tokens": resumed.provider_tokens,
                "initial_normalization": initial.normalization,
                "resumed_normalization": resumed.normalization,
                "initial_candidate_sha256": initial.candidate_sha256s[0],
                "resumed_candidate_sha256": resumed.candidate_sha256s[0],
                "objects": objects,
                "reference_bundle_sha256": reference_bundle_sha256,
                "initial_auxiliary_activity": [
                    dict(activity.document) for activity in initial.tool_activity
                ],
                "resumed_auxiliary_activity": [
                    dict(activity.document) for activity in resumed.tool_activity
                ],
            },
        )
        ledger.seal(
            protocol_adherence="adhered",
            endpoint_observation="qualified",
            endpoint={
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
            },
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
        not audit.integrity
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
                "integrity": audit.integrity,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
