"""Retain terminal faults and seal run checkpoints."""

from __future__ import annotations

from typing import Mapping, Sequence

from open_cake_ir.evidence.store import RunLedger

from .checkpoints import TurnObservation, project_checkpoints
from .faults import RunProtocolFault
from .ralph import RalphController
from .selection import _matched_endpoint_from_checkpoint


def record_run_fault(*, error, live_stage, turn_number, cumulative_tokens, evidence, ledger):
    """Retain the exact fault stage and any available artifacts before sealing."""
    fault = (
        error.protocol_adherence
        if isinstance(error, RunProtocolFault)
        else {
            "provider": "provider_fault",
            "environment": "harness_fault",
            "evaluation": "broker_fault",
        }[live_stage]
    )
    fault_payload: dict[str, object] = {
        "fault": fault,
        "exception_type": type(error).__name__,
        "turn": turn_number,
        "stage": live_stage,
        "terminal_provider_tokens": cumulative_tokens,
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
            fault_payload["objects"] = references
        if rejected_roles:
            fault_payload["artifact_rejections"] = rejected_roles
    ledger.append("run_fault", fault_payload)
    return fault


def _seal_run(
    *,
    ralph_stop_reason: str | None,
    checkpoints: Sequence[int],
    cumulative_tokens: int,
    feedback: Mapping[str, object],
    ledger: RunLedger,
    maximum_turns: int,
    observations: Sequence[TurnObservation],
    protocol_adherence: str,
    ralph: RalphController,
) -> None:
    if ralph_stop_reason is None:
        ralph_stop_reason = ralph.stop_reason(
            turn=min(maximum_turns + 1, len(observations) + 1),
            cumulative_provider_tokens=cumulative_tokens,
        ) or "maximum_turns"

    projected = project_checkpoints(
        turns=observations,
        checkpoints=checkpoints,
        terminal_provider_tokens=cumulative_tokens,
    )
    checkpoint_payload: dict[str, object] = {
        "checkpoints": [
                {
                    "provider_tokens": item.provider_tokens,
                    "state": item.state,
                    "best_candidate_sha256": item.best_candidate_sha256,
                    "best_confirmed_latency_ms": item.best_confirmed_latency_ms,
                }
                for item in projected
            ]
    }
    checkpoint_payload["ralph"] = dict(
        ralph.state_card(
            turn=min(maximum_turns + 1, len(observations) + 1),
            cumulative_provider_tokens=cumulative_tokens,
            feedback=feedback,
            terminal_reason=ralph_stop_reason,
        )
    )
    ledger.append("checkpoints_projected", checkpoint_payload)
    final_checkpoint = projected[-1]
    endpoint_observation, endpoint = _matched_endpoint_from_checkpoint(
        final_checkpoint, protocol_adherence
    )
    ledger.seal(
        protocol_adherence=protocol_adherence,
        endpoint_observation=endpoint_observation,
        endpoint=endpoint,
    )
