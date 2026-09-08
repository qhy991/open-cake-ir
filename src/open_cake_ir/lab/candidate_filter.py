"""Build the complete candidate set and apply only complete pre-GPU ordering."""

from __future__ import annotations

from open_cake_ir.evidence.store import RunLedger

from .environments import AuthoringEnvironment, CandidateSubmission, EnvironmentResult
from .providers import ProviderTurn
from .archive import _candidate_artifact_media_type
from .routing import route_rejection
from .selection import _empirical_filter


def _build_filter_candidates(
    *,
    empirical_enabled: bool,
    environment: AuthoringEnvironment,
    ledger: RunLedger,
    provider_turn: ProviderTurn,
    turn_number: int,
) -> tuple[
    list[tuple[CandidateSubmission, EnvironmentResult]],
    list[int],
    bool,
    list[dict[str, object]],
    dict[str, object] | None,
]:
    built = []
    for payload in provider_turn.candidates:
        entry = CandidateSubmission.seal(environment.media_type, payload)
        built.append((entry, environment.build(entry)))
    launchable_first = [
        index
        for index, (_, result) in enumerate(built)
        if result.disposition == "launchable"
    ]
    # A partial order is not an order over the candidate set. If the
    # model declines any launchable member, moving that unknown behind
    # scored members would let `searches_per_turn` silently reject it as
    # slower. Apply the cost order only when it covers the whole
    # launchable set; otherwise every member keeps provider order.
    cost_order_applied = bool(launchable_first) and all(
        built[index][1].cost is not None
        for index in launchable_first
    )
    if cost_order_applied and not empirical_enabled:
        def complete_cost_order(index: int) -> tuple[tuple, int]:
            cost = built[index][1].cost
            assert cost is not None
            return cost.order, index

        launchable_first.sort(key=complete_cost_order)
    launchable_first.extend(
        index
        for index, (_, result) in enumerate(built)
        if result.disposition != "launchable"
    )
    filter_rows = [
        {
            "candidate_sha256": built[index][0].sha256,
            "disposition": built[index][1].disposition,
            "cost": (
                {
                    "device_fill": round(
                        built[index][1].cost.device_fill, 6
                    ),
                    "binding_resource": built[
                        index
                    ][1].cost.binding_resource,
                }
                if built[index][1].cost is not None
                else None
            ),
            "semantic_sha256": built[index][1].semantic_sha256,
            **({"empirical_cost": (
                dict(built[index][1].empirical_cost)
                if built[index][1].empirical_cost is not None else None
            )} if empirical_enabled else {}),
        }
        for index in (range(len(built)) if empirical_enabled else launchable_first)
    ]
    selection_summary = None
    if empirical_enabled:
        filter_rows, selection_summary = _empirical_filter(filter_rows)
        cost_order_applied = selection_summary["order_applied"]
        by_submission = {entry.sha256: index for index, (entry, _) in enumerate(built)}
        launchable_first = [by_submission[row["candidate_sha256"]] for row in filter_rows]
    ledger.append(
        "candidate_set_filtered",
        {
            "turn": turn_number,
            "submitted": len(built),
            "launchable": sum(
                result.disposition == "launchable"
                for _, result in built
            ),
            "order": filter_rows,
            **({"candidate_selection": selection_summary} if empirical_enabled else {}),
        },
    )
    # A rejected member remains evidence even when another member is
    # launchable. Otherwise the archive would retain only a disposition
    # bit and lose the concrete feedback needed to improve the next set.
    return built, launchable_first, cost_order_applied, filter_rows, selection_summary



def record_candidate_rejections(*, built, evidence, ledger, turn_number):
    """Retain every rejected member, even if another candidate survives."""
    for rejected_submission, rejected_result in built:
        if rejected_result.disposition != "rejected":
            continue
        decision = route_rejection(rejected_result.feedback)
        rejection_payload: dict[str, object] = {
            "turn": turn_number,
            "candidate_sha256": rejected_submission.sha256,
            "feedback": dict(rejected_result.feedback),
            "routed_to": decision.destination,
            "routing_reason": decision.reason,
        }
        if rejected_result.artifact_payloads:
            references = []
            rejected_roles = []
            for role, payload in sorted(
                rejected_result.artifact_payloads.items()
            ):
                try:
                    references.append(
                        evidence.put(
                            payload,
                            media_type=_candidate_artifact_media_type(
                                role.rsplit("_", 1)[-1]
                            ),
                        ).reference(role)
                    )
                except (OSError, ValueError):
                    rejected_roles.append(role)
            if references:
                rejection_payload["objects"] = references
            if rejected_roles:
                rejection_payload["artifact_rejections"] = rejected_roles
        ledger.append("candidate_rejected", rejection_payload)
