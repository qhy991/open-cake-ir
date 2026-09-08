"""Independently check search, confirmation and candidate-selection event order."""

from __future__ import annotations

import json, math
from typing import Mapping, Sequence, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate

from ._documents import _DIGEST, _canonical_json_bytes, _object
from ._policies import _ATTRIBUTION_EVALUATION, _LEGACY_ATTRIBUTION_EVALUATION
from .checkpoints import TurnObservation
from .contracts import CampaignLock
from .selection import (
    _EmpiricalSelection,
    _empirical_filter,
    _receipt_latency_ms,
    _receipt_qualifies,
)
from .replay_outcomes import _expected_matched_diagnoses_v1, _validate_matched_diagnoses_v1


def _replay_candidate_selection(
    *,
    candidate_set_turns: set[int],
    cumulative_by_turn: Mapping[int, int],
    empirical_selection: _EmpiricalSelection | None,
    events: Sequence[Mapping[str, object]],
    fault_turn: int | None,
    launchables: Mapping[tuple[int, str], LaunchableCandidate],
    lock: CampaignLock,
    provider_candidate_bytes: Mapping[tuple[int, str], bytes],
    provider_candidates_by_turn: Mapping[int, tuple[str, ...]],
    receipt_order: Sequence[tuple[int, str, str]],
    receipts: Mapping[tuple[int, str, str], EvaluationReceipt],
    rejected: Mapping[tuple[int, str], Mapping[str, object]],
) -> tuple[list[TurnObservation], int, str | None] | None:
    filter_events = [
        event for event in events if event.get("kind") == "candidate_set_filtered"
    ]
    filters: dict[int, Mapping[str, object]] = {}
    filter_order: dict[int, tuple[str, ...]] = {}
    filter_disposition: dict[int, dict[str, str]] = {}
    for event in filter_events:
        payload = _object(event.get("payload"), "candidate_set_filtered.payload")
        turn = payload.get("turn")
        order = payload.get("order")
        if (
            set(payload) != {"turn", "submitted", "launchable", "order"} | (
                {"candidate_selection"} if empirical_selection is not None else set()
            )
            or not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn in filters
            or turn not in candidate_set_turns
            or not isinstance(order, list)
        ):
            return None
        candidates: list[str] = []
        dispositions: dict[str, str] = {}
        for row in order:
            expected_row_fields = {
                "candidate_sha256",
                "disposition",
                "cost",
            }
            expected_row_fields.add("semantic_sha256")
            if empirical_selection is not None:
                expected_row_fields.add("empirical_cost")
            if not isinstance(row, Mapping) or set(row) != expected_row_fields:
                return None
            candidate_sha256 = row.get("candidate_sha256")
            disposition = row.get("disposition")
            cost = row.get("cost")
            semantic_sha256 = row.get("semantic_sha256")
            if (
                not isinstance(candidate_sha256, str)
                or _DIGEST.fullmatch(candidate_sha256) is None
                or candidate_sha256 in dispositions
                or disposition not in {"launchable", "rejected"}
                or (
                    cost is not None
                    and (
                        not isinstance(cost, Mapping)
                        or set(cost) != {"device_fill", "binding_resource"}
                        or not isinstance(cost.get("device_fill"), (int, float))
                        or isinstance(cost.get("device_fill"), bool)
                        or not math.isfinite(float(cost["device_fill"]))
                        or not 0 < float(cost["device_fill"]) <= 1
                        or not isinstance(cost.get("binding_resource"), str)
                        or not cost.get("binding_resource")
                    )
                )
                or (
                    semantic_sha256 is not None
                    and (
                        not isinstance(semantic_sha256, str)
                        or _DIGEST.fullmatch(semantic_sha256) is None
                    )
                )
            ):
                return None
            candidates.append(candidate_sha256)
            dispositions[candidate_sha256] = cast(str, disposition)
        provider_candidates = provider_candidates_by_turn.get(turn)
        if (
            provider_candidates is None
            or len(candidates) != len(provider_candidates)
            or set(candidates) != set(provider_candidates)
            or payload.get("submitted") != len(provider_candidates)
            or payload.get("launchable")
            != sum(value == "launchable" for value in dispositions.values())
        ):
            return None
        if empirical_selection is not None:
            rows_by_candidate = {row["candidate_sha256"]: row for row in order}
            expected_rows = []
            for candidate_sha256 in provider_candidates:
                retained = rows_by_candidate[candidate_sha256]
                expected = dict(retained)
                expected["empirical_cost"] = (
                    empirical_selection.estimate(json.loads(provider_candidate_bytes[(turn, candidate_sha256)]))
                    if retained["disposition"] == "launchable" else None
                )
                expected_rows.append(expected)
            expected_rows, expected_selection = _empirical_filter(expected_rows)
            if (
                _canonical_json_bytes(order) != _canonical_json_bytes(expected_rows)
                or _canonical_json_bytes(payload["candidate_selection"])
                != _canonical_json_bytes(expected_selection)
            ):
                return None
        filters[turn] = payload
        filter_order[turn] = tuple(candidates)
        filter_disposition[turn] = dispositions

    expected_rejections = {
        (turn, candidate_sha256)
        for turn, dispositions in filter_disposition.items()
        for candidate_sha256, disposition in dispositions.items()
        if disposition == "rejected"
    }
    if set(rejected) != expected_rejections:
        return None

    selection_events = [
        event for event in events if event.get("kind") == "candidate_selected"
    ]
    selections: dict[int, Mapping[str, object]] = {}
    for event in selection_events:
        payload = _object(event.get("payload"), "candidate_selected.payload")
        turn = payload.get("turn")
        candidate_sha256 = payload.get("candidate_sha256")
        qualified = payload.get("qualified_search_candidates")
        if (
            set(payload)
            != {
                "turn",
                "candidate_sha256",
                "qualified_search_candidates",
                "reason",
            }
            or not isinstance(turn, int)
            or isinstance(turn, bool)
            or turn in selections
            or turn not in candidate_set_turns
            or not isinstance(candidate_sha256, str)
            or _DIGEST.fullmatch(candidate_sha256) is None
            or not isinstance(qualified, list)
            or any(
                not isinstance(value, str) or _DIGEST.fullmatch(value) is None
                for value in qualified
            )
        ):
            return None
        selections[turn] = payload

    observations: list[TurnObservation] = []
    evaluation_protocol = _object(
        lock.document["evaluation_protocol"], "evaluation_protocol"
    )
    searches_per_turn = int(
        evaluation_protocol.get("searches_per_turn", 1)
    )
    attribution_evaluation = evaluation_protocol.get("attribution_evaluation")
    expected_searches: dict[int, list[str]] = {}
    expected_diagnoses, expected_searches = _expected_matched_diagnoses_v1(
        filters=filters,
        receipts=receipts,
        receipt_order=receipt_order,
        searches_per_turn=searches_per_turn,
        materiality_ratio=float(
            evaluation_protocol.get("search_materiality_ratio", math.inf)
        ),
    )
    _validate_matched_diagnoses_v1(
        events,
        expected=expected_diagnoses,
        fault_turn=fault_turn,
    )
    for turn, provider_candidates in sorted(provider_candidates_by_turn.items()):
        if turn in candidate_set_turns:
            if turn not in filters:
                if turn != fault_turn:
                    return None
                continue
            selection = selections.get(turn)
            if selection is None:
                if turn != fault_turn:
                    return None
                continue
            selected = cast(str, selection["candidate_sha256"])
            order = filter_order[turn]
            dispositions = filter_disposition[turn]
            if selected not in provider_candidates:
                return None
            search_keys = [
                key
                for key in receipt_order
                if key[0] == turn and key[1] == "search"
            ]
            if selection.get("reason") == "all_candidates_rejected":
                if (
                    selected != order[0]
                    or any(value == "launchable" for value in dispositions.values())
                    or search_keys
                    or selection.get("qualified_search_candidates") != []
                    or (turn, selected) not in rejected
                ):
                    return None
                if turn != fault_turn:
                    observations.append(
                        TurnObservation(
                            turn,
                            cumulative_by_turn[turn],
                            selected,
                            False,
                            None,
                        )
                    )
                continue
            if not search_keys or len(search_keys) > searches_per_turn:
                return None
            searched_candidates = [key[2] for key in search_keys]
            if (
                len(set(searched_candidates)) != len(searched_candidates)
                or any(dispositions.get(value) != "launchable" for value in searched_candidates)
                or [order.index(value) for value in searched_candidates]
                != sorted(order.index(value) for value in searched_candidates)
                or (
                    searched_candidates
                    != (
                        expected_searches[turn][
                            : len(searched_candidates)
                        ]
                        if turn == fault_turn
                        else expected_searches[turn]
                    )
                )
            ):
                return None
            qualified_search = [
                key[2] for key in search_keys if _receipt_qualifies(receipts[key])
            ]
            expected_selected = (
                min(
                    qualified_search,
                    key=lambda value: _receipt_latency_ms(
                        receipts[(turn, "search", value)]
                    )
                    or float("inf"),
                )
                if qualified_search
                else searched_candidates[0]
            )
            expected_reason = (
                "lowest_qualified_search_latency"
                if qualified_search
                else "no_qualified_search_candidate"
            )
            if (
                selection.get("qualified_search_candidates") != qualified_search
                or selected != expected_selected
                or selection.get("reason") != expected_reason
            ):
                return None
            confirms = [
                receipt
                for (candidate_turn, purpose, candidate), receipt in receipts.items()
                if candidate_turn == turn
                and purpose == "confirmatory"
                and candidate == selected
            ]
            foreign_confirms = [
                key
                for key in receipts
                if key[0] == turn
                and key[1] == "confirmatory"
                and key[2] != selected
            ]
            if foreign_confirms or (not qualified_search and confirms):
                return None
            if turn == fault_turn:
                continue
            if len(confirms) != (1 if qualified_search else 0):
                return None
            confirmed = confirms[0] if confirms else None
            qualified = confirmed is not None and _receipt_qualifies(confirmed)
            attributions = [
                key
                for key in receipts
                if key[0] == turn and key[1] == "attribution"
            ]
            expected_attributions = (
                [
                    candidate
                    for candidate in searched_candidates
                    if receipts[(turn, "search", candidate)].correctness_passed
                ]
                if attribution_evaluation == _ATTRIBUTION_EVALUATION
                else (
                    [selected]
                    if qualified
                    and attribution_evaluation
                    == _LEGACY_ATTRIBUTION_EVALUATION
                    else []
                )
            )
            if (
                len(attributions) != len(expected_attributions)
                or {key[2] for key in attributions}
                != set(expected_attributions)
            ):
                return None
            observations.append(
                TurnObservation(
                    turn,
                    cumulative_by_turn[turn],
                    selected,
                    qualified,
                    _receipt_latency_ms(confirmed) if qualified else None,
                )
            )
        else:
            # Historical evidence wrote exactly one candidate per Turn and had no
            # explicit filter/selection events. Keep that bounded spelling readable;
            # new evidence must use the candidate-set contract above.
            if len(provider_candidates) != 1:
                return None
            selected = provider_candidates[0]
            if turn == fault_turn:
                continue
            has_rejection = (turn, selected) in rejected
            has_evaluation = any(
                key[0] == turn and key[2] == selected for key in receipts
            )
            if has_rejection == has_evaluation:
                return None
            confirmed = receipts.get((turn, "confirmatory", selected))
            qualified = confirmed is not None and _receipt_qualifies(confirmed)
            observations.append(
                TurnObservation(
                    turn,
                    cumulative_by_turn[turn],
                    selected,
                    qualified,
                    _receipt_latency_ms(confirmed) if qualified else None,
                )
            )

    if set(filters) != candidate_set_turns - ({fault_turn} if fault_turn else set()):
        # A fault may happen after its filter was written, so the final Turn is the
        # sole allowed extra member on either side of this equality.
        if not (
            fault_turn in candidate_set_turns
            and set(filters) | {fault_turn} == candidate_set_turns
        ):
            return None
    if any(turn not in filters for turn in selections):
        return None
    for turn, candidate_sha256 in rejected:
        if turn in candidate_set_turns:
            # Rejection is a property of each set member, not of the Turn's
            # eventual selection.  A mixed set legitimately records rejected
            # members while selecting a different launchable member.  The exact
            # rejected-member set was checked against the filter dispositions
            # above; the all-rejected selection rule is checked in the selection
            # replay branch.
            if (
                filter_disposition.get(turn, {}).get(candidate_sha256)
                != "rejected"
            ):
                return None

    for turn, candidate_sha256 in launchables:
        if turn in candidate_set_turns and filter_disposition.get(turn, {}).get(
            candidate_sha256
        ) != "launchable":
            return None
    return observations, searches_per_turn, attribution_evaluation
