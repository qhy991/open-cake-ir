"""Independently check search, confirmation and candidate-selection event order."""

from __future__ import annotations

import json, math
from typing import Mapping, Sequence, cast

from open_cake_ir.evaluation import EvaluationReceipt, LaunchableCandidate

from .._documents import _DIGEST, _canonical_json_bytes, _object
from .._policies import _ATTRIBUTION_EVALUATION, _LEGACY_ATTRIBUTION_EVALUATION
from ..checkpoints import TurnObservation
from ..contracts import CampaignLock
from ..selection import (
    _EmpiricalSelection,
    _empirical_filter,
    _receipt_latency_ms,
    _receipt_qualifies,
)
from .outcomes import _expected_matched_diagnoses_v1, _validate_matched_diagnoses_v1
from .refusals import event_location, refuse


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
) -> tuple[list[TurnObservation], int, str | None]:
    filter_events = [
        event for event in events if event.get("kind") == "candidate_set_filtered"
    ]
    filters: dict[int, Mapping[str, object]] = {}
    filter_order: dict[int, tuple[str, ...]] = {}
    filter_disposition: dict[int, dict[str, str]] = {}
    for ordinal, event in enumerate(filter_events):
        location = event_location("candidate_set_filtered", ordinal=ordinal)
        payload = _object(event.get("payload"), "candidate_set_filtered.payload")
        turn = payload.get("turn")
        order = payload.get("order")
        expected_fields = {"turn", "submitted", "launchable", "order"} | (
            {"candidate_selection"} if empirical_selection is not None else set()
        )
        if set(payload) != expected_fields:
            refuse(f"{location}.payload", "fields differ", observed=set(payload),
                   expected=expected_fields)
        if not isinstance(turn, int) or isinstance(turn, bool):
            refuse(f"{location}.payload.turn", "not an integer", observed=turn)
        if turn in filters:
            refuse(f"{location}.payload.turn", "a second filter for one Turn", observed=turn)
        if turn not in candidate_set_turns:
            refuse(f"{location}.payload.turn", "no candidate set was submitted in this Turn",
                   observed=turn, expected=candidate_set_turns)
        location = event_location("candidate_set_filtered", turn=turn)
        if not isinstance(order, list):
            refuse(f"{location}.payload.order", "not a list", observed=type(order).__name__)
        candidates: list[str] = []
        dispositions: dict[str, str] = {}
        for index, row in enumerate(order):
            row_location = f"{location}.payload.order[{index}]"
            expected_row_fields = {
                "candidate_sha256",
                "disposition",
                "cost",
            }
            expected_row_fields.add("semantic_sha256")
            if empirical_selection is not None:
                expected_row_fields.add("empirical_cost")
            if not isinstance(row, Mapping) or set(row) != expected_row_fields:
                refuse(row_location, "row fields differ",
                       observed=set(row) if isinstance(row, Mapping) else type(row).__name__,
                       expected=expected_row_fields)
            candidate_sha256 = row.get("candidate_sha256")
            disposition = row.get("disposition")
            cost = row.get("cost")
            semantic_sha256 = row.get("semantic_sha256")
            if not isinstance(candidate_sha256, str) or _DIGEST.fullmatch(candidate_sha256) is None:
                refuse(f"{row_location}.candidate_sha256", "not a SHA256 digest", observed=candidate_sha256)
            if candidate_sha256 in dispositions:
                refuse(f"{row_location}.candidate_sha256", "a second row for one candidate",
                       observed=candidate_sha256)
            if disposition not in {"launchable", "rejected"}:
                refuse(f"{row_location}.disposition", "not a disposition", observed=disposition,
                       expected={"launchable", "rejected"})
            if cost is not None and (
                not isinstance(cost, Mapping)
                or set(cost) != {"device_fill", "binding_resource"}
                or not isinstance(cost.get("device_fill"), (int, float))
                or isinstance(cost.get("device_fill"), bool)
                or not math.isfinite(float(cost["device_fill"]))
                or not 0 < float(cost["device_fill"]) <= 1
                or not isinstance(cost.get("binding_resource"), str)
                or not cost.get("binding_resource")
            ):
                refuse(f"{row_location}.cost",
                       "not a cost record with device_fill in (0, 1] and a named binding resource",
                       observed=cost)
            if semantic_sha256 is not None and (
                not isinstance(semantic_sha256, str)
                or _DIGEST.fullmatch(semantic_sha256) is None
            ):
                refuse(f"{row_location}.semantic_sha256", "not a SHA256 digest", observed=semantic_sha256)
            candidates.append(candidate_sha256)
            dispositions[candidate_sha256] = cast(str, disposition)
        provider_candidates = provider_candidates_by_turn.get(turn)
        if provider_candidates is None:
            refuse(f"{location}.payload.turn", "no provider Turn completed for this filter", observed=turn)
        if len(candidates) != len(provider_candidates) or set(candidates) != set(provider_candidates):
            refuse(f"{location}.payload.order", "rows differ from the candidates the provider submitted",
                   observed=candidates, expected=provider_candidates)
        if payload.get("submitted") != len(provider_candidates):
            refuse(f"{location}.payload.submitted", "differs from the submitted candidate count",
                   observed=payload.get("submitted"), expected=len(provider_candidates))
        expected_launchable = sum(value == "launchable" for value in dispositions.values())
        if payload.get("launchable") != expected_launchable:
            refuse(f"{location}.payload.launchable", "differs from the launchable rows",
                   observed=payload.get("launchable"), expected=expected_launchable)
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
            if _canonical_json_bytes(order) != _canonical_json_bytes(expected_rows):
                refuse(f"{location}.payload.order",
                       "differs from the empirical order rederived from the retained candidates",
                       observed=order, expected=expected_rows)
            if _canonical_json_bytes(payload["candidate_selection"]) != _canonical_json_bytes(expected_selection):
                refuse(f"{location}.payload.candidate_selection",
                       "differs from the empirical selection rederived from the retained candidates",
                       observed=payload["candidate_selection"], expected=expected_selection)
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
        refuse("candidate_rejected", "rejection events differ from the filters' rejected rows",
               observed={f"turn={turn},candidate={candidate}" for turn, candidate in rejected},
               expected={f"turn={turn},candidate={candidate}" for turn, candidate in expected_rejections})

    selection_events = [
        event for event in events if event.get("kind") == "candidate_selected"
    ]
    selections: dict[int, Mapping[str, object]] = {}
    for ordinal, event in enumerate(selection_events):
        location = event_location("candidate_selected", ordinal=ordinal)
        payload = _object(event.get("payload"), "candidate_selected.payload")
        turn = payload.get("turn")
        candidate_sha256 = payload.get("candidate_sha256")
        qualified = payload.get("qualified_search_candidates")
        expected_fields = {"turn", "candidate_sha256", "qualified_search_candidates", "reason"}
        if set(payload) != expected_fields:
            refuse(f"{location}.payload", "fields differ", observed=set(payload), expected=expected_fields)
        if not isinstance(turn, int) or isinstance(turn, bool):
            refuse(f"{location}.payload.turn", "not an integer", observed=turn)
        if turn in selections:
            refuse(f"{location}.payload.turn", "a second selection for one Turn", observed=turn)
        if turn not in candidate_set_turns:
            refuse(f"{location}.payload.turn", "no candidate set was submitted in this Turn",
                   observed=turn, expected=candidate_set_turns)
        location = event_location("candidate_selected", turn=turn)
        if not isinstance(candidate_sha256, str) or _DIGEST.fullmatch(candidate_sha256) is None:
            refuse(f"{location}.payload.candidate_sha256", "not a SHA256 digest", observed=candidate_sha256)
        if not isinstance(qualified, list) or any(
            not isinstance(value, str) or _DIGEST.fullmatch(value) is None
            for value in qualified
        ):
            refuse(f"{location}.payload.qualified_search_candidates", "not a list of SHA256 digests",
                   observed=qualified)
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
                    refuse(event_location("candidate_set_filtered", turn=turn),
                           "no filter for a Turn that did not fault", expected=fault_turn)
                continue
            selection = selections.get(turn)
            location = event_location("candidate_selected", turn=turn)
            if selection is None:
                if turn != fault_turn:
                    refuse(location, "no selection for a Turn that did not fault", expected=fault_turn)
                continue
            selected = cast(str, selection["candidate_sha256"])
            order = filter_order[turn]
            dispositions = filter_disposition[turn]
            if selected not in provider_candidates:
                refuse(f"{location}.payload.candidate_sha256",
                       "the selected candidate is not one the provider submitted in that Turn",
                       observed=selected, expected=provider_candidates)
            search_keys = [
                key
                for key in receipt_order
                if key[0] == turn and key[1] == "search"
            ]
            if selection.get("reason") == "all_candidates_rejected":
                if selected != order[0]:
                    refuse(f"{location}.payload.candidate_sha256",
                           "an all-rejected Turn selects the first filtered row",
                           observed=selected, expected=order[0])
                if any(value == "launchable" for value in dispositions.values()):
                    refuse(f"{location}.payload.reason", "all_candidates_rejected although a row is launchable",
                           observed=dispositions)
                if search_keys:
                    refuse(f"{location}.payload.reason", "all_candidates_rejected although searches were evaluated",
                           observed=[key[2] for key in search_keys])
                if selection.get("qualified_search_candidates") != []:
                    refuse(f"{location}.payload.qualified_search_candidates",
                           "an all-rejected Turn qualifies no candidate",
                           observed=selection.get("qualified_search_candidates"), expected=[])
                if (turn, selected) not in rejected:
                    refuse(f"{location}.payload.candidate_sha256", "the selected candidate has no rejection event",
                           observed=selected)
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
                refuse(event_location("candidate_evaluated", turn=turn, purpose="search"),
                       "search evaluation count is outside 1..searches_per_turn",
                       observed=len(search_keys), expected=f"1..{searches_per_turn}")
            searched_candidates = [key[2] for key in search_keys]
            search_location = event_location("candidate_evaluated", turn=turn, purpose="search")
            if len(set(searched_candidates)) != len(searched_candidates):
                refuse(search_location, "a candidate was searched twice", observed=searched_candidates)
            if any(dispositions.get(value) != "launchable" for value in searched_candidates):
                refuse(search_location, "a searched candidate is not a launchable row of the filter",
                       observed={value: dispositions.get(value) for value in searched_candidates})
            if [order.index(value) for value in searched_candidates] != sorted(
                order.index(value) for value in searched_candidates
            ):
                refuse(search_location, "searches are not in filter order", observed=searched_candidates,
                       expected=[value for value in order if value in searched_candidates])
            expected_searched = (
                expected_searches[turn][: len(searched_candidates)]
                if turn == fault_turn
                else expected_searches[turn]
            )
            if searched_candidates != expected_searched:
                refuse(search_location, "searched candidates differ from the rederived search plan",
                       observed=searched_candidates, expected=expected_searched)
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
            if selection.get("qualified_search_candidates") != qualified_search:
                refuse(f"{location}.payload.qualified_search_candidates",
                       "differs from the qualification rederived from the search receipts",
                       observed=selection.get("qualified_search_candidates"), expected=qualified_search)
            if selected != expected_selected:
                refuse(f"{location}.payload.candidate_sha256",
                       "differs from the selection rederived from the search receipts",
                       observed=selected, expected=expected_selected)
            if selection.get("reason") != expected_reason:
                refuse(f"{location}.payload.reason", "differs from the rederived selection reason",
                       observed=selection.get("reason"), expected=expected_reason)
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
            confirm_location = event_location("candidate_evaluated", turn=turn, purpose="confirmatory")
            if foreign_confirms:
                refuse(confirm_location, "a confirmatory evaluation of a candidate that was not selected",
                       observed=[key[2] for key in foreign_confirms], expected=selected)
            if not qualified_search and confirms:
                refuse(confirm_location, "a confirmatory evaluation although no search candidate qualified",
                       observed=len(confirms), expected=0)
            if turn == fault_turn:
                continue
            if len(confirms) != (1 if qualified_search else 0):
                refuse(confirm_location, "confirmatory evaluation count differs",
                       observed=len(confirms), expected=1 if qualified_search else 0)
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
                refuse(event_location("candidate_evaluated", turn=turn, purpose="attribution"),
                       "attribution evaluations differ from the protocol's expectation",
                       observed=[key[2] for key in attributions], expected=expected_attributions)
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
                refuse(event_location("provider_turn_completed", turn=turn),
                       "a Turn without a candidate set submits exactly one candidate",
                       observed=len(provider_candidates), expected=1)
            selected = provider_candidates[0]
            if turn == fault_turn:
                continue
            has_rejection = (turn, selected) in rejected
            has_evaluation = any(
                key[0] == turn and key[2] == selected for key in receipts
            )
            if has_rejection == has_evaluation:
                refuse(event_location("provider_turn_completed", turn=turn),
                       "the Turn's candidate was neither rejected nor evaluated, or both",
                       observed={"rejected": has_rejection, "evaluated": has_evaluation})
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
            refuse("candidate_set_filtered", "filtered Turns differ from the Turns with a candidate set",
                   observed=set(filters), expected=candidate_set_turns)
    unfiltered_selections = sorted(turn for turn in selections if turn not in filters)
    if unfiltered_selections:
        refuse("candidate_selected", "a selection in a Turn that was not filtered",
               observed=unfiltered_selections, expected=set(filters))
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
                refuse(event_location("candidate_rejected", turn=turn, candidate=candidate_sha256),
                       "the filter does not dispose this candidate as rejected",
                       observed=filter_disposition.get(turn, {}).get(candidate_sha256), expected="rejected")

    for turn, candidate_sha256 in launchables:
        if turn in candidate_set_turns and filter_disposition.get(turn, {}).get(
            candidate_sha256
        ) != "launchable":
            refuse(event_location("launchable_candidate_sealed", turn=turn, candidate=candidate_sha256),
                   "the filter does not dispose this candidate as launchable",
                   observed=filter_disposition.get(turn, {}).get(candidate_sha256), expected="launchable")
    return observations, searches_per_turn, attribution_evaluation
