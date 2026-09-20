"""Replay logical evaluator invocations without inferring physical device work."""
from __future__ import annotations

from ._documents import _DIGEST
from .selection import _matched_search_plan, _receipt_qualifies
import math
from collections.abc import Mapping


def replay_elapsed_clock(events):
    """One Run clock orders compiler calls, measured receipts and phase boundaries."""
    previous = 0.
    for event in events:
        kind,payload = event['kind'],event['payload']
        if kind in {'candidate_evaluated','compilation_started','compilation_completed'}:
            value = payload.get('elapsed_wall_seconds')
        elif kind == 'search_completed':
            state = payload.get('state')
            value = state.get('elapsed_wall_seconds') if isinstance(state,Mapping) else None
        elif kind == 'checkpoints_projected':
            state = payload.get('ralph')
            value = state.get('elapsed_wall_seconds') if isinstance(state,Mapping) else None
        else:
            continue
        if type(value) not in {int,float} or not math.isfinite(value) or value < previous:
            raise ValueError(f'{kind}: recorded Run clock is not finite and monotone')
        previous = value


def evaluation_origin(payload):
    fields = set(payload) & {'turn', 'source_turn'}
    if len(fields) != 1:
        raise ValueError('Evaluation declares exactly one search turn or terminal source')
    origin = payload[next(iter(fields))]
    if type(origin) is not int or origin <= 0:
        raise ValueError('Evaluation origin must be a positive integer')
    return origin


def replay_evaluation_invocations(events, *, receipts, budget, protocol):
    """Validate sequential start/attempt/receipt transactions and count starts.

    The sole unfinished transaction may precede the final matching evaluation fault.
    A start witnesses Lab invoking its evaluator, not admission or a device dispatch.
    """
    counts = {purpose: 0 for purpose in ("search", "confirmatory", "attribution")}
    starts = set()
    sealed = set()
    filters = {}
    selections = {}
    searches = {}
    completed_receipts = set()
    active = None
    attempt_completed = False
    fault = None
    search_closed = False
    nomination = None
    active_origin_field = None
    for event in events:
        kind, payload = event["kind"], event["payload"]
        if fault is not None:
            if kind not in {"checkpoints_projected", "run_terminal"}:
                raise ValueError("Evaluation history continues after its fault")
            continue
        if kind == "run_fault":
            fault = payload
            if active is not None and (payload.get("stage") != "evaluation" or payload.get(active_origin_field) != active[0]):
                raise ValueError("in-flight Evaluation differs from its terminal fault")
            continue
        if active is not None and kind not in {"evaluation_attempt_completed", "candidate_evaluated"}:
            raise ValueError("Evaluation invocation was interrupted without a fault")
        if search_closed and kind not in {'candidate_nominated', 'evaluation_attempt_started',
                'evaluation_attempt_completed', 'candidate_evaluated', 'checkpoints_projected', 'run_terminal'}:
            raise ValueError('search activity continued after its terminal boundary')
        if kind == 'search_completed':
            search_closed = True
        elif kind == 'candidate_nominated':
            if not search_closed or nomination is not None:
                raise ValueError('nomination must occur exactly once after search')
            nomination = payload
        elif kind == "launchable_candidate_sealed":
            sealed.add((payload["turn"], payload["candidate_sha256"]))
        elif kind == "candidate_set_filtered":
            filters[payload["turn"]] = payload
        elif kind == "candidate_selected":
            selections[payload["turn"]] = payload
        elif kind in {"evaluation_attempt_started", "evaluation_attempt_completed", "candidate_evaluated"}:
            turn, purpose, candidate = evaluation_origin(payload), payload.get("purpose"), payload.get("candidate_sha256")
            origin_field = "source_turn" if "source_turn" in payload else "turn"
            if (type(turn) is not int or turn <= 0 or purpose not in counts
                    or not isinstance(candidate, str) or _DIGEST.fullmatch(candidate) is None):
                raise ValueError("Evaluation invocation identity differs")
            key = (turn, purpose, candidate)
            if kind == "evaluation_attempt_started":
                if set(payload) != {origin_field, "purpose", "candidate_sha256"} or key in starts or (turn, candidate) not in sealed:
                    raise ValueError("Evaluation start is duplicated, unsealed or malformed")
                if 'source_turn' in payload:
                    if (nomination is None or nomination['source_turn'] != turn
                        or nomination['candidate_sha256'] != candidate or purpose == 'search'):
                        raise ValueError('terminal Evaluation differs from the fixed nomination')
                elif search_closed or purpose == 'confirmatory':
                    raise ValueError('confirmation needs a terminal nomination; search cannot resume')
                if purpose == "search":
                    if turn not in filters:
                        raise ValueError("Evaluation search lacks its candidate filter")
                    plan, _ = _matched_search_plan(filters[turn]["order"], protocol.get("searches_per_turn", 1))
                    prior = searches.setdefault(turn, [])
                    if prior + [candidate] != plan[:len(prior) + 1]:
                        raise ValueError("Evaluation start differs from the search plan")
                    prior.append(candidate)
                elif purpose == "confirmatory":
                    if counts['confirmatory'] != 0:
                        raise ValueError('a Run confirms exactly one terminal nominee')
                    selected = selections.get(turn, {})
                    if (selected.get("candidate_sha256") != candidate
                            or candidate not in selected.get("qualified_search_candidates", [])
                            or (turn, "search", candidate) not in completed_receipts):
                        raise ValueError("Evaluation confirmation lacks the selected qualified search")
                else:
                    search = receipts.get((turn, "search", candidate))
                    attribution = protocol.get("attribution_evaluation")
                    if (attribution == "correctness_then_profile_each_search_survivor"
                            and (turn, "search", candidate) in completed_receipts
                            and search is not None and search.correctness_passed):
                        pass
                    elif (attribution == "correctness_then_profile"
                            and (turn, "confirmatory", candidate) in completed_receipts
                            and _receipt_qualifies(receipts[(turn, "confirmatory", candidate)])):
                        pass
                    else:
                        raise ValueError("Evaluation attribution lacks its required observation")
                counts[purpose] += 1
                if counts[purpose] > budget["evaluation_limits"][purpose]:
                    raise ValueError("Evaluation invocation exceeds its budget")
                starts.add(key)
                active, attempt_completed = key, False
                active_origin_field = origin_field
            elif kind == "evaluation_attempt_completed":
                if active != key or origin_field != active_origin_field or attempt_completed:
                    raise ValueError("Evaluation attempt completion lacks its unique start")
                attempt_completed = True
            else:
                if active != key or origin_field != active_origin_field or not attempt_completed or key not in receipts:
                    raise ValueError("Evaluation receipt lacks an ordered completed attempt")
                completed_receipts.add(key)
                active = None
    if active is not None and fault is None:
        raise ValueError("Evaluation invocation has no completion or terminal fault")
    if completed_receipts != set(receipts):
        raise ValueError("Evaluation receipt and invocation sets differ")
    return counts
