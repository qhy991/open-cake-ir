"""Replay logical evaluator invocations without inferring physical device work."""
from __future__ import annotations

from ._documents import _DIGEST
from .selection import _matched_search_plan, _receipt_qualifies


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
    for event in events:
        kind, payload = event["kind"], event["payload"]
        if fault is not None:
            if kind not in {"checkpoints_projected", "run_terminal"}:
                raise ValueError("Evaluation history continues after its fault")
            continue
        if kind == "run_fault":
            fault = payload
            if active is not None and (payload.get("stage") != "evaluation" or payload.get("turn") != active[0]):
                raise ValueError("in-flight Evaluation differs from its terminal fault")
            continue
        if active is not None and kind not in {"evaluation_attempt_completed", "candidate_evaluated"}:
            raise ValueError("Evaluation invocation was interrupted without a fault")
        if kind == "launchable_candidate_sealed":
            sealed.add((payload["turn"], payload["candidate_sha256"]))
        elif kind == "candidate_set_filtered":
            filters[payload["turn"]] = payload
        elif kind == "candidate_selected":
            selections[payload["turn"]] = payload
        elif kind in {"evaluation_attempt_started", "evaluation_attempt_completed", "candidate_evaluated"}:
            turn, purpose, candidate = payload.get("turn"), payload.get("purpose"), payload.get("candidate_sha256")
            if (type(turn) is not int or turn <= 0 or purpose not in counts
                    or not isinstance(candidate, str) or _DIGEST.fullmatch(candidate) is None):
                raise ValueError("Evaluation invocation identity differs")
            key = (turn, purpose, candidate)
            if kind == "evaluation_attempt_started":
                if set(payload) != {"turn", "purpose", "candidate_sha256"} or key in starts or (turn, candidate) not in sealed:
                    raise ValueError("Evaluation start is duplicated, unsealed or malformed")
                if purpose == "search":
                    if turn not in filters:
                        raise ValueError("Evaluation search lacks its candidate filter")
                    plan, _ = _matched_search_plan(filters[turn]["order"], protocol.get("searches_per_turn", 1))
                    prior = searches.setdefault(turn, [])
                    if prior + [candidate] != plan[:len(prior) + 1]:
                        raise ValueError("Evaluation start differs from the search plan")
                    prior.append(candidate)
                elif purpose == "confirmatory":
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
            elif kind == "evaluation_attempt_completed":
                if active != key or attempt_completed:
                    raise ValueError("Evaluation attempt completion lacks its unique start")
                attempt_completed = True
            else:
                if active != key or not attempt_completed or key not in receipts:
                    raise ValueError("Evaluation receipt lacks an ordered completed attempt")
                completed_receipts.add(key)
                active = None
    if active is not None and fault is None:
        raise ValueError("Evaluation invocation has no completion or terminal fault")
    if completed_receipts != set(receipts):
        raise ValueError("Evaluation receipt and invocation sets differ")
    return counts
