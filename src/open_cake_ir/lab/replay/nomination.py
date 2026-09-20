"""Reconstruct the unique terminal nomination from completed search evidence."""
from open_cake_ir.serialization import canonical_json_bytes

from ..nomination import FinalConfirmation, nominate, nomination_document
from ..ralph import RalphBudget, derive_ralph_stop_reason
from ..selection import _receipt_qualifies, _receipt_latency_ms
from .refusals import refuse


def replay_nomination(*, events, observations, launchables, receipts, budget, protocol, terminal_tokens):
    completed = [event['payload'] for event in events if event['kind'] == 'search_completed']
    nominations = [event['payload'] for event in events if event['kind'] == 'candidate_nominated']
    confirmations = [(key, value) for key, value in receipts.items() if key[1] == 'confirmatory']
    faults = [event['payload'] for event in events if event['kind'] == 'run_fault']
    if not completed:
        if nominations or confirmations or not faults:
            refuse('search_completed', 'only a search fault can end without search completion')
        return None, None
    if len(completed) != 1 or len(nominations) != 1 or set(completed[0]) != {'state'}:
        refuse('candidate_nominated', 'a completed search has exactly one closed nomination')
    state = completed[0]['state']
    counts = {name: sum(event['kind']=='evaluation_attempt_started'
                       and event['payload']['purpose']==name for event in
                       events[:next(i for i,e in enumerate(events) if e['kind']=='search_completed')])
              for name in ('search','confirmatory','attribution')}
    if (state.get('evaluation_counts') != counts or counts['confirmatory'] != 0
        or state.get('cumulative_provider_tokens') != terminal_tokens
        or state.get('iteration') != min(budget['maximum_turns']+1,len(observations)+1)):
        refuse('search_completed.state', 'search accounting differs from completed evidence')
    reason = derive_ralph_stop_reason(RalphBudget.from_mapping(budget),turn=state['iteration'],
        cumulative_provider_tokens=terminal_tokens,elapsed_wall_seconds=state['elapsed_wall_seconds'],
        active_authoring_seconds=state['active_authoring_seconds'],evaluation_counts=counts,
        searches_per_turn=protocol.get('searches_per_turn',1),
        profile_each_search_survivor=protocol.get('attribution_evaluation')=='correctness_then_profile_each_search_survivor')
    if reason is None or state.get('terminal_reason') != reason:
        refuse('search_completed.state.terminal_reason', 'search did not stop at its frozen budget')
    nominee = nominate(observations,provider_token_limit=budget['limit'])
    candidate = launchables.get((nominee.turn,nominee.candidate_sha256)) if nominee else None
    expected = nomination_document(nominee,candidate)
    if (nominee and candidate is None) or canonical_json_bytes(nominations[0]) != canonical_json_bytes(expected):
        refuse('candidate_nominated', 'nominee differs from the pre-confirmation search choice')
    if nominee is None:
        if confirmations:
            refuse('candidate_evaluated', 'no search nominee exists for confirmation')
        return None,state
    key = (nominee.turn,'confirmatory',nominee.candidate_sha256)
    terminal_fault = bool(faults and 'source_turn' in faults[0])
    if not confirmations:
        if not terminal_fault:
            refuse('candidate_evaluated', 'the nominee has no independent confirmation or terminal fault')
        return None,state
    if len(confirmations) != 1 or confirmations[0][0] != key:
        refuse('candidate_evaluated', 'confirmation differs from the unique nominated artifact')
    receipt = confirmations[0][1]
    qualified = _receipt_qualifies(receipt)
    return FinalConfirmation(nominee.turn,nominee.candidate_sha256,terminal_tokens,qualified,
                             _receipt_latency_ms(receipt) if qualified else None),state
