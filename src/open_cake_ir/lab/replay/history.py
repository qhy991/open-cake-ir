"""Authenticate author history from earlier events and replayed receipts."""
import json

from .._documents import _canonical_json_bytes, _object
from ..diagnoses import rejected_candidate_feedback
from ..optimization_history import (optimization_history, evaluated_observation,
                                    rejected_observation, action_observation, profile_observation)
from .refusals import refuse


def _check_history(state, expected, location):
    state = _object(state, location)
    try:
        matches = _canonical_json_bytes(state.get('optimization_history')) == _canonical_json_bytes(expected)
    except (TypeError, ValueError):
        matches = False
    if not matches:
        refuse(location + '.optimization_history',
               'differs from earlier Run-local events and independently validated receipts')


def replay_optimization_history(*, events, evidence, receipts, arm, specification=None):
    evaluations, rejections, actions = [], [], []
    by_candidate = {}
    diagnostics_by_candidate = {}
    provider = (specification.document['authoring'].get('provider', {})
                if specification is not None else {})
    for event in events:
        payload = event.get('payload', {})
        kind = event['kind']
        # The reference bundle was sent before this provider event. Its history
        # must therefore use only facts preceding this position in the ledger.
        if kind in {'provider_turn_completed', 'run_fault'}:
            refs = [ref for ref in payload.get('objects', [])
                    if ref.get('role') == 'provider_reference_bundle']
            for ref in refs:
                state = json.loads(evidence.read_object(ref))['state_card']
                expected = optimization_history(evaluations, rejections, actions)
                _check_history(state, expected,
                    f"events[{event['sequence']}].provider_reference_bundle.state_card")
        if (kind == 'run_fault' and payload.get('stage') == 'provider'
                and provider.get('event_contract') == 'responses_messages_v1'):
            # Use the same failed-message request owner as feedback replay. The
            # terminal fault can retain a sent bundle without a completed Turn.
            refs = [ref for ref in payload.get('objects', []) if ref.get('role') == 'provider_stdout']
            if refs:
                from ..message_provider import _json
                location = 'run_fault.payload.objects.provider_stdout.request'
                try:
                    record = _json(evidence.read_object(refs[0]))
                    bundle = _object(_json(record['request']['input'][-1]['content']), location)
                except (UnicodeError, ValueError, KeyError, IndexError, TypeError):
                    refuse(location, 'failed message request lacks its task bundle')
                _check_history(bundle.get('state_card'),
                    optimization_history(evaluations, rejections, actions),
                    location + '.reference_bundle.state_card')
        if kind == 'candidate_set_filtered':
            diagnostics_by_candidate.update({(payload['turn'], row['candidate_sha256']): row['diagnostics']
                for row in payload['order']})
        elif kind == 'candidate_evaluated' and payload['purpose'] == 'search':
            key = (payload['turn'], payload['candidate_sha256'])
            if key not in diagnostics_by_candidate:
                refuse(f"events[{event['sequence']}].candidate_evaluated",
                       'search history lacks preceding candidate-bound filter diagnostics')
            row = evaluated_observation(payload['turn'], receipts[(key[0], 'search', key[1])],
                                        diagnostics=diagnostics_by_candidate[key])
            evaluations.append(row)
            by_candidate[key] = row
        elif (kind == 'candidate_evaluated' and payload['purpose'] == 'attribution'
              and 'turn' in payload):
            key = (payload['turn'], payload['candidate_sha256'])
            if key in by_candidate:
                by_candidate[key]['profile'] = profile_observation(receipts[(key[0], 'attribution', key[1])])
        elif kind == 'candidate_rejected':
            row = rejected_candidate_feedback(payload['candidate_sha256'], payload['feedback'], arm=arm)
            rejections.append(rejected_observation(payload['turn'], row))
        elif kind == 'author_actions_resolved':
            actions.extend(row for action in payload['actions']
                           if (row := action_observation(payload['turn'], action)) is not None)
