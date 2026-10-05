"""Authenticate author history from earlier events and replayed receipts."""
import json

from ..diagnoses import rejected_candidate_feedback
from ..optimization_history import (optimization_history, evaluated_observation,
                                    rejected_observation, action_observation, profile_observation)
from .refusals import refuse


def replay_optimization_history(*, events, evidence, receipts, arm):
    evaluations, rejections, actions = [], [], []
    by_candidate = {}
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
                if state.get('optimization_history') != expected:
                    refuse(f"events[{event['sequence']}].provider_reference_bundle.state_card.optimization_history",
                           'differs from earlier Run-local events and independently validated receipts')
        if kind == 'candidate_evaluated' and payload['purpose'] == 'search':
            key = (payload['turn'], payload['candidate_sha256'])
            row = evaluated_observation(payload['turn'], receipts[(key[0], 'search', key[1])])
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
