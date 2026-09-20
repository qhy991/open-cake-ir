"""Reconstruct action permissions, parent provenance and results from retained inputs.

Compiler rewrites are deterministic domain primitives. Replay does not call the
live writer, build filter or run lifecycle to authenticate its own evidence.
"""
from hashlib import sha256

from .._documents import _canonical_json_bytes
from ..actions import resolve_action_set
from .refusals import refuse, event_location


def replay_actions(*, specification, events, evidence, provider_candidates_by_turn,
                   provider_candidate_bytes, compiler_factory, fault_turn):
    resolutions = {}
    positions = {}
    providers = {}
    for index, event in enumerate(events):
        payload = event.get('payload', {})
        if event.get('kind') == 'provider_turn_completed':
            providers[payload['turn']] = index
        if event.get('kind') != 'author_actions_resolved':
            continue
        turn = payload.get('turn')
        if set(payload) != {'turn', 'actions'} or type(turn) is not int or turn in resolutions:
            refuse(event_location('author_actions_resolved', ordinal=index), 'action resolution record differs')
        resolutions[turn] = payload['actions']
        positions[turn] = index
    if set(resolutions) - set(provider_candidates_by_turn):
        refuse('author_actions_resolved', 'an action resolution has no completed author turn')
    document = specification.document
    baselines = {name: _canonical_json_bytes(program) for name, program in
                 document['reference_inputs'].get('baseline_programs', {}).items()}
    prior = {}
    by_turn, payloads = {}, {}
    for turn, action_ids in sorted(provider_candidates_by_turn.items()):
        location = event_location('author_actions_resolved', turn=turn)
        rows = resolutions.get(turn)
        if rows is None:
            if turn != fault_turn:
                refuse(location, 'completed author turn has no action resolution')
            by_turn[turn] = ()
            continue
        if positions[turn] <= providers[turn] or (turn+1 in providers and positions[turn] >= providers[turn+1]):
            refuse(location, 'action resolution is outside its author turn')
        for index, event in enumerate(events):
            if (event.get('payload', {}).get('turn') == turn
                and event.get('kind') in {'candidate_set_filtered', 'candidate_rejected', 'launchable_candidate_sealed', 'candidate_selected'}
                and index <= positions[turn]):
                refuse(location, 'candidate processing precedes action resolution')
        raw = tuple(provider_candidate_bytes[(turn, identity)] for identity in action_ids)
        derived = resolve_action_set(raw, environment_kind=specification.environment_kind,
            transformations=document['knowledge']['transformations'], candidates=prior,
            baselines=baselines, compiler_factory=compiler_factory,
            allow_python=document["authoring"].get("input_format") == "schedule_or_python_v1")
        if not isinstance(rows, list) or len(rows) != len(derived):
            refuse(location, 'action coverage differs from the author submission')
        current = {}
        for ordinal, (row, result) in enumerate(zip(rows, derived, strict=True)):
            row_location = f'{location}.actions[{ordinal}]'
            expected = {'ordinal': ordinal, **result.document}
            if not isinstance(row, dict) or set(row) != set(expected) | {'objects'}:
                refuse(row_location, 'action result fields differ')
            if {key: row[key] for key in expected} != expected:
                refuse(row_location, 'action result differs from frozen permissions, parent or Compiler rewrite')
            objects = row['objects']
            if result.candidate is None:
                if objects != []:
                    refuse(row_location, 'refused action cannot produce a candidate')
                continue
            identity = sha256(result.candidate).hexdigest()
            if (not isinstance(objects, list) or len(objects) != 1
                or not isinstance(objects[0], dict) or objects[0].get('role') != 'resolved_candidate' or objects[0].get('sha256') != identity
                or evidence.read_object(objects[0]) != result.candidate):
                refuse(row_location, 'resolved candidate bytes differ')
            current[identity] = result.candidate
            payloads[(turn, identity)] = result.candidate
        by_turn[turn] = tuple(current)
        prior.update(current)
    return by_turn, payloads
