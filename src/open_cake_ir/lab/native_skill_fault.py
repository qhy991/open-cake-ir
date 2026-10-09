"""Retain and reconstruct a completed provider turn's rejected native input.

This boundary follows native collection; partial raw rollout collection is outside
its domain. It never qualifies live authoring or changes failed-invocation usage.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

from open_cake_ir.serialization import canonical_json_bytes
from .author_home import ISOLATED_SKILL_PACKAGE_V1
from .faults import RunProtocolFault
from .native_skill_observation import MAX_OBSERVATION_BYTES, _unique
from .native_skill_run import MAX_BINDING_BYTES, validate_run_input

CONTEXT_ROLE = 'native_skill_rejection_context'
INPUT_ROLE = 'rejected_native_skill_input'
BINDING_ROLE = 'rejected_native_skill_binding'
ROLES = frozenset({CONTEXT_ROLE, INPUT_ROLE, BINDING_ROLE})
_LIMITS = {INPUT_ROLE: MAX_OBSERVATION_BYTES, BINDING_ROLE: MAX_BINDING_BYTES}


class NativeSkillRunInputFault(RunProtocolFault):
    """The archive refused returned native skill evidence before Turn completion."""


def rejected_input_fault(error, *, provider_turn, run_id, arm, turn, thread_id):
    payloads, statuses = {}, {}
    for role, raw in ((INPUT_ROLE, provider_turn.native_skill_input),
                      (BINDING_ROLE, provider_turn.native_skill_binding)):
        if raw is None:
            statuses[role] = 'missing'
        elif isinstance(raw, bytes) and len(raw) <= _LIMITS[role]:
            statuses[role] = 'retained'
            payloads[role] = raw
        else:
            # Do not copy an unbounded or non-byte value, or fabricate bytes from
            # its reported size/type. Offline refusal then remains unverified.
            statuses[role] = 'unretained'
    payloads[CONTEXT_ROLE] = canonical_json_bytes({'schema_version': 1,
        'kind': 'native_skill_run_rejection_v1', 'run_id': run_id, 'arm': arm,
        'turn': turn, 'thread_id': thread_id, 'inputs': statuses})
    return NativeSkillRunInputFault('provider_fault', str(error), artifact_payloads=payloads)


def has_native_rejection(payload):
    return (payload.get('exception_type') == NativeSkillRunInputFault.__name__
        or any(isinstance(ref, Mapping) and isinstance(ref.get('role'), str) and ref['role'] in ROLES
               for ref in payload.get('objects', ()))
        or any(isinstance(role, str) and role in ROLES for role in payload.get('artifact_rejections', ())))


def replay_rejection(*, payload, evidence, provider, run_id, task_package,
                     provider_events, observed_usage):
    """Reproduce a refusal from retained inputs after completed Turns were replayed."""
    if not has_native_rejection(payload):
        return
    if (payload.get('stage') != 'provider' or payload.get('fault') != 'provider_fault'
        or payload.get('exception_type') != NativeSkillRunInputFault.__name__
        or provider.get('author_home_policy') != ISOLATED_SKILL_PACKAGE_V1):
        raise ValueError('native skill rejection boundary differs')
    if (task_package is None or run_id != task_package.run_id or observed_usage is None):
        raise ValueError('native skill rejection lacks frozen material or native usage witness')
    refs = {}
    for role in ROLES:
        matching = [ref for ref in payload.get('objects', ())
                    if isinstance(ref, Mapping) and ref.get('role') == role]
        if len(matching) > 1:
            raise ValueError('native skill rejection role is duplicated')
        refs[role] = matching[0] if matching else None
    if refs[CONTEXT_ROLE] is None:
        raise ValueError('native skill rejection context is missing')
    raw = evidence.read_object(refs[CONTEXT_ROLE])
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 16384:
        raise ValueError('native skill rejection context bounds differ')
    try:
        context = json.loads(raw, object_pairs_hook=_unique)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('native skill rejection context JSON differs') from error
    turn = len(provider_events) + 1
    if (not isinstance(context, dict) or set(context) != {
        'schema_version', 'kind', 'run_id', 'arm', 'turn', 'thread_id', 'inputs'}
        or type(context['schema_version']) is not int or context['schema_version'] != 1
        or context['kind'] != 'native_skill_run_rejection_v1'
        or context['run_id'] != run_id or context['arm'] != task_package.arm
        or type(context['turn']) is not int or context['turn'] != turn
        or type(payload.get('turn')) is not int or payload['turn'] != turn
        or context['thread_id'] != observed_usage.thread_id
        or not isinstance(context['inputs'], dict) or set(context['inputs']) != set(_LIMITS)):
        raise ValueError('native skill rejection Run/arm/turn/context differs')
    inputs = {}
    for role, limit in _LIMITS.items():
        status = context['inputs'][role]
        if status not in ('missing', 'retained'):
            raise ValueError('native skill rejected input was not retained; refusal is unverified')
        if (refs[role] is not None) != (status == 'retained') or role in payload.get('artifact_rejections', ()):
            raise ValueError('native skill rejection input retention differs')
        inputs[role] = evidence.read_object(refs[role]) if status == 'retained' else None
        if inputs[role] is not None and len(inputs[role]) > limit:
            raise ValueError('native skill rejection input bounds differ')
    previous_input = previous_binding = None
    if provider_events:
        previous = provider_events[-1]['payload']
        if (previous.get('turn') != turn - 1 or previous.get('thread_id') != observed_usage.thread_id):
            raise ValueError('native skill rejection prior Turn differs')
        prior = []
        for role in ('provider_native_skill_input', 'provider_native_skill_binding'):
            matching = [ref for ref in previous['objects'] if ref.get('role') == role]
            if len(matching) != 1:
                raise ValueError('native skill rejection lacks its validated previous input')
            prior.append(evidence.read_object(matching[0]))
        previous_input, previous_binding = prior
    try:
        validate_run_input(native_input=inputs[INPUT_ROLE], binding=inputs[BINDING_ROLE],
            previous_input=previous_input, previous_binding=previous_binding,
            task_package=task_package, provider=provider, thread_id=observed_usage.thread_id, turn=turn)
    except ValueError as error:
        if payload.get('exception_message') != str(error).strip()[:2048]:
            raise ValueError('native skill rejection reason differs from retained inputs') from error
    else:
        raise ValueError('retained native skill inputs do not reproduce the rejection')
