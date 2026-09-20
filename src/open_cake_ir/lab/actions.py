"""Resolve explicit author intent to candidate bytes; no build, device or acceptance.

The caller supplies a Run-local immutable candidate view. Resolution never searches
another Run, follows a filesystem path, or imports authored Python.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from collections.abc import Mapping

from open_cake_ir.compiler import Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.serialization import canonical_json_bytes


@dataclass(frozen=True)
class ActionResolution:
    action_sha256: str
    kind: str
    candidate: bytes | None
    parent: str | None = None
    transformation: str | None = None
    reason: str = 'submitted'
    message: str = 'Submitted implementation; correctness and performance remain unqualified.'

    @property
    def document(self):
        return {'action_sha256': self.action_sha256, 'kind': self.kind,
                'candidate_sha256': None if self.candidate is None else sha256(self.candidate).hexdigest(),
                'parent': self.parent, 'transformation': self.transformation,
                'reason': self.reason, 'message': self.message}


def candidate_program(payload, *, allow_python=True):
    document = json.loads(payload)
    if isinstance(document, Mapping) and set(document) == {'python_source'}:
        if not allow_python:
            raise ValueError('parent Python syntax is outside the authoring environment')
        document = parse(document['python_source'], filename='parent.cake.py').document
    return Program.from_dict(document) if 'program_id' in document else Program.from_schedule(document)


def resolve_action(payload: bytes, *, environment_kind, transformations, candidates,
                   baselines, compiler_factory, allow_python=False) -> ActionResolution:
    action_sha256 = sha256(payload).hexdigest()
    if environment_kind == 'direct_cuda':
        return ActionResolution(action_sha256, 'submit', payload)
    try:
        document = json.loads(payload)
    except (ValueError, UnicodeError):
        # The representation owner retains the established localized parser feedback.
        return ActionResolution(action_sha256, 'submit', payload)
    if not isinstance(document, Mapping) or 'action' not in document:
        return ActionResolution(action_sha256, 'submit', payload)
    kind = document.get('action')
    def refused(reason, message, *, parent=None, transformation=None):
        return ActionResolution(action_sha256, str(kind), None, parent, transformation, reason, message)
    if kind == 'submit':
        if set(document) != {'action', 'candidate'} or not isinstance(document['candidate'], Mapping):
            return refused('action_shape', 'submit requires one complete candidate object.')
        return ActionResolution(action_sha256, 'submit', canonical_json_bytes(document['candidate']))
    if kind != 'transform' or set(document) != {'action', 'parent', 'transformation', 'parameters'}:
        return refused('action_shape', 'Use submit(candidate) or transform(parent, transformation, parameters).')
    parent, transformation = document['parent'], document['transformation']
    if not isinstance(parent, str) or not parent or not isinstance(transformation, str):
        return refused('action_shape', 'Transform parent and transformation must be names.')
    if environment_kind != 'open_cake' or transformation not in transformations:
        return refused('transform_not_granted', 'This Run does not grant the requested transformation.',
                       parent=parent, transformation=transformation)
    source = (baselines.get(parent.removeprefix('baseline:')) if parent.startswith('baseline:')
              else candidates.get(parent))
    if source is None:
        return refused('parent_not_authorized', 'Parent is not a prior candidate of this Run or an authorized baseline.',
                       parent=parent, transformation=transformation)
    try:
        program = candidate_program(source, allow_python=allow_python)
    except (TypeError, ValueError) as error:
        return refused('parent_not_program', str(error), parent=parent, transformation=transformation)
    result = compiler_factory().rewrite_program(program, transformation, document['parameters'])
    return ActionResolution(action_sha256, 'transform',
        canonical_json_bytes(result.program.document) if result.applied else None,
        parent, transformation, result.reason, result.message)


def resolve_action_set(payloads, **context):
    """All parents come from earlier turns; same-turn proposal ordering grants nothing."""
    return tuple(resolve_action(payload, **context) for payload in payloads)
