"""Executor-owned Run binding and reconstruction of native skill evidence.

This validates retained control-plane and native facts. It grants no qualification,
proves no filesystem read jail, and does not replace Evidence custody verification.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath

from open_cake_ir.serialization import canonical_json_bytes
from .author_home import ISOLATED_SKILL_PACKAGE_V1, system_skills_identity
from .native_skill_observation import _unique, replay_observation
from .provider_documents import invocation_document
from .provider_invocation import codex_argv
from .provider_policy import execution_configuration

MAX_BINDING_BYTES = 2 * 1024 * 1024
_FIELDS = {'schema_version', 'kind', 'run_id', 'arm', 'turn', 'invocation',
           'configuration', 'system_skills_snapshot'}
_INVOCATION_FIELDS = {'argv_without_prompt', 'cwd', 'sandbox', 'provider_revision',
    'removed_environment', 'thread_id', 'codex_home', 'user_home', 'native_skill_package'}


def bind_invocation(*, invocation, request, configuration, system_skills_snapshot) -> bytes:
    """Called by QualifiedRunProvider after its package, lifecycle and tree checks."""
    return canonical_json_bytes({'schema_version': 1, 'kind': 'native_skill_run_binding_v1',
        'run_id': request.run_id, 'arm': request.arm, 'turn': request.turn,
        'invocation': invocation_document(invocation, include_prompt=False),
        'configuration': configuration, 'system_skills_snapshot': system_skills_snapshot})


def _binding(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_BINDING_BYTES:
        raise ValueError('native skill Run binding bounds differ')
    try:
        document = json.loads(raw, object_pairs_hook=_unique)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('native skill Run binding JSON differs') from error
    if (not isinstance(document, dict) or set(document) != _FIELDS
        or type(document['schema_version']) is not int or document['schema_version'] != 1
        or document['kind'] != 'native_skill_run_binding_v1'
        or not isinstance(document['invocation'], dict)
        or set(document['invocation']) != _INVOCATION_FIELDS):
        raise ValueError('native skill Run binding fields differ')
    return document


def _path(value) -> PurePosixPath:
    if not isinstance(value, str) or not 0 < len(value) <= 4096 or '\x00' in value or '\\' in value:
        raise ValueError('native skill invocation path differs')
    path = PurePosixPath(value)
    if not path.is_absolute() or str(path) != value or '..' in path.parts:
        raise ValueError('native skill invocation path is not canonical')
    return path


def _snapshot(value) -> tuple:
    if not isinstance(value, list) or len(value) > 8192:
        raise ValueError('native skill installed snapshot bounds differ')
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError('native skill installed snapshot row differs')
        name, mode, identity = row
        if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
            raise ValueError('native skill installed snapshot path differs')
        path = PurePosixPath(name.rstrip('/'))
        if (path.is_absolute() or '..' in path.parts or str(path) != name.rstrip('/')
            or name.endswith('//') or str(path) == '.' or type(mode) is not int
            or not 0 <= mode <= 0o777 or mode & 0o022
            or not isinstance(identity, str)
            or (identity != '' if name.endswith('/') else
                len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity))):
            raise ValueError('native skill installed snapshot fields differ')
        rows.append(tuple(row))
    if rows != sorted(rows) or len({row[0] for row in rows}) != len(rows):
        raise ValueError('native skill installed snapshot order differs')
    return tuple(rows)


def validate_run_input(*, native_input, binding, previous_input, previous_binding,
                       task_package, provider, thread_id, turn) -> dict | None:
    """Validate one completed Turn against its frozen package and retained invocation.

    The caller must pass only an already validated previous pair.
    Invocation paths are allocated runtime facts; the qualified configuration owns
    content identity. Resume must preserve those allocated paths. This is not proof
    of execution against an adversarial evidence writer; custody remains separate.
    """
    enabled = provider.get('author_home_policy') == ISOLATED_SKILL_PACKAGE_V1
    package = getattr(task_package, 'native_skill_package', None)
    if not enabled:
        if any(value is not None for value in (native_input, binding, previous_input, previous_binding, package)):
            raise ValueError('undeclared native skill Run evidence')
        return None
    if package is None or task_package is None:
        raise ValueError('native skill Run lacks its frozen TaskPackage')
    if (type(turn) is not int or turn < 1 or (previous_input is None) != (turn == 1)
        or (previous_binding is None) != (turn == 1)):
        raise ValueError('native skill Run sequence differs')
    observed = _binding(binding)
    if (type(observed['turn']) is not int or observed['turn'] != turn
        or observed['run_id'] != task_package.run_id or observed['arm'] != task_package.arm):
        raise ValueError('native skill Run/arm/turn binding differs')
    configuration = execution_configuration(provider)
    if canonical_json_bytes(observed['configuration']) != canonical_json_bytes(configuration):
        raise ValueError('native skill frozen configuration differs')
    invocation = observed['invocation']
    if (invocation['provider_revision'] != provider['revision']
        or invocation['sandbox'] != 'workspace-write'
        or invocation['removed_environment'] != configuration['removed_environment']
        or invocation['thread_id'] != (None if turn == 1 else thread_id)
        or invocation['native_skill_package'] != package.reference
        or provider.get('native_skill_package') != package.reference):
        raise ValueError('native skill invocation authority differs')
    cwd, home, codex = (_path(invocation[key]) for key in ('cwd', 'user_home', 'codex_home'))
    paths = (cwd, home, codex)
    if any(a == b or a in b.parents or b in a.parents
           for i, a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError('native skill invocation roots overlap')
    argv = invocation['argv_without_prompt']
    if (not isinstance(argv, list) or not argv or any(not isinstance(v, str) for v in argv)
        or argv.count('--output-schema') != 1):
        raise ValueError('native skill invocation command differs')
    index = argv.index('--output-schema')
    if index + 1 == len(argv):
        raise ValueError('native skill invocation schema argument is missing')
    executable, schema = _path(argv[0]), _path(argv[index+1])
    declared_schema = PurePosixPath(provider['output_schema']['path'])
    if (not declared_schema.parts or '..' in declared_schema.parts
        or (schema != declared_schema if declared_schema.is_absolute() else
            schema.parts[-len(declared_schema.parts):] != declared_schema.parts)):
        raise ValueError('native skill invocation schema path differs from declaration')
    expected_argv = codex_argv(executable=executable, output_schema=schema,
        model=configuration['model'], reasoning_effort=configuration['reasoning_effort'],
        service_tier=configuration['service_tier'], disabled_features=configuration['disabled_features'],
        event_contract=configuration.get('event_contract', 'closed_file_change_v1'),
        thread_id=None if turn == 1 else thread_id)
    if argv != list(expected_argv):
        raise ValueError('native skill invocation command differs from frozen policy')
    snapshot = _snapshot(observed['system_skills_snapshot'])
    if previous_binding is None:
        # This handoff reuses the owner's existing inventory in place of reopening
        # the original installation. A mismatch with the qualified tree refuses.
        if system_skills_identity(snapshot) != provider.get('system_skills_sha256'):
            raise ValueError('native skill installed tree differs from qualification')
    else:
        prior = _binding(previous_binding)
        if (prior['run_id'] != task_package.run_id or prior['arm'] != task_package.arm
            or type(prior['turn']) is not int or prior['turn'] != turn - 1
            or observed['system_skills_snapshot'] != prior['system_skills_snapshot']
            or observed['configuration'] != prior['configuration']):
            raise ValueError('native skill previous Run binding or installed tree differs')
        for key in ('cwd', 'user_home', 'codex_home', 'native_skill_package', 'provider_revision'):
            if invocation[key] != prior['invocation'][key]:
                raise ValueError('native skill invocation paths changed between turns')
        previous_argv = prior['invocation']['argv_without_prompt']
        if argv[0] != previous_argv[0] or argv[index+1] != previous_argv[previous_argv.index('--output-schema')+1]:
            raise ValueError('native skill executable or schema path changed between turns')
    entries = {f'skills/{name}/SKILL.md' for name in package.entry_names}
    return replay_observation(native_input, previous=previous_input, thread_id=thread_id,
        cwd=str(cwd), model=configuration['model'], effort=configuration['reasoning_effort'], turn=turn,
        package_files={str(home/'.agents'/item.path): item.payload for item in package.files if item.path in entries},
        system_paths=tuple(str(codex/'skills/.system'/name) for name, _, _ in snapshot
                           if len(name.split('/')) == 2 and name.endswith('/SKILL.md')))
