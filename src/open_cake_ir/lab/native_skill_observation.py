"""Bounded Codex 0.159.2 skill input observation, not provider qualification.

Read the same invocation's native rollout, retaining only skill frames and their
native turn bindings. No credential, full prompt, tool output or reasoning log is
copied. This is version-specific retained-input evidence, not an HTTP wire tap,
model-use claim or filesystem read-isolation proof.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from open_cake_ir.serialization import canonical_json_bytes

VERSION = '0.159.2'
MAX_ROLLOUT_BYTES = 32 * 1024 * 1024
MAX_RECORD_BYTES = 8 * 1024 * 1024
MAX_RECORDS = 100000
MAX_OBSERVATION_BYTES = 64 * 1024 * 1024
MAX_SKILLS = 128
_ID = re.compile(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('native skill observation has duplicate JSON keys')
        result[key] = value
    return result


def _read(path: Path) -> bytes:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
        or info.st_uid != os.geteuid() or info.st_mode & 0o022
        or not 0 < info.st_size <= MAX_ROLLOUT_BYTES):
        raise ValueError('native skill rollout owner, type or size differs')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        identity = lambda value: (value.st_dev, value.st_ino, value.st_mode,
            value.st_nlink, value.st_uid, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if identity(info) != identity(os.fstat(descriptor)):
            raise ValueError('native skill rollout changed before read')
        chunks, remaining = [], info.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError('native skill rollout ended early')
            chunks.append(chunk)
            remaining -= len(chunk)
        if (os.read(descriptor, 1) or identity(info) != identity(os.fstat(descriptor))
            or identity(info) != identity(path.lstat())):
            raise ValueError('native skill rollout changed while reading')
        return b''.join(chunks)
    finally:
        os.close(descriptor)


def _rollout_path(home: Path, thread_id: str) -> Path | None:
    if not isinstance(thread_id, str) or _ID.fullmatch(thread_id) is None:
        raise ValueError('native skill thread identity differs')
    home = Path(home)
    if not home.is_absolute() or home.resolve(strict=True) != home:
        raise ValueError('native skill CODEX_HOME is not canonical')
    sessions = home/'sessions'
    if not sessions.exists() and not sessions.is_symlink():
        return None
    pending, found, count = [sessions], [], 0
    while pending:
        directory = pending.pop()
        info = directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_mode & 0o022):
            raise ValueError('native skill session directory custody differs')
        for path in directory.iterdir():
            count += 1
            if count > 4096:
                raise ValueError('native skill session tree exceeds bound')
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('native skill session tree contains link or special file')
            elif path.name.endswith('-' + thread_id + '.jsonl'):
                found.append(path)
    if len(found) > 1:
        raise ValueError('native skill thread has multiple rollouts')
    return found[0] if found else None


def before_invocation(invocation) -> bytes | None:
    """Keep the resume prefix in memory; absence can never produce acceptance."""
    if invocation.thread_id is None:
        return None
    path = _rollout_path(invocation.codex_home, invocation.thread_id)
    return _read(path) if path is not None else None


def _records(raw: bytes) -> list[dict]:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_ROLLOUT_BYTES or not raw.endswith(b'\n'):
        raise ValueError('native skill rollout bounds or final record differ')
    lines = raw.splitlines()
    if len(lines) > MAX_RECORDS or any(not line or len(line) > MAX_RECORD_BYTES for line in lines):
        raise ValueError('native skill rollout record bounds differ')
    try:
        records = [json.loads(line, object_pairs_hook=_unique) for line in lines]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('native skill rollout JSON differs') from error
    if any(not isinstance(row, dict) or not isinstance(row.get('payload'), dict) for row in records):
        raise ValueError('native skill rollout record differs')
    return records


def _path(locator: str, roots: dict[str, str]) -> str:
    if not isinstance(locator, str) or not 0 < len(locator) <= 4096 or '\\' in locator:
        raise ValueError('native skill locator differs')
    path = PurePosixPath(locator)
    if '..' in path.parts or path.as_posix() != locator:
        raise ValueError('native skill locator is not canonical')
    if not path.is_absolute():
        alias, separator, suffix = locator.partition('/')
        if not separator or alias not in roots:
            raise ValueError('native skill root alias is unknown')
        path = PurePosixPath(roots[alias])/suffix
    if not path.is_absolute():
        raise ValueError('native skill root is not absolute')
    return str(path)


def _catalog(text: str) -> list[dict]:
    if not text.startswith('<skills_instructions>\n') or not text.endswith('\n</skills_instructions>'):
        raise ValueError('native skill catalog frame differs')
    roots, entries, active = {}, [], False
    for line in text.splitlines():
        alias = re.fullmatch(r'- `([^`]+)` = `([^`]+)`', line)
        if alias:
            if alias[1] in roots:
                raise ValueError('native skill root alias is duplicated')
            roots[alias[1]] = _path(alias[2], {})
        if line == '### Available skills':
            active = True
            continue
        if active and line.startswith('### '):
            active = False
        if not active or not line.startswith('- '):
            continue
        row = re.fullmatch(r'- (.+?): .*\(file: (.+)\)', line)
        if row is None or len(row[1]) > 160 or len(entries) >= MAX_SKILLS:
            raise ValueError('native skill catalog entry differs')
        entries.append({'name': row[1], 'path': _path(row[2], roots)})
    if (not entries or len({e['name'] for e in entries}) != len(entries)
        or len({e['path'] for e in entries}) != len(entries)):
        raise ValueError('native skill catalog is empty or duplicated')
    return entries



def _retained_records(rows: list[dict]) -> list[dict]:
    """Keep only the native fields needed for skill-input semantic reconstruction.

    Original line positions distinguish this turn from prior turns. Missing rows
    are deliberately outside retained coverage; they are not transcript evidence.
    """
    retained = []
    for index, row in enumerate(rows):
        kind, payload = row.get('type'), row['payload']
        selected = None
        if kind == 'session_meta':
            selected = {key: payload.get(key) for key in ('id', 'cwd', 'cli_version')}
        elif kind == 'event_msg' and payload.get('type') in {'task_started', 'task_complete'}:
            selected = {key: payload.get(key) for key in ('type', 'turn_id')}
        elif kind == 'turn_context':
            selected = {key: payload.get(key) for key in ('turn_id', 'cwd', 'model', 'effort')}
        elif kind == 'world_state':
            state = payload['state']
            selected = {'full': payload['full'], 'state': {}}
            if 'host_skills' in state:
                selected['state']['host_skills'] = {
                    key: state['host_skills'].get(key) for key in ('body', 'includeInstructions')}
        elif kind == 'response_item':
            metadata = payload.get('internal_chat_message_metadata_passthrough') or {}
            kinds = metadata.get('content_item_kinds', []) if isinstance(metadata, dict) else []
            content = payload.get('content', [])
            if isinstance(content, list) and isinstance(kinds, list):
                positions = [i for i, item_kind in enumerate(kinds)
                             if item_kind in {'host_skills.instructions', 'skills.selected_skill_instructions'}]
                if positions:
                    selected = {'role': payload['role'],
                        'content': [{key: content[i][key] for key in ('type', 'text')} for i in positions],
                        'internal_chat_message_metadata_passthrough': {
                            'turn_id': metadata['turn_id'], 'content_item_kinds': [kinds[i] for i in positions]}}
        if selected is not None:
            retained.append({'line': index + 1, 'record': {'type': kind, 'payload': selected}})
    return retained


def _restore_records(document: dict) -> bytes:
    count, facts = document.get('record_count'), document.get('native_records')
    if (type(count) is not int or not 0 < count <= MAX_RECORDS
        or not isinstance(facts, list) or not 0 < len(facts) <= count):
        raise ValueError('retained native skill record bounds differ')
    rows = [{'payload': {}} for _ in range(count)]
    last = 0
    for fact in facts:
        if (not isinstance(fact, dict) or set(fact) != {'line', 'record'}
            or type(fact['line']) is not int or not last < fact['line'] <= count
            or not isinstance(fact['record'], dict)
            or set(fact['record']) != {'type', 'payload'}
            or not isinstance(fact['record']['payload'], dict)):
            raise ValueError('retained native skill record shape or order differs')
        rows[fact['line'] - 1] = fact['record']
        last = fact['line']
    return b''.join(canonical_json_bytes(row) + b'\n' for row in rows)


def _observation(raw: bytes) -> dict:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_OBSERVATION_BYTES:
        raise ValueError('retained native skill observation bounds differ')
    try:
        document = json.loads(raw, object_pairs_hook=_unique)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('retained native skill observation JSON differs') from error
    if (not isinstance(document, dict) or type(document.get('schema_version')) is not int
        or document['schema_version'] != 2 or document.get('kind') != 'codex_skill_input_observation_v2'):
        raise ValueError('retained native skill observation is not replayable v2')
    return document


def replay_observation(raw: bytes, *, previous: bytes | None, thread_id: str,
                       cwd: str, model: str, effort: str, turn: int,
                       package_files: dict[str, bytes], system_paths: tuple[str, ...]) -> dict:
    """Rederive retained skill semantics using externally supplied invocation authority.

    The caller supplies frozen package bytes and installed system entry paths. The
    preceding observation must have been validated by that caller in sequence.
    Full raw-prefix continuity is checked during capture, not proved by this
    filtered replay. No live filesystem, CLI process or model is consulted here.
    """
    if type(turn) is not int or turn < 1 or (previous is None) != (turn == 1):
        raise ValueError('retained native skill turn sequence differs')
    document = _observation(raw)
    prior = _observation(previous) if previous is not None else None
    if prior is not None and (prior.get('prior_turn_count') != turn - 2
        or any(prior.get(key) != value for key, value in (
            ('thread_id', thread_id), ('cwd', cwd), ('model', model), ('reasoning_effort', effort)))):
        raise ValueError('retained native skill previous invocation differs')
    restored = _restore_records(document)
    old = _restore_records(prior) if prior is not None else None
    projected = project_rollout(restored, previous=old, thread_id=thread_id,
        cwd=cwd, model=model, effort=effort, resumed=turn > 1,
        package_paths=tuple(package_files), system_paths=system_paths)
    if (projected['prior_turn_count'] != turn - 1
        or canonical_json_bytes(document) != canonical_json_bytes(projected)):
        raise ValueError('retained native skill projection differs from its source facts')
    # Check every retained package body, including history, against the frozen input.
    # project_rollout has already parsed and checked the native frame structure.
    for fact in document['native_records']:
        payload = fact['record']['payload']
        if fact['record']['type'] != 'response_item':
            continue
        for content in payload['content']:
            match = re.fullmatch(r'<skill>\n<name>([^\n]+)</name>\n<path>([^\n]+)</path>\n(.*)\n</skill>', content['text'], re.S)
            if match and match[2] in package_files and match[3].encode() != package_files[match[2]]:
                raise ValueError('native selected skill body differs from frozen package')
    return projected


def project_rollout(raw: bytes, *, previous: bytes | None, thread_id: str,
                    cwd: str, model: str, effort: str, resumed: bool,
                    package_paths: tuple[str, ...], system_paths: tuple[str, ...]) -> dict:
    """Pure projection with exact source and turn binding, reusable by replay.

    All package entry points must be in the effective catalog. Thus explicit-only
    or disabled catalog entries are not supported by this observation contract.
    Bodies loaded this turn are distinct from catalog carried across a resume.
    """
    rows = _records(raw)
    if resumed:
        if previous is None or not raw.startswith(previous) or raw == previous:
            raise ValueError('native skill resume lacks an unchanged prior prefix')
        prior_count = len(_records(previous))
    else:
        if previous is not None:
            raise ValueError('native skill initial invocation has prior state')
        prior_count = 0
    meta = rows[0]
    if (meta.get('type') != 'session_meta' or meta['payload'].get('id') != thread_id
        or meta['payload'].get('cwd') != cwd or meta['payload'].get('cli_version') != VERSION
        or sum(row.get('type') == 'session_meta' for row in rows) != 1):
        raise ValueError('native skill session version, thread or workspace differs')
    allowed = set(package_paths) | set(system_paths)
    if not package_paths or len(set(package_paths)) != len(package_paths) or len(allowed) != len(package_paths) + len(system_paths):
        raise ValueError('native skill declared entry points differ')
    active = None
    completed = []
    catalog, catalog_turn, catalog_text = None, None, None
    world_state_seen = False
    current_bodies, retained = [], []
    turn_contexts = {}
    appended_starts = []
    for index, row in enumerate(rows):
        kind, payload = row.get('type'), row['payload']
        event = payload.get('type')
        if any(word in str(kind) + ' ' + str(event) for word in ('compact', 'rollback', 'rolled_back')):
            raise ValueError('native skill compaction or rollback is outside observation coverage')
        if kind == 'event_msg' and event == 'task_started':
            native_turn = payload.get('turn_id')
            if active is not None or not isinstance(native_turn, str) or _ID.fullmatch(native_turn) is None or native_turn in completed:
                raise ValueError('native skill turn start differs')
            active = native_turn
            if index >= prior_count:
                appended_starts.append(active)
        elif kind == 'event_msg' and event == 'task_complete':
            if active is None or payload.get('turn_id') != active or active not in turn_contexts:
                raise ValueError('native skill turn completion differs')
            completed.append(active)
            active = None
        elif kind == 'turn_context':
            if (active is None or payload.get('turn_id') != active or active in turn_contexts
                or payload.get('cwd') != cwd or payload.get('model') != model or payload.get('effort') != effort):
                raise ValueError('native skill turn context differs')
            turn_contexts[active] = index
        elif kind == 'world_state':
            state = payload.get('state')
            if not isinstance(state, dict) or type(payload.get('full')) is not bool:
                raise ValueError('native skill world state differs')
            if 'host_skills' in state:
                skills = state['host_skills']
                if (active is None or catalog_text is None or not isinstance(skills, dict)
                    or skills.get('includeInstructions') is not True
                    or not isinstance(skills.get('body'), str)
                    or '<skills_instructions>' + skills['body'] + '</skills_instructions>' != catalog_text):
                    raise ValueError('native skill world state differs from input catalog')
                world_state_seen = True
            elif payload['full']:
                raise ValueError('native skill full world state lacks skill input')
        elif kind == 'response_item':
            contents = payload.get('content', [])
            if not isinstance(contents, list):
                continue
            metadata = payload.get('internal_chat_message_metadata_passthrough') or {}
            kinds = metadata.get('content_item_kinds', []) if isinstance(metadata, dict) else []
            for position, content in enumerate(contents):
                if not isinstance(content, dict):
                    continue
                text = content.get('text', '')
                item_kind = kinds[position] if isinstance(kinds, list) and position < len(kinds) else None
                has_frame = isinstance(text, str) and text.startswith(('<skills_instructions>', '<skill>'))
                is_skill = item_kind in {'host_skills.instructions', 'skills.selected_skill_instructions'}
                if not has_frame and not is_skill:
                    continue
                if (not is_skill or not has_frame or active is None
                    or metadata.get('turn_id') != active or len(kinds) != len(contents)
                    or content.get('type') != 'input_text'):
                    raise ValueError('native skill frame lacks its current native turn binding')
                if item_kind == 'host_skills.instructions':
                    if payload.get('role') != 'developer':
                        raise ValueError('native skill catalog role differs')
                    observed = _catalog(text)
                    paths = {entry['path'] for entry in observed}
                    if paths - allowed or not set(package_paths) <= paths:
                        raise ValueError('native skill catalog has undeclared or missing sources')
                    if catalog is not None and (observed != catalog or text != catalog_text):
                        raise ValueError('native skill catalog drifted across turns')
                    catalog, catalog_turn, catalog_text = observed, active, text
                    retained.append({'line': index + 1, 'turn_id': active,
                                     'kind': item_kind, 'role': 'developer', 'text': text})
                else:
                    match = re.fullmatch(r'<skill>\n<name>([^\n]+)</name>\n<path>([^\n]+)</path>\n(.*)\n</skill>', text, re.S)
                    if (payload.get('role') != 'user' or match is None
                        or active not in turn_contexts or catalog is None
                        or {'name': match[1], 'path': match[2]} not in catalog):
                        raise ValueError('native selected skill differs from the bound catalog')
                    if index >= prior_count:
                        current_bodies.append({'name': match[1], 'path': match[2], 'body': match[3],
                                               'line': index + 1, 'turn_id': active})
                        retained.append({'line': index + 1, 'turn_id': active,
                            'kind': item_kind, 'role': 'user', 'text': text})
    if (active is not None or len(appended_starts) != 1 or not completed
        or completed[-1] != appended_starts[0] or catalog is None or not world_state_seen
        or not resumed and len(completed) != 1
        or resumed and len(completed) < 2):
        raise ValueError('native skill invocation does not contain exactly one complete new turn')
    return {'schema_version': 2, 'kind': 'codex_skill_input_observation_v2',
            'record_count': len(rows), 'prior_record_count': prior_count,
            'native_records': _retained_records(rows),
            'cli_version': VERSION, 'thread_id': thread_id, 'turn_id': completed[-1],
            'cwd': cwd, 'model': model, 'reasoning_effort': effort, 'resumed': resumed,
            'prior_turn_count': len(completed) - 1, 'catalog_turn_id': catalog_turn,
            'catalog': catalog, 'loaded_this_turn': current_bodies, 'native_skill_frames': retained,
            'system_entrypoints_not_in_catalog': sorted(set(system_paths) - {item['path'] for item in catalog}),
            'coverage': 'retained_native_skill_input_not_wire_capture_or_model_use'}


def after_invocation(invocation, *, thread_id: str, previous: bytes | None) -> bytes:
    """Collect after the native process exits and before accepting its candidate."""
    from .author_home import verify_user_home
    if invocation.thread_id is not None and invocation.thread_id != thread_id:
        raise ValueError('native skill resumed thread differs from invocation')
    verify_user_home(invocation.user_home, invocation.native_skill_package)
    path = _rollout_path(invocation.codex_home, thread_id)
    if path is None:
        raise ValueError('native skill invocation produced no matching rollout')
    package = invocation.native_skill_package
    entrypoints = {f'skills/{name}/SKILL.md' for name in package.entry_names}
    package_files = {str(invocation.user_home/'.agents'/item.path): item.payload
                     for item in package.files if item.path in entrypoints}
    system_root = invocation.codex_home/'skills/.system'
    # The builder's existing post-turn system-skills check owns tree custody and
    # identity. Discover entry paths here, without a second digest inventory.
    system_paths = tuple(sorted(str(path) for path in system_root.glob('*/SKILL.md')))
    argv = invocation.argv
    models = [argv[index+1] for index, item in enumerate(argv[:-1]) if item == '--model']
    efforts = [json.loads(argv[index+1].split('=', 1)[1]) for index, item in enumerate(argv[:-1])
               if item == '--config' and argv[index+1].startswith('model_reasoning_effort=')]
    if len(models) != 1 or len(efforts) != 1:
        raise ValueError('native skill invocation model configuration differs')
    observation = project_rollout(_read(path), previous=previous, thread_id=thread_id,
        cwd=str(invocation.cwd), model=models[0], effort=efforts[0], resumed=invocation.thread_id is not None,
        package_paths=tuple(package_files), system_paths=system_paths)
    for selected in observation['loaded_this_turn']:
        expected = package_files.get(selected['path'])
        if expected is not None and selected['body'].encode() != expected:
            raise ValueError('native selected skill body differs from frozen package')
    encoded = canonical_json_bytes(observation)
    _observation(encoded)  # Enforce the retained-object bound before accepting the Turn.
    return encoded
