"""Reconstruct qualification native inputs from retained objects, without live homes.

This is a semantic check, not an admission grant. Its caller owns authority/receipt
anchoring and historical Evidence custody. No model use or filesystem jail is proved.
"""
from hashlib import sha256
import json
import re

from open_cake_ir.serialization import canonical_json_bytes
from .author_home import ISOLATED_SKILL_PACKAGE_V1
from .native_skill_observation import _unique
from .native_skill_run import validate_run_input, _path
from .native_skills import NativeSkillPackage
from .provider_documents import NATIVE_SKILL_QUALIFICATION_V1
from .provider_events import parse_codex_turn_events
from .provider_policy import execution_configuration
from .task_package import TaskPackage, render_task_request


def selection_instruction(names):
    """Names are supplied by the treatment, never guessed from package directories."""
    if (not isinstance(names, list) or not 0 < len(names) <= 32
        or any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9:_-]{0,127}', name)
               for name in names) or len(set(names)) != len(names)):
        raise ValueError('qualification native skill selection differs')
    return 'Use these native skills in every turn: ' + ', '.join('$' + name for name in names) + '.\n'


def reconstruct_qualification_inputs(*, evidence, run_id, authority, payload, receipt, requested_names=None):
    """Recheck each declared arm's initial/resume pair; do not reopen its original files.

    Reuses Run command-policy/native reconstruction with an in-memory binding derived
    from the retained invocation. The binding is not a second archived record.
    Receipt and authority are expected to be independently anchored by the caller.
    Requested names, when supplied for reuse, must be current package bodies in
    every arm and both turns; a system skill of the same name cannot satisfy them.
    """
    if (authority.get('author_home_policy') != ISOLATED_SKILL_PACKAGE_V1
        or authority.get('harness') != 'codex'
        or authority.get('kind') != 'codex_provider_two_turn_qualification'
        or authority.get('cwd_policy') != 'same_new_task_workspace'
        or authority.get('turns') != ['initial_add', 'same_thread_resume_update']
        or authority.get('gpu_execution_authorized') is not False):
        raise ValueError('native qualification authority differs')
    context = authority.get('native_skill_context')
    if (not isinstance(context, dict)
        or set(context) != {'kind', 'executable', 'output_schema', 'arms', 'selected_names'}
        or context['kind'] != NATIVE_SKILL_QUALIFICATION_V1):
        raise ValueError('native qualification lacks its frozen context')
    instruction = selection_instruction(context['selected_names'])
    if requested_names is not None:
        selection_instruction(requested_names)
    # A reused qualification must cover this request, not just its original TASK.
    names = list(dict.fromkeys(context['selected_names'] + (requested_names or [])))
    arms = authority.get('arms')
    if (not isinstance(arms, list) or not 0 < len(arms) <= 2
        or any(not isinstance(arm, str) or not arm for arm in arms) or len(set(arms)) != len(arms)
        or not isinstance(context['arms'], dict) or set(context['arms']) != set(arms)
        or not isinstance(payload.get('arms'), dict) or set(payload['arms']) != set(arms)):
        raise ValueError('native qualification arm set differs')
    refs = payload.get('objects')
    if (not isinstance(refs, list) or any(not isinstance(ref, dict) or not isinstance(ref.get('role'), str)
                                        for ref in refs)):
        raise ValueError('native qualification object roles differ')
    objects = {ref['role']: ref for ref in refs}
    required = {'native_skill_package', 'system_skills_snapshot', 'qualification_reference', 'qualification_receipt'}
    for arm in arms:
        required.update(f'{arm}_{phase}_{role}' for phase in ('initial', 'resumed')
                        for role in ('invocation', 'task_projection', 'provider_events', 'native_skill_input'))
    if len(objects) != len(refs) or not required <= set(objects):
        raise ValueError('native qualification missing or duplicate object role')
    # Other qualification evidence keeps its existing owner. Reject foreign native arms.
    if any(role.endswith('_native_skill_input') and role not in required for role in objects):
        raise ValueError('native qualification foreign native input role')
    def raw(role):
        return evidence.read_object(objects[role])
    def document(role):
        return json.loads(raw(role), object_pairs_hook=_unique)
    if document('qualification_receipt') != receipt.document:
        raise ValueError('native qualification retained receipt differs')
    if (receipt.provider_revision != authority['provider_revision']
        or receipt.executable_sha256 != authority['executable_sha256']
        or receipt.scope != authority['qualification_scope']
        or not all((receipt.qualified, receipt.initial_and_resume_equivalent,
                    receipt.file_lifecycle_observed, receipt.usage_observed))):
        raise ValueError('native qualification receipt authority differs')
    bundle = raw('qualification_reference')
    if (sha256(bundle).hexdigest() != authority['reference_bundle_sha256']
        or payload.get('reference_bundle_sha256') != authority['reference_bundle_sha256']):
        raise ValueError('native qualification reference differs')
    reference = json.loads(bundle, object_pairs_hook=_unique)
    if not isinstance(reference, dict) or set(reference) != set(arms):
        raise ValueError('native qualification reference arm set differs')
    package = NativeSkillPackage.from_bytes(raw('native_skill_package'), authority['native_skill_package'])
    provider = {key: authority[key] for key in (
        'model', 'reasoning_effort', 'service_tier', 'removed_environment', 'sandbox',
        'reference_visibility', 'disabled_features', 'code_mode_host', 'submission_contract',
        'author_home_policy', 'native_skill_package')}
    provider.update(revision=authority['provider_revision'], qualification=None, qualification_anchor=None,
        executable_sha256=authority['executable_sha256'], system_skills_sha256=receipt.system_skills_sha256,
        cwd_policy='independent_task_workspace', output_schema={
            'path': str(_path(context['output_schema'])), 'sha256': authority['output_schema_sha256']})
    if authority['event_contract'] == 'closed_file_change_v1':
        provider['web_search'] = authority['web_search']
    else:
        provider['event_contract'] = authority['event_contract']
    configuration = execution_configuration(provider)
    if sha256(canonical_json_bytes(configuration)).hexdigest() != receipt.configuration_sha256:
        raise ValueError('native qualification configuration differs from receipt')
    snapshot = document('system_skills_snapshot')
    all_roots, threads, result = [], set(), {}
    for arm in arms:
        paths = context['arms'][arm]
        if not isinstance(paths, dict) or set(paths) != {'cwd', 'user_home', 'codex_home'}:
            raise ValueError('native qualification arm paths differ')
        for path in map(_path, paths.values()):
            if any(path == old or path in old.parents or old in path.parents for old in all_roots):
                raise ValueError('native qualification arm roots overlap')
            all_roots.append(path)
        texts = reference[arm]
        if (not isinstance(texts, dict) or set(texts) != {'task_markdown', 'agents_markdown'}
            or any(not isinstance(value, str) for value in texts.values())
            or not texts['task_markdown'].endswith(instruction)):
            raise ValueError('native qualification task selection differs')
        task = TaskPackage(f'{run_id}-{arm}', arm, **texts, native_skill_package=package)
        previous_input = previous_binding = thread_id = None
        result[arm] = []
        for number, phase in enumerate(('initial', 'resumed'), 1):
            prefix = f'{arm}_{phase}_'
            invocation = document(prefix + 'invocation')
            prompt, projection = render_task_request(task, {'turn': number})
            if (not isinstance(invocation, dict) or not isinstance(invocation.get('argv'), list)
                or len(invocation['argv']) < 2 or invocation['argv'][-1] != prompt
                or invocation['argv'][0] != str(_path(context['executable']))
                or any(invocation.get(key) != value for key, value in paths.items())
                or raw(prefix + 'task_projection') != projection):
                raise ValueError('native qualification invocation/task projection differs')
            terminal = {'arm': arm, 'candidate_written': True, 'kind': 'open_cake_ir_turn', 'turn': number}
            if authority['event_contract'] == 'closed_file_change_v1':
                terminal['tool_calls'] = 1
            parsed = parse_codex_turn_events(raw(prefix + 'provider_events'),
                expected_terminal_message=canonical_json_bytes(terminal).decode(),
                event_contract=authority['event_contract'])
            if number == 1:
                thread_id = parsed.thread_id
                if thread_id in threads:
                    raise ValueError('native qualification arm threads overlap')
                threads.add(thread_id)
            if parsed.thread_id != thread_id or payload['arms'][arm].get('thread_id') != thread_id:
                raise ValueError('native qualification provider thread differs')
            invocation = dict(invocation)
            invocation['argv_without_prompt'] = invocation.pop('argv')[:-1]
            binding = canonical_json_bytes({'schema_version': 1, 'kind': 'native_skill_run_binding_v1',
                'run_id': task.run_id, 'arm': arm, 'turn': number, 'invocation': invocation,
                'configuration': configuration, 'system_skills_snapshot': snapshot})
            native_input = raw(prefix + 'native_skill_input')
            observation = validate_run_input(native_input=native_input, binding=binding,
                previous_input=previous_input, previous_binding=previous_binding,
                task_package=task, provider=provider, thread_id=thread_id, turn=number)
            entry_paths = {str(_path(paths['user_home'])/'.agents/skills'/name/'SKILL.md')
                           for name in package.entry_names}
            for name in names:
                matches = [item for item in observation['catalog']
                           if item['name'] == name and item['path'] in entry_paths]
                if (len(matches) != 1 or not any(item['name'] == name and item['path'] == matches[0]['path']
                    for item in observation['loaded_this_turn'])):
                    raise ValueError(f'native qualification selected body was not delivered this turn: {arm}/{phase}/{name}')
            result[arm].append(observation)
            previous_input, previous_binding = native_input, binding
    return result


def verify_qualification_evidence(*, qualification, anchor, requested_names=None,
                                  required_environment_kinds=()):
    """Open actual sealed evidence and verify the receipt's native-input capability.

    Scope admission and the external anchor reference belong to admission. This
    verifier preserves the receipt's own scope: fixture evidence remains fixture
    evidence. A capability name or an immediate-audit flag cannot replace this check.
    """
    from open_cake_ir.evidence import EvidenceStore
    if (not isinstance(required_environment_kinds, (tuple, list))
        or any(not isinstance(kind, str) or not kind for kind in required_environment_kinds)):
        raise ValueError('native qualification requested environment kinds differ')
    if requested_names is not None:
        selection_instruction(requested_names)
    if qualification.native_skill_input_contract != NATIVE_SKILL_QUALIFICATION_V1:
        raise ValueError('native skill input capability is not qualified by this receipt')
    if (not isinstance(anchor, dict)
        or set(anchor) != {'schema_version', 'kind', 'run_id', 'evidence_root', 'authority_sha256',
                           'qualification_receipt_sha256', 'immediate_audit_integrity', 'terminal_seal_sha256'}
        or anchor.get('schema_version') != 1 or anchor.get('immediate_audit_integrity') is not True
        or anchor.get('kind') != 'codex_provider_qualification_evidence_anchor'
        or anchor.get('qualification_receipt_sha256') != qualification.canonical_sha256
        or not isinstance(anchor.get('run_id'), str) or not anchor['run_id']
        or not isinstance(anchor.get('evidence_root'), str) or not anchor['evidence_root']):
        raise ValueError('native qualification anchor differs from receipt')
    evidence = EvidenceStore.open(anchor['evidence_root'])
    audit = evidence.audit_run(anchor['run_id'])
    if (not audit.archive_integrity or not audit.filesystem_custody_verified
        or audit.authority_sha256 != anchor.get('authority_sha256')
        or audit.terminal_seal_sha256 != anchor.get('terminal_seal_sha256')
        or audit.protocol_adherence != 'adhered' or audit.endpoint_observation != 'qualified'
        or not isinstance(audit.endpoint, dict)):
        raise ValueError('native qualification archive integrity, custody or outcome is unverified')
    authority = evidence.replay_authority(audit.run_id)
    endpoint = audit.endpoint
    if (endpoint.get('qualification_receipt_sha256') != qualification.canonical_sha256
        or endpoint.get('qualification_scope') != qualification.scope
        or endpoint.get('arms_qualified') != authority.get('arms')
        or endpoint.get('harness') != 'codex'
        or endpoint.get('gpu_execution_authorized') is not False
        or any(endpoint.get(key) is not True for key in (
            'add_observed', 'update_observed', 'thread_continuity_observed', 'usage_observed',
            'sandbox_observed', 'cwd_observed', 'candidate_changed', 'reference_visibility_observed'))
        or any(endpoint.get(key) != authority.get(key) for key in (
            'event_contract', 'feature_policy', 'submission_contract', 'maximum_candidates_per_turn'))):
        raise ValueError('native qualification terminal endpoint differs')
    events = evidence.replay_events(audit.run_id)
    observations = [event['payload'] for event in events if event['kind'] == 'provider_qualification_observed']
    if len(observations) != 1 or any(event['kind'] == 'provider_qualification_failed' for event in events):
        raise ValueError('native qualification must retain exactly one successful observation')
    result = reconstruct_qualification_inputs(evidence=evidence, run_id=audit.run_id,
        authority=authority, payload=observations[0], receipt=qualification, requested_names=requested_names)
    missing = set(required_environment_kinds) - set(result)
    if missing:
        raise ValueError('native qualification has no retained turns for environment: ' + ', '.join(sorted(missing)))
    return result


def require_live_native_receipt(*, qualification, anchor):
    """Reject absent/old/fixture receipts early; this never grants admission."""
    if (anchor is None or not getattr(qualification, "qualified", False)
        or getattr(qualification, "scope", None) not in {
            "live_two_turn_current_provider", "live_two_turn_tool_rich_provider"}
        or getattr(qualification, "native_skill_input_contract", None) != NATIVE_SKILL_QUALIFICATION_V1):
        raise ValueError("native skill discovery and actual initial/resume delivery are not qualified; live evidence is required")


def verify_live_qualification_evidence(*, qualification, anchor, required_environment_kinds=(),
                                       expected_configuration=None):
    """Live construction needs the archive itself; receipt fields are not permission."""
    require_live_native_receipt(qualification=qualification, anchor=anchor)
    if (expected_configuration is not None
        and sha256(canonical_json_bytes(expected_configuration)).hexdigest() != qualification.configuration_sha256):
        raise ValueError('native provider qualification bytes or capability differ from the runtime configuration')
    return verify_qualification_evidence(qualification=qualification, anchor=anchor,
        required_environment_kinds=required_environment_kinds)
