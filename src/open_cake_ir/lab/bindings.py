"""External execution locators resolved once into the Campaign Lock.

Study policy remains stable. Only explicit campaign-owned runtime leaves can be
filled; source references and provider treatment are never refreshed here.
"""
from __future__ import annotations

import json
import shlex
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping

from open_cake_ir.evaluation.paired import candidate_from_identity, candidate_identity
from .executor import ExecutorRevision

CAMPAIGN_BINDING = {'binding': 'campaign_lock'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def external_file(project_root, value, context):
    """Admit an existing canonical external file, rejecting traversal and symlinks."""
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError(f'{context} external path differs')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError(f'{context} requires an absolute external path')
    resolved = path.resolve(strict=True)
    root = Path(project_root).resolve(strict=True)
    if resolved != path or not path.is_file() or root in path.parents:
        raise ValueError(f'{context} external file custody differs')
    # Another source worktree is not an external evidence store either.
    if any((parent / '.git').exists() for parent in path.parents):
        raise ValueError(f'{context} must stay outside source worktrees')
    return path


def qualification_path(project_root, value, context):
    """Only qualification references admit external files; source paths stay strict."""
    if isinstance(value, str) and Path(value).is_absolute():
        path = external_file(project_root, value, context)
        return str(path), path
    from .executor import _relative_file
    return _relative_file(Path(project_root), value, context)


def load_baseline_bundle(project_root, bundle_path):
    path = external_file(project_root, str(bundle_path), 'fixed baseline bundle')
    document = json.loads(path.read_bytes())
    if not isinstance(document, Mapping):
        raise ValueError('baseline bundle must be an object')
    # The canonical broker request is already a complete sealed artifact bundle.
    # Its historical Workload/Executor request is preserved, never rewritten.
    identity = document.get('candidate')
    if identity is None:
        fields = ('candidate_sha256', 'target', 'entry_point', 'artifact_roles',
                  'launch_spec_sha256', 'candidate_record_sha256')
        identity = {key: document.get(key) for key in fields}
    paths = document.get('artifact_paths')
    if not isinstance(paths, Mapping):
        raise ValueError('baseline bundle artifact paths differ')
    payloads = {}
    for role, value in paths.items():
        if not isinstance(value, str) or not value or '\\' in value:
            raise ValueError('baseline artifact path differs')
        relative = PurePosixPath(value)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('baseline artifact path is unsafe')
        artifact = external_file(project_root, str(path.parent / value), 'baseline artifact')
        if path.parent not in artifact.parents:
            raise ValueError('baseline artifact escapes bundle')
        payloads[role] = artifact.read_bytes()
    candidate = candidate_from_identity(identity, payloads)
    return candidate


def resolve_execution_bindings(
    project_root: str | Path, study, bindings_path: str | Path | None
) -> tuple[dict[str, object], ExecutorRevision | None]:
    """Return resolved runtime leaves and the Executor already validated for them."""
    from .runtime import broker_execution_sha256
    from .providers import ProviderQualificationReceipt, resolve_codex_code_mode_host
    from .triton_build import IsolatedTritonCompiler

    document = json.loads(canonical(study.document))
    arms = document['arms']
    if set(arms) != {'open_cake', 'native_triton'}:
        if bindings_path is not None:
            raise ValueError('external execution binding requires paired Triton Study')
        return document, None
    execution = document['execution']
    provider_fields = ('revision', 'executable_sha256', 'qualification', 'qualification_anchor', 'code_mode_host')
    leaves = [arms[name]['provider'].get(field) for name in arms for field in provider_fields]
    leaves += [arms[name].get('toolchain_sha256') for name in arms]
    leaves += [execution.get('broker_execution_sha256'), execution.get('fixed_baseline')]
    markers = [value == CAMPAIGN_BINDING for value in leaves]
    if not any(markers):
        if bindings_path is not None:
            raise ValueError('exact or historical Study rejects execution overrides')
        return document, None
    if study.state != 'template' or not all(markers) or bindings_path is None:
        raise ValueError('complete campaign binding markers require external execution bindings')
    path = external_file(project_root, str(bindings_path), 'execution bindings')
    bindings = json.loads(path.read_bytes())
    if not isinstance(bindings, Mapping) or set(bindings) != {
        'schema_version', 'qualification_path', 'qualification_anchor_path',
        'runtime_config_path', 'fixed_baseline_bundle_path',
    } or type(bindings['schema_version']) is not int or bindings['schema_version'] != 1:
        raise ValueError('external execution binding fields differ')
    receipt_path = external_file(project_root, bindings['qualification_path'], 'qualification')
    anchor_path = external_file(project_root, bindings['qualification_anchor_path'], 'qualification anchor')
    runtime_path = external_file(project_root, bindings['runtime_config_path'], 'runtime configuration')
    receipt = ProviderQualificationReceipt.load(receipt_path)
    anchor = json.loads(anchor_path.read_bytes())
    config = json.loads(runtime_path.read_bytes())
    if (not isinstance(config, Mapping) or set(config) != {'schema_version', 'provider', 'toolchain', 'broker'}
        or type(config['schema_version']) is not int or config['schema_version'] != 1
        or set(config['provider']) != {'executable', 'workspace_root'}
        or set(config['toolchain']) != {'python', 'bubblewrap', 'runtime_roots', 'triton_version', 'timeout_seconds'}
        or set(config['broker']) != {'command', 'cwd', 'timeout_seconds', 'service_user', 'service_group'}):
        raise ValueError('external runtime configuration fields differ')
    executable = Path(config['provider']['executable']).resolve(strict=True)
    if sha256(executable.read_bytes()).hexdigest() != receipt.executable_sha256:
        raise ValueError('runtime provider executable differs from qualification')
    code_mode_host = resolve_codex_code_mode_host(executable)
    for arm in arms.values():
        provider = arm['provider']
        provider.update(revision=receipt.provider_revision, executable_sha256=receipt.executable_sha256,
            code_mode_host=code_mode_host,
            qualification={'path': str(receipt_path), 'canonical_sha256': receipt.canonical_sha256},
            qualification_anchor={'path': str(anchor_path), 'canonical_sha256': sha256(canonical(anchor)).hexdigest()})
    # current_release resolves once, with the existing Executor resolver's closure checks.
    from .core import _resolve_executor_reference
    executor_reference = _resolve_executor_reference(Path(project_root), execution['executor_revision'],
        'study.execution', template=True)
    executor = ExecutorRevision.load(project_root, Path(project_root) / executor_reference['path'])
    if dict(executor.reference) != executor_reference:
        raise ValueError('Executor changed during external execution binding')
    toolchain = IsolatedTritonCompiler(**config['toolchain'])
    toolchain.check_executor(executor, author_workspace=config['provider']['workspace_root'])
    for arm in arms.values():
        arm['toolchain_sha256'] = toolchain.canonical_sha256
    broker = config['broker']
    command = tuple(broker['command']) if isinstance(broker['command'], list) else tuple(shlex.split(broker['command']))
    execution['executor_revision'] = executor_reference
    execution['broker_execution_sha256'] = broker_execution_sha256(command,
        cwd=Path(broker['cwd']).resolve(strict=True), project_root=Path(project_root),
        timeout_seconds=broker['timeout_seconds'], service_user=broker['service_user'], service_group=broker['service_group'])
    baseline = load_baseline_bundle(project_root, bindings['fixed_baseline_bundle_path'])
    execution['fixed_baseline'] = {'bundle_path': str(external_file(project_root,
        bindings['fixed_baseline_bundle_path'], 'fixed baseline bundle')), 'candidate': candidate_identity(baseline)}
    execution['runtime_config'] = {'path': str(runtime_path), 'sha256': sha256(runtime_path.read_bytes()).hexdigest()}
    return document, executor
