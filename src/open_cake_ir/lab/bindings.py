"""External execution locators resolved once into the Campaign Lock.

Study policy remains stable. Only explicit campaign-owned runtime leaves can be
filled; source references and provider treatment are never refreshed here.
"""
from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes

import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping

from open_cake_ir.evaluation.paired import candidate_from_identity, candidate_identity
from .executor import ExecutorRevision
from open_cake_ir.compiler import Compiler, CorpusGateReport
from ._documents import _object, _digest, _project_path

CAMPAIGN_BINDING = {'binding': 'campaign_lock'}
CURRENT_RELEASE_BINDING = {'binding': 'current_release'}


def canonical(value):
    return canonical_json_bytes(value)


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


def source_reference_path(project_root, value, context):
    """Read a frozen task input from source or an explicit external publication.

    Absolute aliases back into any checkout are refused. Compiler revisions still
    use their own source-only admission; this path is for task inputs and schema.
    """
    if isinstance(value, str) and Path(value).is_absolute():
        path = external_file(project_root, value, context)
        return str(path), path
    from .executor import _relative_file
    return _relative_file(Path(project_root), value, context)


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


def current_executor_reference(root: Path, target: str) -> dict[str, object]:
    """Return the one current Executor owned by an exact hardware target."""
    if not isinstance(target, str) or not target:
        raise ValueError("current Executor resolution requires an exact target")
    inventory = json.loads((root / "inventory/EXECUTOR_REVISIONS.json").read_text())
    if not isinstance(inventory, Mapping) or inventory.get("schema_version") != 2:
        raise ValueError("Executor inventory schema differs")
    currents = inventory.get("current_by_target")
    if not isinstance(currents, Mapping):
        raise ValueError("Executor inventory current_by_target must be an object")
    current = currents.get(target)
    if not isinstance(current, Mapping):
        raise ValueError(f"no current Executor is published for exact target {target!r}")
    fields = ("executor_id", "path", "canonical_sha256")
    try:
        return {field: current[field] for field in fields}
    except KeyError as error:
        raise ValueError(f"current Executor for exact target {target!r} differs") from error


def resolve_executor(
    root: Path, value: object, context: str, *, template: bool,
    target: str | None = None,
) -> ExecutorRevision:
    """Select the Study's current or frozen reference and return its verified object."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.executor_revision must be an object")
    if template:
        if value != CURRENT_RELEASE_BINDING:
            raise ValueError("Study template Executor binding differs")
        exact = current_executor_reference(root, target)
    else:
        if value == CURRENT_RELEASE_BINDING:
            raise ValueError("frozen Study cannot follow the current Executor")
        exact = value
    return ExecutorRevision.load_reference(root, exact, f"{context}.executor_revision")


def resolve_execution_bindings(
    project_root: str | Path, study, bindings_path: str | Path | None
) -> tuple[dict[str, object], ExecutorRevision | None]:
    """Return resolved runtime leaves and the Executor already validated for them."""
    from .runtime_config import broker_execution_sha256, load_runtime_config
    from .providers import ProviderQualificationReceipt, resolve_codex_code_mode_host
    from .pairing import comparison_arm, native_backend, backend_policy

    document = json.loads(canonical(study.document))
    arms = document['arms']
    single = set(arms) == {"open_cake"} and document["claim_scope"] == "artifact_optimization_only"
    policy = native_backend(comparison_arm(arms))
    if policy is None and not single:
        if bindings_path is not None:
            raise ValueError('external execution binding requires a same-backend native Study')
        return document, None
    execution = document['execution']
    from .provider_policy import provider_harness
    harness = provider_harness(arms['open_cake']['provider'])
    provider_fields = ('revision', 'executable_sha256', 'qualification', 'qualification_anchor') + (() if harness == 'claude-code' else ('code_mode_host',))
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
    if single:
        route = arms["open_cake"].get("lowering_route")
        if not isinstance(route, Mapping) or route.get("backend") not in {"metal", "triton"}:
            raise ValueError("single-environment lowering route differs")
        backend = route["backend"]
    else:
        backend = policy.backend
    config = load_runtime_config(runtime_path, toolchain_kind=backend)
    executable = Path(config['provider']['executable']).resolve(strict=True)
    if sha256(executable.read_bytes()).hexdigest() != receipt.executable_sha256:
        raise ValueError('runtime provider executable differs from qualification')
    code_mode_host = resolve_codex_code_mode_host(executable) if harness == "codex" else None
    for arm in arms.values():
        provider = arm['provider']
        provider.update(revision=receipt.provider_revision, executable_sha256=receipt.executable_sha256,
            qualification={'path': str(receipt_path), 'canonical_sha256': receipt.canonical_sha256},
            qualification_anchor={'path': str(anchor_path), 'canonical_sha256': sha256(canonical(anchor)).hexdigest()})
        if code_mode_host is not None:
            provider['code_mode_host'] = code_mode_host
    # current_release resolves once, with the existing Executor resolver's closure checks.
    executor = resolve_executor(Path(project_root), execution['executor_revision'],
        'study.execution', template=True, target=execution['target'])
    executor_reference = dict(executor.reference)
    if backend == "metal":
        from .metal_build import MetalArchiveHost
        toolchain = MetalArchiveHost.from_executor(executor)
    else:
        toolchain = backend_policy(backend).isolated_compiler(config['toolchain'])
        toolchain.check_executor(executor, author_workspace=config['provider']['workspace_root'])
    for arm in arms.values():
        arm['toolchain_sha256'] = toolchain.canonical_sha256
    broker = config['broker']
    command = broker['command']
    execution['executor_revision'] = executor_reference
    execution['broker_execution_sha256'] = broker_execution_sha256(command,
        cwd=Path(broker['cwd']).resolve(strict=True), project_root=Path(project_root),
        timeout_seconds=broker['timeout_seconds'], service_user=broker['service_user'], service_group=broker['service_group'])
    baseline = load_baseline_bundle(project_root, bindings['fixed_baseline_bundle_path'])
    execution['fixed_baseline'] = {'bundle_path': str(external_file(project_root,
        bindings['fixed_baseline_bundle_path'], 'fixed baseline bundle')), 'candidate': candidate_identity(baseline)}
    execution['runtime_config'] = {'path': str(runtime_path), 'sha256': sha256(runtime_path.read_bytes()).hexdigest()}
    return document, executor


def _resolve_compiler_reference(
    root: Path,
    value: object,
    context: str,
    *,
    template: bool,
) -> tuple[CorpusGateReport, str, dict[str, object]]:
    """Resolve a template binding or verify one frozen Compiler reference."""

    reference = _object(value, context)
    if template:
        if reference != CURRENT_RELEASE_BINDING:
            raise ValueError("Study template Compiler binding differs")
        relative = "compiler/revision.lock.json"
        path = (root / relative).resolve(strict=True)
    else:
        if reference == CURRENT_RELEASE_BINDING:
            raise ValueError("frozen Study cannot follow the current Compiler")
        if set(reference) not in (
            {"path", "canonical_sha256"},
            {"path", "canonical_sha256", "revision_id"},
        ):
            raise ValueError("Compiler Revision reference fields differ")
        relative, path = _project_path(root, reference["path"], f"{context}.path")

    compiler = Compiler.load(root, path)
    gate = compiler.check_corpus()
    if compiler.state != "released" or not gate.passed:
        raise ValueError("Study Contract requires a released gated Compiler Revision")
    if not template and (
        gate.compiler_revision_sha256
        != _digest(reference["canonical_sha256"], f"{context}.canonical_sha256")
        or (
            "revision_id" in reference
            and reference["revision_id"] != gate.compiler_revision_id
        )
    ):
        raise ValueError("Study Contract Compiler Revision differs")
    exact = {
        "revision_id": gate.compiler_revision_id,
        "path": relative,
        "canonical_sha256": gate.compiler_revision_sha256,
    }
    return gate, relative, exact


def load_compiler_reference(root: Path, value: object, context: str):
    """Verify a Campaign's exact Compiler dependency at a process handoff.

    Compiler owns its transitive sources and targets. This does not run the Corpus
    Gate again; release/preflight owns that gate, while this boundary verifies the
    released manifest and its source closure before runtime interpretation.
    """
    from open_cake_ir.compiler.revision import load_revision
    reference = _object(value, context)
    if set(reference) != {"path", "canonical_sha256", "revision_id"}:
        raise ValueError(f"{context} Compiler reference fields differ")
    _, path = _project_path(root, reference["path"], f"{context}.path")
    revision = load_revision(root, path)
    if (revision.state != "released" or revision.revision_id != reference["revision_id"]
            or revision.canonical_sha256 != _digest(reference["canonical_sha256"], context)):
        raise ValueError(f"{context} Compiler Revision differs")
    return revision
