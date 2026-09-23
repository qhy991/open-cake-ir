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
from .executor import ExecutorRevision, _relative_file
from open_cake_ir.compiler import Compiler, CorpusGateReport
from open_cake_ir.compiler.revision import load_revision
from ._documents import _object, _project_path, differs
from .pairing import comparison_arm, native_backend
from .runtime_config import broker_execution_sha256, load_runtime_config
from .toolchains import single_environment_toolchain, toolchain_for

CAMPAIGN_BINDING = {'binding': 'campaign_lock'}
CURRENT_RELEASE_BINDING = {'binding': 'current_release'}
COMPILER_REVISION_PATH = "compiler/revision.json"


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
    return _relative_file(Path(project_root), value, context)


def qualification_path(project_root, value, context):
    """Only qualification references admit external files; source paths stay strict."""
    if isinstance(value, str) and Path(value).is_absolute():
        path = external_file(project_root, value, context)
        return str(path), path
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
        raise differs('baseline bundle artifact paths differ', expected='<object>', observed=paths)
    payloads = {}
    for role, value in paths.items():
        if not isinstance(value, str) or not value or '\\' in value:
            raise differs(f'baseline artifact path for role {role!r}', expected='<non-empty posix path>', observed=value)
        relative = PurePosixPath(value)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('baseline artifact path is unsafe')
        artifact = external_file(project_root, str(path.parent / value), 'baseline artifact')
        if path.parent not in artifact.parents:
            raise ValueError('baseline artifact escapes bundle')
        payloads[role] = artifact.read_bytes()
    candidate = candidate_from_identity(identity, payloads)
    return candidate


def load_prepared_baseline(project_root: Path, path: str | Path):
    """Read one pre-provider handoff without reselecting or rebuilding its baseline."""
    from .incumbents import validate_baseline_selection

    document = json.loads(external_file(project_root, str(path), 'prepared baseline').read_bytes())
    if (not isinstance(document, Mapping) or set(document) != {
        'schema_version', 'fixed_baseline_bundle_path', 'fixed_baseline_candidate',
        'fixed_baseline_selection',
    } or type(document.get('schema_version')) is not int or document['schema_version'] != 1):
        raise differs(
            'prepared baseline fields differ',
            expected={'schema_version': 1, 'fields': ['fixed_baseline_bundle_path', 'fixed_baseline_candidate',
                                                     'fixed_baseline_selection', 'schema_version']},
            observed={'schema_version': document.get('schema_version') if isinstance(document, Mapping) else None,
                      'fields': sorted(document) if isinstance(document, Mapping) else document},
        )
    bundle = external_file(project_root, document['fixed_baseline_bundle_path'], 'prepared baseline bundle')
    baseline = load_baseline_bundle(project_root, bundle)
    if candidate_identity(baseline) != document['fixed_baseline_candidate']:
        raise differs(
            'prepared baseline candidate differs from its sealed selection',
            expected=document['fixed_baseline_candidate'], observed=candidate_identity(baseline),
        )
    return bundle, baseline, validate_baseline_selection(document['fixed_baseline_selection'])


def current_executor_reference(root: Path, target: str) -> dict[str, object]:
    """Return the reference of the Executor for one exact target at this checkout."""
    return dict(ExecutorRevision.for_target(root, target).reference)


def resolve_executor(
    root: Path, value: object, context: str, *, template: bool,
    target: str | None = None,
) -> ExecutorRevision:
    """Select the Study's current or frozen reference and return its verified object."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.executor_revision must be an object")
    if template:
        if value != CURRENT_RELEASE_BINDING:
            raise differs("Study template Executor binding", expected=CURRENT_RELEASE_BINDING, observed=value)
        return ExecutorRevision.for_target(root, target)
    if value == CURRENT_RELEASE_BINDING:
        raise ValueError("frozen Study cannot follow the current Executor")
    return ExecutorRevision.load_reference(root, value, f"{context}.executor_revision")


def bind_cli_provider(project_root,provider,row,*,runtime_path,receipt_path,anchor_path):
    """Resolve the CLI provider's existing qualification into one authoring value."""
    from .providers import ProviderQualificationReceipt, resolve_codex_code_mode_host
    from .provider_policy import provider_harness
    harness = provider_harness(provider)
    if harness not in {'codex','claude-code'}:
        raise ValueError('CLI task preparation requires a CLI provider')
    config = load_runtime_config(runtime_path,toolchain_kind=row.runtime_kind)
    receipt = ProviderQualificationReceipt.load(receipt_path)
    anchor = json.loads(Path(anchor_path).read_bytes())
    executable = Path(config['provider']['executable']).resolve(strict=True)
    observed = sha256(executable.read_bytes()).hexdigest()
    if observed != receipt.executable_sha256:
        raise differs('runtime provider executable differs from qualification',
                      expected=receipt.executable_sha256,observed=observed)
    bound = json.loads(canonical(provider))
    bound.update(revision=receipt.provider_revision,executable_sha256=receipt.executable_sha256,
        qualification={'path':str(receipt_path),'canonical_sha256':receipt.canonical_sha256},
        qualification_anchor={'path':str(anchor_path),'canonical_sha256':sha256(canonical(anchor)).hexdigest()})
    if harness=='codex':
        bound['code_mode_host'] = resolve_codex_code_mode_host(executable)
    return bound,config


def bind_runtime_execution(project_root,row,executor,config,runtime_path):
    """Freeze one admitted toolchain and broker, shared by Run and Campaign preparation."""
    toolchain = row.bind(config['toolchain'],executor,author_workspace=config['provider']['workspace_root'])
    broker = config['broker']
    execution = {'executor_revision':dict(executor.reference),
        'broker_execution_sha256':broker_execution_sha256(broker['command'],
            cwd=Path(broker['cwd']).resolve(strict=True),project_root=Path(project_root),
            timeout_seconds=broker['timeout_seconds'],service_user=broker['service_user'],service_group=broker['service_group']),
        'runtime_config':{'path':str(runtime_path),'sha256':sha256(Path(runtime_path).read_bytes()).hexdigest()}}
    return toolchain.canonical_sha256,execution


def bind_fixed_baseline(project_root,path,selection=None):
    baseline = load_baseline_bundle(project_root,path)
    fixed = {'bundle_path':str(external_file(project_root,str(path),'fixed baseline bundle')),
             'candidate':candidate_identity(baseline)}
    if selection is not None:
        from .incumbents import validate_baseline_selection
        fixed['selection'] = validate_baseline_selection(selection)
    return fixed


def resolve_execution_bindings(
    project_root: str | Path, study, bindings_path: str | Path | None
) -> tuple[dict[str, object], ExecutorRevision | None]:
    """Return resolved runtime leaves and the Executor already validated for them."""
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
    version = bindings.get('schema_version') if isinstance(bindings, Mapping) else None
    expected = ({'schema_version', 'qualification_paths', 'qualification_anchor_paths',
                 'runtime_config_path', 'fixed_baseline_bundle_path'}
                if version == 3 else
                {'schema_version', 'qualification_path', 'qualification_anchor_path',
                 'runtime_config_path', 'fixed_baseline_bundle_path'}) | (
                    {'fixed_baseline_selection'} if (version == 2 or
                    (version == 3 and isinstance(bindings, Mapping)
                     and 'fixed_baseline_selection' in bindings)) else set())
    if (not isinstance(bindings, Mapping) or set(bindings) != expected
            or type(version) is not int or version not in {1, 2, 3}):
        raise differs(
            'external execution binding fields differ',
            expected={'schema_version': [1, 2, 3], 'fields': sorted(expected)},
            observed={'schema_version': version,
                      'fields': sorted(bindings) if isinstance(bindings, Mapping) else bindings},
        )
    if version != 3 and any(
        arm['provider'].get('submission_contract')
        != arms['open_cake']['provider'].get('submission_contract')
        for arm in arms.values()
    ):
        raise ValueError('mixed provider transports require per-arm execution bindings v3')
    runtime_path = external_file(project_root, bindings['runtime_config_path'], 'runtime configuration')
    if single:
        route = arms["open_cake"].get("lowering_route")
        if not isinstance(route, Mapping) or "backend" not in route:
            raise differs("single-environment lowering route", expected={"backend": "<name>"}, observed=route)
        row = single_environment_toolchain(route["backend"])
    else:
        row = toolchain_for(policy.backend)
    if version == 3:
        if (single or not isinstance(bindings['qualification_paths'], Mapping)
            or not isinstance(bindings['qualification_anchor_paths'], Mapping)
            or set(bindings['qualification_paths']) != set(arms)
            or set(bindings['qualification_anchor_paths']) != set(arms)):
            raise ValueError('per-arm qualification binding must name each paired arm')
        configurations = []
        for name, arm in arms.items():
            receipt_path = external_file(project_root, bindings['qualification_paths'][name],
                                         f'{name} qualification')
            anchor_path = external_file(project_root, bindings['qualification_anchor_paths'][name],
                                        f'{name} qualification anchor')
            provider, config = bind_cli_provider(project_root, arm['provider'], row,
                runtime_path=runtime_path, receipt_path=receipt_path, anchor_path=anchor_path)
            arm['provider'] = json.loads(canonical(provider))
            configurations.append(config)
        if any(other != configurations[0] for other in configurations[1:]):
            raise ValueError('paired arms use different runtime configurations')
        config = configurations[0]
    else:
        receipt_path = external_file(project_root, bindings['qualification_path'], 'qualification')
        anchor_path = external_file(project_root, bindings['qualification_anchor_path'], 'qualification anchor')
        provider, config = bind_cli_provider(project_root, arms['open_cake']['provider'], row,
            runtime_path=runtime_path, receipt_path=receipt_path, anchor_path=anchor_path)
        for arm in arms.values():
            arm['provider'] = json.loads(canonical(provider))
    executor = resolve_executor(Path(project_root),execution['executor_revision'],
        'study.execution',template=True,target=execution['target'])
    toolchain_sha256,bound_execution = bind_runtime_execution(project_root,row,executor,config,runtime_path)
    for arm in arms.values():
        arm['toolchain_sha256'] = toolchain_sha256
    execution.update(bound_execution)
    execution['fixed_baseline'] = bind_fixed_baseline(project_root,bindings['fixed_baseline_bundle_path'],
        bindings.get('fixed_baseline_selection') if version in {2, 3} else None)
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
            raise differs("Study template Compiler binding", expected=CURRENT_RELEASE_BINDING, observed=reference)
        relative = COMPILER_REVISION_PATH
        path = (root / relative).resolve(strict=True)
    else:
        if reference == CURRENT_RELEASE_BINDING:
            raise ValueError("frozen Study cannot follow the current Compiler")
        if set(reference) != {"path", "revision_id"}:
            raise differs(
                "Compiler Revision reference fields differ",
                expected=["path", "revision_id"], observed=sorted(reference),
            )
        relative, path = _project_path(root, reference["path"], f"{context}.path")

    compiler = Compiler.load(root, path)
    if compiler.commit is None:
        raise ValueError("Study Contract requires a Compiler at a clean committed checkout")
    gate = compiler.check_corpus()
    if not gate.passed:
        raise ValueError("Study Contract requires a Compiler whose full Corpus Gate passes")
    if not template and reference["revision_id"] != gate.compiler_revision_id:
        raise differs(
            "Study Contract Compiler Revision",
            expected=reference["revision_id"], observed=gate.compiler_revision_id,
        )
    exact = {"revision_id": gate.compiler_revision_id, "path": relative}
    return gate, relative, exact


def load_compiler_reference(root: Path, value: object, context: str):
    """Verify a Campaign's exact Compiler dependency at a process handoff.

    This does not run the Corpus Gate again; preflight owns that gate. The boundary
    checks that the checkout is clean at the pinned commit and that the manifest and
    declared Targets are the ones the Campaign bound.
    """
    reference = _object(value, context)
    if set(reference) != {"path", "revision_id"}:
        raise differs(
            f"{context} Compiler reference fields differ",
            expected=["path", "revision_id"], observed=sorted(reference),
        )
    _, path = _project_path(root, reference["path"], f"{context}.path")
    revision = load_revision(root, path)
    if revision.commit is None:
        raise ValueError(f"{context} requires a clean committed checkout")
    if revision.revision_id != reference["revision_id"]:
        raise ValueError(
            f"{context} Compiler Revision differs: the reference pins "
            f"{reference['revision_id']!r} and this checkout provides {revision.revision_id!r}"
        )
    return revision
