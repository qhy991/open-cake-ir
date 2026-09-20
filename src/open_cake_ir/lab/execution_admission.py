"""Validate live dependencies against a resolved Campaign Lock."""

from __future__ import annotations

from hashlib import sha256
import json

from open_cake_ir.evaluation.paired import paired_protocol

from ._documents import _canonical_json_bytes, _digest, _name, _object
from .bindings import qualification_path as _qualification_path
from .pairing import comparison_arm, native_backend
from .providers import (
    CODEX_DISABLED_FEATURES,
    ProviderQualificationReceipt,
)

from .provider_policy import provider_configuration


def campaign_provider_binding(lock,project_root):
    """Retain matched-Campaign policy at its external input boundary."""
    arms = lock.document['resolved_inputs']['arm_environments']
    evaluation_protocol = lock.document['evaluation_protocol']
    first_arm = _object(arms["open_cake"], "arm_environments.open_cake")
    provider_document = _object(
        first_arm["provider"], "arm_environments.open_cake.provider"
    )
    qualification_ref = _object(
        provider_document["qualification"],
        "arm_environments.open_cake.provider_qualification",
    )
    _, qualification_path = _qualification_path(
        project_root,
        qualification_ref["path"],
        "arm_environments.open_cake.provider_qualification.path",
    )
    qualification = ProviderQualificationReceipt.load(qualification_path)
    if (native_backend(comparison_arm(arms)) is not None and qualification.scope != 'zero_gpu_contract_fixture_only'
        and (paired_protocol(evaluation_protocol) is None
             or provider_document['disabled_features'] != list(CODEX_DISABLED_FEATURES))):
        raise ValueError('new live native execution requires paired policy and current closed provider surface')
    configuration = provider_configuration(provider_document,lock.claim_scope,arms=arms)
    return provider_document,qualification_ref,qualification,configuration


def validate_execution_bindings(
    *,
    lock,
    resolved_inputs,
    evaluation_protocol,
    expected_protocol_sha256,
    provider,
    evaluator,
    environments,
    project_root,
    workload_loader,
    validate_authoring,
):
    """Check live provider/evaluator/environment identity before creating evidence."""
    if (
        getattr(evaluator, "protocol", None) != evaluation_protocol
        or getattr(evaluator, "protocol_sha256", None) != expected_protocol_sha256
    ):
        raise ValueError("Run Evaluator does not match the Campaign Lock")
    arms = _object(resolved_inputs["arm_environments"], "resolved_inputs.arm_environments")
    validate_authoring(workload_loader(project_root / str(lock.document["workload"]["path"])), arms)
    arm_hashes = _object(
        resolved_inputs["arm_environment_sha256"],
        "resolved_inputs.arm_environment_sha256",
    )
    provider_documents = {
        name: _object(
            _object(arms[name], f"arm_environments.{name}").get("provider"),
            f"arm_environments.{name}.provider",
        )
        for name in environments
    }
    provider_revisions = {
        _name(value.get("revision"), f"arm_environments.{name}.provider.revision")
        for name, value in provider_documents.items()
    }
    qualification_digests = {
        _digest(
            _object(
                value.get("qualification"),
                f"arm_environments.{name}.provider.qualification",
            ).get("canonical_sha256"),
            f"arm_environments.{name}.provider.qualification.sha256",
        )
        for name, value in provider_documents.items()
    }
    if (
        provider_revisions != {getattr(provider, "provider_revision", None)}
        or qualification_digests != {getattr(provider, "qualification_sha256", None)}
    ):
        raise ValueError("Run Provider does not match the Campaign Lock")
    provider_document,qualification_ref,qualification,expected_provider_configuration = campaign_provider_binding(lock,project_root)
    if getattr(provider, "executable_sha256", None) != qualification.executable_sha256:
        raise ValueError("Run Provider executable does not match its qualification")
    if (
        getattr(provider, "configuration", None) != expected_provider_configuration
        or qualification.canonical_sha256
        != qualification_ref["canonical_sha256"]
        or qualification.configuration_sha256
        != sha256(
            _canonical_json_bytes(expected_provider_configuration)
        ).hexdigest()
    ):
        raise ValueError("Run Provider configuration does not match the Campaign Lock")
    for name, environment in environments.items():
        if (
            getattr(environment, "authority_document", None) != arms[name]
            or getattr(environment, "canonical_sha256", None) != arm_hashes[name]
        ):
            raise ValueError(f"{name} Authoring Environment does not match the Campaign Lock")
    return arms, provider_document


def validate_run_bindings(specification, *, project_root, workload_loader, provider, environment, evaluator,
                          task_package):
    """Resolve the live closure for one Run before evidence or process side effects."""
    from .bindings import source_reference_path
    from .provider_policy import execution_configuration
    from .admission import admit_run_inputs

    document = specification.document
    protocol = document['evaluation_protocol']
    if (getattr(evaluator, 'protocol', None) != protocol
        or getattr(evaluator, 'protocol_sha256', None) != sha256(_canonical_json_bytes(protocol)).hexdigest()):
        raise ValueError('Run Evaluator differs from its frozen protocol')
    workload, qualification = admit_run_inputs(specification, project_root=project_root,
                                               workload_loader=workload_loader)
    authority = document['authoring']
    if (getattr(environment, 'authority_document', None) != authority
        or getattr(environment, 'canonical_sha256', None) != sha256(_canonical_json_bytes(authority)).hexdigest()):
        raise ValueError('Run Authoring Environment differs from its frozen authority')
    declared = authority['provider']
    reference = declared['qualification']
    configuration = execution_configuration(declared)
    message_provider = declared.get('harness') == 'responses'
    if message_provider:
        from .message_provider import ResponsesRunProvider
        if type(provider) is not ResponsesRunProvider:
            raise ValueError('message-only authoring requires the owned Responses provider')
        provider.validate_transport(qualification)
        provider.validate_task_package(task_package(specification, specification.run_id))
    if (qualification.canonical_sha256 != reference['canonical_sha256']
        or qualification.provider_revision != declared['revision']
        or qualification.configuration_sha256 != sha256(_canonical_json_bytes(configuration)).hexdigest()
        or provider.provider_revision != declared['revision']
        or provider.qualification_sha256 != qualification.canonical_sha256
        or (not message_provider and provider.executable_sha256 != qualification.executable_sha256)
        or provider.configuration != configuration):
        raise ValueError('Run Provider differs from its frozen qualification and configuration')
    return workload
