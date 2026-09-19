"""Study admission checks grouped by their external authority boundaries."""

from __future__ import annotations

import ast
import json
from hashlib import sha256
from typing import Mapping, cast

from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.toolchain import triton_route
from open_cake_ir.evaluation.paired import (
    METAL_KINDS, candidate_identity, paired_protocol, validate_pair_candidates, validation_case_ids,
)

from ._documents import _canonical_json_bytes, _digest, _name, _object, differs
from ._policies import _ATTRIBUTION_EVALUATION
from .bindings import load_baseline_bundle, qualification_path as _qualification_path, source_reference_path
from .build import _hidden_pointers
from .incumbents import admit_baseline_selection
from .providers import (
    ProviderQualificationReceipt,
    required_live_provider_qualification_scope,
)
from .pairing import native_source, native_block, backend_policy

from .provider_policy import provider_configuration, provider_harness


def validate_provider(*, open_cake, policy, project_root, study):
    """Validate provider qualification and declared authoring capabilities."""
    provider = _object(open_cake.get("provider"), "study.arms.provider")
    provider_revision = _name(provider.get("revision"), "study.arms.provider.revision")
    claim_scope = cast(str, study.document["claim_scope"])
    expected_provider_configuration = provider_configuration(provider, claim_scope, arms=study.document["arms"])
    executable_sha256 = _digest(
        provider.get("executable_sha256"), "study.arms.provider.executable_sha256"
    )
    qualification_ref = _object(
        provider.get("qualification"),
        "study.arms.provider.qualification",
    )
    if set(qualification_ref) != {"path", "canonical_sha256"}:
        raise differs(
            "provider qualification reference fields differ",
            expected=["canonical_sha256", "path"], observed=sorted(qualification_ref),
        )
    _, qualification_path = _qualification_path(
        project_root,
        qualification_ref.get("path"),
        "study.arms.provider.qualification.path",
    )
    qualification = ProviderQualificationReceipt.load(qualification_path)
    expected_configuration_sha256 = sha256(
        _canonical_json_bytes(expected_provider_configuration)).hexdigest()
    admitted_scopes = {
        "zero_gpu_contract_fixture_only",
        required_live_provider_qualification_scope(claim_scope),
    }
    expected_qualification = {
        "provider_revision": provider_revision, "executable_sha256": executable_sha256,
        "configuration_sha256": expected_configuration_sha256,
        "initial_and_resume_equivalent": True, "file_lifecycle_observed": True,
        "usage_observed": True, "qualified": True, "scope": sorted(admitted_scopes),
        "canonical_sha256": qualification_ref.get("canonical_sha256"),
    }
    observed_qualification = {
        "provider_revision": qualification.provider_revision,
        "executable_sha256": qualification.executable_sha256,
        "configuration_sha256": qualification.configuration_sha256,
        "initial_and_resume_equivalent": qualification.initial_and_resume_equivalent,
        "file_lifecycle_observed": qualification.file_lifecycle_observed,
        "usage_observed": qualification.usage_observed, "qualified": qualification.qualified,
        "scope": qualification.scope, "canonical_sha256": qualification.canonical_sha256,
    }
    if (
        qualification.provider_revision != provider_revision
        or qualification.executable_sha256 != executable_sha256
        or qualification.configuration_sha256 != expected_configuration_sha256
        or not qualification.initial_and_resume_equivalent
        or not qualification.file_lifecycle_observed
        or not qualification.usage_observed
        or not qualification.qualified
        or qualification.scope not in admitted_scopes
        or qualification_ref.get("canonical_sha256") != qualification.canonical_sha256
    ):
        raise differs(
            "provider qualification bytes or capability",
            expected=expected_qualification, observed=observed_qualification,
        )
    if (policy is not None and qualification.scope != 'zero_gpu_contract_fixture_only'
        and paired_protocol(study.document['evaluation_protocol']) is None):
        raise ValueError('new live native Campaign requires explicit fixed-baseline paired policy')
    qualification_anchor = provider.get("qualification_anchor")
    if qualification.scope == "zero_gpu_contract_fixture_only":
        if qualification_anchor is not None:
            raise ValueError("fixture provider qualification anchor must be null")
    else:
        anchor_reference = _object(
            qualification_anchor,
            "study.arms.provider.qualification_anchor",
        )
        if set(anchor_reference) != {"path", "canonical_sha256"}:
            raise differs(
                "provider qualification anchor reference",
                expected=["canonical_sha256", "path"], observed=sorted(anchor_reference),
            )
        _, anchor_path = _qualification_path(
            project_root,
            anchor_reference.get("path"),
            "study.arms.provider.qualification_anchor.path",
        )
        anchor = _object(
            json.loads(anchor_path.read_text(encoding="utf-8")),
            "study.arms.provider.qualification_anchor.document",
        )
        anchor_fields = {
            "schema_version", "kind", "run_id", "evidence_root", "authority_sha256",
            "qualification_receipt_sha256", "immediate_audit_integrity", "terminal_seal_sha256",
        }
        if set(anchor) != anchor_fields:
            raise differs(
                "provider qualification anchor fields differ",
                expected=sorted(anchor_fields), observed=sorted(anchor),
            )
        if (
            anchor.get("schema_version") != 1
            or anchor.get("kind")
            != ("provider_qualification_evidence_anchor" if provider_harness(provider) == "claude-code" else "codex_provider_qualification_evidence_anchor")
            or not isinstance(anchor.get("run_id"), str)
            or not anchor["run_id"]
            or not isinstance(anchor.get("evidence_root"), str)
            or not anchor["evidence_root"]
            or _digest(
                anchor.get("authority_sha256"),
                "provider qualification anchor authority",
            )
            != anchor.get("authority_sha256")
            or anchor.get("qualification_receipt_sha256")
            != qualification.canonical_sha256
            or anchor.get("immediate_audit_integrity") is not True
            or _digest(
                anchor.get("terminal_seal_sha256"),
                "provider qualification anchor terminal seal",
            )
            != anchor.get("terminal_seal_sha256")
            or _digest(
                anchor_reference.get("canonical_sha256"),
                "study.arms.provider.qualification_anchor.canonical_sha256",
            )
            != sha256(_canonical_json_bytes(anchor)).hexdigest()
        ):
            raise ValueError("provider qualification anchor evidence differs")
    for field in (("output_schema",) if provider_harness(provider) == "codex" else ()):
        reference = _object(provider.get(field), f"study.arms.provider.{field}")
        if set(reference) != {"path", "sha256"}:
            raise differs(
                f"Study Contract provider {field} reference",
                expected=["path", "sha256"], observed=sorted(reference),
            )
        _, path = source_reference_path(
            project_root, reference.get("path"), f"study.arms.provider.{field}.path"
        )
        expected_sha256 = _digest(reference.get("sha256"), f"study.arms.provider.{field}.sha256")
        observed_sha256 = sha256(path.read_bytes()).hexdigest()
        if expected_sha256 != observed_sha256:
            raise differs(
                f"Study Contract provider {field} bytes differ",
                expected=expected_sha256, observed=observed_sha256,
            )
    return claim_scope


def validate_evaluation(
    *,
    attribution_evaluation,
    baseline_lowering,
    manifest_parser,
    policy,
    project_root,
    skeleton_document,
    study,
    workload,
):
    """Validate the numerical assay against the Workload and its execution admission.

    What the assay says by its own bytes (searches, materiality, Ralph limits) is
    `StudyContract.load`'s; this is the half that needs the Workload, the sealed
    baseline and the checkout.
    """
    evaluation = study.evaluation_protocol
    workload.case(_name(evaluation.get("case_id"), "study.evaluation_protocol.case_id"))
    assay = paired_protocol(evaluation)
    single_environment = study.comparison is None
    route = study.arms["open_cake"]["lowering_route"]
    if assay is not None and policy is None and not single_environment:
        raise ValueError('fixed-baseline assay requires the same-backend native Study')
    if single_environment and assay is None:
        # A single-arm optimization campaign selects on getting faster, so with no timed
        # assay there is nothing to select on. Say which of the two reasons applies: a
        # Study that declares a measurement-coverage limitation is not misconfigured, it
        # is running against a target no timer has been named for, and reporting that as
        # a missing assay sends the reader to fix a configuration that is already correct.
        coverage = evaluation.get("measurement_coverage")
        if isinstance(coverage, Mapping) and coverage.get("timed_assay") == "unavailable":
            raise ValueError(
                "single-environment optimization requires a timed assay, and this Study "
                f"reports none is available: {coverage.get('reason')}. Correctness "
                "evaluation of a sealed candidate for this target does not need one; a "
                "campaign that selects on latency does."
            )
        raise ValueError("single-environment optimization requires an explicit fixed-baseline paired assay")
    if route["backend"] == "metal":
        if (not single_environment or evaluation.get("paired_timing", {}).get("kind") not in METAL_KINDS
                or validation_case_ids(evaluation) != tuple(workload.case_ids)
                or attribution_evaluation != _ATTRIBUTION_EVALUATION):
            raise ValueError("Metal optimization must bind its paired assay, all Workload cases and attribution")
    elif evaluation.get("paired_timing", {}).get("kind") in METAL_KINDS:
        raise ValueError("Metal paired assay cannot evaluate a different backend")
    elif ("validation_case_ids" in evaluation
          or single_environment and workload.document["validation"].get("all_cases_required") is True):
        if (validation_case_ids(evaluation) != tuple(workload.case_ids)
                or workload.document["validation"].get("all_cases_required") is not True):
            raise differs(
                "CUDA validation case projection differs from Workload validation",
                expected={"validation_case_ids": tuple(workload.case_ids), "all_cases_required": True},
                observed={"validation_case_ids": validation_case_ids(evaluation),
                          "all_cases_required": workload.document["validation"].get("all_cases_required")},
            )
    execution = study.execution
    expected_execution_fields = {'target', 'executor_revision', 'broker_execution_sha256', 'gpu', 'sandbox'}
    if assay is not None:
        expected_execution_fields.update({'fixed_baseline', 'runtime_config'})
    if set(execution) != expected_execution_fields:
        raise differs(
            "Study Contract execution fields differ",
            expected=sorted(expected_execution_fields), observed=sorted(execution),
        )
    if assay is not None:
        fixed = _object(execution['fixed_baseline'], 'execution.fixed_baseline')
        sealed_baseline = load_baseline_bundle(project_root, fixed['bundle_path'])
        validate_pair_candidates(sealed_baseline, sealed_baseline, workload, str(evaluation['case_id']))
        selection = fixed.get('selection')
        incumbent_baseline = False
        if selection is not None:
            incumbent_baseline = admit_baseline_selection(
                selection, candidate=fixed['candidate'], workload=workload,
                case_id=str(evaluation['case_id']), backend=str(route['backend']),
                evaluation_protocol=evaluation,
            )
        requirements = baseline_lowering.toolchain_requirements
        source = sealed_baseline.artifact_payloads.get('lowered_source')
        if source is None:
            raise ValueError('fixed baseline requires retained Compiler lowering source')
        manifest = manifest_parser(json.loads(sealed_baseline.artifact_payloads['launch_manifest']))
        if route["backend"] == "metal":
            source_matches = source == baseline_lowering.source.encode()
            grid, block = requirements["threadgroups_per_grid"], tuple(requirements["threads_per_threadgroup"])
        else:
            expected_source = native_source(baseline_lowering.source.encode(), requirements)
            observed_source = native_source(source, requirements)
            source_matches = ast.dump(ast.parse(observed_source)) == ast.dump(ast.parse(expected_source))
            # The width comes from the Target that declares it. Reading a shared 32 here
            # refused every wave64 baseline, and said the Compiler kernel differed.
            _, target_path = source_reference_path(
                project_root, f"compiler/targets/{workload.target}.json",
                'paired baseline target')
            grid = requirements['grid']
            block = tuple(native_block(
                requirements, warp_size=Target.load(target_path).warp_size))
            # How many pointers the kernel takes beyond its tensors is the kernel's own
            # fact, not a per-backend constant. For AMDGCN it is in the sealed assembly's
            # `.amdgpu_metadata`; the table's 2 was right for every Triton target there
            # was when it was written, and is still the CUDA route's. The route is read
            # from the frozen Compiler kernel's own compile contract, which carries the
            # code object, architecture and lane width of the Target it was lowered for.
            if route["backend"] == "triton":
                expected_hidden = _hidden_pointers(
                    triton_route(requirements), sealed_baseline.artifact_payloads,
                    len(workload.tensor_abi(str(evaluation['case_id']))))
            else:
                expected_hidden = backend_policy(route["backend"]).hidden_null_pointer_parameters
            if manifest.hidden_null_pointer_parameters != expected_hidden:
                raise differs(
                    'fixed baseline hidden pointer commitments differ',
                    expected=expected_hidden, observed=manifest.hidden_null_pointer_parameters,
                )
        reference_differs = (not source_matches or list(manifest.grid) != list(grid)
                             or manifest.block != block)
        if (fixed['candidate'] != candidate_identity(sealed_baseline)
                or (not incumbent_baseline and reference_differs)):
            raise differs(
                'fixed baseline differs from the frozen Compiler kernel or launch commitments',
                expected={'candidate': fixed['candidate'], 'source_matches': True,
                          'grid': list(grid), 'block': block},
                observed={'candidate': candidate_identity(sealed_baseline), 'source_matches': source_matches,
                          'grid': list(manifest.grid), 'block': manifest.block},
            )
    provider_sandbox = study.arms["open_cake"]["provider"].get("sandbox")
    if (
        execution.get("target") != workload.target
        or execution.get("target") != skeleton_document.get("target")
        or execution.get("sandbox") != provider_sandbox
    ):
        raise differs(
            "Study Contract execution authority",
            expected={"target": workload.target, "skeleton_target": workload.target,
                      "sandbox": provider_sandbox},
            observed={"target": execution.get("target"),
                      "skeleton_target": skeleton_document.get("target"),
                      "sandbox": execution.get("sandbox")},
        )
    _digest(
        execution.get("broker_execution_sha256"),
        "study.execution.broker_execution_sha256",
    )
    return evaluation, execution
