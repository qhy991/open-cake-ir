#!/usr/bin/env python3
"""Validate the hardware-informed design stage without network access."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence, cast

from validate_abstraction_extraction import (
    AbstractionExtractionValidationError,
    validate_abstraction_extraction,
)
from validate_operator_library import OperatorLibraryValidationError


_ROOT_FILES = frozenset({"README.md", "schema.json", "manifest.json"})
_ROOT_DIRECTORIES = frozenset({"evidence", "proposals", "review_requests"})
_SCHEMA_ROOT_KEYS = frozenset({"$schema", "$id", "oneOf", "$defs"})
_SCHEMA_DEFINITION_NAMES = frozenset(
    {
        "manifest",
        "paper_stage",
        "input_candidate_closure",
        "counts",
        "evidence",
        "pinned_pdf_source",
        "unverified_web_source",
        "fact",
        "local_probe_observation",
        "local_probe_repository_snapshot",
        "pipeline_observation",
        "hierarchical_reduction_probe_observation",
        "hierarchical_probe_repository_snapshot",
        "hierarchical_probe_retained_artifact",
        "hierarchical_probe_normalized_output",
        "hierarchical_probe_device",
        "hierarchical_probe_pipeline",
        "hierarchical_probe_coverage",
        "hierarchical_probe_oracle",
        "hierarchical_probe_case_result",
        "hierarchical_probe_aggregate",
        "local_falsifier_outcome",
        "non_evidentiary_context",
        "non_evidentiary_repository_snapshot",
        "review_request",
        "review_stage_binding",
        "review_manifest_binding",
        "review_evidence_binding",
        "review_proposal_binding",
        "review_closure",
        "review_closure_counts",
        "review_item",
        "phase_order_ambiguity",
        "preimplementation_standalone_metal_prototypes_option",
        "amend_gate_after_bounded_implementation_option",
        "absent_decision_artifact",
        "external_human_decision_contract",
        "per_item_verdict_contract",
        "ambiguity_resolution_contract",
        "review_authorizations",
        "hardware_review_decision",
        "review_decision_request_binding",
        "reviewer_attestation",
        "review_item_verdict",
        "review_ambiguity_resolution",
        "review_scope_acknowledgements",
        "proposal",
        "candidate_binding",
        "evidence_binding",
        "design",
        "width_policy",
        "resource_derivation",
        "dynamic_allocation_formula",
        "variant_resource_requirements",
        "rms_norm_resource_requirement",
        "layer_norm_resource_requirement",
        "reduction_sequence",
        "scratch_lifecycle",
        "final_reducer_variants",
        "reducer_variant",
        "numeric_contract",
        "decision",
        "hypothesis",
        "hypothesis_falsifier",
        "observed_scope_finding",
        "falsifier",
        "gate_separation",
        "validation_gate",
        "readiness_gate",
        "state",
        "claim_scope",
        "disposition",
        "next_gate",
        "safe_path",
        "https_url",
        "id",
        "id_list",
        "git_commit",
        "human_review_text",
        "sha256",
        "nonempty_strings",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "design_stage_id",
        "paper_stage",
        "paper_source_path",
        "state",
        "claim_scope",
        "disposition",
        "next_gate",
        "input_candidate_closure",
        "evidence",
        "proposals",
        "review_requests",
        "gate_separation",
        "counts",
        "non_claims",
    }
)
_PAPER_STAGE_KEYS = frozenset(
    {"paper_revision", "ordinal", "name", "predecessor", "successor"}
)
_INPUT_CLOSURE_KEYS = frozenset(
    {
        "candidate_id",
        "candidate_path",
        "observation_paths",
        "canonicalization_contract",
        "candidate_closure_sha256",
    }
)
_COUNT_KEYS = frozenset(
    {
        "evidence_count",
        "proposal_count",
        "review_request_count",
        "candidate_count",
        "observation_count",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "evidence_id",
        "state",
        "claim_scope",
        "disposition",
        "next_gate",
        "sources",
        "facts",
        "local_probe_observation",
        "hierarchical_reduction_probe_observation",
        "non_evidentiary_context",
        "assumptions",
        "gate_separation",
        "non_claims",
    }
)
_PINNED_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "kind",
        "title",
        "url",
        "revision",
        "locators",
        "content_sha256",
        "content_verification",
    }
)
_WEB_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "kind",
        "title",
        "url",
        "locators",
        "offline_content_verification",
    }
)
_FACT_KEYS = frozenset({"fact_id", "statement", "scope", "authority", "source_ids"})
_LOCAL_PROBE_KEYS = frozenset(
    {
        "observation_id",
        "observed_at",
        "repository_snapshot",
        "device_class",
        "supports_apple9_or_newer",
        "maximum_threadgroup_memory_bytes",
        "pipelines",
        "raw_output_retained",
        "performance_measured",
        "scientific_claim_authorized",
        "scope",
    }
)
_LOCAL_PROBE_REPOSITORY_SNAPSHOT_KEYS = frozenset(
    {"git_parent_revision", "probe_path", "binding_scope", "reproducibility_note"}
)
_PIPELINE_KEYS = frozenset(
    {"kind", "thread_execution_width", "max_total_threads_per_threadgroup"}
)
_HIERARCHICAL_PROBE_KEYS = frozenset(
    {
        "observation_id",
        "observed_at",
        "repository_snapshot",
        "retained_artifact",
        "probe_scope",
        "normalized_output",
        "falsifier_outcomes",
        "normalized_probe_output_retained",
        "raw_runner_output_retained",
        "stable_device_identifiers_retained",
        "external_workload_oracle_used",
        "performance_measured",
        "evaluation_evidence",
        "scientific_claim_authorized",
        "compiler_change_authorized",
        "human_hardware_review_cleared",
        "promotion_authorized",
        "scope",
    }
)
_HIERARCHICAL_PROBE_REPOSITORY_SNAPSHOT_KEYS = frozenset(
    {
        "repository_revision",
        "worktree_clean_at_observation",
        "source_paths",
        "binding_scope",
    }
)
_HIERARCHICAL_PROBE_RETAINED_ARTIFACT_KEYS = frozenset(
    {
        "storage_scope",
        "relative_path",
        "artifact_kind",
        "reference_status",
        "offline_validator_access",
    }
)
_HIERARCHICAL_PROBE_NORMALIZED_OUTPUT_KEYS = frozenset(
    {"status", "device", "pipeline", "coverage", "oracle", "case_results", "aggregate"}
)
_HIERARCHICAL_PROBE_DEVICE_KEYS = frozenset(
    {
        "device_class",
        "supports_apple9_or_newer",
        "max_threadgroup_memory_bytes",
    }
)
_HIERARCHICAL_PROBE_PIPELINE_KEYS = frozenset(
    {
        "entry_point",
        "thread_execution_width",
        "max_total_threads_per_threadgroup",
        "static_threadgroup_memory_bytes",
        "dynamic_threadgroup_memory_bytes",
    }
)
_HIERARCHICAL_PROBE_COVERAGE_KEYS = frozenset(
    {
        "simdgroup_counts",
        "threads_per_threadgroup",
        "epochs_per_dispatch",
        "input_patterns",
        "dirty_sentinels",
        "single_owner_final_scalar_broadcast",
        "scratch_reused_within_dispatch",
        "uniform_reuse_barrier",
        "layer_all_simdgroups_tested",
        "bfloat_input_conversion_tested",
    }
)
_HIERARCHICAL_PROBE_ORACLE_KEYS = frozenset({"dtype", "tolerance_rule", "kind"})
_HIERARCHICAL_PROBE_CASE_RESULT_KEYS = frozenset(
    {
        "simdgroup_count",
        "threads_per_threadgroup",
        "expected_owner_sums",
        "observed_owner_sums",
        "broadcast_thread_counts",
        "dirty_sentinel_preserved",
        "owner_mismatch_count",
        "broadcast_mismatch_count",
        "dirty_snapshot_mismatch_count",
        "command_buffer_status",
        "passed",
    }
)
_HIERARCHICAL_PROBE_AGGREGATE_KEYS = frozenset(
    {
        "case_count",
        "epoch_result_count",
        "broadcast_consumer_check_count",
        "owner_mismatch_count",
        "broadcast_mismatch_count",
        "dirty_snapshot_mismatch_count",
        "all_command_buffers_completed",
        "all_cases_passed",
    }
)
_LOCAL_FALSIFIER_OUTCOME_KEYS = frozenset({"falsifier_id", "outcome", "basis"})
_TARGET_CONTEXT_KEYS = frozenset(
    {
        "role",
        "repository_snapshot",
        "target_id",
        "target_execution_group_width",
        "target_maximum_threads_per_workgroup",
        "target_maximum_threadgroup_memory_bytes",
        "current_lowering_supports_threadgroup_allocation",
        "current_lowering_supports_threadgroup_barrier",
        "context_note",
    }
)
_NON_EVIDENTIARY_REPOSITORY_SNAPSHOT_KEYS = frozenset(
    {"git_parent_revision", "target_path", "lowering_path", "binding_scope"}
)
_REVIEW_REQUEST_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "review_request_id",
        "state",
        "authority",
        "stage_binding",
        "closure",
        "review_items",
        "phase_order_ambiguity",
        "decision_artifact",
        "external_human_decision_contract",
        "authorizations",
        "non_claims",
    }
)
_REVIEW_STAGE_BINDING_KEYS = frozenset({"manifest", "evidence", "proposal"})
_REVIEW_MANIFEST_BINDING_KEYS = frozenset({"design_stage_id", "path"})
_REVIEW_EVIDENCE_BINDING_KEYS = frozenset({"evidence_id", "path"})
_REVIEW_PROPOSAL_BINDING_KEYS = frozenset({"proposal_id", "path"})
_REVIEW_CLOSURE_KEYS = frozenset(
    {
        "fact_ids",
        "decision_ids",
        "hypothesis_ids",
        "observed_scope_finding_ids",
        "falsifier_ids",
        "counts",
    }
)
_REVIEW_CLOSURE_COUNT_KEYS = frozenset(
    {
        "fact_count",
        "decision_count",
        "hypothesis_count",
        "observed_scope_finding_count",
        "falsifier_count",
    }
)
_REVIEW_ITEM_KEYS = frozenset(
    {
        "review_item_id",
        "question",
        "fact_ids",
        "decision_ids",
        "hypothesis_ids",
        "observed_scope_finding_ids",
        "falsifier_ids",
        "required_human_judgment",
    }
)
_PHASE_ORDER_AMBIGUITY_KEYS = frozenset(
    {
        "ambiguity_id",
        "status",
        "contradiction",
        "resolution_options",
        "selection_required",
        "automation_may_select",
    }
)
_PHASE_ORDER_RESOLUTION_OPTION_KEYS = frozenset({"option_id", "action", "consequence"})
_ABSENT_DECISION_ARTIFACT_KEYS = frozenset({"status", "path", "decision_id"})
_EXTERNAL_HUMAN_DECISION_CONTRACT_KEYS = frozenset(
    {
        "decision_authority",
        "global_dispositions",
        "required_review_item_ids",
        "per_item_verdict_contract",
        "reviewed_git_revision_required",
        "ambiguity_resolution_contract",
        "stage_transition_rule",
        "automation_may_write_decision",
    }
)
_PER_ITEM_VERDICT_CONTRACT_KEYS = frozenset(
    {
        "per_item_verdict_required",
        "verdict_values",
        "global_disposition_aggregation",
        "verdict_severity_order",
        "localized_reason_required",
        "stage_clear_requires_all_item_verdicts_approved",
        "non_approved_item_blocks_stage_transition",
    }
)
_AMBIGUITY_RESOLUTION_CONTRACT_KEYS = frozenset(
    {
        "required_ambiguity_id",
        "allowed_option_ids",
        "localized_reason_required",
        "approval_compatible_option_id",
        "successor_required_option_id",
        "successor_required_option_allowed_global_dispositions",
        "successor_required_option_required_non_approved_review_item_id",
    }
)
_REVIEW_AUTHORIZATION_KEYS = frozenset(
    {
        "human_hardware_review_cleared",
        "principle_driven_iteration_authorized",
        "compiler_change_authorized",
        "implementation_authorized",
        "evaluation_authorized",
        "performance_claim_authorized",
        "scientific_claim_authorized",
        "promotion_authorized",
    }
)
_HARDWARE_REVIEW_DECISION_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "decision_id",
        "kind",
        "authority",
        "binding",
        "reviewer_attestation",
        "global_disposition",
        "item_verdicts",
        "ambiguity_resolution",
        "scope_acknowledgements",
    }
)
_REVIEW_DECISION_REQUEST_BINDING_KEYS = frozenset(
    {
        "design_stage_id",
        "request_id",
        "path",
        "reviewed_git_revision",
        "reviewed_artifact_paths",
    }
)
_REVIEWER_ATTESTATION_KEYS = frozenset(
    {"reviewer_identity", "authorship_attestation", "decision_basis"}
)
_REVIEW_ITEM_VERDICT_KEYS = frozenset({"review_item_id", "verdict", "localized_reason"})
_REVIEW_AMBIGUITY_RESOLUTION_KEYS = frozenset(
    {"ambiguity_id", "selected_option_id", "localized_reason"}
)
_REVIEW_SCOPE_ACKNOWLEDGEMENT_KEYS = frozenset(
    {
        "approval_scope",
        "principle_driven_iteration_required",
        "resource_accounting_hypothesis_resolved",
        "compiler_change_authorized",
        "implementation_authorized",
        "evaluation_authorized",
        "performance_claim_authorized",
        "scientific_claim_authorized",
        "promotion_authorized",
    }
)
_PROPOSAL_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "proposal_id",
        "state",
        "claim_scope",
        "disposition",
        "next_gate",
        "candidate_binding",
        "evidence_binding",
        "design",
        "decisions",
        "assumptions",
        "hypotheses",
        "observed_scope_findings",
        "falsifiers",
        "open_questions",
        "gate_separation",
        "non_claims",
    }
)
_CANDIDATE_BINDING_KEYS = frozenset(
    {
        "candidate_id",
        "pattern_signature",
        "observation_ids",
        "expected_state",
        "expected_next_gate",
        "candidate_width_contract",
    }
)
_EVIDENCE_BINDING_KEYS = frozenset({"evidence_id", "fact_ids"})
_DESIGN_KEYS = frozenset(
    {
        "width_policy",
        "resource_derivation",
        "reduction_sequence",
        "scratch_lifecycle",
        "final_reducer_variants",
        "numeric_contract",
    }
)
_WIDTH_POLICY_KEYS = frozenset(
    {
        "candidate_contract",
        "specialization",
        "required_runtime_width",
        "runtime_query",
        "mismatch_action",
        "threadgroup_multiple_requirement",
        "partial_capacity_formula",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "family_thread_ceiling",
        "specialized_width",
        "maximum_simdgroups_at_family_ceiling",
        "calculation",
        "partial_dtype",
        "partial_bytes",
        "scratch_bytes_per_accumulator",
        "scratch_calculation",
        "dynamic_allocation_formula",
        "variant_resource_requirements",
        "variant_input_scope",
        "threadgroup_allocation_alignment_bytes",
        "total_threadgroup_memory_ceiling_bytes",
        "runtime_thread_limit_gate",
    }
)
_DYNAMIC_ALLOCATION_FORMULA_KEYS = frozenset(
    {"raw_dynamic_bytes", "aligned_dynamic_bytes", "allocation_strategy"}
)
_VARIANT_RESOURCE_REQUIREMENTS_KEYS = frozenset({"rms_norm", "layer_norm"})
_VARIANT_RESOURCE_REQUIREMENT_KEYS = frozenset(
    {
        "accumulator_count",
        "broadcast_scalar_bytes",
        "raw_dynamic_bytes",
        "aligned_dynamic_bytes",
        "layout",
        "total_memory_gate",
    }
)
_REDUCTION_KEYS = frozenset(
    {"steps", "publication_barrier", "barrier_participation", "early_return_policy"}
)
_SCRATCH_KEYS = frozenset(
    {
        "initialization",
        "initialization_barrier",
        "unused_final_lanes",
        "reuse_rule",
        "final_scalar_visibility_rule",
    }
)
_REDUCER_VARIANTS_KEYS = frozenset({"rms_norm", "layer_norm"})
_REDUCER_VARIANT_KEYS = frozenset({"ownership", "publication", "consumer_barrier"})
_NUMERIC_KEYS = frozenset(
    {
        "accumulator_dtype",
        "bfloat_input_policy",
        "bitwise_reduction_order_guaranteed",
        "correctness_requirement",
    }
)
_DECISION_KEYS = frozenset(
    {
        "decision_id",
        "statement",
        "candidate_effect",
        "fact_ids",
        "rationale",
        "alternatives",
    }
)
_HYPOTHESIS_KEYS = frozenset(
    {
        "hypothesis_id",
        "statement",
        "related_decision_ids",
        "status",
        "required_before",
        "falsifier",
        "result_binding",
    }
)
_HYPOTHESIS_FALSIFIER_KEYS = frozenset(
    {"method", "observable", "reject_when", "retained_result_contract"}
)
_OBSERVED_SCOPE_FINDING_KEYS = frozenset(
    {
        "finding_id",
        "fact_ids",
        "status",
        "scope",
        "reported_outcome",
        "does_not_resolve_hypothesis_ids",
        "scope_limit",
    }
)
_FALSIFIER_KEYS = frozenset({"falsifier_id", "condition", "required_action"})
_GATE_KEYS = frozenset({"validation", "readiness"})
_VALIDATION_GATE_KEYS = frozenset({"kind", "checks", "can_set_readiness"})
_READINESS_GATE_KEYS = frozenset({"ready", "authority", "blocking_reviews"})

_SCHEMA_OBJECT_DEFINITION_KEYS = frozenset(
    {"type", "additionalProperties", "required", "properties"}
)
_SCHEMA_OBJECT_KEYS_BY_DEFINITION = {
    "manifest": _MANIFEST_KEYS,
    "paper_stage": _PAPER_STAGE_KEYS,
    "input_candidate_closure": _INPUT_CLOSURE_KEYS,
    "counts": _COUNT_KEYS,
    "evidence": _EVIDENCE_KEYS,
    "pinned_pdf_source": _PINNED_SOURCE_KEYS,
    "unverified_web_source": _WEB_SOURCE_KEYS,
    "fact": _FACT_KEYS,
    "local_probe_observation": _LOCAL_PROBE_KEYS,
    "local_probe_repository_snapshot": _LOCAL_PROBE_REPOSITORY_SNAPSHOT_KEYS,
    "pipeline_observation": _PIPELINE_KEYS,
    "hierarchical_reduction_probe_observation": _HIERARCHICAL_PROBE_KEYS,
    "hierarchical_probe_repository_snapshot": (
        _HIERARCHICAL_PROBE_REPOSITORY_SNAPSHOT_KEYS
    ),
    "hierarchical_probe_retained_artifact": (
        _HIERARCHICAL_PROBE_RETAINED_ARTIFACT_KEYS
    ),
    "hierarchical_probe_normalized_output": (
        _HIERARCHICAL_PROBE_NORMALIZED_OUTPUT_KEYS
    ),
    "hierarchical_probe_device": _HIERARCHICAL_PROBE_DEVICE_KEYS,
    "hierarchical_probe_pipeline": _HIERARCHICAL_PROBE_PIPELINE_KEYS,
    "hierarchical_probe_coverage": _HIERARCHICAL_PROBE_COVERAGE_KEYS,
    "hierarchical_probe_oracle": _HIERARCHICAL_PROBE_ORACLE_KEYS,
    "hierarchical_probe_case_result": _HIERARCHICAL_PROBE_CASE_RESULT_KEYS,
    "hierarchical_probe_aggregate": _HIERARCHICAL_PROBE_AGGREGATE_KEYS,
    "local_falsifier_outcome": _LOCAL_FALSIFIER_OUTCOME_KEYS,
    "non_evidentiary_context": _TARGET_CONTEXT_KEYS,
    "non_evidentiary_repository_snapshot": (_NON_EVIDENTIARY_REPOSITORY_SNAPSHOT_KEYS),
    "review_request": _REVIEW_REQUEST_KEYS,
    "review_stage_binding": _REVIEW_STAGE_BINDING_KEYS,
    "review_manifest_binding": _REVIEW_MANIFEST_BINDING_KEYS,
    "review_evidence_binding": _REVIEW_EVIDENCE_BINDING_KEYS,
    "review_proposal_binding": _REVIEW_PROPOSAL_BINDING_KEYS,
    "review_closure": _REVIEW_CLOSURE_KEYS,
    "review_closure_counts": _REVIEW_CLOSURE_COUNT_KEYS,
    "review_item": _REVIEW_ITEM_KEYS,
    "phase_order_ambiguity": _PHASE_ORDER_AMBIGUITY_KEYS,
    "preimplementation_standalone_metal_prototypes_option": (
        _PHASE_ORDER_RESOLUTION_OPTION_KEYS
    ),
    "amend_gate_after_bounded_implementation_option": (
        _PHASE_ORDER_RESOLUTION_OPTION_KEYS
    ),
    "absent_decision_artifact": _ABSENT_DECISION_ARTIFACT_KEYS,
    "external_human_decision_contract": _EXTERNAL_HUMAN_DECISION_CONTRACT_KEYS,
    "per_item_verdict_contract": _PER_ITEM_VERDICT_CONTRACT_KEYS,
    "ambiguity_resolution_contract": _AMBIGUITY_RESOLUTION_CONTRACT_KEYS,
    "review_authorizations": _REVIEW_AUTHORIZATION_KEYS,
    "hardware_review_decision": _HARDWARE_REVIEW_DECISION_KEYS,
    "review_decision_request_binding": _REVIEW_DECISION_REQUEST_BINDING_KEYS,
    "reviewer_attestation": _REVIEWER_ATTESTATION_KEYS,
    "review_item_verdict": _REVIEW_ITEM_VERDICT_KEYS,
    "review_ambiguity_resolution": _REVIEW_AMBIGUITY_RESOLUTION_KEYS,
    "review_scope_acknowledgements": _REVIEW_SCOPE_ACKNOWLEDGEMENT_KEYS,
    "proposal": _PROPOSAL_KEYS,
    "candidate_binding": _CANDIDATE_BINDING_KEYS,
    "evidence_binding": _EVIDENCE_BINDING_KEYS,
    "design": _DESIGN_KEYS,
    "width_policy": _WIDTH_POLICY_KEYS,
    "resource_derivation": _RESOURCE_KEYS,
    "dynamic_allocation_formula": _DYNAMIC_ALLOCATION_FORMULA_KEYS,
    "variant_resource_requirements": _VARIANT_RESOURCE_REQUIREMENTS_KEYS,
    "rms_norm_resource_requirement": _VARIANT_RESOURCE_REQUIREMENT_KEYS,
    "layer_norm_resource_requirement": _VARIANT_RESOURCE_REQUIREMENT_KEYS,
    "reduction_sequence": _REDUCTION_KEYS,
    "scratch_lifecycle": _SCRATCH_KEYS,
    "final_reducer_variants": _REDUCER_VARIANTS_KEYS,
    "reducer_variant": _REDUCER_VARIANT_KEYS,
    "numeric_contract": _NUMERIC_KEYS,
    "decision": _DECISION_KEYS,
    "hypothesis": _HYPOTHESIS_KEYS,
    "hypothesis_falsifier": _HYPOTHESIS_FALSIFIER_KEYS,
    "observed_scope_finding": _OBSERVED_SCOPE_FINDING_KEYS,
    "falsifier": _FALSIFIER_KEYS,
    "gate_separation": _GATE_KEYS,
    "validation_gate": _VALIDATION_GATE_KEYS,
    "readiness_gate": _READINESS_GATE_KEYS,
}

_ID = re.compile(r"[a-z0-9][a-z0-9._-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SAFE_PATH_CHARACTERS = re.compile(r"[A-Za-z0-9._/-]+")
_HTTPS_URL = re.compile(r"https://[^ ]+")
_HUMAN_REVIEW_TEXT = re.compile(
    r"[^\x00-\x1f\x7f-\x9f]*[^\s\x00-\x1f\x7f-\x9f][^\x00-\x1f\x7f-\x9f]*"
)
_HUMAN_REVIEW_TEXT_MAX_LENGTH = 4096

_STATE = "review_pending"
_CLAIM_SCOPE = "hardware_mapping_only"
_DISPOSITION = "await_human_hardware_review"
_NEXT_GATE = "principle_driven_iteration"
_REVIEW_REQUEST_ID = "hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1"
_REVIEW_REQUEST_PATH = (
    "review_requests/hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json"
)
_REVIEW_EVIDENCE_PATH = "evidence/apple-metal-hierarchical-reduction-v1.json"
_REVIEW_PROPOSAL_PATH = (
    "proposals/hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json"
)
_HARDWARE_REVIEW_DECISION_SCHEMA = (
    "https://open-cake-ir.local/hardware-informed-design/schema-v1.json"
    "#/$defs/hardware_review_decision"
)
_HARDWARE_REVIEW_ARTIFACT_PATHS = (
    "hardware_informed_design/schema.json",
    "hardware_informed_design/manifest.json",
    "hardware_informed_design/evidence/apple-metal-hierarchical-reduction-v1.json",
    "hardware_informed_design/proposals/"
    "hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json",
    "hardware_informed_design/review_requests/"
    "hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json",
)
_PHASE_ORDER_AMBIGUITY_ID = "resource-accounting-phase-order"
_RESOURCE_PHASE_CONTRADICTION = (
    "The resource-accounting-fits-selected-pipelines hypothesis is marked "
    "required_before=implementation, but its falsifier requires compiling every "
    "retained future variant and inspecting pipeline-created static threadgroup "
    "memory. The current finite Compiler lowering cannot create the required scratch "
    "allocation or barriers, so the word implementation cannot simultaneously mean "
    "both any materialization code and the accepted Compiler port."
)
_PHASE_ORDER_OPTIONS = (
    {
        "option_id": "preimplementation-standalone-metal-prototypes",
        "action": (
            "Keep required_before=implementation literal by compiling separately "
            "reviewed standalone Metal prototype pipelines for every retained RMS "
            "and Layer variant before any Compiler implementation."
        ),
        "consequence": (
            "Append-only resource observations may resolve the hypothesis without "
            "granting Compiler-change, correctness, performance, or promotion "
            "authority; the later P1-P8 review still governs the compiler design."
        ),
    },
    {
        "option_id": "amend-gate-to-port-acceptance-after-bounded-implementation",
        "action": (
            "Require a human-requested successor proposal that moves the resource "
            "hypothesis to port acceptance and authorizes only the bounded "
            "implementation needed to materialize retained pipelines after P1-P8 "
            "acceptance."
        ),
        "consequence": (
            "The first implementation cannot be accepted, released, evaluated, or "
            "used for a claim until every retained variant's resource observation "
            "passes; automation may not silently reinterpret the existing "
            "required_before field."
        ),
    },
)
_PHASE_ORDER_OPTION_IDS = tuple(option["option_id"] for option in _PHASE_ORDER_OPTIONS)
_RESOURCE_HYPOTHESIS_FALSIFIER_CONTRACT = {
    "method": (
        "Compile every retained variant and inspect "
        "pipeline.staticThreadgroupMemoryLength, that variant's aligned dynamic "
        "allocation, device.maxThreadgroupMemoryLength, and "
        "maxTotalThreadsPerThreadgroup."
    ),
    "observable": (
        "Total static-plus-dynamic bytes, alignment padding, requested threads, and "
        "both runtime ceilings for each pipeline variant."
    ),
    "reject_when": (
        "Any retained configuration exceeds either runtime limit or the 32768-byte "
        "family ceiling."
    ),
    "retained_result_contract": (
        "Append-only per-variant resource observations including refused "
        "configurations; no regenerated expectation may erase an overflow."
    ),
}
_STAGE_TRANSITION_RULE = (
    "Stage 3 clears only when an external human decision is bound to the reviewed "
    "Git revision, covers every review item with a localized reason, has "
    "global_disposition=approved equal to the maximum review-item verdict severity, "
    "has every review-item verdict approved, and selects "
    "preimplementation-standalone-metal-prototypes. Selecting "
    "amend-gate-to-port-acceptance-after-bounded-implementation requires a "
    "non-approved resource-accounting-and-runtime-gates item, permits only "
    "changes_requested or rejected globally, and requires a successor Stage-3 "
    "proposal and review request; it cannot clear the current Stage-3 closure."
)
_REVIEW_DISPOSITIONS = ("approved", "changes_requested", "rejected")
_REVIEW_DISPOSITION_SEVERITY = {
    disposition: severity for severity, disposition in enumerate(_REVIEW_DISPOSITIONS)
}
_APPROVAL_COMPATIBLE_OPTION_ID = _PHASE_ORDER_OPTION_IDS[0]
_SUCCESSOR_REQUIRED_OPTION_ID = _PHASE_ORDER_OPTION_IDS[1]
_SUCCESSOR_REQUIRED_GLOBAL_DISPOSITIONS = ("changes_requested", "rejected")
_SUCCESSOR_REQUIRED_NON_APPROVED_REVIEW_ITEM_ID = (
    "resource-accounting-and-runtime-gates"
)
_GIT_PARENT_REVISION = "2238610a4e8923330d73d125e79fd374fc7d2397"
_PROBE_PATH = "tools/probe_metal_operators.py"
_LOCAL_PROBE_OBSERVATION_ID = "local-apple-m4-metal-operator-probe-2026-08-26"
_HIERARCHICAL_PROBE_OBSERVATION_ID = (
    "local-apple-m4-hierarchical-reduction-probe-2026-08-26"
)
_HIERARCHICAL_PROBE_REPOSITORY_REVISION = "f858612c7ee695b57d72115c76723c7bacc22a9b"
_HIERARCHICAL_PROBE_SOURCE_PATHS = (
    "tools/probe_metal_hierarchical_reduction.py",
    "tools/metal_hierarchical_reduction_probe/hierarchical_reduction.metal",
    "tools/metal_hierarchical_reduction_probe/run_hierarchical_reduction.swift",
)
_HIERARCHICAL_PROBE_ARTIFACT_RELATIVE_PATH = (
    "f858612c7ee695b57d72115c76723c7bacc22a9b/metal-hierarchical-reduction.json"
)
_HIERARCHICAL_PROBE_GROUP_COUNTS = (1, 2, 4, 8, 16, 32)
_HIERARCHICAL_PROBE_THREAD_COUNTS = (32, 64, 128, 256, 512, 1024)
_HIERARCHICAL_PROBE_EXPECTED_SUMS = (
    (13.375, -25.25),
    (31.25, -50.25),
    (62.625, -99.5),
    (125.75, -195.0),
    (253.5, -387.75),
    (510.125, -769.25),
)
_HIERARCHICAL_PROBE_FACT_IDS = frozenset(
    {
        "observed-m4-hierarchical-exact-owner-and-broadcast",
        "observed-m4-hierarchical-runtime-resource-gates",
        "observed-m4-hierarchical-two-epoch-scratch-reuse",
        "observed-m4-hierarchical-width32-case-matrix",
    }
)
_HIERARCHICAL_PROBE_FALSIFIER_OUTCOMES = {
    "runtime-width-not-32": "not_triggered_in_observed_cases",
    "pipeline-thread-limit-exceeded": "not_triggered_in_observed_cases",
    "partial-final-simdgroup-outside-v1": ("precondition_satisfied_in_observed_cases"),
    "partial-set-exceeds-final-simdgroup": "not_triggered_in_observed_cases",
    "threadgroup-memory-budget-exceeded": "not_triggered_in_observed_cases",
    "barrier-not-uniform": "requires_source_review",
    "scratch-read-before-publication": "not_observed_in_tested_outputs",
    "scratch-reuse-race": "not_observed_in_tested_outputs",
    "bfloat-simd-sum-input": "outside_probe_scope",
    "ambiguous-final-reducer-owner": "not_triggered_for_single_owner_probe_only",
    "bitwise-order-required": "outside_probe_scope",
    "current-lowering-capability-missing": "condition_still_present",
}
_OBSERVED_SCOPE_FINDINGS = {
    "observed-width32-case-matrix": (
        "observed-m4-hierarchical-width32-case-matrix",
        ("width32-specialization-covers-intended-devices",),
    ),
    "observed-exact-owner-and-broadcast": (
        "observed-m4-hierarchical-exact-owner-and-broadcast",
        (
            "floating-reassociation-meets-workload-tolerances",
            "one-partial-per-simdgroup-is-expressible",
        ),
    ),
    "observed-two-epoch-scratch-reuse": (
        "observed-m4-hierarchical-two-epoch-scratch-reuse",
        ("one-partial-per-simdgroup-is-expressible",),
    ),
    "observed-runtime-resource-gates": (
        "observed-m4-hierarchical-runtime-resource-gates",
        (
            "resource-accounting-fits-selected-pipelines",
            "specialization-is-performance-competitive",
        ),
    ),
}
_REVIEW_ITEM_REFERENCES: dict[str, dict[str, tuple[str, ...]]] = {
    "vendor-fact-applicability": {
        "fact_ids": (
            "apple-m4-is-gpu-family9",
            "apple-family9-supports-simdgroup-reductions",
            "apple-family9-thread-ceiling-1024",
            "apple-family9-threadgroup-memory-ceiling-32768",
            "apple-family9-threadgroup-memory-alignment-16",
            "metal-threadgroup-memory-scope",
            "metal-simd-sum-active-thread-scope",
            "metal-simd-sum-excludes-bfloat-input",
            "metal-threadgroup-publication-barrier",
            "metal-simd-width-must-be-queried",
            "metal-pipeline-thread-limit-must-be-queried",
            "metal-total-threadgroup-memory-runtime-gate",
            "metal-family-query-is-lower-bound",
            "observed-m4-supports-apple9-or-newer",
        ),
        "decision_ids": (
            "preserve-width-neutral-candidate",
            "refine-width32-fail-closed-specialization",
        ),
        "hypothesis_ids": ("width32-specialization-covers-intended-devices",),
        "observed_scope_finding_ids": ("observed-width32-case-matrix",),
        "falsifier_ids": ("runtime-width-not-32",),
    },
    "width-policy-and-fail-closed-selection": {
        "fact_ids": (
            "apple-family9-thread-ceiling-1024",
            "metal-simd-width-must-be-queried",
            "metal-pipeline-thread-limit-must-be-queried",
            "observed-m4-pipeline-width-32",
            "observed-m4-pipeline-thread-limit-1024",
            "observed-m4-hierarchical-width32-case-matrix",
        ),
        "decision_ids": (
            "preserve-width-neutral-candidate",
            "refine-width32-fail-closed-specialization",
        ),
        "hypothesis_ids": ("width32-specialization-covers-intended-devices",),
        "observed_scope_finding_ids": ("observed-width32-case-matrix",),
        "falsifier_ids": (
            "runtime-width-not-32",
            "pipeline-thread-limit-exceeded",
            "partial-final-simdgroup-outside-v1",
            "partial-set-exceeds-final-simdgroup",
        ),
    },
    "partial-cardinality-and-final-reduction": {
        "fact_ids": (
            "apple-family9-supports-simdgroup-reductions",
            "metal-simd-sum-active-thread-scope",
            "observed-m4-hierarchical-exact-owner-and-broadcast",
        ),
        "decision_ids": (
            "refine-metal-two-level-reduction-sequence",
            "split-final-reducer-ownership",
        ),
        "hypothesis_ids": ("one-partial-per-simdgroup-is-expressible",),
        "observed_scope_finding_ids": ("observed-exact-owner-and-broadcast",),
        "falsifier_ids": (
            "partial-set-exceeds-final-simdgroup",
            "scratch-read-before-publication",
            "ambiguous-final-reducer-owner",
        ),
    },
    "barrier-participation-and-publication": {
        "fact_ids": (
            "metal-threadgroup-memory-scope",
            "metal-threadgroup-publication-barrier",
            "observed-m4-hierarchical-exact-owner-and-broadcast",
            "observed-m4-hierarchical-two-epoch-scratch-reuse",
        ),
        "decision_ids": (
            "refine-metal-two-level-reduction-sequence",
            "split-final-reducer-ownership",
        ),
        "hypothesis_ids": ("one-partial-per-simdgroup-is-expressible",),
        "observed_scope_finding_ids": (
            "observed-exact-owner-and-broadcast",
            "observed-two-epoch-scratch-reuse",
        ),
        "falsifier_ids": (
            "barrier-not-uniform",
            "scratch-read-before-publication",
            "scratch-reuse-race",
        ),
    },
    "scratch-initialization-lifetime-and-reuse": {
        "fact_ids": (
            "metal-threadgroup-memory-scope",
            "metal-threadgroup-publication-barrier",
            "observed-m4-hierarchical-two-epoch-scratch-reuse",
        ),
        "decision_ids": (
            "refine-metal-two-level-reduction-sequence",
            "refine-explicit-resource-accounting",
        ),
        "hypothesis_ids": ("one-partial-per-simdgroup-is-expressible",),
        "observed_scope_finding_ids": ("observed-two-epoch-scratch-reuse",),
        "falsifier_ids": (
            "scratch-read-before-publication",
            "scratch-reuse-race",
        ),
    },
    "variant-owner-split": {
        "fact_ids": (
            "metal-simd-sum-active-thread-scope",
            "metal-threadgroup-memory-scope",
            "metal-threadgroup-publication-barrier",
            "observed-m4-hierarchical-exact-owner-and-broadcast",
        ),
        "decision_ids": ("split-final-reducer-ownership",),
        "hypothesis_ids": ("one-partial-per-simdgroup-is-expressible",),
        "observed_scope_finding_ids": ("observed-exact-owner-and-broadcast",),
        "falsifier_ids": ("ambiguous-final-reducer-owner",),
    },
    "resource-accounting-and-runtime-gates": {
        "fact_ids": (
            "apple-family9-thread-ceiling-1024",
            "apple-family9-threadgroup-memory-ceiling-32768",
            "apple-family9-threadgroup-memory-alignment-16",
            "metal-pipeline-thread-limit-must-be-queried",
            "metal-total-threadgroup-memory-runtime-gate",
            "observed-m4-threadgroup-memory-32768",
            "observed-m4-pipeline-thread-limit-1024",
            "observed-m4-hierarchical-runtime-resource-gates",
        ),
        "decision_ids": ("refine-explicit-resource-accounting",),
        "hypothesis_ids": (
            "resource-accounting-fits-selected-pipelines",
            "specialization-is-performance-competitive",
        ),
        "observed_scope_finding_ids": ("observed-runtime-resource-gates",),
        "falsifier_ids": (
            "pipeline-thread-limit-exceeded",
            "threadgroup-memory-budget-exceeded",
            "current-lowering-capability-missing",
        ),
    },
    "numeric-scope-and-oracle-boundary": {
        "fact_ids": (
            "metal-simd-sum-active-thread-scope",
            "metal-simd-sum-excludes-bfloat-input",
            "observed-m4-hierarchical-exact-owner-and-broadcast",
        ),
        "decision_ids": ("refine-metal-two-level-reduction-sequence",),
        "hypothesis_ids": ("floating-reassociation-meets-workload-tolerances",),
        "observed_scope_finding_ids": ("observed-exact-owner-and-broadcast",),
        "falsifier_ids": ("bfloat-simd-sum-input", "bitwise-order-required"),
    },
    "observed-scope-and-authority-boundary": {
        "fact_ids": (
            "observed-m4-supports-apple9-or-newer",
            "observed-m4-threadgroup-memory-32768",
            "observed-m4-pipeline-width-32",
            "observed-m4-pipeline-thread-limit-1024",
            "observed-m4-hierarchical-width32-case-matrix",
            "observed-m4-hierarchical-exact-owner-and-broadcast",
            "observed-m4-hierarchical-two-epoch-scratch-reuse",
            "observed-m4-hierarchical-runtime-resource-gates",
        ),
        "decision_ids": (
            "preserve-width-neutral-candidate",
            "refine-width32-fail-closed-specialization",
            "refine-metal-two-level-reduction-sequence",
            "split-final-reducer-ownership",
            "refine-explicit-resource-accounting",
        ),
        "hypothesis_ids": (
            "width32-specialization-covers-intended-devices",
            "one-partial-per-simdgroup-is-expressible",
            "resource-accounting-fits-selected-pipelines",
            "floating-reassociation-meets-workload-tolerances",
            "specialization-is-performance-competitive",
        ),
        "observed_scope_finding_ids": (
            "observed-width32-case-matrix",
            "observed-exact-owner-and-broadcast",
            "observed-two-epoch-scratch-reuse",
            "observed-runtime-resource-gates",
        ),
        "falsifier_ids": (
            "runtime-width-not-32",
            "pipeline-thread-limit-exceeded",
            "partial-final-simdgroup-outside-v1",
            "partial-set-exceeds-final-simdgroup",
            "threadgroup-memory-budget-exceeded",
            "barrier-not-uniform",
            "scratch-read-before-publication",
            "scratch-reuse-race",
            "bfloat-simd-sum-input",
            "ambiguous-final-reducer-owner",
            "bitwise-order-required",
            "current-lowering-capability-missing",
        ),
    },
}
_REVIEW_ITEM_IDS = tuple(_REVIEW_ITEM_REFERENCES)
_TARGET_PATH = "compiler/targets/apple_gpu_family9.json"
_LOWERING_PATH = "src/open_cake_ir/compiler/emit_metal.py"
_CANDIDATE_ID = "hierarchical-simdgroup-threadgroup-reduction"
_CANDIDATE_PATH = (
    "abstraction_extraction/candidates/"
    "hierarchical-simdgroup-threadgroup-reduction.json"
)
_OBSERVATION_PATHS = (
    "abstraction_extraction/observations/mlx-layer-norm-hierarchical-reduction.json",
    "abstraction_extraction/observations/mlx-rms-norm-hierarchical-reduction.json",
)
_PINNED_PDF_IDENTITIES = {
    "apple-metal-shading-language-specification-2026-06-04": {
        "url": "https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf",
        "revision": "2026-06-04",
        "content_sha256": "41538b30d2f1140a5b2a0c84ce0a9f7b67bf0c707e224cfea0bfe5a44aa26cf5",
    },
    "apple-metal-feature-set-tables-2026-05-21": {
        "url": "https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf",
        "revision": "2026-05-21",
        "content_sha256": "9f31df15dd6827545702c5a0845f6e36e1889878cd0e534123bd70211e5c00a8",
    },
}
_FALSIFIER_IDS = frozenset(
    {
        "ambiguous-final-reducer-owner",
        "barrier-not-uniform",
        "bfloat-simd-sum-input",
        "bitwise-order-required",
        "current-lowering-capability-missing",
        "partial-final-simdgroup-outside-v1",
        "partial-set-exceeds-final-simdgroup",
        "pipeline-thread-limit-exceeded",
        "runtime-width-not-32",
        "scratch-read-before-publication",
        "scratch-reuse-race",
        "threadgroup-memory-budget-exceeded",
    }
)
_SCHEMA_NON_OBJECT_CONTRACTS: dict[str, dict[str, object]] = {
    "state": {"const": _STATE},
    "claim_scope": {"const": _CLAIM_SCOPE},
    "disposition": {"const": _DISPOSITION},
    "next_gate": {"const": _NEXT_GATE},
    "safe_path": {
        "type": "string",
        "pattern": r"^(?!/)(?!.*(?:^|/)\.\.(?:/|$))[A-Za-z0-9._/-]+$",
    },
    "https_url": {"type": "string", "pattern": r"^https://[^ ]+$"},
    "id": {"type": "string", "pattern": r"^[a-z0-9][a-z0-9._-]+$"},
    "id_list": {
        "type": "array",
        "uniqueItems": True,
        "items": {"$ref": "#/$defs/id"},
    },
    "git_commit": {"type": "string", "pattern": r"^[0-9a-f]{40}$"},
    "human_review_text": {
        "type": "string",
        "minLength": 1,
        "maxLength": _HUMAN_REVIEW_TEXT_MAX_LENGTH,
        "pattern": (
            r"^[^\u0000-\u001F\u007F-\u009F]*"
            r"[^\s\u0000-\u001F\u007F-\u009F]"
            r"[^\u0000-\u001F\u007F-\u009F]*$"
        ),
    },
    "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
    "nonempty_strings": {
        "type": "array",
        "minItems": 1,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1},
    },
}

_SCHEMA_CRITICAL_PROPERTY_CONTRACTS: dict[str, dict[str, object]] = {
    "review_request": {
        "$schema": {"const": "../schema.json#/$defs/review_request"},
        "schema_version": {"const": 1},
        "review_request_id": {"const": _REVIEW_REQUEST_ID},
        "state": {"const": "awaiting_human_review"},
        "authority": {"const": "external_human_hardware_reviewer_only"},
        "stage_binding": {"$ref": "#/$defs/review_stage_binding"},
        "closure": {"$ref": "#/$defs/review_closure"},
        "review_items": {
            "type": "array",
            "minItems": 9,
            "maxItems": 9,
            "uniqueItems": True,
            "items": {"$ref": "#/$defs/review_item"},
        },
        "phase_order_ambiguity": {"$ref": "#/$defs/phase_order_ambiguity"},
        "decision_artifact": {"$ref": "#/$defs/absent_decision_artifact"},
        "external_human_decision_contract": {
            "$ref": "#/$defs/external_human_decision_contract"
        },
        "authorizations": {"$ref": "#/$defs/review_authorizations"},
        "non_claims": {"$ref": "#/$defs/nonempty_strings"},
    },
    "review_stage_binding": {
        "manifest": {"$ref": "#/$defs/review_manifest_binding"},
        "evidence": {"$ref": "#/$defs/review_evidence_binding"},
        "proposal": {"$ref": "#/$defs/review_proposal_binding"},
    },
    "review_manifest_binding": {
        "design_stage_id": {"const": "apple-metal-hardware-informed-design-v1"},
        "path": {"const": "hardware_informed_design/manifest.json"},
    },
    "review_evidence_binding": {
        "evidence_id": {"const": "apple-metal-hierarchical-reduction-v1"},
        "path": {
            "const": (
                "hardware_informed_design/evidence/"
                "apple-metal-hierarchical-reduction-v1.json"
            )
        },
    },
    "review_proposal_binding": {
        "proposal_id": {"const": _REVIEW_REQUEST_ID},
        "path": {
            "const": (
                "hardware_informed_design/proposals/"
                "hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json"
            )
        },
    },
    "review_closure": {
        "fact_ids": {
            "type": "array",
            "minItems": 21,
            "maxItems": 21,
            "uniqueItems": True,
            "items": {"$ref": "#/$defs/id"},
        },
        "decision_ids": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "uniqueItems": True,
            "items": {"$ref": "#/$defs/id"},
        },
        "hypothesis_ids": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "uniqueItems": True,
            "items": {"$ref": "#/$defs/id"},
        },
        "observed_scope_finding_ids": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "uniqueItems": True,
            "items": {"$ref": "#/$defs/id"},
        },
        "falsifier_ids": {
            "type": "array",
            "minItems": 12,
            "maxItems": 12,
            "uniqueItems": True,
            "items": {"$ref": "#/$defs/id"},
        },
        "counts": {"$ref": "#/$defs/review_closure_counts"},
    },
    "review_closure_counts": {
        "fact_count": {"const": 21},
        "decision_count": {"const": 5},
        "hypothesis_count": {"const": 5},
        "observed_scope_finding_count": {"const": 4},
        "falsifier_count": {"const": 12},
    },
    "review_item": {
        "review_item_id": {"$ref": "#/$defs/id"},
        "question": {"type": "string", "minLength": 1},
        "fact_ids": {"$ref": "#/$defs/id_list"},
        "decision_ids": {"$ref": "#/$defs/id_list"},
        "hypothesis_ids": {"$ref": "#/$defs/id_list"},
        "observed_scope_finding_ids": {"$ref": "#/$defs/id_list"},
        "falsifier_ids": {"$ref": "#/$defs/id_list"},
        "required_human_judgment": {"type": "string", "minLength": 1},
    },
    "phase_order_ambiguity": {
        "ambiguity_id": {"const": _PHASE_ORDER_AMBIGUITY_ID},
        "status": {"const": "unresolved_requires_human_choice"},
        "contradiction": {"const": _RESOURCE_PHASE_CONTRADICTION},
        "resolution_options": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "prefixItems": [
                {
                    "$ref": (
                        "#/$defs/preimplementation_standalone_metal_prototypes_option"
                    )
                },
                {"$ref": "#/$defs/amend_gate_after_bounded_implementation_option"},
            ],
            "items": False,
        },
        "selection_required": {"const": True},
        "automation_may_select": {"const": False},
    },
    "preimplementation_standalone_metal_prototypes_option": {
        field: {"const": value} for field, value in _PHASE_ORDER_OPTIONS[0].items()
    },
    "amend_gate_after_bounded_implementation_option": {
        field: {"const": value} for field, value in _PHASE_ORDER_OPTIONS[1].items()
    },
    "absent_decision_artifact": {
        "status": {"const": "absent"},
        "path": {"type": "null"},
        "decision_id": {"type": "null"},
    },
    "external_human_decision_contract": {
        "decision_authority": {"const": "external_human_hardware_reviewer_only"},
        "global_dispositions": {"const": list(_REVIEW_DISPOSITIONS)},
        "required_review_item_ids": {
            "type": "array",
            "minItems": 9,
            "maxItems": 9,
            "uniqueItems": True,
            "items": {"$ref": "#/$defs/id"},
        },
        "per_item_verdict_contract": {"$ref": "#/$defs/per_item_verdict_contract"},
        "reviewed_git_revision_required": {"const": True},
        "ambiguity_resolution_contract": {
            "$ref": "#/$defs/ambiguity_resolution_contract"
        },
        "stage_transition_rule": {"const": _STAGE_TRANSITION_RULE},
        "automation_may_write_decision": {"const": False},
    },
    "per_item_verdict_contract": {
        "per_item_verdict_required": {"const": True},
        "verdict_values": {"const": list(_REVIEW_DISPOSITIONS)},
        "global_disposition_aggregation": {
            "const": "maximum_review_item_verdict_severity"
        },
        "verdict_severity_order": {"const": list(_REVIEW_DISPOSITIONS)},
        "localized_reason_required": {"const": True},
        "stage_clear_requires_all_item_verdicts_approved": {"const": True},
        "non_approved_item_blocks_stage_transition": {"const": True},
    },
    "ambiguity_resolution_contract": {
        "required_ambiguity_id": {"const": _PHASE_ORDER_AMBIGUITY_ID},
        "allowed_option_ids": {"const": list(_PHASE_ORDER_OPTION_IDS)},
        "localized_reason_required": {"const": True},
        "approval_compatible_option_id": {"const": _APPROVAL_COMPATIBLE_OPTION_ID},
        "successor_required_option_id": {"const": _SUCCESSOR_REQUIRED_OPTION_ID},
        "successor_required_option_allowed_global_dispositions": {
            "const": list(_SUCCESSOR_REQUIRED_GLOBAL_DISPOSITIONS)
        },
        "successor_required_option_required_non_approved_review_item_id": {
            "const": _SUCCESSOR_REQUIRED_NON_APPROVED_REVIEW_ITEM_ID
        },
    },
    "review_authorizations": {
        field: {"const": False} for field in _REVIEW_AUTHORIZATION_KEYS
    },
    "hardware_review_decision": {
        "$schema": {"const": _HARDWARE_REVIEW_DECISION_SCHEMA},
        "schema_version": {"const": 1},
        "decision_id": {"$ref": "#/$defs/id"},
        "kind": {"const": "external_human_hardware_review_decision"},
        "authority": {"const": "external_human_hardware_reviewer_only"},
        "binding": {"$ref": "#/$defs/review_decision_request_binding"},
        "reviewer_attestation": {"$ref": "#/$defs/reviewer_attestation"},
        "global_disposition": {"enum": list(_REVIEW_DISPOSITIONS)},
        "item_verdicts": {
            "type": "array",
            "minItems": 9,
            "maxItems": 9,
            "uniqueItems": True,
            "items": {"$ref": "#/$defs/review_item_verdict"},
        },
        "ambiguity_resolution": {"$ref": "#/$defs/review_ambiguity_resolution"},
        "scope_acknowledgements": {"$ref": "#/$defs/review_scope_acknowledgements"},
    },
    "review_decision_request_binding": {
        "design_stage_id": {"const": "apple-metal-hardware-informed-design-v1"},
        "request_id": {"const": _REVIEW_REQUEST_ID},
        "path": {"const": f"hardware_informed_design/{_REVIEW_REQUEST_PATH}"},
        "reviewed_git_revision": {"$ref": "#/$defs/git_commit"},
        "reviewed_artifact_paths": {"const": list(_HARDWARE_REVIEW_ARTIFACT_PATHS)},
    },
    "reviewer_attestation": {
        "reviewer_identity": {"$ref": "#/$defs/human_review_text"},
        "authorship_attestation": {"const": "external-human-outside-automation"},
        "decision_basis": {"$ref": "#/$defs/human_review_text"},
    },
    "review_item_verdict": {
        "review_item_id": {"enum": list(_REVIEW_ITEM_IDS)},
        "verdict": {"enum": list(_REVIEW_DISPOSITIONS)},
        "localized_reason": {"$ref": "#/$defs/human_review_text"},
    },
    "review_ambiguity_resolution": {
        "ambiguity_id": {"const": _PHASE_ORDER_AMBIGUITY_ID},
        "selected_option_id": {"enum": list(_PHASE_ORDER_OPTION_IDS)},
        "localized_reason": {"$ref": "#/$defs/human_review_text"},
    },
    "review_scope_acknowledgements": {
        "approval_scope": {"const": "stage3_hardware_review_only"},
        "principle_driven_iteration_required": {"const": True},
        "resource_accounting_hypothesis_resolved": {"const": False},
        "compiler_change_authorized": {"const": False},
        "implementation_authorized": {"const": False},
        "evaluation_authorized": {"const": False},
        "performance_claim_authorized": {"const": False},
        "scientific_claim_authorized": {"const": False},
        "promotion_authorized": {"const": False},
    },
    "hierarchical_probe_aggregate": {
        "all_command_buffers_completed": {"const": True},
        "all_cases_passed": {"const": True},
    },
}


class HardwareInformedDesignValidationError(ValueError):
    """The deterministic set of hardware-design validation failures."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(sorted(set(errors)))
        super().__init__("\n".join(self.errors))


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True)
class HardwareInformedDesignSummary:
    design_stage_id: str
    state: str
    disposition: str
    next_gate: str
    counts: dict[str, int]
    proposal_ids: tuple[str, ...]
    verification_scope: str = "offline_metadata_and_derivation_validation"
    ready_for_principle_review: bool = False
    hardware_semantics_verified: bool = False
    performance_claim_authorized: bool = False
    compiler_change_authorized: bool = False
    remote_source_bytes_verified: bool = False
    engineering_observation_record_consistent: bool = True
    retained_local_probe_reported_outcome: bool = True
    retained_local_probe_case_count: int = 6
    retained_local_probe_epoch_count: int = 12
    retained_local_probe_scope: str = "observed_apple_m4_fp32_rms_probe_pipeline_only"
    review_request_count: int = 1
    hardware_review_decision_present: bool = False
    hardware_review_decision_id: str | None = None
    hardware_review_decision_disposition: str | None = None
    hardware_reviewed_git_revision: str | None = None
    hardware_reviewed_revision_bound: bool = False
    non_approved_review_item_ids: tuple[str, ...] = ()
    resource_phase_option_id: str | None = None
    successor_stage3_proposal_required: bool = False
    resource_accounting_hypothesis_resolved: bool = False
    reviewer_authorship_assurance: str = "no_external_decision_supplied"
    stage_transition_applied: bool = False
    next_action: str = "obtain_external_human_hardware_review"
    resource_phase_ambiguity_exposed: bool = True
    hardware_review_complete: bool = False
    implementation_authorized: bool = False
    evaluation_authorized: bool = False
    scientific_claim_authorized: bool = False
    promotion_authorized: bool = False
    evaluation_evidence_present: bool = False

    def report(self) -> dict[str, object]:
        return {"valid": True, **asdict(self)}


@dataclass(frozen=True)
class _HardwareReviewDecisionResult:
    decision_id: str
    disposition: str
    reviewed_git_revision: str
    non_approved_review_item_ids: tuple[str, ...]
    selected_option_id: str
    successor_stage3_proposal_required: bool
    hardware_review_clear: bool
    next_action: str


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_non_json_number(value: str) -> object:
    raise ValueError(f"non-JSON number {value}")


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value}")
    return parsed


class _Validator:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.errors: list[str] = []

    def error(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    def root_closure(self) -> None:
        allowed = _ROOT_FILES | _ROOT_DIRECTORIES
        try:
            entries = {entry.name: entry for entry in self.root.iterdir()}
        except OSError as error:
            self.error(
                "hardware_informed_design", f"could not enumerate directory ({error})"
            )
            return
        unknown = sorted(set(entries) - allowed)
        if unknown:
            self.error(
                "hardware_informed_design",
                f"unlisted hardware_informed_design entries: {', '.join(unknown)}",
            )
        for name in sorted(_ROOT_FILES):
            entry = entries.get(name)
            if entry is None or entry.is_symlink() or not entry.is_file():
                self.error(
                    f"hardware_informed_design/{name}",
                    "must be a regular non-symlink file",
                )
        for name in sorted(_ROOT_DIRECTORIES):
            entry = entries.get(name)
            if entry is None or entry.is_symlink() or not entry.is_dir():
                self.error(
                    f"hardware_informed_design/{name}",
                    "must be a regular non-symlink directory",
                )

    def read(self, path: Path, display: str) -> dict[str, object] | None:
        try:
            valid_file = not path.is_symlink() and path.is_file()
        except OSError as error:
            self.error(display, f"could not inspect file ({error})")
            return None
        if not valid_file:
            self.error(display, "must be a regular non-symlink file")
            return None
        try:
            value = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_json_number,
                parse_float=_parse_json_float,
            )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            _DuplicateKeyError,
            ValueError,
        ) as error:
            self.error(display, f"invalid UTF-8 JSON ({error})")
            return None
        if not isinstance(value, dict):
            self.error(display, "top level must be an object")
            return None
        try:
            _canonical_json_bytes(value)
        except (TypeError, UnicodeError, ValueError) as error:
            self.error(display, f"cannot be canonicalized as UTF-8 JSON ({error})")
            return None
        return value

    def exact(
        self, value: object, path: str, keys: frozenset[str]
    ) -> dict[str, object] | None:
        if not isinstance(value, dict):
            self.error(path, "must be an object")
            return None
        missing = sorted(keys - set(value))
        unknown = sorted(set(value) - keys)
        if missing:
            self.error(path, f"missing fields: {', '.join(missing)}")
        if unknown:
            self.error(path, f"unknown fields: {', '.join(unknown)}")
        return value

    def literal(self, value: object, path: str, expected: object) -> bool:
        if type(value) is not type(expected) or value != expected:  # noqa: E721
            self.error(path, f"must equal {expected!r}")
            return False
        return True

    def string(
        self,
        value: object,
        path: str,
        pattern: re.Pattern[str] | None = None,
    ) -> str | None:
        if not isinstance(value, str) or not value:
            self.error(path, "must be a non-empty string")
            return None
        if pattern is not None and pattern.fullmatch(value) is None:
            self.error(path, f"does not match {pattern.pattern!r}")
            return None
        try:
            value.encode("utf-8")
        except UnicodeError as error:
            self.error(path, f"must be valid UTF-8 ({error})")
            return None
        return value

    def human_review_text(self, value: object, path: str) -> str | None:
        parsed = self.string(value, path)
        if parsed is None:
            return None
        if len(parsed) > _HUMAN_REVIEW_TEXT_MAX_LENGTH:
            self.error(
                path,
                f"must contain at most {_HUMAN_REVIEW_TEXT_MAX_LENGTH} characters",
            )
            return None
        if _HUMAN_REVIEW_TEXT.fullmatch(parsed) is None:
            self.error(
                path,
                "must contain a non-whitespace character and no C0 or C1 controls",
            )
            return None
        return parsed

    def integer(self, value: object, path: str, minimum: int = 0) -> int | None:
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            self.error(path, f"must be an integer >= {minimum}")
            return None
        return value

    def strings(
        self,
        value: object,
        path: str,
        *,
        nonempty: bool,
        pattern: re.Pattern[str] | None = None,
        sorted_required: bool = False,
    ) -> list[str]:
        if not isinstance(value, list):
            self.error(path, "must be an array")
            return []
        if nonempty and not value:
            self.error(path, "must not be empty")
        result: list[str] = []
        for index, item in enumerate(value):
            parsed = self.string(item, f"{path}[{index}]", pattern)
            if parsed is not None:
                result.append(parsed)
        if len(result) == len(value):
            if len(set(result)) != len(result):
                self.error(path, "must contain unique values")
            if sorted_required and result != sorted(result):
                self.error(path, "must be sorted lexicographically")
        return result

    def safe_path(self, value: object, path: str) -> str | None:
        parsed = self.string(value, path)
        if parsed is None:
            return None
        pure = PurePosixPath(parsed)
        if (
            _SAFE_PATH_CHARACTERS.fullmatch(parsed) is None
            or pure.is_absolute()
            or parsed != pure.as_posix()
            or any(part in {"", ".", ".."} for part in parsed.split("/"))
        ):
            self.error(path, "must be a canonical safe relative path without traversal")
            return None
        return parsed

    def listed_paths(self, value: object, field: str, directory: str) -> list[str]:
        base = f"manifest.json.{field}"
        listed = self.strings(value, base, nonempty=True, sorted_required=True)
        result: list[str] = []
        for index, relative in enumerate(listed):
            parsed = self.safe_path(relative, f"{base}[{index}]")
            if parsed is None:
                continue
            pure = PurePosixPath(parsed)
            if (
                len(pure.parts) != 2
                or pure.parts[0] != directory
                or pure.suffix != ".json"
            ):
                self.error(f"{base}[{index}]", f"must name one {directory}/*.json file")
                continue
            candidate = self.root / parsed
            try:
                valid_file = not candidate.is_symlink() and candidate.is_file()
            except OSError as error:
                self.error(f"{base}[{index}]", f"could not inspect file ({error})")
                valid_file = False
            if not valid_file:
                self.error(
                    f"{base}[{index}]",
                    f"must reference a regular non-symlink file: {parsed}",
                )
            result.append(parsed)
        location = self.root / directory
        try:
            valid_directory = location.is_dir() and not location.is_symlink()
        except OSError as error:
            self.error(base, f"could not inspect directory ({error})")
            valid_directory = False
        if valid_directory:
            try:
                actual = sorted(
                    child.relative_to(self.root).as_posix()
                    for child in location.iterdir()
                )
            except OSError as error:
                self.error(base, f"could not enumerate directory ({error})")
            else:
                unlisted = sorted(set(actual) - set(result))
                if unlisted:
                    self.error(
                        base, f"unlisted directory entries: {', '.join(unlisted)}"
                    )
        return result


def _stage_literals(
    validator: _Validator,
    document: Mapping[str, object],
    path: str,
    *,
    claim_scope: str = _CLAIM_SCOPE,
) -> None:
    validator.literal(document.get("state"), f"{path}.state", _STATE)
    validator.literal(document.get("claim_scope"), f"{path}.claim_scope", claim_scope)
    validator.literal(document.get("disposition"), f"{path}.disposition", _DISPOSITION)
    validator.literal(document.get("next_gate"), f"{path}.next_gate", _NEXT_GATE)


def _gate(validator: _Validator, value: object, path: str) -> dict[str, object] | None:
    gate = validator.exact(value, path, _GATE_KEYS)
    if gate is None:
        return None
    validation = validator.exact(
        gate.get("validation"), f"{path}.validation", _VALIDATION_GATE_KEYS
    )
    if validation is not None:
        validator.literal(
            validation.get("kind"),
            f"{path}.validation.kind",
            "offline_metadata_and_derivation_validation",
        )
        validator.strings(
            validation.get("checks"),
            f"{path}.validation.checks",
            nonempty=True,
        )
        validator.literal(
            validation.get("can_set_readiness"),
            f"{path}.validation.can_set_readiness",
            False,
        )
    readiness = validator.exact(
        gate.get("readiness"), f"{path}.readiness", _READINESS_GATE_KEYS
    )
    if readiness is not None:
        validator.literal(readiness.get("ready"), f"{path}.readiness.ready", False)
        validator.literal(
            readiness.get("authority"),
            f"{path}.readiness.authority",
            "human_hardware_review",
        )
        validator.strings(
            readiness.get("blocking_reviews"),
            f"{path}.readiness.blocking_reviews",
            nonempty=True,
        )
    return gate


def _public_schema(validator: _Validator, schema: Mapping[str, object]) -> None:
    validator.exact(schema, "schema.json", _SCHEMA_ROOT_KEYS)
    validator.literal(
        schema.get("$schema"),
        "schema.json.$schema",
        "https://json-schema.org/draft/2020-12/schema",
    )
    validator.literal(
        schema.get("$id"),
        "schema.json.$id",
        "https://open-cake-ir.local/hardware-informed-design/schema-v1.json",
    )
    validator.literal(
        schema.get("oneOf"),
        "schema.json.oneOf",
        [
            {"$ref": "#/$defs/manifest"},
            {"$ref": "#/$defs/evidence"},
            {"$ref": "#/$defs/proposal"},
            {"$ref": "#/$defs/review_request"},
            {"$ref": "#/$defs/hardware_review_decision"},
        ],
    )
    definitions = validator.exact(
        schema.get("$defs"), "schema.json.$defs", _SCHEMA_DEFINITION_NAMES
    )
    if definitions is None:
        return

    covered_definitions = set(_SCHEMA_OBJECT_KEYS_BY_DEFINITION) | set(
        _SCHEMA_NON_OBJECT_CONTRACTS
    )
    if covered_definitions != set(_SCHEMA_DEFINITION_NAMES):
        validator.error(
            "schema.json.$defs",
            "validator's semantic definition closure differs from the public schema closure",
        )

    for name, expected_properties in sorted(_SCHEMA_OBJECT_KEYS_BY_DEFINITION.items()):
        path = f"schema.json.$defs.{name}"
        definition = validator.exact(
            definitions.get(name), path, _SCHEMA_OBJECT_DEFINITION_KEYS
        )
        if definition is None:
            continue
        validator.literal(definition.get("type"), f"{path}.type", "object")
        validator.literal(
            definition.get("additionalProperties"),
            f"{path}.additionalProperties",
            False,
        )
        required = validator.strings(
            definition.get("required"), f"{path}.required", nonempty=True
        )
        if frozenset(required) != expected_properties:
            validator.error(
                f"{path}.required",
                "required field set must equal the validator's exact-key contract",
            )
        properties = validator.exact(
            definition.get("properties"), f"{path}.properties", expected_properties
        )
        if properties is not None:
            for property_name in sorted(expected_properties):
                property_schema = properties.get(property_name)
                if not isinstance(property_schema, dict) or not property_schema:
                    validator.error(
                        f"{path}.properties.{property_name}",
                        "must be a non-empty schema object",
                    )

    for definition_name, property_contracts in sorted(
        _SCHEMA_CRITICAL_PROPERTY_CONTRACTS.items()
    ):
        definition = definitions.get(definition_name)
        if not isinstance(definition, dict):
            continue
        properties = definition.get("properties")
        if not isinstance(properties, dict):
            continue
        for property_name, expected_contract in sorted(property_contracts.items()):
            validator.literal(
                properties.get(property_name),
                f"schema.json.$defs.{definition_name}.properties.{property_name}",
                expected_contract,
            )

    for name, expected_contract in sorted(_SCHEMA_NON_OBJECT_CONTRACTS.items()):
        validator.literal(
            definitions.get(name),
            f"schema.json.$defs.{name}",
            expected_contract,
        )


def _candidate_documents(
    validator: _Validator,
    closure: Mapping[str, object],
    extraction_root: Path,
) -> tuple[dict[str, object] | None, dict[str, dict[str, object]]]:
    validator.literal(
        closure.get("candidate_id"),
        "manifest.json.input_candidate_closure.candidate_id",
        _CANDIDATE_ID,
    )
    validator.literal(
        closure.get("candidate_path"),
        "manifest.json.input_candidate_closure.candidate_path",
        _CANDIDATE_PATH,
    )
    observation_paths = validator.strings(
        closure.get("observation_paths"),
        "manifest.json.input_candidate_closure.observation_paths",
        nonempty=True,
        sorted_required=True,
    )
    if tuple(observation_paths) != _OBSERVATION_PATHS:
        validator.error(
            "manifest.json.input_candidate_closure.observation_paths",
            f"must equal the candidate's two observation paths {list(_OBSERVATION_PATHS)!r}",
        )
    validator.literal(
        closure.get("canonicalization_contract"),
        "manifest.json.input_candidate_closure.canonicalization_contract",
        "candidate-plus-sorted-observation-documents.canonical-json-v1",
    )
    observed_digest = validator.string(
        closure.get("candidate_closure_sha256"),
        "manifest.json.input_candidate_closure.candidate_closure_sha256",
        _SHA256,
    )

    candidate = validator.read(
        extraction_root
        / "candidates"
        / "hierarchical-simdgroup-threadgroup-reduction.json",
        _CANDIDATE_PATH,
    )
    observations: dict[str, dict[str, object]] = {}
    for repository_path in _OBSERVATION_PATHS:
        local_path = extraction_root / PurePosixPath(repository_path).relative_to(
            "abstraction_extraction"
        )
        document = validator.read(local_path, repository_path)
        if document is not None:
            observations[repository_path] = document
    if candidate is not None:
        validator.literal(
            candidate.get("candidate_id"),
            f"{_CANDIDATE_PATH}.candidate_id",
            _CANDIDATE_ID,
        )
        validator.literal(
            candidate.get("state"), f"{_CANDIDATE_PATH}.state", "extracted_unreviewed"
        )
        validator.literal(
            candidate.get("claim_scope"),
            f"{_CANDIDATE_PATH}.claim_scope",
            "source_pattern_only",
        )
        validator.literal(
            candidate.get("next_gate"),
            f"{_CANDIDATE_PATH}.next_gate",
            "hardware_informed_design",
        )
        observation_ids = validator.strings(
            candidate.get("observation_ids"),
            f"{_CANDIDATE_PATH}.observation_ids",
            nonempty=True,
            pattern=_ID,
            sorted_required=True,
        )
        derived_ids = sorted(
            cast(str, item.get("observation_id"))
            for item in observations.values()
            if isinstance(item.get("observation_id"), str)
        )
        if observation_ids != derived_ids:
            validator.error(
                f"{_CANDIDATE_PATH}.observation_ids",
                "does not equal IDs from the candidate-closure observation documents",
            )
    if candidate is not None and len(observations) == len(_OBSERVATION_PATHS):
        try:
            derived_digest = sha256(
                _canonical_json_bytes(
                    {"candidate": candidate, "observations": observations}
                )
            ).hexdigest()
        except (UnicodeError, ValueError) as error:
            validator.error(
                "manifest.json.input_candidate_closure.candidate_closure_sha256",
                f"candidate closure cannot be canonicalized ({error})",
            )
        else:
            if observed_digest is not None and observed_digest != derived_digest:
                validator.error(
                    "manifest.json.input_candidate_closure.candidate_closure_sha256",
                    "candidate closure canonical digest differs",
                )
    return candidate, observations


def _source(
    validator: _Validator, value: object, path: str
) -> tuple[str | None, str | None]:
    if not isinstance(value, dict):
        validator.error(path, "must be an object")
        return None, None
    kind = value.get("kind")
    keys = _PINNED_SOURCE_KEYS if kind == "vendor_pdf" else _WEB_SOURCE_KEYS
    source = validator.exact(value, path, keys)
    if source is None:
        return None, None
    source_id = validator.string(source.get("source_id"), f"{path}.source_id", _ID)
    if kind == "vendor_pdf":
        validator.literal(source.get("kind"), f"{path}.kind", "vendor_pdf")
        validator.string(source.get("title"), f"{path}.title")
        validator.string(source.get("url"), f"{path}.url", _HTTPS_URL)
        validator.string(source.get("revision"), f"{path}.revision")
        validator.strings(source.get("locators"), f"{path}.locators", nonempty=True)
        validator.string(
            source.get("content_sha256"), f"{path}.content_sha256", _SHA256
        )
        validator.literal(
            source.get("content_verification"),
            f"{path}.content_verification",
            "downloaded_content_identity_verified_once",
        )
        if source_id is not None:
            identity = _PINNED_PDF_IDENTITIES.get(source_id)
            if identity is None:
                validator.error(
                    f"{path}.source_id",
                    "is not one of the two pinned vendor PDF identities",
                )
            else:
                for field, expected in identity.items():
                    validator.literal(source.get(field), f"{path}.{field}", expected)
    else:
        validator.literal(
            source.get("kind"), f"{path}.kind", "vendor_web_documentation"
        )
        validator.string(source.get("title"), f"{path}.title")
        validator.string(source.get("url"), f"{path}.url", _HTTPS_URL)
        validator.strings(source.get("locators"), f"{path}.locators", nonempty=True)
        validator.literal(
            source.get("offline_content_verification"),
            f"{path}.offline_content_verification",
            "not_performed",
        )
    return source_id, cast(str | None, kind)


def _source_ids(
    validator: _Validator,
    value: object,
    path: str,
    known_sources: set[str],
) -> list[str]:
    values = validator.strings(
        value, path, nonempty=True, pattern=_ID, sorted_required=True
    )
    unknown = sorted(set(values) - known_sources)
    if unknown:
        validator.error(path, f"unknown source IDs: {', '.join(unknown)}")
    return values


def _hierarchical_probe_observation(
    validator: _Validator, value: object, path: str
) -> None:
    observation = validator.exact(value, path, _HIERARCHICAL_PROBE_KEYS)
    if observation is None:
        return

    observation_literals = {
        "observation_id": _HIERARCHICAL_PROBE_OBSERVATION_ID,
        "observed_at": "2026-08-26",
        "probe_scope": "fp32_rms_style_single_owner_hierarchical_reduction_only",
        "normalized_probe_output_retained": True,
        "raw_runner_output_retained": False,
        "stable_device_identifiers_retained": False,
        "external_workload_oracle_used": False,
        "performance_measured": False,
        "evaluation_evidence": False,
        "scientific_claim_authorized": False,
        "compiler_change_authorized": False,
        "human_hardware_review_cleared": False,
        "promotion_authorized": False,
        "scope": "local_engineering_observation_only",
    }
    for field, expected in observation_literals.items():
        validator.literal(observation.get(field), f"{path}.{field}", expected)

    snapshot = validator.exact(
        observation.get("repository_snapshot"),
        f"{path}.repository_snapshot",
        _HIERARCHICAL_PROBE_REPOSITORY_SNAPSHOT_KEYS,
    )
    if snapshot is not None:
        validator.literal(
            snapshot.get("repository_revision"),
            f"{path}.repository_snapshot.repository_revision",
            _HIERARCHICAL_PROBE_REPOSITORY_REVISION,
        )
        validator.literal(
            snapshot.get("worktree_clean_at_observation"),
            f"{path}.repository_snapshot.worktree_clean_at_observation",
            True,
        )
        source_paths = validator.strings(
            snapshot.get("source_paths"),
            f"{path}.repository_snapshot.source_paths",
            nonempty=True,
        )
        if tuple(source_paths) != _HIERARCHICAL_PROBE_SOURCE_PATHS:
            validator.error(
                f"{path}.repository_snapshot.source_paths",
                "must equal the complete ordered tracked probe source closure",
            )
        for index, source_path in enumerate(source_paths):
            validator.safe_path(
                source_path, f"{path}.repository_snapshot.source_paths[{index}]"
            )
        validator.literal(
            snapshot.get("binding_scope"),
            f"{path}.repository_snapshot.binding_scope",
            "complete_tracked_probe_source_closure",
        )

    artifact = validator.exact(
        observation.get("retained_artifact"),
        f"{path}.retained_artifact",
        _HIERARCHICAL_PROBE_RETAINED_ARTIFACT_KEYS,
    )
    if artifact is not None:
        artifact_literals = {
            "storage_scope": "external_generated_artifact_outside_checkout",
            "relative_path": _HIERARCHICAL_PROBE_ARTIFACT_RELATIVE_PATH,
            "artifact_kind": "open_cake_metal_hierarchical_reduction_probe_v1",
            "reference_status": "unverified_external_artifact_reference",
            "offline_validator_access": "not_available",
        }
        for field, expected in artifact_literals.items():
            validator.literal(
                artifact.get(field), f"{path}.retained_artifact.{field}", expected
            )
        validator.safe_path(
            artifact.get("relative_path"), f"{path}.retained_artifact.relative_path"
        )
        # This caller-root-relative path is an unverified convenience
        # reference. The offline stage validator has no artifact root and does
        # not locate, read, or establish byte identity for external results.

    normalized = validator.exact(
        observation.get("normalized_output"),
        f"{path}.normalized_output",
        _HIERARCHICAL_PROBE_NORMALIZED_OUTPUT_KEYS,
    )
    if normalized is not None:
        validator.literal(
            normalized.get("status"), f"{path}.normalized_output.status", "passed"
        )

        device = validator.exact(
            normalized.get("device"),
            f"{path}.normalized_output.device",
            _HIERARCHICAL_PROBE_DEVICE_KEYS,
        )
        if device is not None:
            device_literals = {
                "device_class": "Apple M4",
                "supports_apple9_or_newer": True,
                "max_threadgroup_memory_bytes": 32768,
            }
            for field, expected in device_literals.items():
                validator.literal(
                    device.get(field),
                    f"{path}.normalized_output.device.{field}",
                    expected,
                )

        pipeline = validator.exact(
            normalized.get("pipeline"),
            f"{path}.normalized_output.pipeline",
            _HIERARCHICAL_PROBE_PIPELINE_KEYS,
        )
        if pipeline is not None:
            pipeline_literals = {
                "entry_point": "hierarchical_reduction_w32_probe",
                "thread_execution_width": 32,
                "max_total_threads_per_threadgroup": 1024,
                "static_threadgroup_memory_bytes": 0,
                "dynamic_threadgroup_memory_bytes": 144,
            }
            for field, expected in pipeline_literals.items():
                validator.literal(
                    pipeline.get(field),
                    f"{path}.normalized_output.pipeline.{field}",
                    expected,
                )

        coverage = validator.exact(
            normalized.get("coverage"),
            f"{path}.normalized_output.coverage",
            _HIERARCHICAL_PROBE_COVERAGE_KEYS,
        )
        if coverage is not None:
            coverage_literals = {
                "simdgroup_counts": list(_HIERARCHICAL_PROBE_GROUP_COUNTS),
                "threads_per_threadgroup": list(_HIERARCHICAL_PROBE_THREAD_COUNTS),
                "epochs_per_dispatch": 2,
                "input_patterns": [
                    "((thread_mod_13)-6)*0.125+0.5",
                    "((thread_mod_11)-5)*0.25-0.75",
                ],
                "dirty_sentinels": [-12345.25, 9876.5],
                "single_owner_final_scalar_broadcast": True,
                "scratch_reused_within_dispatch": True,
                "uniform_reuse_barrier": "threadgroup_barrier(mem_threadgroup)",
                "layer_all_simdgroups_tested": False,
                "bfloat_input_conversion_tested": False,
            }
            for field, expected in coverage_literals.items():
                validator.literal(
                    coverage.get(field),
                    f"{path}.normalized_output.coverage.{field}",
                    expected,
                )

        oracle = validator.exact(
            normalized.get("oracle"),
            f"{path}.normalized_output.oracle",
            _HIERARCHICAL_PROBE_ORACLE_KEYS,
        )
        if oracle is not None:
            oracle_literals = {
                "dtype": "float32",
                "tolerance_rule": "exact_fp32_bits_for_dyadic_fixture",
                "kind": "internal_finite_probe_oracle_not_workload_oracle",
            }
            for field, expected in oracle_literals.items():
                validator.literal(
                    oracle.get(field),
                    f"{path}.normalized_output.oracle.{field}",
                    expected,
                )

        case_results = normalized.get("case_results")
        parsed_cases: list[dict[str, object]] = []
        if not isinstance(case_results, list):
            validator.error(
                f"{path}.normalized_output.case_results", "must be an array"
            )
        else:
            if len(case_results) != len(_HIERARCHICAL_PROBE_GROUP_COUNTS):
                validator.error(
                    f"{path}.normalized_output.case_results",
                    "must contain exactly 6 ordered SIMDgroup-count cases",
                )
            for index, case_value in enumerate(case_results):
                case_path = f"{path}.normalized_output.case_results[{index}]"
                case = validator.exact(
                    case_value, case_path, _HIERARCHICAL_PROBE_CASE_RESULT_KEYS
                )
                if case is None:
                    continue
                parsed_cases.append(case)
                if index >= len(_HIERARCHICAL_PROBE_GROUP_COUNTS):
                    continue
                groups = _HIERARCHICAL_PROBE_GROUP_COUNTS[index]
                threads = _HIERARCHICAL_PROBE_THREAD_COUNTS[index]
                expected_sums = [
                    sum(((thread % 13) - 6) * 0.125 + 0.5 for thread in range(threads)),
                    sum(((thread % 11) - 5) * 0.25 - 0.75 for thread in range(threads)),
                ]
                if tuple(expected_sums) != _HIERARCHICAL_PROBE_EXPECTED_SUMS[index]:
                    validator.error(
                        case_path,
                        "validator's independently derived finite-oracle sums differ "
                        "from its closed expected-sum contract",
                    )
                case_literals = {
                    "simdgroup_count": groups,
                    "threads_per_threadgroup": threads,
                    "expected_owner_sums": expected_sums,
                    "observed_owner_sums": expected_sums,
                    "broadcast_thread_counts": [threads, threads],
                    "dirty_sentinel_preserved": [True, True],
                    "owner_mismatch_count": 0,
                    "broadcast_mismatch_count": 0,
                    "dirty_snapshot_mismatch_count": 0,
                    "command_buffer_status": "completed",
                    "passed": True,
                }
                for field, expected in case_literals.items():
                    validator.literal(case.get(field), f"{case_path}.{field}", expected)

        aggregate = validator.exact(
            normalized.get("aggregate"),
            f"{path}.normalized_output.aggregate",
            _HIERARCHICAL_PROBE_AGGREGATE_KEYS,
        )
        if aggregate is not None:
            broadcast_checks = 2 * sum(_HIERARCHICAL_PROBE_THREAD_COUNTS)
            aggregate_literals = {
                "case_count": len(_HIERARCHICAL_PROBE_GROUP_COUNTS),
                "epoch_result_count": 2 * len(_HIERARCHICAL_PROBE_GROUP_COUNTS),
                "broadcast_consumer_check_count": broadcast_checks,
                "owner_mismatch_count": 0,
                "broadcast_mismatch_count": 0,
                "dirty_snapshot_mismatch_count": 0,
                "all_command_buffers_completed": True,
                "all_cases_passed": True,
            }
            for field, expected in aggregate_literals.items():
                validator.literal(
                    aggregate.get(field),
                    f"{path}.normalized_output.aggregate.{field}",
                    expected,
                )
            if len(parsed_cases) == len(_HIERARCHICAL_PROBE_GROUP_COUNTS):
                expected_sum_lists = [
                    item.get("expected_owner_sums") for item in parsed_cases
                ]
                broadcast_count_lists = [
                    item.get("broadcast_thread_counts") for item in parsed_cases
                ]
                mismatch_fields = (
                    "owner_mismatch_count",
                    "broadcast_mismatch_count",
                    "dirty_snapshot_mismatch_count",
                )
                if all(isinstance(item, list) for item in expected_sum_lists) and all(
                    isinstance(item, list) for item in broadcast_count_lists
                ):
                    derived_aggregate: dict[str, object] = {
                        "case_count": len(parsed_cases),
                        "epoch_result_count": sum(
                            len(cast(list[object], item)) for item in expected_sum_lists
                        ),
                        "broadcast_consumer_check_count": sum(
                            sum(cast(list[int], item))
                            for item in broadcast_count_lists
                            if all(
                                isinstance(count, int) and not isinstance(count, bool)
                                for count in cast(list[object], item)
                            )
                        ),
                        "all_command_buffers_completed": all(
                            item.get("command_buffer_status") == "completed"
                            for item in parsed_cases
                        ),
                        "all_cases_passed": all(
                            item.get("passed") is True for item in parsed_cases
                        ),
                    }
                    for field in mismatch_fields:
                        values = [item.get(field) for item in parsed_cases]
                        if all(
                            isinstance(item, int) and not isinstance(item, bool)
                            for item in values
                        ):
                            derived_aggregate[field] = sum(cast(list[int], values))
                    for field, expected in derived_aggregate.items():
                        validator.literal(
                            aggregate.get(field),
                            f"{path}.normalized_output.aggregate.{field}",
                            expected,
                        )

        if pipeline is not None and device is not None:
            width = pipeline.get("thread_execution_width")
            pipeline_limit = pipeline.get("max_total_threads_per_threadgroup")
            static_bytes = pipeline.get("static_threadgroup_memory_bytes")
            dynamic_bytes = pipeline.get("dynamic_threadgroup_memory_bytes")
            device_limit = device.get("max_threadgroup_memory_bytes")
            if isinstance(width, int) and not isinstance(width, bool) and width > 0:
                for index, (groups, threads) in enumerate(
                    zip(
                        _HIERARCHICAL_PROBE_GROUP_COUNTS,
                        _HIERARCHICAL_PROBE_THREAD_COUNTS,
                        strict=True,
                    )
                ):
                    if groups * width != threads:
                        validator.error(
                            f"{path}.normalized_output.case_results[{index}]."
                            "threads_per_threadgroup",
                            "does not derive as simdgroup_count * pipeline thread_execution_width",
                        )
            if (
                isinstance(pipeline_limit, int)
                and not isinstance(pipeline_limit, bool)
                and max(_HIERARCHICAL_PROBE_THREAD_COUNTS) > pipeline_limit
            ):
                validator.error(
                    f"{path}.normalized_output.pipeline.max_total_threads_per_threadgroup",
                    "does not admit the complete observed thread-count matrix",
                )
            if all(
                isinstance(number, int) and not isinstance(number, bool)
                for number in (static_bytes, dynamic_bytes, device_limit)
            ):
                assert isinstance(static_bytes, int)
                assert isinstance(dynamic_bytes, int)
                assert isinstance(device_limit, int)
                if static_bytes + dynamic_bytes > device_limit:
                    validator.error(
                        f"{path}.normalized_output.pipeline.dynamic_threadgroup_memory_bytes",
                        "static-plus-dynamic threadgroup memory exceeds the observed device limit",
                    )

    outcomes_value = observation.get("falsifier_outcomes")
    observed_outcome_ids: list[str] = []
    if not isinstance(outcomes_value, list):
        validator.error(f"{path}.falsifier_outcomes", "must be an array")
    else:
        if len(outcomes_value) != len(_HIERARCHICAL_PROBE_FALSIFIER_OUTCOMES):
            validator.error(
                f"{path}.falsifier_outcomes",
                "must contain exactly 12 local falsifier outcomes",
            )
        for index, outcome_value in enumerate(outcomes_value):
            outcome_path = f"{path}.falsifier_outcomes[{index}]"
            outcome = validator.exact(
                outcome_value, outcome_path, _LOCAL_FALSIFIER_OUTCOME_KEYS
            )
            if outcome is None:
                continue
            falsifier_id = validator.string(
                outcome.get("falsifier_id"), f"{outcome_path}.falsifier_id", _ID
            )
            if falsifier_id is not None:
                observed_outcome_ids.append(falsifier_id)
                expected_outcome = _HIERARCHICAL_PROBE_FALSIFIER_OUTCOMES.get(
                    falsifier_id
                )
                if expected_outcome is None:
                    validator.error(
                        f"{outcome_path}.falsifier_id",
                        "does not name a proposal falsifier in the local outcome contract",
                    )
                else:
                    validator.literal(
                        outcome.get("outcome"),
                        f"{outcome_path}.outcome",
                        expected_outcome,
                    )
            validator.string(outcome.get("basis"), f"{outcome_path}.basis")
        if len(set(observed_outcome_ids)) != len(observed_outcome_ids):
            validator.error(
                f"{path}.falsifier_outcomes", "falsifier outcome IDs must be unique"
            )
        if set(observed_outcome_ids) != set(_HIERARCHICAL_PROBE_FALSIFIER_OUTCOMES):
            missing = sorted(
                set(_HIERARCHICAL_PROBE_FALSIFIER_OUTCOMES) - set(observed_outcome_ids)
            )
            unexpected = sorted(
                set(observed_outcome_ids) - set(_HIERARCHICAL_PROBE_FALSIFIER_OUTCOMES)
            )
            details: list[str] = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            validator.error(
                f"{path}.falsifier_outcomes",
                "local falsifier outcome closure differs (" + "; ".join(details) + ")",
            )


def _evidence_document(
    validator: _Validator,
    document: dict[str, object],
    relative: str,
    target_path: Path,
) -> tuple[str | None, dict[str, dict[str, object]]]:
    validator.exact(document, relative, _EVIDENCE_KEYS)
    validator.literal(
        document.get("$schema"), f"{relative}.$schema", "../schema.json#/$defs/evidence"
    )
    validator.literal(document.get("schema_version"), f"{relative}.schema_version", 1)
    evidence_id = validator.string(
        document.get("evidence_id"), f"{relative}.evidence_id", _ID
    )
    validator.literal(
        evidence_id, f"{relative}.evidence_id", "apple-metal-hierarchical-reduction-v1"
    )
    _stage_literals(validator, document, relative, claim_scope="hardware_evidence_only")

    sources_value = document.get("sources")
    source_ids: list[str] = []
    source_kinds: list[str] = []
    pinned_pdf_ids: set[str] = set()
    if not isinstance(sources_value, list):
        validator.error(f"{relative}.sources", "must be an array")
    else:
        if len(sources_value) < 2:
            validator.error(f"{relative}.sources", "must contain at least 2 sources")
        for index, source_value in enumerate(sources_value):
            source_id, source_kind = _source(
                validator, source_value, f"{relative}.sources[{index}]"
            )
            if source_id is not None:
                source_ids.append(source_id)
            if source_kind is not None:
                source_kinds.append(source_kind)
            if source_id is not None and source_kind == "vendor_pdf":
                pinned_pdf_ids.add(source_id)
        if len(set(source_ids)) != len(source_ids):
            validator.error(f"{relative}.sources", "source IDs must be unique")
        if (
            "vendor_pdf" not in source_kinds
            or "vendor_web_documentation" not in source_kinds
        ):
            validator.error(
                f"{relative}.sources",
                "must include both pinned vendor PDFs and unverified vendor web documentation",
            )
        expected_pdf_ids = set(_PINNED_PDF_IDENTITIES)
        if pinned_pdf_ids != expected_pdf_ids:
            missing = sorted(expected_pdf_ids - pinned_pdf_ids)
            unexpected = sorted(pinned_pdf_ids - expected_pdf_ids)
            details: list[str] = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            validator.error(
                f"{relative}.sources",
                "pinned PDF identity closure differs (" + "; ".join(details) + ")",
            )
    vendor_sources = set(source_ids)
    local_observation_ids = {
        _LOCAL_PROBE_OBSERVATION_ID,
        _HIERARCHICAL_PROBE_OBSERVATION_ID,
    }
    known_sources = vendor_sources | local_observation_ids

    facts_value = document.get("facts")
    facts_by_id: dict[str, dict[str, object]] = {}
    referenced_sources: set[str] = set()
    hierarchical_probe_fact_ids: set[str] = set()
    if not isinstance(facts_value, list):
        validator.error(f"{relative}.facts", "must be an array")
    else:
        if not facts_value:
            validator.error(f"{relative}.facts", "must not be empty")
        for index, fact_value in enumerate(facts_value):
            path = f"{relative}.facts[{index}]"
            fact = validator.exact(fact_value, path, _FACT_KEYS)
            if fact is None:
                continue
            fact_id = validator.string(fact.get("fact_id"), f"{path}.fact_id", _ID)
            validator.string(fact.get("statement"), f"{path}.statement")
            scope = validator.string(fact.get("scope"), f"{path}.scope", _ID)
            authority = validator.string(
                fact.get("authority"), f"{path}.authority", _ID
            )
            if scope not in {
                "language_semantics",
                "gpu_family",
                "observed_device_pipeline",
            }:
                validator.error(f"{path}.scope", "must name a closed fact scope")
            if authority not in {
                "apple_vendor_documentation",
                "local_engineering_observation",
            }:
                validator.error(
                    f"{path}.authority", "must name a closed fact authority"
                )
            fact_source_ids = _source_ids(
                validator,
                fact.get("source_ids"),
                f"{path}.source_ids",
                known_sources,
            )
            referenced_sources.update(fact_source_ids)
            if authority == "local_engineering_observation":
                if scope != "observed_device_pipeline":
                    validator.error(
                        f"{path}.scope",
                        f"local fact {fact_id!r} cannot be promoted from an observed device or pipeline to a GPU-family or language-semantics scope",
                    )
                expected_local_source = (
                    _HIERARCHICAL_PROBE_OBSERVATION_ID
                    if fact_id in _HIERARCHICAL_PROBE_FACT_IDS
                    else _LOCAL_PROBE_OBSERVATION_ID
                )
                if fact_source_ids != [expected_local_source]:
                    validator.error(
                        f"{path}.source_ids",
                        "a local engineering fact must reference only its exact "
                        "local probe observation",
                    )
                if fact_id is not None and fact_source_ids == [
                    _HIERARCHICAL_PROBE_OBSERVATION_ID
                ]:
                    hierarchical_probe_fact_ids.add(fact_id)
            elif authority == "apple_vendor_documentation":
                if scope == "observed_device_pipeline":
                    validator.error(
                        f"{path}.scope",
                        "vendor documentation cannot be labeled as a local observed-device fact",
                    )
                non_vendor = sorted(set(fact_source_ids) - vendor_sources)
                if non_vendor:
                    validator.error(
                        f"{path}.source_ids",
                        "vendor facts may reference only listed vendor sources",
                    )
            if fact_id is not None:
                if fact_id in facts_by_id:
                    validator.error(path, f"duplicate fact_id {fact_id!r}")
                else:
                    facts_by_id[fact_id] = fact
    if hierarchical_probe_fact_ids != _HIERARCHICAL_PROBE_FACT_IDS:
        missing = sorted(_HIERARCHICAL_PROBE_FACT_IDS - hierarchical_probe_fact_ids)
        unexpected = sorted(hierarchical_probe_fact_ids - _HIERARCHICAL_PROBE_FACT_IDS)
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        validator.error(
            f"{relative}.facts",
            "hierarchical-probe local fact closure differs ("
            + "; ".join(details)
            + ")",
        )
    unused_sources = sorted(known_sources - referenced_sources)
    if unused_sources:
        validator.error(
            f"{relative}.sources",
            f"vendor sources or the local observation are not referenced by any hardware fact: {', '.join(unused_sources)}",
        )

    probe = validator.exact(
        document.get("local_probe_observation"),
        f"{relative}.local_probe_observation",
        _LOCAL_PROBE_KEYS,
    )
    if probe is not None:
        probe_literals = {
            "observation_id": _LOCAL_PROBE_OBSERVATION_ID,
            "observed_at": "2026-08-26",
            "device_class": "Apple M4",
            "supports_apple9_or_newer": True,
            "maximum_threadgroup_memory_bytes": 32768,
            "raw_output_retained": False,
            "performance_measured": False,
            "scientific_claim_authorized": False,
            "scope": "local_engineering_observation_only",
        }
        for field, expected in probe_literals.items():
            validator.literal(
                probe.get(field),
                f"{relative}.local_probe_observation.{field}",
                expected,
            )
        repository_snapshot = validator.exact(
            probe.get("repository_snapshot"),
            f"{relative}.local_probe_observation.repository_snapshot",
            _LOCAL_PROBE_REPOSITORY_SNAPSHOT_KEYS,
        )
        if repository_snapshot is not None:
            snapshot_literals = {
                "git_parent_revision": _GIT_PARENT_REVISION,
                "probe_path": _PROBE_PATH,
                "binding_scope": "captured_source_identity_only",
            }
            for field, expected in snapshot_literals.items():
                validator.literal(
                    repository_snapshot.get(field),
                    f"{relative}.local_probe_observation.repository_snapshot.{field}",
                    expected,
                )
            validator.string(
                repository_snapshot.get("reproducibility_note"),
                f"{relative}.local_probe_observation.repository_snapshot.reproducibility_note",
            )
        pipelines = probe.get("pipelines")
        observed_kinds: list[str] = []
        if not isinstance(pipelines, list):
            validator.error(
                f"{relative}.local_probe_observation.pipelines", "must be an array"
            )
        else:
            if len(pipelines) != 2:
                validator.error(
                    f"{relative}.local_probe_observation.pipelines",
                    "must contain exactly 2 pipelines",
                )
            for index, pipeline_value in enumerate(pipelines):
                path = f"{relative}.local_probe_observation.pipelines[{index}]"
                pipeline = validator.exact(pipeline_value, path, _PIPELINE_KEYS)
                if pipeline is None:
                    continue
                kind = pipeline.get("kind")
                if kind not in {"indexed_gather", "weighted_combine"}:
                    validator.error(
                        f"{path}.kind", "must name an observed probe pipeline"
                    )
                elif isinstance(kind, str):
                    observed_kinds.append(kind)
                validator.literal(
                    pipeline.get("thread_execution_width"),
                    f"{path}.thread_execution_width",
                    32,
                )
                validator.literal(
                    pipeline.get("max_total_threads_per_threadgroup"),
                    f"{path}.max_total_threads_per_threadgroup",
                    1024,
                )
            if sorted(observed_kinds) != ["indexed_gather", "weighted_combine"]:
                validator.error(
                    f"{relative}.local_probe_observation.pipelines",
                    "must contain each observed pipeline exactly once",
                )

    _hierarchical_probe_observation(
        validator,
        document.get("hierarchical_reduction_probe_observation"),
        f"{relative}.hierarchical_reduction_probe_observation",
    )

    context = validator.exact(
        document.get("non_evidentiary_context"),
        f"{relative}.non_evidentiary_context",
        _TARGET_CONTEXT_KEYS,
    )
    target = validator.read(target_path, "compiler/targets/apple_gpu_family9.json")
    if context is not None:
        context_literals = {
            "role": "non_evidentiary_context_only",
            "target_id": "apple_gpu_family9",
            "target_execution_group_width": 32,
            "target_maximum_threads_per_workgroup": 1024,
            "target_maximum_threadgroup_memory_bytes": 32768,
            "current_lowering_supports_threadgroup_allocation": False,
            "current_lowering_supports_threadgroup_barrier": False,
        }
        for field, expected in context_literals.items():
            validator.literal(
                context.get(field),
                f"{relative}.non_evidentiary_context.{field}",
                expected,
            )
        repository_snapshot = validator.exact(
            context.get("repository_snapshot"),
            f"{relative}.non_evidentiary_context.repository_snapshot",
            _NON_EVIDENTIARY_REPOSITORY_SNAPSHOT_KEYS,
        )
        if repository_snapshot is not None:
            snapshot_literals = {
                "git_parent_revision": _GIT_PARENT_REVISION,
                "target_path": _TARGET_PATH,
                "lowering_path": _LOWERING_PATH,
                "binding_scope": "non_evidentiary_implementation_context_only",
            }
            for field, expected in snapshot_literals.items():
                validator.literal(
                    repository_snapshot.get(field),
                    f"{relative}.non_evidentiary_context.repository_snapshot.{field}",
                    expected,
                )
        validator.string(
            context.get("context_note"),
            f"{relative}.non_evidentiary_context.context_note",
        )
        if target is not None:
            resources = target.get("resource_limits")
            if not isinstance(resources, Mapping):
                validator.error(
                    "compiler/targets/apple_gpu_family9.json.resource_limits",
                    "must be an object",
                )
                resources = {}
            comparisons = {
                "target_id": (
                    "target_id",
                    target.get("target_id"),
                    "apple_gpu_family9",
                ),
                "target_execution_group_width": (
                    "execution_group_width",
                    target.get("execution_group_width"),
                    32,
                ),
                "target_maximum_threads_per_workgroup": (
                    "resource_limits.maximum_threads_per_workgroup",
                    resources.get("maximum_threads_per_workgroup"),
                    1024,
                ),
                "target_maximum_threadgroup_memory_bytes": (
                    "resource_limits.maximum_threadgroup_memory_bytes",
                    resources.get("maximum_threadgroup_memory_bytes"),
                    32768,
                ),
            }
            for field, (target_field, observed, expected) in comparisons.items():
                if type(observed) is not type(expected) or observed != expected:  # noqa: E721
                    validator.error(
                        f"compiler/targets/apple_gpu_family9.json.{target_field}",
                        f"target semantic context must equal {expected!r}; observed {observed!r}",
                    )
                validator.literal(
                    context.get(field),
                    f"{relative}.non_evidentiary_context.{field}",
                    observed,
                )

    validator.strings(
        document.get("assumptions"), f"{relative}.assumptions", nonempty=True
    )
    _gate(
        validator,
        document.get("gate_separation"),
        f"{relative}.gate_separation",
    )
    validator.strings(
        document.get("non_claims"), f"{relative}.non_claims", nonempty=True
    )
    return evidence_id, facts_by_id


def _proposal_document(
    validator: _Validator,
    document: dict[str, object],
    relative: str,
    candidate: Mapping[str, object] | None,
    observations: Mapping[str, Mapping[str, object]],
    evidence_facts: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> str | None:
    validator.exact(document, relative, _PROPOSAL_KEYS)
    validator.literal(
        document.get("$schema"), f"{relative}.$schema", "../schema.json#/$defs/proposal"
    )
    validator.literal(document.get("schema_version"), f"{relative}.schema_version", 1)
    proposal_id = validator.string(
        document.get("proposal_id"), f"{relative}.proposal_id", _ID
    )
    validator.literal(
        proposal_id,
        f"{relative}.proposal_id",
        "hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1",
    )
    _stage_literals(validator, document, relative)

    binding = validator.exact(
        document.get("candidate_binding"),
        f"{relative}.candidate_binding",
        _CANDIDATE_BINDING_KEYS,
    )
    if binding is not None:
        validator.literal(
            binding.get("candidate_id"),
            f"{relative}.candidate_binding.candidate_id",
            _CANDIDATE_ID,
        )
        expected_signature = (
            candidate.get("pattern_signature")
            if candidate is not None
            else "simdgroup-partial.threadgroup-publish.simdgroup-final"
        )
        validator.literal(
            binding.get("pattern_signature"),
            f"{relative}.candidate_binding.pattern_signature",
            expected_signature,
        )
        observation_ids = validator.strings(
            binding.get("observation_ids"),
            f"{relative}.candidate_binding.observation_ids",
            nonempty=True,
            pattern=_ID,
            sorted_required=True,
        )
        expected_observation_ids = sorted(
            cast(str, item.get("observation_id"))
            for item in observations.values()
            if isinstance(item.get("observation_id"), str)
        )
        if observation_ids != expected_observation_ids:
            validator.error(
                f"{relative}.candidate_binding.observation_ids",
                "does not equal the input candidate's observation IDs",
            )
        validator.literal(
            binding.get("candidate_width_contract"),
            f"{relative}.candidate_binding.candidate_width_contract",
            "width_neutral",
        )
        expected_state = (
            candidate.get("state") if candidate is not None else "extracted_unreviewed"
        )
        expected_next_gate = (
            candidate.get("next_gate")
            if candidate is not None
            else "hardware_informed_design"
        )
        validator.literal(
            binding.get("expected_state"),
            f"{relative}.candidate_binding.expected_state",
            expected_state,
        )
        validator.literal(
            binding.get("expected_next_gate"),
            f"{relative}.candidate_binding.expected_next_gate",
            expected_next_gate,
        )

    evidence_binding = validator.exact(
        document.get("evidence_binding"),
        f"{relative}.evidence_binding",
        _EVIDENCE_BINDING_KEYS,
    )
    bound_fact_ids: set[str] = set()
    if evidence_binding is not None:
        evidence_id = validator.string(
            evidence_binding.get("evidence_id"),
            f"{relative}.evidence_binding.evidence_id",
            _ID,
        )
        if evidence_id is not None and evidence_id not in evidence_facts:
            validator.error(
                f"{relative}.evidence_binding.evidence_id",
                "does not reference a listed evidence artifact",
            )
        fact_ids = validator.strings(
            evidence_binding.get("fact_ids"),
            f"{relative}.evidence_binding.fact_ids",
            nonempty=True,
            pattern=_ID,
            sorted_required=True,
        )
        bound_fact_ids = set(fact_ids)
        known_fact_ids = (
            set(evidence_facts[evidence_id])
            if evidence_id is not None and evidence_id in evidence_facts
            else set()
        )
        unknown_facts = sorted(bound_fact_ids - known_fact_ids)
        if unknown_facts:
            validator.error(
                f"{relative}.evidence_binding.fact_ids",
                f"unknown fact IDs: {', '.join(unknown_facts)}",
            )

    design = validator.exact(document.get("design"), f"{relative}.design", _DESIGN_KEYS)
    if design is not None:
        width = validator.exact(
            design.get("width_policy"),
            f"{relative}.design.width_policy",
            _WIDTH_POLICY_KEYS,
        )
        if width is not None:
            width_literals = {
                "candidate_contract": "width_neutral",
                "specialization": "apple-family9-w32-guarded",
                "required_runtime_width": 32,
                "runtime_query": "MTLComputePipelineState.threadExecutionWidth",
                "mismatch_action": "fail_closed_before_dispatch",
                "threadgroup_multiple_requirement": "threads_per_threadgroup_mod_runtime_width_equals_zero",
                "partial_capacity_formula": "ceil(threads_per_threadgroup/runtime_width)<=runtime_width",
            }
            for field, expected in width_literals.items():
                validator.literal(
                    width.get(field),
                    f"{relative}.design.width_policy.{field}",
                    expected,
                )
        resource = validator.exact(
            design.get("resource_derivation"),
            f"{relative}.design.resource_derivation",
            _RESOURCE_KEYS,
        )
        if resource is not None:
            resource_literals = {
                "family_thread_ceiling": 1024,
                "specialized_width": 32,
                "maximum_simdgroups_at_family_ceiling": 32,
                "calculation": "1024/32=32",
                "partial_dtype": "float32",
                "partial_bytes": 4,
                "scratch_bytes_per_accumulator": 128,
                "scratch_calculation": "32*4=128",
                "variant_input_scope": "accumulator_count_is_current_variant_input_not_candidate_invariant",
                "threadgroup_allocation_alignment_bytes": 16,
                "total_threadgroup_memory_ceiling_bytes": 32768,
                "runtime_thread_limit_gate": "threads_per_threadgroup<=pipeline.maxTotalThreadsPerThreadgroup",
            }
            for field, expected in resource_literals.items():
                validator.literal(
                    resource.get(field),
                    f"{relative}.design.resource_derivation.{field}",
                    expected,
                )
            formula = validator.exact(
                resource.get("dynamic_allocation_formula"),
                f"{relative}.design.resource_derivation.dynamic_allocation_formula",
                _DYNAMIC_ALLOCATION_FORMULA_KEYS,
            )
            if formula is not None:
                formula_literals = {
                    "raw_dynamic_bytes": "raw_dynamic_bytes=accumulator_count*128+broadcast_scalar_bytes",
                    "aligned_dynamic_bytes": "aligned_dynamic_bytes=align_up(raw_dynamic_bytes,16)",
                    "allocation_strategy": "one_combined_threadgroup_region_per_variant",
                }
                for field, expected in formula_literals.items():
                    validator.literal(
                        formula.get(field),
                        f"{relative}.design.resource_derivation.dynamic_allocation_formula.{field}",
                        expected,
                    )
            variant_requirements = validator.exact(
                resource.get("variant_resource_requirements"),
                f"{relative}.design.resource_derivation.variant_resource_requirements",
                _VARIANT_RESOURCE_REQUIREMENTS_KEYS,
            )
            expected_variant_resources = {
                "rms_norm": {
                    "accumulator_count": 1,
                    "broadcast_scalar_bytes": 4,
                    "raw_dynamic_bytes": 132,
                    "aligned_dynamic_bytes": 144,
                    "layout": "partial[0]=bytes[0,128);broadcast_scalar=bytes[128,132);padding=bytes[132,144)",
                    "total_memory_gate": "pipeline.staticThreadgroupMemoryLength+144<=device.maxThreadgroupMemoryLength_and<=32768_family_ceiling",
                },
                "layer_norm": {
                    "accumulator_count": 2,
                    "broadcast_scalar_bytes": 0,
                    "raw_dynamic_bytes": 256,
                    "aligned_dynamic_bytes": 256,
                    "layout": "partial[0]=bytes[0,128);partial[1]=bytes[128,256);broadcast_scalar=none;padding=none",
                    "total_memory_gate": "pipeline.staticThreadgroupMemoryLength+256<=device.maxThreadgroupMemoryLength_and<=32768_family_ceiling",
                },
            }
            parsed_variant_resources: dict[str, dict[str, object]] = {}
            if variant_requirements is not None:
                for name, expected_fields in expected_variant_resources.items():
                    variant_path = (
                        f"{relative}.design.resource_derivation."
                        f"variant_resource_requirements.{name}"
                    )
                    variant_resource = validator.exact(
                        variant_requirements.get(name),
                        variant_path,
                        _VARIANT_RESOURCE_REQUIREMENT_KEYS,
                    )
                    if variant_resource is None:
                        continue
                    parsed_variant_resources[name] = variant_resource
                    for field, expected in expected_fields.items():
                        validator.literal(
                            variant_resource.get(field),
                            f"{variant_path}.{field}",
                            expected,
                        )
            threads = resource.get("family_thread_ceiling")
            width_value = resource.get("specialized_width")
            partial_bytes = resource.get("partial_bytes")
            groups = resource.get("maximum_simdgroups_at_family_ceiling")
            scratch = resource.get("scratch_bytes_per_accumulator")
            if all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (threads, width_value, partial_bytes, groups, scratch)
            ):
                assert isinstance(threads, int) and isinstance(width_value, int)
                assert (
                    isinstance(partial_bytes, int)
                    and isinstance(groups, int)
                    and isinstance(scratch, int)
                )
                if (
                    width_value <= 0
                    or threads // width_value != groups
                    or threads % width_value != 0
                ):
                    validator.error(
                        f"{relative}.design.resource_derivation.maximum_simdgroups_at_family_ceiling",
                        "does not derive exactly as 1024 / 32",
                    )
                if groups * partial_bytes != scratch:
                    validator.error(
                        f"{relative}.design.resource_derivation.scratch_bytes_per_accumulator",
                        "does not derive as maximum_simdgroups * partial_bytes",
                    )
                alignment = resource.get("threadgroup_allocation_alignment_bytes")
                ceiling = resource.get("total_threadgroup_memory_ceiling_bytes")
                if (
                    isinstance(alignment, int)
                    and alignment > 0
                    and scratch % alignment != 0
                ):
                    validator.error(
                        f"{relative}.design.resource_derivation.scratch_bytes_per_accumulator",
                        "must respect threadgroup allocation alignment",
                    )
                if isinstance(ceiling, int) and scratch > ceiling:
                    validator.error(
                        f"{relative}.design.resource_derivation.scratch_bytes_per_accumulator",
                        "must not exceed total threadgroup memory ceiling",
                    )
                if (
                    isinstance(alignment, int)
                    and not isinstance(alignment, bool)
                    and alignment > 0
                    and isinstance(ceiling, int)
                    and not isinstance(ceiling, bool)
                ):
                    for name, variant_resource in parsed_variant_resources.items():
                        count = variant_resource.get("accumulator_count")
                        scalar_bytes = variant_resource.get("broadcast_scalar_bytes")
                        raw_bytes = variant_resource.get("raw_dynamic_bytes")
                        aligned_bytes = variant_resource.get("aligned_dynamic_bytes")
                        if not all(
                            isinstance(value, int) and not isinstance(value, bool)
                            for value in (
                                count,
                                scalar_bytes,
                                raw_bytes,
                                aligned_bytes,
                            )
                        ):
                            continue
                        assert isinstance(count, int)
                        assert isinstance(scalar_bytes, int)
                        assert isinstance(raw_bytes, int)
                        assert isinstance(aligned_bytes, int)
                        derived_raw = count * scratch + scalar_bytes
                        variant_path = (
                            f"{relative}.design.resource_derivation."
                            f"variant_resource_requirements.{name}"
                        )
                        if raw_bytes != derived_raw:
                            validator.error(
                                f"{variant_path}.raw_dynamic_bytes",
                                "does not derive as accumulator_count * 128 + broadcast_scalar_bytes",
                            )
                        derived_aligned = (
                            (derived_raw + alignment - 1) // alignment
                        ) * alignment
                        if aligned_bytes != derived_aligned:
                            validator.error(
                                f"{variant_path}.aligned_dynamic_bytes",
                                "does not derive as align_up(raw_dynamic_bytes, 16)",
                            )
                        if aligned_bytes > ceiling:
                            validator.error(
                                f"{variant_path}.aligned_dynamic_bytes",
                                "must not exceed the total threadgroup memory ceiling",
                            )
        reduction = validator.exact(
            design.get("reduction_sequence"),
            f"{relative}.design.reduction_sequence",
            _REDUCTION_KEYS,
        )
        if reduction is not None:
            steps = validator.strings(
                reduction.get("steps"),
                f"{relative}.design.reduction_sequence.steps",
                nonempty=True,
            )
            if len(steps) < 5:
                validator.error(
                    f"{relative}.design.reduction_sequence.steps",
                    "must contain at least 5 steps",
                )
            validator.literal(
                reduction.get("publication_barrier"),
                f"{relative}.design.reduction_sequence.publication_barrier",
                "threadgroup_barrier(mem_threadgroup)",
            )
            validator.literal(
                reduction.get("barrier_participation"),
                f"{relative}.design.reduction_sequence.barrier_participation",
                "uniform_across_all_live_threads",
            )
            validator.literal(
                reduction.get("early_return_policy"),
                f"{relative}.design.reduction_sequence.early_return_policy",
                "no_early_return_or_divergent_control_around_threadgroup_barriers",
            )
        scratch_lifecycle = validator.exact(
            design.get("scratch_lifecycle"),
            f"{relative}.design.scratch_lifecycle",
            _SCRATCH_KEYS,
        )
        if scratch_lifecycle is not None:
            for field in (
                "initialization",
                "initialization_barrier",
                "reuse_rule",
                "final_scalar_visibility_rule",
            ):
                validator.string(
                    scratch_lifecycle.get(field),
                    f"{relative}.design.scratch_lifecycle.{field}",
                )
            validator.literal(
                scratch_lifecycle.get("unused_final_lanes"),
                f"{relative}.design.scratch_lifecycle.unused_final_lanes",
                "load_float32_zero_instead_of_unwritten_scratch",
            )
        variants = validator.exact(
            design.get("final_reducer_variants"),
            f"{relative}.design.final_reducer_variants",
            _REDUCER_VARIANTS_KEYS,
        )
        if variants is not None:
            for name in sorted(_REDUCER_VARIANTS_KEYS):
                variant = validator.exact(
                    variants.get(name),
                    f"{relative}.design.final_reducer_variants.{name}",
                    _REDUCER_VARIANT_KEYS,
                )
                if variant is not None:
                    for field in sorted(_REDUCER_VARIANT_KEYS):
                        validator.string(
                            variant.get(field),
                            f"{relative}.design.final_reducer_variants.{name}.{field}",
                        )
        numeric = validator.exact(
            design.get("numeric_contract"),
            f"{relative}.design.numeric_contract",
            _NUMERIC_KEYS,
        )
        if numeric is not None:
            numeric_literals = {
                "accumulator_dtype": "float32",
                "bfloat_input_policy": "convert_to_float32_before_simd_sum",
                "bitwise_reduction_order_guaranteed": False,
                "correctness_requirement": "external_oracle_with_explicit_tolerances_required_later",
            }
            for field, expected in numeric_literals.items():
                validator.literal(
                    numeric.get(field),
                    f"{relative}.design.numeric_contract.{field}",
                    expected,
                )

    decisions_value = document.get("decisions")
    decision_ids: set[str] = set()
    decision_fact_ids: set[str] = set()
    if not isinstance(decisions_value, list):
        validator.error(f"{relative}.decisions", "must be an array")
    else:
        if not decisions_value:
            validator.error(f"{relative}.decisions", "must not be empty")
        for index, decision_value in enumerate(decisions_value):
            path = f"{relative}.decisions[{index}]"
            decision = validator.exact(decision_value, path, _DECISION_KEYS)
            if decision is None:
                continue
            decision_id = validator.string(
                decision.get("decision_id"), f"{path}.decision_id", _ID
            )
            if decision_id is not None:
                if decision_id in decision_ids:
                    validator.error(path, f"duplicate decision_id {decision_id!r}")
                decision_ids.add(decision_id)
            validator.string(decision.get("statement"), f"{path}.statement")
            candidate_effect = validator.string(
                decision.get("candidate_effect"), f"{path}.candidate_effect", _ID
            )
            if candidate_effect not in {"preserve", "refine", "split", "reject"}:
                validator.error(
                    f"{path}.candidate_effect",
                    "must be one of preserve, refine, split, or reject",
                )
            fact_ids = validator.strings(
                decision.get("fact_ids"),
                f"{path}.fact_ids",
                nonempty=True,
                pattern=_ID,
                sorted_required=True,
            )
            unknown_facts = sorted(set(fact_ids) - bound_fact_ids)
            if unknown_facts:
                validator.error(
                    f"{path}.fact_ids",
                    f"decision references unknown or unbound fact IDs: {', '.join(unknown_facts)}",
                )
            decision_fact_ids.update(fact_ids)
            validator.string(decision.get("rationale"), f"{path}.rationale")
            validator.strings(
                decision.get("alternatives"),
                f"{path}.alternatives",
                nonempty=True,
            )
    validator.strings(
        document.get("assumptions"), f"{relative}.assumptions", nonempty=True
    )

    hypotheses_value = document.get("hypotheses")
    hypothesis_ids: set[str] = set()
    if not isinstance(hypotheses_value, list):
        validator.error(f"{relative}.hypotheses", "must be an array")
    else:
        if not hypotheses_value:
            validator.error(f"{relative}.hypotheses", "must not be empty")
        for index, hypothesis_value in enumerate(hypotheses_value):
            path = f"{relative}.hypotheses[{index}]"
            hypothesis = validator.exact(hypothesis_value, path, _HYPOTHESIS_KEYS)
            if hypothesis is None:
                continue
            hypothesis_id = validator.string(
                hypothesis.get("hypothesis_id"), f"{path}.hypothesis_id", _ID
            )
            if hypothesis_id is not None:
                if hypothesis_id in hypothesis_ids:
                    validator.error(path, f"duplicate hypothesis_id {hypothesis_id!r}")
                hypothesis_ids.add(hypothesis_id)
            validator.string(hypothesis.get("statement"), f"{path}.statement")
            related_ids = validator.strings(
                hypothesis.get("related_decision_ids"),
                f"{path}.related_decision_ids",
                nonempty=True,
                pattern=_ID,
                sorted_required=True,
            )
            unknown_decisions = sorted(set(related_ids) - decision_ids)
            if unknown_decisions:
                validator.error(
                    f"{path}.related_decision_ids",
                    "hypothesis references unknown decision IDs: "
                    + ", ".join(unknown_decisions),
                )
            validator.literal(hypothesis.get("status"), f"{path}.status", "unresolved")
            required_before = validator.string(
                hypothesis.get("required_before"), f"{path}.required_before", _ID
            )
            if required_before not in {
                "principle_review",
                "implementation",
                "port_acceptance",
                "performance_claim",
            }:
                validator.error(
                    f"{path}.required_before",
                    "must be one of principle_review, implementation, port_acceptance, or performance_claim",
                )
            falsifier = validator.exact(
                hypothesis.get("falsifier"),
                f"{path}.falsifier",
                _HYPOTHESIS_FALSIFIER_KEYS,
            )
            if falsifier is not None:
                for field in sorted(_HYPOTHESIS_FALSIFIER_KEYS):
                    validator.string(falsifier.get(field), f"{path}.falsifier.{field}")
            validator.literal(
                hypothesis.get("result_binding"), f"{path}.result_binding", None
            )

    findings_value = document.get("observed_scope_findings")
    finding_ids: list[str] = []
    finding_fact_ids: set[str] = set()
    if not isinstance(findings_value, list):
        validator.error(f"{relative}.observed_scope_findings", "must be an array")
    else:
        if len(findings_value) != len(_OBSERVED_SCOPE_FINDINGS):
            validator.error(
                f"{relative}.observed_scope_findings",
                "must contain exactly 4 observed-scope findings",
            )
        for index, finding_value in enumerate(findings_value):
            path = f"{relative}.observed_scope_findings[{index}]"
            finding = validator.exact(finding_value, path, _OBSERVED_SCOPE_FINDING_KEYS)
            if finding is None:
                continue
            finding_id = validator.string(
                finding.get("finding_id"), f"{path}.finding_id", _ID
            )
            expected_binding = (
                _OBSERVED_SCOPE_FINDINGS.get(finding_id)
                if finding_id is not None
                else None
            )
            if finding_id is not None:
                finding_ids.append(finding_id)
                if expected_binding is None:
                    validator.error(
                        f"{path}.finding_id",
                        "does not name an observed-scope finding in the closed contract",
                    )
            fact_ids = validator.strings(
                finding.get("fact_ids"),
                f"{path}.fact_ids",
                nonempty=True,
                pattern=_ID,
                sorted_required=True,
            )
            finding_fact_ids.update(fact_ids)
            unknown_facts = sorted(set(fact_ids) - bound_fact_ids)
            if unknown_facts:
                validator.error(
                    f"{path}.fact_ids",
                    "finding references unknown or unbound fact IDs: "
                    + ", ".join(unknown_facts),
                )
            does_not_resolve_hypothesis_ids = validator.strings(
                finding.get("does_not_resolve_hypothesis_ids"),
                f"{path}.does_not_resolve_hypothesis_ids",
                nonempty=True,
                pattern=_ID,
                sorted_required=True,
            )
            unknown_hypotheses = sorted(
                set(does_not_resolve_hypothesis_ids) - hypothesis_ids
            )
            if unknown_hypotheses:
                validator.error(
                    f"{path}.does_not_resolve_hypothesis_ids",
                    "finding references unknown hypothesis IDs: "
                    + ", ".join(unknown_hypotheses),
                )
            if expected_binding is not None:
                expected_fact_id, expected_hypothesis_ids = expected_binding
                validator.literal(
                    fact_ids,
                    f"{path}.fact_ids",
                    [expected_fact_id],
                )
                validator.literal(
                    does_not_resolve_hypothesis_ids,
                    f"{path}.does_not_resolve_hypothesis_ids",
                    list(expected_hypothesis_ids),
                )
            validator.literal(
                finding.get("status"),
                f"{path}.status",
                "reported_in_observed_scope",
            )
            validator.literal(
                finding.get("scope"),
                f"{path}.scope",
                "observed_apple_m4_fp32_rms_probe_pipeline_only",
            )
            validator.string(
                finding.get("reported_outcome"), f"{path}.reported_outcome"
            )
            validator.string(finding.get("scope_limit"), f"{path}.scope_limit")
        if len(set(finding_ids)) != len(finding_ids):
            validator.error(
                f"{relative}.observed_scope_findings", "finding IDs must be unique"
            )
        if set(finding_ids) != set(_OBSERVED_SCOPE_FINDINGS):
            missing = sorted(set(_OBSERVED_SCOPE_FINDINGS) - set(finding_ids))
            unexpected = sorted(set(finding_ids) - set(_OBSERVED_SCOPE_FINDINGS))
            details: list[str] = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            validator.error(
                f"{relative}.observed_scope_findings",
                "observed-scope finding closure differs (" + "; ".join(details) + ")",
            )

    unused_bound_facts = sorted(bound_fact_ids - (decision_fact_ids | finding_fact_ids))
    if unused_bound_facts:
        validator.error(
            f"{relative}.evidence_binding.fact_ids",
            "bound facts are not referenced by a design decision or observed-scope finding: "
            + ", ".join(unused_bound_facts),
        )

    falsifiers_value = document.get("falsifiers")
    falsifier_ids: list[str] = []
    if not isinstance(falsifiers_value, list):
        validator.error(f"{relative}.falsifiers", "must be an array")
    else:
        if not falsifiers_value:
            validator.error(f"{relative}.falsifiers", "must not be empty")
        for index, value in enumerate(falsifiers_value):
            path = f"{relative}.falsifiers[{index}]"
            falsifier = validator.exact(value, path, _FALSIFIER_KEYS)
            if falsifier is None:
                continue
            falsifier_id = validator.string(
                falsifier.get("falsifier_id"), f"{path}.falsifier_id", _ID
            )
            if falsifier_id is not None:
                falsifier_ids.append(falsifier_id)
            validator.string(falsifier.get("condition"), f"{path}.condition")
            validator.string(
                falsifier.get("required_action"), f"{path}.required_action"
            )
        if len(set(falsifier_ids)) != len(falsifier_ids):
            validator.error(f"{relative}.falsifiers", "falsifier IDs must be unique")
        observed_falsifier_ids = set(falsifier_ids)
        if observed_falsifier_ids != _FALSIFIER_IDS:
            missing = sorted(_FALSIFIER_IDS - observed_falsifier_ids)
            unexpected = sorted(observed_falsifier_ids - _FALSIFIER_IDS)
            details: list[str] = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            validator.error(
                f"{relative}.falsifiers",
                "required falsifier ID closure differs (" + "; ".join(details) + ")",
            )
    validator.strings(
        document.get("open_questions"), f"{relative}.open_questions", nonempty=True
    )
    _gate(validator, document.get("gate_separation"), f"{relative}.gate_separation")
    validator.strings(
        document.get("non_claims"), f"{relative}.non_claims", nonempty=True
    )
    return proposal_id


def _cross_document_falsifier_closure(
    validator: _Validator,
    evidence_documents: Sequence[tuple[str, dict[str, object]]],
    proposal_documents: Sequence[tuple[str, dict[str, object]]],
) -> None:
    """Relate local outcomes to the actual proposal, not a copied ID list."""

    if len(evidence_documents) != 1 or len(proposal_documents) != 1:
        return
    evidence_relative, evidence = evidence_documents[0]
    proposal_relative, proposal = proposal_documents[0]
    observation = evidence.get("hierarchical_reduction_probe_observation")
    if not isinstance(observation, dict):
        return
    outcomes = observation.get("falsifier_outcomes")
    falsifiers = proposal.get("falsifiers")
    if not isinstance(outcomes, list) or not isinstance(falsifiers, list):
        return

    def identifiers(values: list[object]) -> set[str] | None:
        result: set[str] = set()
        for value in values:
            if not isinstance(value, dict):
                return None
            identifier = value.get("falsifier_id")
            if not isinstance(identifier, str):
                return None
            result.add(identifier)
        return result

    outcome_ids = identifiers(outcomes)
    proposal_ids = identifiers(falsifiers)
    if outcome_ids is None or proposal_ids is None or outcome_ids == proposal_ids:
        return
    missing = sorted(proposal_ids - outcome_ids)
    unexpected = sorted(outcome_ids - proposal_ids)
    details: list[str] = []
    if missing:
        details.append("missing local outcomes: " + ", ".join(missing))
    if unexpected:
        details.append("outcomes absent from proposal: " + ", ".join(unexpected))
    validator.error(
        "cross_document_falsifier_closure",
        f"{evidence_relative} local outcome IDs differ from "
        f"{proposal_relative} falsifier IDs (" + "; ".join(details) + ")",
    )


def _identifier_sequence(
    document: Mapping[str, object], collection: str, identifier: str
) -> list[str] | None:
    values = document.get(collection)
    if not isinstance(values, list):
        return None
    result: list[str] = []
    for value in values:
        if not isinstance(value, dict):
            return None
        item = value.get(identifier)
        if not isinstance(item, str):
            return None
        result.append(item)
    return result


def _review_request_document(
    validator: _Validator,
    document: Mapping[str, object],
    relative: str,
    design_stage_id: str | None,
    evidence_documents: Sequence[tuple[str, dict[str, object]]],
    proposal_documents: Sequence[tuple[str, dict[str, object]]],
) -> str | None:
    validator.exact(document, relative, _REVIEW_REQUEST_KEYS)
    validator.literal(
        document.get("$schema"),
        f"{relative}.$schema",
        "../schema.json#/$defs/review_request",
    )
    validator.literal(document.get("schema_version"), f"{relative}.schema_version", 1)
    review_request_id = validator.string(
        document.get("review_request_id"), f"{relative}.review_request_id", _ID
    )
    validator.literal(
        review_request_id, f"{relative}.review_request_id", _REVIEW_REQUEST_ID
    )
    validator.literal(
        document.get("state"), f"{relative}.state", "awaiting_human_review"
    )
    validator.literal(
        document.get("authority"),
        f"{relative}.authority",
        "external_human_hardware_reviewer_only",
    )

    stage_binding = validator.exact(
        document.get("stage_binding"),
        f"{relative}.stage_binding",
        _REVIEW_STAGE_BINDING_KEYS,
    )
    if stage_binding is not None:
        manifest_binding = validator.exact(
            stage_binding.get("manifest"),
            f"{relative}.stage_binding.manifest",
            _REVIEW_MANIFEST_BINDING_KEYS,
        )
        if manifest_binding is not None:
            validator.literal(
                manifest_binding.get("design_stage_id"),
                f"{relative}.stage_binding.manifest.design_stage_id",
                design_stage_id,
            )
            validator.literal(
                manifest_binding.get("path"),
                f"{relative}.stage_binding.manifest.path",
                "hardware_informed_design/manifest.json",
            )

        evidence_binding = validator.exact(
            stage_binding.get("evidence"),
            f"{relative}.stage_binding.evidence",
            _REVIEW_EVIDENCE_BINDING_KEYS,
        )
        if evidence_binding is not None:
            if len(evidence_documents) != 1:
                validator.error(
                    f"{relative}.stage_binding.evidence",
                    "requires exactly one actual evidence document",
                )
            else:
                evidence_relative, evidence = evidence_documents[0]
                validator.literal(
                    evidence_binding.get("evidence_id"),
                    f"{relative}.stage_binding.evidence.evidence_id",
                    evidence.get("evidence_id"),
                )
                validator.literal(
                    evidence_binding.get("path"),
                    f"{relative}.stage_binding.evidence.path",
                    f"hardware_informed_design/{evidence_relative}",
                )

        proposal_binding = validator.exact(
            stage_binding.get("proposal"),
            f"{relative}.stage_binding.proposal",
            _REVIEW_PROPOSAL_BINDING_KEYS,
        )
        if proposal_binding is not None:
            if len(proposal_documents) != 1:
                validator.error(
                    f"{relative}.stage_binding.proposal",
                    "requires exactly one actual proposal document",
                )
            else:
                proposal_relative, proposal = proposal_documents[0]
                validator.literal(
                    proposal_binding.get("proposal_id"),
                    f"{relative}.stage_binding.proposal.proposal_id",
                    proposal.get("proposal_id"),
                )
                validator.literal(
                    proposal_binding.get("path"),
                    f"{relative}.stage_binding.proposal.path",
                    f"hardware_informed_design/{proposal_relative}",
                )

    actual_closure: dict[str, list[str]] | None = None
    if len(evidence_documents) == 1 and len(proposal_documents) == 1:
        evidence = evidence_documents[0][1]
        proposal = proposal_documents[0][1]
        candidate_closure = {
            "fact_ids": _identifier_sequence(evidence, "facts", "fact_id"),
            "decision_ids": _identifier_sequence(proposal, "decisions", "decision_id"),
            "hypothesis_ids": _identifier_sequence(
                proposal, "hypotheses", "hypothesis_id"
            ),
            "observed_scope_finding_ids": _identifier_sequence(
                proposal, "observed_scope_findings", "finding_id"
            ),
            "falsifier_ids": _identifier_sequence(
                proposal, "falsifiers", "falsifier_id"
            ),
        }
        if all(values is not None for values in candidate_closure.values()):
            actual_closure = cast(dict[str, list[str]], candidate_closure)

    closure = validator.exact(
        document.get("closure"), f"{relative}.closure", _REVIEW_CLOSURE_KEYS
    )
    observed_closure: dict[str, list[str]] = {}
    if closure is not None:
        for field in (
            "fact_ids",
            "decision_ids",
            "hypothesis_ids",
            "observed_scope_finding_ids",
            "falsifier_ids",
        ):
            values = validator.strings(
                closure.get(field),
                f"{relative}.closure.{field}",
                nonempty=True,
                pattern=_ID,
            )
            observed_closure[field] = values
            if actual_closure is not None and values != actual_closure[field]:
                validator.error(
                    f"{relative}.closure.{field}",
                    "must equal the ordered ID closure derived from the actual "
                    "evidence or proposal document",
                )

        closure_counts = validator.exact(
            closure.get("counts"),
            f"{relative}.closure.counts",
            _REVIEW_CLOSURE_COUNT_KEYS,
        )
        if closure_counts is not None:
            count_fields = {
                "fact_count": "fact_ids",
                "decision_count": "decision_ids",
                "hypothesis_count": "hypothesis_ids",
                "observed_scope_finding_count": "observed_scope_finding_ids",
                "falsifier_count": "falsifier_ids",
            }
            required_counts = {
                "fact_count": 21,
                "decision_count": 5,
                "hypothesis_count": 5,
                "observed_scope_finding_count": 4,
                "falsifier_count": 12,
            }
            for count_field, closure_field in count_fields.items():
                expected_count = (
                    len(actual_closure[closure_field])
                    if actual_closure is not None
                    else required_counts[count_field]
                )
                validator.literal(
                    closure_counts.get(count_field),
                    f"{relative}.closure.counts.{count_field}",
                    expected_count,
                )
                validator.literal(
                    closure_counts.get(count_field),
                    f"{relative}.closure.counts.{count_field}",
                    required_counts[count_field],
                )

    review_items = document.get("review_items")
    review_item_ids: list[str] = []
    covered_references = {
        field: set()
        for field in (
            "fact_ids",
            "decision_ids",
            "hypothesis_ids",
            "observed_scope_finding_ids",
            "falsifier_ids",
        )
    }
    if not isinstance(review_items, list):
        validator.error(f"{relative}.review_items", "must be an array")
    else:
        if len(review_items) != len(_REVIEW_ITEM_IDS):
            validator.error(
                f"{relative}.review_items",
                f"must contain exactly {len(_REVIEW_ITEM_IDS)} review items",
            )
        for index, value in enumerate(review_items):
            path = f"{relative}.review_items[{index}]"
            item = validator.exact(value, path, _REVIEW_ITEM_KEYS)
            if item is None:
                continue
            review_item_id = validator.string(
                item.get("review_item_id"), f"{path}.review_item_id", _ID
            )
            if review_item_id is not None:
                review_item_ids.append(review_item_id)
            validator.string(item.get("question"), f"{path}.question")
            validator.string(
                item.get("required_human_judgment"),
                f"{path}.required_human_judgment",
            )
            expected_references = (
                _REVIEW_ITEM_REFERENCES.get(review_item_id)
                if review_item_id is not None
                else None
            )
            if review_item_id is not None and expected_references is None:
                validator.error(
                    f"{path}.review_item_id",
                    "does not name a review item in the closed request contract",
                )
            for field in covered_references:
                references = validator.strings(
                    item.get(field),
                    f"{path}.{field}",
                    nonempty=True,
                    pattern=_ID,
                )
                covered_references[field].update(references)
                if actual_closure is not None:
                    unknown = sorted(set(references) - set(actual_closure[field]))
                    if unknown:
                        validator.error(
                            f"{path}.{field}",
                            "contains IDs outside the actual stage closure: "
                            + ", ".join(unknown),
                        )
                if expected_references is not None and references != list(
                    expected_references[field]
                ):
                    validator.error(
                        f"{path}.{field}",
                        f"must equal the closed {review_item_id!r} reference mapping",
                    )

        if review_item_ids != list(_REVIEW_ITEM_IDS):
            validator.error(
                f"{relative}.review_items",
                "review item IDs must equal the ordered closed request contract",
            )
        if len(set(review_item_ids)) != len(review_item_ids):
            validator.error(
                f"{relative}.review_items", "review item IDs must be unique"
            )
        if actual_closure is not None:
            for field, covered in covered_references.items():
                if covered != set(actual_closure[field]):
                    missing = sorted(set(actual_closure[field]) - covered)
                    unexpected = sorted(covered - set(actual_closure[field]))
                    details: list[str] = []
                    if missing:
                        details.append("missing: " + ", ".join(missing))
                    if unexpected:
                        details.append("unexpected: " + ", ".join(unexpected))
                    validator.error(
                        f"{relative}.review_items",
                        f"{field} union differs from the actual stage closure ("
                        + "; ".join(details)
                        + ")",
                    )

    ambiguity = validator.exact(
        document.get("phase_order_ambiguity"),
        f"{relative}.phase_order_ambiguity",
        _PHASE_ORDER_AMBIGUITY_KEYS,
    )
    option_ids: list[str] = []
    if ambiguity is not None:
        validator.literal(
            ambiguity.get("ambiguity_id"),
            f"{relative}.phase_order_ambiguity.ambiguity_id",
            _PHASE_ORDER_AMBIGUITY_ID,
        )
        validator.literal(
            ambiguity.get("status"),
            f"{relative}.phase_order_ambiguity.status",
            "unresolved_requires_human_choice",
        )
        validator.literal(
            ambiguity.get("contradiction"),
            f"{relative}.phase_order_ambiguity.contradiction",
            _RESOURCE_PHASE_CONTRADICTION,
        )
        options = ambiguity.get("resolution_options")
        if not isinstance(options, list):
            validator.error(
                f"{relative}.phase_order_ambiguity.resolution_options",
                "must be an array",
            )
        else:
            if len(options) != 2:
                validator.error(
                    f"{relative}.phase_order_ambiguity.resolution_options",
                    "must contain exactly two human-selectable options",
                )
            for index, value in enumerate(options):
                path = f"{relative}.phase_order_ambiguity.resolution_options[{index}]"
                option = validator.exact(
                    value, path, _PHASE_ORDER_RESOLUTION_OPTION_KEYS
                )
                if option is None:
                    continue
                option_id = validator.string(
                    option.get("option_id"), f"{path}.option_id", _ID
                )
                if option_id is not None:
                    option_ids.append(option_id)
                if index < len(_PHASE_ORDER_OPTIONS):
                    expected_option = _PHASE_ORDER_OPTIONS[index]
                    for field in ("option_id", "action", "consequence"):
                        validator.literal(
                            option.get(field),
                            f"{path}.{field}",
                            expected_option[field],
                        )
            if option_ids != list(_PHASE_ORDER_OPTION_IDS):
                validator.error(
                    f"{relative}.phase_order_ambiguity.resolution_options",
                    "option IDs must equal the two ordered human resolution choices",
                )
            if len(set(option_ids)) != len(option_ids):
                validator.error(
                    f"{relative}.phase_order_ambiguity.resolution_options",
                    "option IDs must be unique",
                )
        validator.literal(
            ambiguity.get("selection_required"),
            f"{relative}.phase_order_ambiguity.selection_required",
            True,
        )
        validator.literal(
            ambiguity.get("automation_may_select"),
            f"{relative}.phase_order_ambiguity.automation_may_select",
            False,
        )

    if len(proposal_documents) == 1:
        proposal_relative, proposal_document = proposal_documents[0]
        hypotheses = proposal_document.get("hypotheses")
        resource_hypothesis = None
        resource_hypothesis_index: int | None = None
        if isinstance(hypotheses, list):
            for index, item in enumerate(hypotheses):
                if (
                    isinstance(item, dict)
                    and item.get("hypothesis_id")
                    == "resource-accounting-fits-selected-pipelines"
                ):
                    resource_hypothesis = item
                    resource_hypothesis_index = index
                    break
        if not isinstance(resource_hypothesis, dict):
            validator.error(
                f"{relative}.phase_order_ambiguity",
                "actual proposal lacks the resource-accounting hypothesis",
            )
        else:
            hypothesis_path = (
                f"{proposal_relative}.hypotheses[{resource_hypothesis_index}]"
            )
            if resource_hypothesis.get("required_before") != "implementation":
                validator.error(
                    f"{hypothesis_path}.required_before",
                    "resource-accounting-fits-selected-pipelines must retain "
                    "required_before='implementation'",
                )
            falsifier = resource_hypothesis.get("falsifier")
            if not isinstance(falsifier, dict):
                validator.error(
                    f"{hypothesis_path}.falsifier",
                    "must retain the resource phase-order falsifier contract",
                )
            else:
                for field, expected in sorted(
                    _RESOURCE_HYPOTHESIS_FALSIFIER_CONTRACT.items()
                ):
                    if falsifier.get(field) == expected:
                        continue
                    validator.error(
                        f"{hypothesis_path}.falsifier.{field}",
                        "resource-accounting-fits-selected-pipelines must retain "
                        f"the exact {field} semantics",
                    )

    decision_artifact = validator.exact(
        document.get("decision_artifact"),
        f"{relative}.decision_artifact",
        _ABSENT_DECISION_ARTIFACT_KEYS,
    )
    if decision_artifact is not None:
        validator.literal(
            decision_artifact.get("status"),
            f"{relative}.decision_artifact.status",
            "absent",
        )
        validator.literal(
            decision_artifact.get("path"),
            f"{relative}.decision_artifact.path",
            None,
        )
        validator.literal(
            decision_artifact.get("decision_id"),
            f"{relative}.decision_artifact.decision_id",
            None,
        )

    decision_contract = validator.exact(
        document.get("external_human_decision_contract"),
        f"{relative}.external_human_decision_contract",
        _EXTERNAL_HUMAN_DECISION_CONTRACT_KEYS,
    )
    if decision_contract is not None:
        validator.literal(
            decision_contract.get("decision_authority"),
            f"{relative}.external_human_decision_contract.decision_authority",
            "external_human_hardware_reviewer_only",
        )
        validator.literal(
            decision_contract.get("global_dispositions"),
            f"{relative}.external_human_decision_contract.global_dispositions",
            list(_REVIEW_DISPOSITIONS),
        )
        required_item_ids = validator.strings(
            decision_contract.get("required_review_item_ids"),
            f"{relative}.external_human_decision_contract.required_review_item_ids",
            nonempty=True,
            pattern=_ID,
        )
        if required_item_ids != list(_REVIEW_ITEM_IDS):
            validator.error(
                f"{relative}.external_human_decision_contract.required_review_item_ids",
                "must equal the ordered review item closure",
            )
        per_item_contract = validator.exact(
            decision_contract.get("per_item_verdict_contract"),
            f"{relative}.external_human_decision_contract.per_item_verdict_contract",
            _PER_ITEM_VERDICT_CONTRACT_KEYS,
        )
        if per_item_contract is not None:
            validator.literal(
                per_item_contract.get("per_item_verdict_required"),
                f"{relative}.external_human_decision_contract."
                "per_item_verdict_contract.per_item_verdict_required",
                True,
            )
            validator.literal(
                per_item_contract.get("verdict_values"),
                f"{relative}.external_human_decision_contract."
                "per_item_verdict_contract.verdict_values",
                list(_REVIEW_DISPOSITIONS),
            )
            validator.literal(
                per_item_contract.get("global_disposition_aggregation"),
                f"{relative}.external_human_decision_contract."
                "per_item_verdict_contract.global_disposition_aggregation",
                "maximum_review_item_verdict_severity",
            )
            validator.literal(
                per_item_contract.get("verdict_severity_order"),
                f"{relative}.external_human_decision_contract."
                "per_item_verdict_contract.verdict_severity_order",
                list(_REVIEW_DISPOSITIONS),
            )
            validator.literal(
                per_item_contract.get("localized_reason_required"),
                f"{relative}.external_human_decision_contract."
                "per_item_verdict_contract.localized_reason_required",
                True,
            )
            validator.literal(
                per_item_contract.get(
                    "stage_clear_requires_all_item_verdicts_approved"
                ),
                f"{relative}.external_human_decision_contract."
                "per_item_verdict_contract."
                "stage_clear_requires_all_item_verdicts_approved",
                True,
            )
            validator.literal(
                per_item_contract.get("non_approved_item_blocks_stage_transition"),
                f"{relative}.external_human_decision_contract."
                "per_item_verdict_contract."
                "non_approved_item_blocks_stage_transition",
                True,
            )
        validator.literal(
            decision_contract.get("reviewed_git_revision_required"),
            f"{relative}.external_human_decision_contract."
            "reviewed_git_revision_required",
            True,
        )
        ambiguity_contract = validator.exact(
            decision_contract.get("ambiguity_resolution_contract"),
            f"{relative}.external_human_decision_contract."
            "ambiguity_resolution_contract",
            _AMBIGUITY_RESOLUTION_CONTRACT_KEYS,
        )
        if ambiguity_contract is not None:
            validator.literal(
                ambiguity_contract.get("required_ambiguity_id"),
                f"{relative}.external_human_decision_contract."
                "ambiguity_resolution_contract.required_ambiguity_id",
                _PHASE_ORDER_AMBIGUITY_ID,
            )
            validator.literal(
                ambiguity_contract.get("allowed_option_ids"),
                f"{relative}.external_human_decision_contract."
                "ambiguity_resolution_contract.allowed_option_ids",
                list(_PHASE_ORDER_OPTION_IDS),
            )
            validator.literal(
                ambiguity_contract.get("localized_reason_required"),
                f"{relative}.external_human_decision_contract."
                "ambiguity_resolution_contract.localized_reason_required",
                True,
            )
            ambiguity_literals = {
                "approval_compatible_option_id": _APPROVAL_COMPATIBLE_OPTION_ID,
                "successor_required_option_id": _SUCCESSOR_REQUIRED_OPTION_ID,
                "successor_required_option_allowed_global_dispositions": list(
                    _SUCCESSOR_REQUIRED_GLOBAL_DISPOSITIONS
                ),
                "successor_required_option_required_non_approved_review_item_id": (
                    _SUCCESSOR_REQUIRED_NON_APPROVED_REVIEW_ITEM_ID
                ),
            }
            for field, expected in ambiguity_literals.items():
                validator.literal(
                    ambiguity_contract.get(field),
                    f"{relative}.external_human_decision_contract."
                    f"ambiguity_resolution_contract.{field}",
                    expected,
                )
        validator.literal(
            decision_contract.get("stage_transition_rule"),
            f"{relative}.external_human_decision_contract.stage_transition_rule",
            _STAGE_TRANSITION_RULE,
        )
        validator.literal(
            decision_contract.get("automation_may_write_decision"),
            f"{relative}.external_human_decision_contract."
            "automation_may_write_decision",
            False,
        )

    authorizations = validator.exact(
        document.get("authorizations"),
        f"{relative}.authorizations",
        _REVIEW_AUTHORIZATION_KEYS,
    )
    if authorizations is not None:
        for field in sorted(_REVIEW_AUTHORIZATION_KEYS):
            validator.literal(
                authorizations.get(field), f"{relative}.authorizations.{field}", False
            )
    validator.strings(
        document.get("non_claims"), f"{relative}.non_claims", nonempty=True
    )
    return review_request_id


def _run_git(
    validator: _Validator,
    repository_root: Path,
    arguments: Sequence[str],
    diagnostic_path: str,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repository_root),
                *arguments,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=_git_environment(),
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as error:
        validator.error(
            diagnostic_path, f"could not inspect local Git repository ({error})"
        )
        return None


def _run_git_bytes(
    validator: _Validator,
    repository_root: Path,
    arguments: Sequence[str],
    diagnostic_path: str,
) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repository_root),
                *arguments,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_git_environment(),
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        validator.error(
            diagnostic_path, f"could not inspect local Git repository ({error})"
        )
        return None


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_NO_LAZY_FETCH"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def _verify_reviewed_git_revision(
    validator: _Validator,
    repository_root: Path,
    reviewed_git_revision: str,
) -> bool:
    diagnostic_path = "review-decision.binding.reviewed_git_revision"
    top_level = _run_git(
        validator,
        repository_root,
        ("rev-parse", "--show-toplevel"),
        "repository-root",
    )
    if top_level is None:
        return False
    if top_level.returncode != 0:
        detail = top_level.stderr.strip() or "not a non-bare Git worktree"
        validator.error(
            "repository-root",
            f"must be the root of a non-bare Git worktree ({detail})",
        )
        return False
    discovered_text = top_level.stdout.strip()
    if not discovered_text or "\n" in discovered_text:
        validator.error(
            "repository-root", "Git worktree discovery returned an invalid path"
        )
        return False
    try:
        discovered_root = Path(discovered_text).resolve(strict=True)
        same_repository = discovered_root.samefile(repository_root)
    except OSError as error:
        validator.error(
            "repository-root", f"could not verify discovered Git worktree ({error})"
        )
        return False
    if not same_repository:
        validator.error(
            "repository-root",
            "Git worktree discovery must resolve to the supplied repository root",
        )
        return False
    verified = _run_git(
        validator,
        repository_root,
        (
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{reviewed_git_revision}^{{commit}}",
        ),
        diagnostic_path,
    )
    if verified is None:
        return False
    if verified.returncode != 0 or verified.stdout.strip() != reviewed_git_revision:
        detail = verified.stderr.strip() or "commit does not exist"
        validator.error(
            diagnostic_path,
            f"must name an existing full commit in this repository ({detail})",
        )
        return False

    tree = _run_git(
        validator,
        repository_root,
        (
            "ls-tree",
            "-r",
            "--full-tree",
            reviewed_git_revision,
            "--",
            *_HARDWARE_REVIEW_ARTIFACT_PATHS,
        ),
        diagnostic_path,
    )
    if tree is None:
        return False
    if tree.returncode != 0:
        detail = tree.stderr.strip() or "git ls-tree failed"
        validator.error(
            diagnostic_path,
            f"could not inspect the reviewed artifact tree ({detail})",
        )
        return False

    entries: dict[str, tuple[str, str, str]] = {}
    malformed_entries: list[str] = []
    for line in tree.stdout.splitlines():
        try:
            metadata, path = line.split("\t", 1)
            mode, object_type, object_id = metadata.split(" ", 2)
        except ValueError:
            malformed_entries.append(line)
            continue
        if path in entries:
            malformed_entries.append(path)
            continue
        entries[path] = (mode, object_type, object_id)
    if malformed_entries:
        validator.error(
            diagnostic_path,
            "reviewed tree returned malformed or duplicate entries: "
            + ", ".join(sorted(malformed_entries)),
        )
    expected_paths = set(_HARDWARE_REVIEW_ARTIFACT_PATHS)
    actual_paths = set(entries)
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing:
        validator.error(
            diagnostic_path,
            "reviewed commit is missing required closure paths: " + ", ".join(missing),
        )
    if unexpected:
        validator.error(
            diagnostic_path,
            "reviewed tree returned unexpected closure paths: " + ", ".join(unexpected),
        )
    for path in sorted(expected_paths & actual_paths):
        mode, object_type, object_id = entries[path]
        if mode != "100644" or object_type != "blob":
            validator.error(
                f"review-decision.binding.reviewed_artifact_paths[{path}]",
                "reviewed tree entry must be a 100644 blob",
            )
        if _GIT_COMMIT.fullmatch(object_id) is None:
            validator.error(
                f"review-decision.binding.reviewed_artifact_paths[{path}]",
                "reviewed tree blob must have a full 40-character object ID",
            )

    live_bytes: dict[str, bytes] = {}
    for path in _HARDWARE_REVIEW_ARTIFACT_PATHS:
        live_path = repository_root / path
        try:
            live_is_file = not live_path.is_symlink() and live_path.is_file()
        except OSError as error:
            validator.error(
                f"review-decision.binding.reviewed_artifact_paths[{path}]",
                f"could not inspect live closure path ({error})",
            )
            continue
        if not live_is_file:
            validator.error(
                f"review-decision.binding.reviewed_artifact_paths[{path}]",
                "live closure path must be a regular non-symlink file",
            )
            continue
        try:
            live_bytes[path] = live_path.read_bytes()
        except OSError as error:
            validator.error(
                f"review-decision.binding.reviewed_artifact_paths[{path}]",
                f"could not read live closure bytes ({error})",
            )

    if validator.errors:
        return False
    drifted_paths: list[str] = []
    for path in _HARDWARE_REVIEW_ARTIFACT_PATHS:
        _mode, _object_type, object_id = entries[path]
        blob = _run_git_bytes(
            validator,
            repository_root,
            ("cat-file", "blob", object_id),
            f"review-decision.binding.reviewed_artifact_paths[{path}]",
        )
        if blob is None:
            continue
        if blob.returncode != 0:
            detail = blob.stderr.decode("utf-8", errors="replace").strip()
            validator.error(
                f"review-decision.binding.reviewed_artifact_paths[{path}]",
                "could not read reviewed tree blob"
                + (f" ({detail})" if detail else ""),
            )
            continue
        if blob.stdout != live_bytes[path]:
            drifted_paths.append(path)
    if drifted_paths:
        validator.error(
            diagnostic_path,
            "reviewed commit closure differs byte-for-byte from the live worktree: "
            + ", ".join(drifted_paths),
        )
    return not validator.errors


def _review_decision_document(
    validator: _Validator,
    document: Mapping[str, object],
    design_stage_id: str,
    repository_root: Path,
) -> _HardwareReviewDecisionResult | None:
    start_error_count = len(validator.errors)
    relative = "review-decision"
    validator.exact(document, relative, _HARDWARE_REVIEW_DECISION_KEYS)
    validator.literal(
        document.get("$schema"),
        f"{relative}.$schema",
        _HARDWARE_REVIEW_DECISION_SCHEMA,
    )
    validator.literal(document.get("schema_version"), f"{relative}.schema_version", 1)
    decision_id = validator.string(
        document.get("decision_id"), f"{relative}.decision_id", _ID
    )
    validator.literal(
        document.get("kind"),
        f"{relative}.kind",
        "external_human_hardware_review_decision",
    )
    validator.literal(
        document.get("authority"),
        f"{relative}.authority",
        "external_human_hardware_reviewer_only",
    )

    reviewed_git_revision: str | None = None
    binding = validator.exact(
        document.get("binding"),
        f"{relative}.binding",
        _REVIEW_DECISION_REQUEST_BINDING_KEYS,
    )
    if binding is not None:
        validator.literal(
            binding.get("design_stage_id"),
            f"{relative}.binding.design_stage_id",
            design_stage_id,
        )
        validator.literal(
            binding.get("request_id"),
            f"{relative}.binding.request_id",
            _REVIEW_REQUEST_ID,
        )
        validator.literal(
            binding.get("path"),
            f"{relative}.binding.path",
            f"hardware_informed_design/{_REVIEW_REQUEST_PATH}",
        )
        reviewed_git_revision = validator.string(
            binding.get("reviewed_git_revision"),
            f"{relative}.binding.reviewed_git_revision",
            _GIT_COMMIT,
        )
        validator.literal(
            binding.get("reviewed_artifact_paths"),
            f"{relative}.binding.reviewed_artifact_paths",
            list(_HARDWARE_REVIEW_ARTIFACT_PATHS),
        )

    attestation = validator.exact(
        document.get("reviewer_attestation"),
        f"{relative}.reviewer_attestation",
        _REVIEWER_ATTESTATION_KEYS,
    )
    if attestation is not None:
        validator.human_review_text(
            attestation.get("reviewer_identity"),
            f"{relative}.reviewer_attestation.reviewer_identity",
        )
        validator.literal(
            attestation.get("authorship_attestation"),
            f"{relative}.reviewer_attestation.authorship_attestation",
            "external-human-outside-automation",
        )
        validator.human_review_text(
            attestation.get("decision_basis"),
            f"{relative}.reviewer_attestation.decision_basis",
        )

    disposition = validator.string(
        document.get("global_disposition"), f"{relative}.global_disposition"
    )
    if disposition is not None and disposition not in _REVIEW_DISPOSITIONS:
        validator.error(
            f"{relative}.global_disposition",
            "must be one of " + ", ".join(_REVIEW_DISPOSITIONS),
        )

    item_ids: list[str] = []
    item_dispositions: list[str] = []
    item_disposition_by_id: dict[str, str] = {}
    item_values = document.get("item_verdicts")
    if not isinstance(item_values, list):
        validator.error(f"{relative}.item_verdicts", "must be an array")
    else:
        if len(item_values) != len(_REVIEW_ITEM_IDS):
            validator.error(
                f"{relative}.item_verdicts",
                f"must contain exactly {len(_REVIEW_ITEM_IDS)} review-item verdicts",
            )
        for index, value in enumerate(item_values):
            path = f"{relative}.item_verdicts[{index}]"
            item = validator.exact(value, path, _REVIEW_ITEM_VERDICT_KEYS)
            if item is None:
                continue
            item_id = validator.string(
                item.get("review_item_id"), f"{path}.review_item_id", _ID
            )
            verdict = validator.string(item.get("verdict"), f"{path}.verdict")
            if verdict is not None and verdict not in _REVIEW_DISPOSITIONS:
                validator.error(
                    f"{path}.verdict",
                    "must be one of " + ", ".join(_REVIEW_DISPOSITIONS),
                )
            validator.human_review_text(
                item.get("localized_reason"), f"{path}.localized_reason"
            )
            if item_id is not None:
                item_ids.append(item_id)
            if item_id is not None and verdict in _REVIEW_DISPOSITIONS:
                item_disposition_by_id[item_id] = verdict
            if verdict in _REVIEW_DISPOSITIONS:
                item_dispositions.append(verdict)
        if item_ids != list(_REVIEW_ITEM_IDS):
            validator.error(
                f"{relative}.item_verdicts",
                "review_item_id values must equal the ordered review-item closure",
            )

    selected_option_id: str | None = None
    ambiguity = validator.exact(
        document.get("ambiguity_resolution"),
        f"{relative}.ambiguity_resolution",
        _REVIEW_AMBIGUITY_RESOLUTION_KEYS,
    )
    if ambiguity is not None:
        validator.literal(
            ambiguity.get("ambiguity_id"),
            f"{relative}.ambiguity_resolution.ambiguity_id",
            _PHASE_ORDER_AMBIGUITY_ID,
        )
        selected_option_id = validator.string(
            ambiguity.get("selected_option_id"),
            f"{relative}.ambiguity_resolution.selected_option_id",
            _ID,
        )
        if (
            selected_option_id is not None
            and selected_option_id not in _PHASE_ORDER_OPTION_IDS
        ):
            validator.error(
                f"{relative}.ambiguity_resolution.selected_option_id",
                "must be one of " + ", ".join(_PHASE_ORDER_OPTION_IDS),
            )
        validator.human_review_text(
            ambiguity.get("localized_reason"),
            f"{relative}.ambiguity_resolution.localized_reason",
        )

    acknowledgements = validator.exact(
        document.get("scope_acknowledgements"),
        f"{relative}.scope_acknowledgements",
        _REVIEW_SCOPE_ACKNOWLEDGEMENT_KEYS,
    )
    if acknowledgements is not None:
        acknowledgement_literals = {
            "approval_scope": "stage3_hardware_review_only",
            "principle_driven_iteration_required": True,
            "resource_accounting_hypothesis_resolved": False,
            "compiler_change_authorized": False,
            "implementation_authorized": False,
            "evaluation_authorized": False,
            "performance_claim_authorized": False,
            "scientific_claim_authorized": False,
            "promotion_authorized": False,
        }
        for field, expected in acknowledgement_literals.items():
            validator.literal(
                acknowledgements.get(field),
                f"{relative}.scope_acknowledgements.{field}",
                expected,
            )

    if disposition in _REVIEW_DISPOSITIONS and len(item_dispositions) == len(
        _REVIEW_ITEM_IDS
    ):
        aggregate = max(item_dispositions, key=_REVIEW_DISPOSITION_SEVERITY.__getitem__)
        if disposition != aggregate:
            validator.error(
                f"{relative}.global_disposition",
                f"must equal the maximum review-item verdict severity {aggregate!r}",
            )
    if (
        disposition == "approved"
        and selected_option_id != _APPROVAL_COMPATIBLE_OPTION_ID
    ):
        validator.error(
            f"{relative}.ambiguity_resolution.selected_option_id",
            "an approved decision must select the approval-compatible option",
        )
    if selected_option_id == _SUCCESSOR_REQUIRED_OPTION_ID:
        if disposition not in _SUCCESSOR_REQUIRED_GLOBAL_DISPOSITIONS:
            validator.error(
                f"{relative}.global_disposition",
                "the successor-required option permits only changes_requested or rejected",
            )
        resource_verdict = item_disposition_by_id.get(
            _SUCCESSOR_REQUIRED_NON_APPROVED_REVIEW_ITEM_ID
        )
        if resource_verdict == "approved":
            validator.error(
                f"{relative}.item_verdicts["
                f"{_REVIEW_ITEM_IDS.index(_SUCCESSOR_REQUIRED_NON_APPROVED_REVIEW_ITEM_ID)}].verdict",
                "the successor-required option requires this review item to be non-approved",
            )

    if len(validator.errors) != start_error_count:
        return None
    assert decision_id is not None
    assert disposition is not None
    assert reviewed_git_revision is not None
    assert selected_option_id is not None
    if not _verify_reviewed_git_revision(
        validator, repository_root, reviewed_git_revision
    ):
        return None
    non_approved = tuple(
        item_id
        for item_id in _REVIEW_ITEM_IDS
        if item_disposition_by_id[item_id] != "approved"
    )
    hardware_review_clear = (
        disposition == "approved"
        and not non_approved
        and selected_option_id == _APPROVAL_COMPATIBLE_OPTION_ID
    )
    successor_required = selected_option_id == _SUCCESSOR_REQUIRED_OPTION_ID
    if hardware_review_clear:
        next_action = "begin_principle_driven_iteration"
    elif successor_required:
        next_action = "author_successor_stage3_proposal_and_review_request"
    elif disposition == "changes_requested":
        next_action = "revise_stage3_closure_and_request_new_human_review"
    else:
        next_action = "stop_or_replace_stage3_proposal_before_new_review"
    return _HardwareReviewDecisionResult(
        decision_id=decision_id,
        disposition=disposition,
        reviewed_git_revision=reviewed_git_revision,
        non_approved_review_item_ids=non_approved,
        selected_option_id=selected_option_id,
        successor_stage3_proposal_required=successor_required,
        hardware_review_clear=hardware_review_clear,
        next_action=next_action,
    )


def _external_review_decision(
    validator: _Validator,
    decision_path: Path,
    repository_root: Path,
    design_root: Path,
    design_stage_id: str,
) -> _HardwareReviewDecisionResult | None:
    diagnostic_path = "review-decision"
    if not decision_path.is_absolute():
        validator.error(diagnostic_path, "path must be absolute and caller-managed")
        return None
    if repository_root.is_symlink():
        validator.error("repository-root", "must be a regular non-symlink directory")
        return None
    try:
        resolved_repository_root = repository_root.resolve(strict=True)
    except OSError as error:
        validator.error("repository-root", f"directory does not exist ({error})")
        return None
    if not resolved_repository_root.is_dir():
        validator.error("repository-root", "must be a directory")
        return None
    if design_root != resolved_repository_root / "hardware_informed_design":
        validator.error(
            "repository-root",
            "must own the validated hardware_informed_design directory",
        )
        return None
    try:
        if decision_path.is_symlink() or not decision_path.is_file():
            validator.error(
                diagnostic_path, "must be a regular non-symlink external file"
            )
            return None
        resolved_decision_path = decision_path.resolve(strict=True)
    except OSError as error:
        validator.error(diagnostic_path, f"could not inspect external file ({error})")
        return None
    inside_checkout = resolved_decision_path.is_relative_to(resolved_repository_root)
    if not inside_checkout:
        for parent in resolved_decision_path.parents:
            try:
                if parent.samefile(resolved_repository_root):
                    inside_checkout = True
                    break
            except OSError as error:
                validator.error(
                    diagnostic_path,
                    "could not establish that the decision is outside the source "
                    f"checkout ({error})",
                )
                return None
    if inside_checkout:
        validator.error(diagnostic_path, "must remain outside the source checkout")
        return None
    document = validator.read(resolved_decision_path, diagnostic_path)
    if document is None:
        return None
    return _review_decision_document(
        validator,
        document,
        design_stage_id,
        resolved_repository_root,
    )


def validate_hardware_informed_design(
    design_root: Path,
    extraction_root: Path,
    library_root: Path,
    target_path: Path,
    decision_path: Path | None = None,
    repository_root: Path | None = None,
) -> HardwareInformedDesignSummary:
    """Validate stage-three artifacts after validating stage-two inputs."""

    # The predecessor validator intentionally runs before any stage-three I/O.
    validate_abstraction_extraction(extraction_root, library_root)

    if design_root.is_symlink():
        raise HardwareInformedDesignValidationError(
            ["hardware_informed_design: must be a regular non-symlink directory"]
        )
    try:
        root = design_root.resolve(strict=True)
    except OSError as error:
        raise HardwareInformedDesignValidationError(
            [f"hardware_informed_design: directory does not exist ({error})"]
        ) from error
    if not root.is_dir():
        raise HardwareInformedDesignValidationError(
            ["hardware_informed_design: must be a directory"]
        )
    validator = _Validator(root)
    validator.root_closure()

    # The public schema is parsed for UTF-8/JSON closure here. Its five document
    # definitions are exercised by the contract tests; this validator keeps
    # the live stage boundary to the single candidate-closure digest.
    schema = validator.read(root / "schema.json", "schema.json")
    if schema is not None:
        _public_schema(validator, schema)

    manifest = validator.read(root / "manifest.json", "manifest.json")
    if manifest is None:
        raise HardwareInformedDesignValidationError(validator.errors)
    validator.exact(manifest, "manifest.json", _MANIFEST_KEYS)
    validator.literal(
        manifest.get("$schema"), "manifest.json.$schema", "schema.json#/$defs/manifest"
    )
    validator.literal(manifest.get("schema_version"), "manifest.json.schema_version", 1)
    design_stage_id = validator.string(
        manifest.get("design_stage_id"), "manifest.json.design_stage_id", _ID
    )
    validator.literal(
        design_stage_id,
        "manifest.json.design_stage_id",
        "apple-metal-hardware-informed-design-v1",
    )
    paper = validator.exact(
        manifest.get("paper_stage"), "manifest.json.paper_stage", _PAPER_STAGE_KEYS
    )
    if paper is not None:
        paper_literals = {
            "paper_revision": "arxiv:2608.12629v1",
            "ordinal": 3,
            "name": "hardware_informed_design",
            "predecessor": "abstraction_extraction",
            "successor": "principle_driven_iteration",
        }
        for field, expected in paper_literals.items():
            validator.literal(
                paper.get(field), f"manifest.json.paper_stage.{field}", expected
            )
    validator.literal(
        manifest.get("paper_source_path"),
        "manifest.json.paper_source_path",
        "operator_library/sources/cake-paper-v1.json",
    )
    _stage_literals(validator, manifest, "manifest.json")

    closure = validator.exact(
        manifest.get("input_candidate_closure"),
        "manifest.json.input_candidate_closure",
        _INPUT_CLOSURE_KEYS,
    )
    candidate: dict[str, object] | None = None
    observations: dict[str, dict[str, object]] = {}
    if closure is not None:
        candidate, observations = _candidate_documents(
            validator, closure, extraction_root
        )

    evidence_paths = validator.listed_paths(
        manifest.get("evidence"), "evidence", "evidence"
    )
    proposal_paths = validator.listed_paths(
        manifest.get("proposals"), "proposals", "proposals"
    )
    evidence_documents: list[tuple[str, dict[str, object]]] = []
    evidence_facts: dict[str, dict[str, dict[str, object]]] = {}
    for relative in evidence_paths:
        document = validator.read(root / relative, relative)
        if document is None:
            continue
        evidence_documents.append((relative, document))
        evidence_id, facts_by_id = _evidence_document(
            validator, document, relative, target_path
        )
        if evidence_id is not None:
            if evidence_id in evidence_facts:
                validator.error(relative, f"duplicate evidence_id {evidence_id!r}")
            else:
                evidence_facts[evidence_id] = facts_by_id

    proposal_documents: list[tuple[str, dict[str, object]]] = []
    proposal_ids: list[str] = []
    for relative in proposal_paths:
        document = validator.read(root / relative, relative)
        if document is None:
            continue
        proposal_documents.append((relative, document))
        proposal_id = _proposal_document(
            validator,
            document,
            relative,
            candidate,
            observations,
            evidence_facts,
        )
        if proposal_id is not None:
            proposal_ids.append(proposal_id)
    if len(set(proposal_ids)) != len(proposal_ids):
        validator.error("manifest.json.proposals", "proposal IDs must be unique")

    _cross_document_falsifier_closure(validator, evidence_documents, proposal_documents)

    review_request_paths = validator.listed_paths(
        manifest.get("review_requests"), "review_requests", "review_requests"
    )
    if review_request_paths != [_REVIEW_REQUEST_PATH]:
        validator.error(
            "manifest.json.review_requests",
            f"must equal the single closed review request path {_REVIEW_REQUEST_PATH!r}",
        )
    review_request_documents: list[tuple[str, dict[str, object]]] = []
    review_request_ids: list[str] = []
    for relative in review_request_paths:
        document = validator.read(root / relative, relative)
        if document is None:
            continue
        review_request_documents.append((relative, document))
        review_request_id = _review_request_document(
            validator,
            document,
            relative,
            design_stage_id,
            evidence_documents,
            proposal_documents,
        )
        if review_request_id is not None:
            review_request_ids.append(review_request_id)
    if len(set(review_request_ids)) != len(review_request_ids):
        validator.error(
            "manifest.json.review_requests", "review request IDs must be unique"
        )

    _gate(validator, manifest.get("gate_separation"), "manifest.json.gate_separation")

    counts = validator.exact(
        manifest.get("counts"), "manifest.json.counts", _COUNT_KEYS
    )
    derived_counts = {
        "evidence_count": len(evidence_documents),
        "proposal_count": len(proposal_documents),
        "review_request_count": len(review_request_documents),
        "candidate_count": 1 if candidate is not None else 0,
        "observation_count": len(observations),
    }
    if counts is not None:
        expected_counts = {
            "evidence_count": 1,
            "proposal_count": 1,
            "review_request_count": 1,
            "candidate_count": 1,
            "observation_count": 2,
        }
        for field, expected in expected_counts.items():
            validator.literal(
                counts.get(field), f"manifest.json.counts.{field}", expected
            )
        if counts != derived_counts:
            validator.error(
                "manifest.json.counts",
                "does not equal counts derived from listed artifacts",
            )
    validator.strings(
        manifest.get("non_claims"), "manifest.json.non_claims", nonempty=True
    )

    if validator.errors:
        raise HardwareInformedDesignValidationError(validator.errors)
    assert design_stage_id is not None
    decision_result: _HardwareReviewDecisionResult | None = None
    if decision_path is not None:
        decision_result = _external_review_decision(
            validator,
            decision_path,
            repository_root if repository_root is not None else root.parent,
            root,
            design_stage_id,
        )
        if validator.errors:
            raise HardwareInformedDesignValidationError(validator.errors)
        assert decision_result is not None
    review_clear = (
        decision_result.hardware_review_clear if decision_result is not None else False
    )
    return HardwareInformedDesignSummary(
        design_stage_id=design_stage_id,
        state=_STATE,
        disposition=_DISPOSITION,
        next_gate=_NEXT_GATE,
        counts=derived_counts,
        proposal_ids=tuple(sorted(proposal_ids)),
        verification_scope=(
            "offline_metadata_derivation_and_external_decision_binding_validation"
            if decision_result is not None
            else "offline_metadata_and_derivation_validation"
        ),
        ready_for_principle_review=review_clear,
        review_request_count=derived_counts["review_request_count"],
        hardware_review_decision_present=decision_result is not None,
        hardware_review_decision_id=(
            decision_result.decision_id if decision_result is not None else None
        ),
        hardware_review_decision_disposition=(
            decision_result.disposition if decision_result is not None else None
        ),
        hardware_reviewed_git_revision=(
            decision_result.reviewed_git_revision
            if decision_result is not None
            else None
        ),
        hardware_reviewed_revision_bound=decision_result is not None,
        non_approved_review_item_ids=(
            decision_result.non_approved_review_item_ids
            if decision_result is not None
            else ()
        ),
        resource_phase_option_id=(
            decision_result.selected_option_id if decision_result is not None else None
        ),
        successor_stage3_proposal_required=(
            decision_result.successor_stage3_proposal_required
            if decision_result is not None
            else False
        ),
        reviewer_authorship_assurance=(
            "external_human_process_attested_not_cryptographically_verified"
            if decision_result is not None
            else "no_external_decision_supplied"
        ),
        next_action=(
            decision_result.next_action
            if decision_result is not None
            else "obtain_external_human_hardware_review"
        ),
        resource_phase_ambiguity_exposed=True,
        hardware_review_complete=review_clear,
    )


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _text(summary: HardwareInformedDesignSummary) -> str:
    lines = [
        f"hardware-informed design metadata consistent: {summary.design_stage_id} ({summary.state})",
        f"verification scope: {summary.verification_scope}",
        f"ready for principle review: {_yes_no(summary.ready_for_principle_review)}",
        "hardware semantics verified: no",
        "performance claim authorized: no",
        "compiler change authorized: no",
        "implementation authorized: no",
        "evaluation authorized: no",
        "scientific claim authorized: no",
        "promotion authorized: no",
        "remote source bytes verified: no",
        "engineering observation record consistent: yes",
        "retained local probe reported outcome: yes (6 cases, 12 epochs)",
        f"retained local probe scope: {summary.retained_local_probe_scope}",
        f"review requests: {summary.review_request_count}",
        "hardware review decision present: "
        f"{_yes_no(summary.hardware_review_decision_present)}",
        "hardware reviewed revision bound: "
        f"{_yes_no(summary.hardware_reviewed_revision_bound)}",
        "resource phase ambiguity exposed: yes",
        f"hardware review complete: {_yes_no(summary.hardware_review_complete)}",
        "resource accounting hypothesis resolved: no",
        f"stage transition applied: {_yes_no(summary.stage_transition_applied)}",
        f"next action: {summary.next_action}",
        "evaluation evidence present: no",
        f"evidence artifacts: {summary.counts['evidence_count']}",
        f"proposals: {summary.counts['proposal_count']}",
    ]
    if summary.hardware_review_decision_present:
        lines.extend(
            (
                f"hardware review decision ID: {summary.hardware_review_decision_id}",
                "hardware review decision disposition: "
                f"{summary.hardware_review_decision_disposition}",
                f"reviewed Git revision: {summary.hardware_reviewed_git_revision}",
                f"resource phase option: {summary.resource_phase_option_id}",
                "successor Stage-3 proposal required: "
                f"{_yes_no(summary.successor_stage3_proposal_required)}",
                "non-approved review items: "
                + (
                    ", ".join(summary.non_approved_review_item_ids)
                    if summary.non_approved_review_item_ids
                    else "none"
                ),
                f"reviewer authorship assurance: {summary.reviewer_authorship_assurance}",
            )
        )
    lines.append("proposal IDs:")
    lines.extend(f"  {proposal_id}" for proposal_id in summary.proposal_ids)
    return "\n".join(lines)


def _markdown(summary: HardwareInformedDesignSummary) -> str:
    lines = [
        "# Hardware-informed design metadata consistency",
        "",
        f"Stage: `{summary.design_stage_id}` ({summary.state})",
        "",
        f"Verification scope: `{summary.verification_scope}`.",
        "",
        "| Boundary | Value |",
        "| --- | --- |",
        "| Ready for principle review | "
        f"{_yes_no(summary.ready_for_principle_review).title()} |",
        "| Hardware semantics verified | No |",
        "| Performance claim authorized | No |",
        "| Compiler change authorized | No |",
        "| Implementation authorized | No |",
        "| Evaluation authorized | No |",
        "| Scientific claim authorized | No |",
        "| Promotion authorized | No |",
        "| Remote source bytes verified | No |",
        "| Engineering observation record consistent | Yes |",
        "| Retained local probe reported outcome | Yes (6 cases, 12 epochs) |",
        f"| Retained local probe scope | `{summary.retained_local_probe_scope}` |",
        f"| Review requests | {summary.review_request_count} |",
        "| Hardware review decision present | "
        f"{_yes_no(summary.hardware_review_decision_present).title()} |",
        "| Hardware reviewed revision bound | "
        f"{_yes_no(summary.hardware_reviewed_revision_bound).title()} |",
        "| Resource phase ambiguity exposed | Yes |",
        "| Hardware review complete | "
        f"{_yes_no(summary.hardware_review_complete).title()} |",
        "| Resource accounting hypothesis resolved | No |",
        "| Stage transition applied | No |",
        f"| Next action | `{summary.next_action}` |",
        "| Evaluation evidence present | No |",
        f"| Evidence artifacts | {summary.counts['evidence_count']} |",
        f"| Proposals | {summary.counts['proposal_count']} |",
    ]
    if summary.hardware_review_decision_present:
        lines.extend(
            (
                f"| Hardware review decision ID | `{summary.hardware_review_decision_id}` |",
                "| Hardware review decision disposition | "
                f"`{summary.hardware_review_decision_disposition}` |",
                f"| Reviewed Git revision | `{summary.hardware_reviewed_git_revision}` |",
                f"| Resource phase option | `{summary.resource_phase_option_id}` |",
                "| Successor Stage-3 proposal required | "
                f"{_yes_no(summary.successor_stage3_proposal_required).title()} |",
                "| Non-approved review items | "
                + (
                    ", ".join(
                        f"`{item_id}`"
                        for item_id in summary.non_approved_review_item_ids
                    )
                    if summary.non_approved_review_item_ids
                    else "None"
                )
                + " |",
                "| Reviewer authorship assurance | "
                f"`{summary.reviewer_authorship_assurance}` |",
            )
        )
    lines.extend(("", "## Proposal IDs", ""))
    lines.extend(f"- `{proposal_id}`" for proposal_id in summary.proposal_ids)
    return "\n".join(lines)


def _failure(
    error: HardwareInformedDesignValidationError
    | AbstractionExtractionValidationError
    | OperatorLibraryValidationError,
    output_format: str,
) -> None:
    if output_format == "json":
        print(
            json.dumps({"valid": False, "errors": list(error.errors)}, sort_keys=True)
        )
    elif output_format == "markdown":
        print("# Hardware-informed design validation failed", file=sys.stderr)
        for message in error.errors:
            print(f"- {message}", file=sys.stderr)
    else:
        for message in error.errors:
            print(f"error: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-root", type=Path, default=project_root / "operator_library"
    )
    parser.add_argument(
        "--extraction-root", type=Path, default=project_root / "abstraction_extraction"
    )
    parser.add_argument(
        "--design-root", type=Path, default=project_root / "hardware_informed_design"
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        default=project_root / "compiler" / "targets" / "apple_gpu_family9.json",
    )
    parser.add_argument("--repository-root", type=Path, default=project_root)
    parser.add_argument(
        "--review-decision",
        type=Path,
        help="absolute path to a caller-managed external human review decision",
    )
    parser.add_argument(
        "--require-hardware-review-clear",
        action="store_true",
        help="exit 3 when the valid reviewed state does not clear Stage 3",
    )
    parser.add_argument(
        "--format", choices=("text", "markdown", "json"), default="text"
    )
    arguments = parser.parse_args(argv)
    try:
        summary = validate_hardware_informed_design(
            arguments.design_root,
            arguments.extraction_root,
            arguments.library_root,
            arguments.target_path,
            arguments.review_decision,
            arguments.repository_root,
        )
    except (
        HardwareInformedDesignValidationError,
        AbstractionExtractionValidationError,
        OperatorLibraryValidationError,
    ) as error:
        _failure(error, arguments.format)
        return 1
    if arguments.format == "json":
        print(json.dumps(summary.report(), indent=2, sort_keys=True))
    elif arguments.format == "markdown":
        print(_markdown(summary))
    else:
        print(_text(summary))
    if arguments.require_hardware_review_clear and not summary.hardware_review_complete:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
