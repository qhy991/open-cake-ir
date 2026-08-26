#!/usr/bin/env python3
"""Validate the hardware-informed design stage without network access."""

from __future__ import annotations

import argparse
import json
import math
import re
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
_ROOT_DIRECTORIES = frozenset({"evidence", "proposals"})
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
        "non_evidentiary_context",
        "non_evidentiary_repository_snapshot",
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
    {"evidence_count", "proposal_count", "candidate_count", "observation_count"}
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
    "non_evidentiary_context": _TARGET_CONTEXT_KEYS,
    "non_evidentiary_repository_snapshot": (_NON_EVIDENTIARY_REPOSITORY_SNAPSHOT_KEYS),
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
    "falsifier": _FALSIFIER_KEYS,
    "gate_separation": _GATE_KEYS,
    "validation_gate": _VALIDATION_GATE_KEYS,
    "readiness_gate": _READINESS_GATE_KEYS,
}

_ID = re.compile(r"[a-z0-9][a-z0-9._-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_PATH_CHARACTERS = re.compile(r"[A-Za-z0-9._/-]+")
_HTTPS_URL = re.compile(r"https://[^ ]+")

_STATE = "review_pending"
_CLAIM_SCOPE = "hardware_mapping_only"
_DISPOSITION = "await_human_hardware_review"
_NEXT_GATE = "principle_driven_iteration"
_GIT_PARENT_REVISION = "2238610a4e8923330d73d125e79fd374fc7d2397"
_PROBE_PATH = "tools/probe_metal_operators.py"
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
    "sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
    "nonempty_strings": {
        "type": "array",
        "minItems": 1,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1},
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

    def report(self) -> dict[str, object]:
        return {"valid": True, **asdict(self)}


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
    local_observation_id = "local-apple-m4-metal-operator-probe-2026-08-26"
    vendor_sources = set(source_ids)
    known_sources = vendor_sources | {local_observation_id}

    facts_value = document.get("facts")
    facts_by_id: dict[str, dict[str, object]] = {}
    referenced_sources: set[str] = set()
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
                if fact_source_ids != [local_observation_id]:
                    validator.error(
                        f"{path}.source_ids",
                        "a local engineering fact must reference only its local probe observation",
                    )
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
            "observation_id": "local-apple-m4-metal-operator-probe-2026-08-26",
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
    unused_bound_facts = sorted(bound_fact_ids - decision_fact_ids)
    if unused_bound_facts:
        validator.error(
            f"{relative}.evidence_binding.fact_ids",
            "bound facts are not referenced by a design decision: "
            + ", ".join(unused_bound_facts),
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


def validate_hardware_informed_design(
    design_root: Path,
    extraction_root: Path,
    library_root: Path,
    target_path: Path,
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

    # The public schema is parsed for UTF-8/JSON closure here. Its three
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

    _gate(validator, manifest.get("gate_separation"), "manifest.json.gate_separation")

    counts = validator.exact(
        manifest.get("counts"), "manifest.json.counts", _COUNT_KEYS
    )
    derived_counts = {
        "evidence_count": len(evidence_documents),
        "proposal_count": len(proposal_documents),
        "candidate_count": 1 if candidate is not None else 0,
        "observation_count": len(observations),
    }
    if counts is not None:
        expected_counts = {
            "evidence_count": 1,
            "proposal_count": 1,
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
    return HardwareInformedDesignSummary(
        design_stage_id=design_stage_id,
        state=_STATE,
        disposition=_DISPOSITION,
        next_gate=_NEXT_GATE,
        counts=derived_counts,
        proposal_ids=tuple(sorted(proposal_ids)),
    )


def _text(summary: HardwareInformedDesignSummary) -> str:
    lines = [
        f"hardware-informed design metadata consistent: {summary.design_stage_id} ({summary.state})",
        f"verification scope: {summary.verification_scope}",
        "ready for principle review: no",
        "hardware semantics verified: no",
        "performance claim authorized: no",
        "compiler change authorized: no",
        "remote source bytes verified: no",
        f"evidence artifacts: {summary.counts['evidence_count']}",
        f"proposals: {summary.counts['proposal_count']}",
        "proposal IDs:",
    ]
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
        "| Ready for principle review | No |",
        "| Hardware semantics verified | No |",
        "| Performance claim authorized | No |",
        "| Compiler change authorized | No |",
        "| Remote source bytes verified | No |",
        f"| Evidence artifacts | {summary.counts['evidence_count']} |",
        f"| Proposals | {summary.counts['proposal_count']} |",
        "",
        "## Proposal IDs",
        "",
    ]
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
