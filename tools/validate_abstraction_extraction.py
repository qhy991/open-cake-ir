#!/usr/bin/env python3
"""Validate the source-grounded abstraction extraction without network access."""

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

from validate_operator_library import (
    OperatorLibraryValidationError,
    validate_operator_library,
)


_ROOT_FILES = frozenset({"README.md", "schema.json", "manifest.json"})
_ROOT_DIRECTORIES = frozenset({"observations", "candidates"})
_MANIFEST_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "extraction_id",
        "state",
        "paper_source_path",
        "input_library",
        "selection_policy",
        "observations",
        "candidates",
        "counts",
    }
)
_INPUT_LIBRARY_KEYS = frozenset(
    {"manifest_path", "library_id", "canonical_sha256"}
)
_SELECTION_POLICY_KEYS = frozenset(
    {
        "name",
        "minimum_distinct_observations",
        "threshold_authority",
        "inclusions",
        "exclusions",
        "deviations",
        "non_claims",
    }
)
_COUNT_KEYS = frozenset(
    {"observation_count", "candidate_count", "implementation_count", "source_count"}
)
_OBSERVATION_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "observation_id",
        "pattern_signature",
        "implementation_id",
        "operator_id",
        "source_id",
        "source_locator",
        "source_span",
        "observed_structure",
        "observed_primitives",
        "variant_notes",
        "review_status",
        "claim_scope",
    }
)
_SOURCE_LOCATOR_KEYS = frozenset({"path", "revision", "git_object", "symbol"})
_SOURCE_SPAN_KEYS = frozenset({"start_line", "end_line", "sha256", "hash_contract"})
_CANDIDATE_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "candidate_id",
        "pattern_signature",
        "summary",
        "normalized_steps",
        "observed_variants",
        "observation_ids",
        "evidence_breadth",
        "state",
        "claim_scope",
        "next_gate",
    }
)
_BREADTH_KEYS = frozenset(
    {
        "observation_count",
        "implementation_ids",
        "operator_ids",
        "source_ids",
        "source_revisions",
        "git_objects",
        "families",
        "hardware_targets",
    }
)
_ID = re.compile(r"[a-z0-9][a-z0-9._-]+")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_PATH_CHARACTERS = re.compile(r"[A-Za-z0-9._/-]+")
_SCHEMA_CANONICAL_SHA256 = (
    "16b20e16c8f5a8b9717c1a562f0f6b625c0528d95fc863eb2d1459d118628a34"
)


class AbstractionExtractionValidationError(ValueError):
    """The deterministic set of extraction validation failures."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(sorted(set(errors)))
        super().__init__("\n".join(self.errors))


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True)
class AbstractionExtractionSummary:
    extraction_id: str
    state: str
    counts: dict[str, int]
    candidate_patterns: tuple[str, ...]
    verification_scope: str = "metadata_consistency"
    source_bytes_verified: bool = False

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
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError(f"duplicate object key {key!r}")
        value[key] = item
    return value


def _reject_non_json_number(value: str) -> object:
    raise ValueError(f"non-JSON number {value}")


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value}")
    return parsed


class _Validator:
    def __init__(self, extraction_root: Path) -> None:
        self.root = extraction_root
        self.errors: list[str] = []

    def error(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    def root_closure(self) -> None:
        allowed = _ROOT_FILES | _ROOT_DIRECTORIES
        try:
            entries = {entry.name: entry for entry in self.root.iterdir()}
        except OSError as error:
            self.error("abstraction_extraction", f"could not enumerate directory ({error})")
            return
        unknown = sorted(set(entries) - allowed)
        if unknown:
            self.error(
                "abstraction_extraction",
                f"unlisted abstraction_extraction entries: {', '.join(unknown)}",
            )
        for name in sorted(_ROOT_FILES):
            entry = entries.get(name)
            if entry is None or entry.is_symlink() or not entry.is_file():
                self.error(
                    f"abstraction_extraction/{name}",
                    "must be a regular non-symlink file",
                )
        for name in sorted(_ROOT_DIRECTORIES):
            entry = entries.get(name)
            if entry is None or entry.is_symlink() or not entry.is_dir():
                self.error(
                    f"abstraction_extraction/{name}",
                    "must be a regular non-symlink directory",
                )

    def read(self, path: Path, display: str) -> dict[str, object] | None:
        if path.is_symlink() or not path.is_file():
            self.error(display, "must be a regular non-symlink file")
            return None
        try:
            text = path.read_text(encoding="utf-8")
            value = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_json_number,
                parse_float=_parse_json_float,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as error:
            self.error(display, f"invalid UTF-8 JSON ({error})")
            return None
        if not isinstance(value, dict):
            self.error(display, "top level must be an object")
            return None
        try:
            _canonical_json_bytes(value)
        except (TypeError, UnicodeError, ValueError) as error:
            self.error(
                display,
                f"cannot be canonicalized as UTF-8 JSON ({error})",
            )
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
        return value

    def integer(self, value: object, path: str, minimum: int) -> int | None:
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
        ids: bool = False,
        pattern: re.Pattern[str] | None = None,
        sorted_required: bool,
    ) -> list[str]:
        if not isinstance(value, list):
            self.error(path, "must be an array")
            return []
        if nonempty and not value:
            self.error(path, "must not be empty")
        parsed: list[str] = []
        item_pattern = _ID if ids else pattern
        for index, item in enumerate(value):
            result = self.string(item, f"{path}[{index}]", item_pattern)
            if result is not None:
                parsed.append(result)
        if len(parsed) == len(value):
            if len(set(parsed)) != len(parsed):
                self.error(path, "must contain unique values")
            if sorted_required and parsed != sorted(parsed):
                self.error(path, "must be sorted lexicographically")
        return parsed

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
        values = self.strings(
            value,
            base,
            nonempty=field == "observations",
            sorted_required=True,
        )
        result: list[str] = []
        for index, value_path in enumerate(values):
            relative = self.safe_path(value_path, f"{base}[{index}]")
            if relative is None:
                continue
            pure = PurePosixPath(relative)
            if len(pure.parts) != 2 or pure.parts[0] != directory or pure.suffix != ".json":
                self.error(f"{base}[{index}]", f"must name one {directory}/*.json file")
                continue
            candidate = self.root / relative
            if candidate.is_symlink() or not candidate.is_file():
                self.error(
                    f"{base}[{index}]",
                    f"must reference a regular non-symlink file: {relative}",
                )
            result.append(relative)
        location = self.root / directory
        if location.is_dir() and not location.is_symlink():
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


def _library_documents(
    library_root: Path,
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    str,
]:
    validate_operator_library(library_root)
    root = library_root.resolve(strict=True)
    manifest = cast(
        dict[str, object], json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    source_documents = {
        cast(str, relative): cast(
            dict[str, object], json.loads((root / cast(str, relative)).read_text(encoding="utf-8"))
        )
        for relative in cast(list[object], manifest["sources"])
    }
    operator_documents = {
        cast(str, relative): cast(
            dict[str, object], json.loads((root / cast(str, relative)).read_text(encoding="utf-8"))
        )
        for relative in cast(list[object], manifest["operators"])
    }
    sources_by_id = {
        cast(str, value["source_id"]): value for value in source_documents.values()
    }
    operators_by_id = {
        cast(str, value["implementation_id"]): value
        for value in operator_documents.values()
    }
    try:
        digest = sha256(
            _canonical_json_bytes(
                {
                    "manifest": manifest,
                    "sources": source_documents,
                    "operators": operator_documents,
                }
            )
        ).hexdigest()
    except (UnicodeError, ValueError) as error:
        raise AbstractionExtractionValidationError(
            [f"operator_library: canonical digest input cannot be canonicalized ({error})"]
        ) from error
    return manifest, sources_by_id, operators_by_id, operator_documents, digest


def _identity(value: Mapping[str, object], field: str) -> str | None:
    item = value.get(field)
    return item if isinstance(item, str) else None


def _sorted_values(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def validate_abstraction_extraction(
    extraction_root: Path, library_root: Path
) -> AbstractionExtractionSummary:
    """Validate extraction artifacts after validating their operator-library input."""

    (
        library_manifest,
        sources_by_id,
        operators_by_id,
        _operator_documents,
        library_digest,
    ) = _library_documents(library_root)
    if extraction_root.is_symlink():
        raise AbstractionExtractionValidationError(
            ["abstraction_extraction: must be a regular non-symlink directory"]
        )
    try:
        root = extraction_root.resolve(strict=True)
    except OSError as error:
        raise AbstractionExtractionValidationError(
            [f"abstraction_extraction: directory does not exist ({error})"]
        ) from error
    if not root.is_dir():
        raise AbstractionExtractionValidationError(
            ["abstraction_extraction: must be a directory"]
        )
    validator = _Validator(root)
    validator.root_closure()
    schema = validator.read(root / "schema.json", "schema.json")
    if schema is not None:
        try:
            schema_digest = sha256(_canonical_json_bytes(schema)).hexdigest()
        except (UnicodeError, ValueError) as error:
            validator.error("schema.json", f"cannot be canonicalized ({error})")
        else:
            if schema_digest != _SCHEMA_CANONICAL_SHA256:
                validator.error(
                    "schema.json",
                    "schema differs from the canonical extraction schema",
                )
    manifest = validator.read(root / "manifest.json", "manifest.json")
    if manifest is None:
        raise AbstractionExtractionValidationError(validator.errors)
    validator.exact(manifest, "manifest.json", _MANIFEST_KEYS)
    validator.literal(manifest.get("$schema"), "manifest.json.$schema", "schema.json#/$defs/manifest")
    validator.literal(manifest.get("schema_version"), "manifest.json.schema_version", 1)
    extraction_id = validator.string(
        manifest.get("extraction_id"), "manifest.json.extraction_id", _ID
    )
    validator.literal(manifest.get("state"), "manifest.json.state", "in_progress")
    expected_method_source = "sources/cake-paper-v1.json"
    validator.literal(
        library_manifest.get("method_source"),
        "operator_library/manifest.json.method_source",
        expected_method_source,
    )
    library_source_paths = library_manifest.get("sources")
    if (
        not isinstance(library_source_paths, list)
        or expected_method_source not in library_source_paths
    ):
        validator.error(
            "operator_library/manifest.json.sources",
            "must list the CAKE paper method source",
        )
    validator.literal(
        manifest.get("paper_source_path"),
        "manifest.json.paper_source_path",
        f"operator_library/{expected_method_source}",
    )

    input_library = validator.exact(
        manifest.get("input_library"), "manifest.json.input_library", _INPUT_LIBRARY_KEYS
    )
    if input_library is not None:
        validator.literal(
            input_library.get("manifest_path"),
            "manifest.json.input_library.manifest_path",
            "operator_library/manifest.json",
        )
        validator.literal(
            input_library.get("library_id"),
            "manifest.json.input_library.library_id",
            library_manifest.get("library_id"),
        )
        observed_digest = validator.string(
            input_library.get("canonical_sha256"),
            "manifest.json.input_library.canonical_sha256",
            _SHA256,
        )
        if observed_digest is not None and observed_digest != library_digest:
            validator.error(
                "manifest.json.input_library.canonical_sha256",
                "operator-library canonical digest differs",
            )

    policy = validator.exact(
        manifest.get("selection_policy"),
        "manifest.json.selection_policy",
        _SELECTION_POLICY_KEYS,
    )
    if policy is not None:
        validator.string(policy.get("name"), "manifest.json.selection_policy.name")
        validator.literal(
            policy.get("minimum_distinct_observations"),
            "manifest.json.selection_policy.minimum_distinct_observations",
            2,
        )
        validator.literal(
            policy.get("threshold_authority"),
            "manifest.json.selection_policy.threshold_authority",
            "local_repository_policy_not_paper_requirement",
        )
        for field in ("inclusions", "exclusions", "deviations", "non_claims"):
            validator.strings(
                policy.get(field),
                f"manifest.json.selection_policy.{field}",
                nonempty=True,
                sorted_required=False,
            )

    observation_paths = validator.listed_paths(
        manifest.get("observations"), "observations", "observations"
    )
    candidate_paths = validator.listed_paths(
        manifest.get("candidates"), "candidates", "candidates"
    )

    observations: list[dict[str, object]] = []
    observation_paths_by_id: dict[str, str] = {}
    for relative in observation_paths:
        document = validator.read(root / relative, relative)
        if document is None:
            continue
        validator.exact(document, relative, _OBSERVATION_KEYS)
        validator.literal(document.get("$schema"), f"{relative}.$schema", "../schema.json#/$defs/observation")
        validator.literal(document.get("schema_version"), f"{relative}.schema_version", 1)
        observation_id = validator.string(document.get("observation_id"), f"{relative}.observation_id", _ID)
        validator.string(document.get("pattern_signature"), f"{relative}.pattern_signature", _ID)
        implementation_id = validator.string(document.get("implementation_id"), f"{relative}.implementation_id", _ID)
        operator_id = validator.string(document.get("operator_id"), f"{relative}.operator_id", _ID)
        source_id = validator.string(document.get("source_id"), f"{relative}.source_id", _ID)
        occurrence = operators_by_id.get(implementation_id) if implementation_id else None
        if implementation_id is not None and occurrence is None:
            validator.error(f"{relative}.implementation_id", "does not reference an operator-library occurrence")
        if occurrence is not None:
            if operator_id != occurrence.get("operator_id"):
                validator.error(f"{relative}.operator_id", "differs from the referenced occurrence")
            if source_id != occurrence.get("source_id"):
                validator.error(f"{relative}.source_id", "differs from the referenced occurrence")
        if source_id is not None and source_id not in sources_by_id:
            validator.error(f"{relative}.source_id", "does not reference an operator-library source")

        locator = validator.exact(document.get("source_locator"), f"{relative}.source_locator", _SOURCE_LOCATOR_KEYS)
        locator_path: str | None = None
        revision: str | None = None
        git_object: str | None = None
        if locator is not None:
            locator_path = validator.safe_path(locator.get("path"), f"{relative}.source_locator.path")
            revision = validator.string(locator.get("revision"), f"{relative}.source_locator.revision", _HEX40)
            git_object = validator.string(locator.get("git_object"), f"{relative}.source_locator.git_object", _HEX40)
            validator.string(locator.get("symbol"), f"{relative}.source_locator.symbol")
        if occurrence is not None and locator_path and revision and git_object:
            matching = any(
                isinstance(item, dict)
                and item.get("kind") == "repository_path"
                and item.get("value") == locator_path
                and item.get("revision") == revision
                and item.get("git_object") == git_object
                for item in cast(list[object], occurrence.get("locators", []))
            )
            if not matching:
                validator.error(
                    f"{relative}.source_locator",
                    "does not match one repository_path locator on the referenced occurrence",
                )

        span = validator.exact(document.get("source_span"), f"{relative}.source_span", _SOURCE_SPAN_KEYS)
        if span is not None:
            start = validator.integer(span.get("start_line"), f"{relative}.source_span.start_line", 1)
            end = validator.integer(span.get("end_line"), f"{relative}.source_span.end_line", 1)
            if start is not None and end is not None and end < start:
                validator.error(f"{relative}.source_span", "end_line must be >= start_line")
            validator.string(span.get("sha256"), f"{relative}.source_span.sha256", _SHA256)
            validator.literal(
                span.get("hash_contract"),
                f"{relative}.source_span.hash_contract",
                "sha256-utf8-lf-inclusive-lines-v1",
            )
        validator.strings(document.get("observed_structure"), f"{relative}.observed_structure", nonempty=True, sorted_required=False)
        validator.strings(document.get("observed_primitives"), f"{relative}.observed_primitives", nonempty=True, ids=True, sorted_required=True)
        validator.strings(document.get("variant_notes"), f"{relative}.variant_notes", nonempty=True, sorted_required=False)
        validator.literal(document.get("review_status"), f"{relative}.review_status", "semantics_reviewed")
        validator.literal(document.get("claim_scope"), f"{relative}.claim_scope", "source_structure_only")
        if observation_id is not None:
            if observation_id in observation_paths_by_id:
                validator.error(relative, f"duplicate observation_id also used by {observation_paths_by_id[observation_id]}")
            else:
                observation_paths_by_id[observation_id] = relative
        observations.append(document)

    observations_by_id = {
        cast(str, item["observation_id"]): item
        for item in observations
        if isinstance(item.get("observation_id"), str)
    }
    candidates: list[dict[str, object]] = []
    candidate_paths_by_id: dict[str, str] = {}
    for relative in candidate_paths:
        document = validator.read(root / relative, relative)
        if document is None:
            continue
        validator.exact(document, relative, _CANDIDATE_KEYS)
        validator.literal(document.get("$schema"), f"{relative}.$schema", "../schema.json#/$defs/candidate")
        validator.literal(document.get("schema_version"), f"{relative}.schema_version", 1)
        candidate_id = validator.string(document.get("candidate_id"), f"{relative}.candidate_id", _ID)
        signature = validator.string(document.get("pattern_signature"), f"{relative}.pattern_signature", _ID)
        validator.string(document.get("summary"), f"{relative}.summary")
        validator.strings(document.get("normalized_steps"), f"{relative}.normalized_steps", nonempty=True, sorted_required=False)
        validator.strings(document.get("observed_variants"), f"{relative}.observed_variants", nonempty=True, sorted_required=False)
        observation_ids = validator.strings(
            document.get("observation_ids"),
            f"{relative}.observation_ids",
            nonempty=True,
            ids=True,
            sorted_required=True,
        )
        if len(observation_ids) < 2:
            validator.error(f"{relative}.observation_ids", "candidate requires at least 2 observations")
        evidence = [observations_by_id[item] for item in observation_ids if item in observations_by_id]
        unknown = sorted(set(observation_ids) - set(observations_by_id))
        if unknown:
            validator.error(f"{relative}.observation_ids", f"unknown observations: {', '.join(unknown)}")
        if signature is not None and any(item.get("pattern_signature") != signature for item in evidence):
            validator.error(f"{relative}.pattern_signature", "differs from referenced observation pattern_signature")

        implementation_ids = _sorted_values(
            cast(str, item["implementation_id"])
            for item in evidence
            if isinstance(item.get("implementation_id"), str)
        )
        git_objects = _sorted_values(
            cast(str, cast(Mapping[str, object], item["source_locator"])["git_object"])
            for item in evidence
            if isinstance(item.get("source_locator"), Mapping)
            and isinstance(cast(Mapping[str, object], item["source_locator"]).get("git_object"), str)
        )
        if len(implementation_ids) < 2:
            validator.error(f"{relative}.observation_ids", "candidate requires distinct implementation_id evidence")
        if len(git_objects) < 2:
            validator.error(f"{relative}.observation_ids", "candidate requires distinct git_object evidence")
        derived_breadth: dict[str, object] = {
            "observation_count": len(evidence),
            "implementation_ids": implementation_ids,
            "operator_ids": _sorted_values(
                cast(str, item["operator_id"])
                for item in evidence
                if isinstance(item.get("operator_id"), str)
            ),
            "source_ids": _sorted_values(
                cast(str, item["source_id"])
                for item in evidence
                if isinstance(item.get("source_id"), str)
            ),
            "source_revisions": _sorted_values(
                cast(str, cast(Mapping[str, object], item["source_locator"])["revision"])
                for item in evidence
                if isinstance(item.get("source_locator"), Mapping)
                and isinstance(cast(Mapping[str, object], item["source_locator"]).get("revision"), str)
            ),
            "git_objects": git_objects,
            "families": _sorted_values(
                cast(str, operators_by_id[cast(str, item["implementation_id"])]["family"])
                for item in evidence
                if isinstance(item.get("implementation_id"), str)
                and cast(str, item["implementation_id"]) in operators_by_id
            ),
            "hardware_targets": _sorted_values(
                target
                for item in evidence
                if isinstance(item.get("implementation_id"), str)
                and cast(str, item["implementation_id"]) in operators_by_id
                for target in cast(
                    list[str],
                    operators_by_id[cast(str, item["implementation_id"])]["hardware_targets"],
                )
            ),
        }
        breadth = validator.exact(document.get("evidence_breadth"), f"{relative}.evidence_breadth", _BREADTH_KEYS)
        if breadth is not None:
            validator.integer(
                breadth.get("observation_count"),
                f"{relative}.evidence_breadth.observation_count",
                2,
            )
            for field in (
                "implementation_ids",
                "operator_ids",
                "source_ids",
                "families",
                "hardware_targets",
            ):
                values = validator.strings(
                    breadth.get(field),
                    f"{relative}.evidence_breadth.{field}",
                    nonempty=True,
                    ids=True,
                    sorted_required=True,
                )
                if field == "implementation_ids" and len(values) < 2:
                    validator.error(
                        f"{relative}.evidence_breadth.{field}",
                        "must contain at least 2 implementation IDs",
                    )
            for field in ("source_revisions", "git_objects"):
                values = validator.strings(
                    breadth.get(field),
                    f"{relative}.evidence_breadth.{field}",
                    nonempty=True,
                    pattern=_HEX40,
                    sorted_required=True,
                )
                if field == "git_objects" and len(values) < 2:
                    validator.error(
                        f"{relative}.evidence_breadth.{field}",
                        "must contain at least 2 Git objects",
                    )
            if breadth != derived_breadth:
                validator.error(
                    f"{relative}.evidence_breadth",
                    "does not equal breadth derived from referenced observations",
                )
        validator.literal(document.get("state"), f"{relative}.state", "extracted_unreviewed")
        validator.literal(document.get("claim_scope"), f"{relative}.claim_scope", "source_pattern_only")
        validator.literal(document.get("next_gate"), f"{relative}.next_gate", "hardware_informed_design")
        if candidate_id is not None:
            if candidate_id in candidate_paths_by_id:
                validator.error(relative, f"duplicate candidate_id also used by {candidate_paths_by_id[candidate_id]}")
            else:
                candidate_paths_by_id[candidate_id] = relative
        candidates.append(document)

    counts = validator.exact(manifest.get("counts"), "manifest.json.counts", _COUNT_KEYS)
    derived_counts = {
        "observation_count": len(observations),
        "candidate_count": len(candidates),
        "implementation_count": len(
            {
                item.get("implementation_id")
                for item in observations
                if isinstance(item.get("implementation_id"), str)
            }
        ),
        "source_count": len(
            {
                item.get("source_id")
                for item in observations
                if isinstance(item.get("source_id"), str)
            }
        ),
    }
    if counts is not None:
        for field, minimum in (
            ("observation_count", 1),
            ("candidate_count", 0),
            ("implementation_count", 1),
            ("source_count", 1),
        ):
            validator.integer(counts.get(field), f"manifest.json.counts.{field}", minimum)
        if counts != derived_counts:
            validator.error("manifest.json.counts", "does not equal counts derived from listed artifacts")

    if validator.errors:
        raise AbstractionExtractionValidationError(validator.errors)
    assert extraction_id is not None
    return AbstractionExtractionSummary(
        extraction_id=extraction_id,
        state="in_progress",
        counts=derived_counts,
        candidate_patterns=tuple(
            sorted(
                cast(str, item["pattern_signature"])
                for item in candidates
                if isinstance(item.get("pattern_signature"), str)
            )
        ),
    )


def _text(summary: AbstractionExtractionSummary) -> str:
    lines = [
        f"abstraction extraction metadata consistent: {summary.extraction_id} ({summary.state})",
        f"verification scope: {summary.verification_scope}",
        "source bytes verified: no",
        f"observations: {summary.counts['observation_count']}",
        f"candidates: {summary.counts['candidate_count']}",
        f"implementations: {summary.counts['implementation_count']}",
        f"sources: {summary.counts['source_count']}",
        "candidate patterns:",
    ]
    lines.extend(f"  {pattern}" for pattern in summary.candidate_patterns)
    return "\n".join(lines)


def _markdown(summary: AbstractionExtractionSummary) -> str:
    lines = [
        "# Abstraction extraction metadata consistency",
        "",
        f"Extraction: `{summary.extraction_id}` ({summary.state})",
        "",
        f"Verification scope: `{summary.verification_scope}`. Source bytes verified: **no**.",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Observations | {summary.counts['observation_count']} |",
        f"| Candidates | {summary.counts['candidate_count']} |",
        f"| Implementations | {summary.counts['implementation_count']} |",
        f"| Sources | {summary.counts['source_count']} |",
        "",
        "## Candidate patterns",
        "",
    ]
    lines.extend(f"- `{pattern}`" for pattern in summary.candidate_patterns)
    return "\n".join(lines)


def _failure(
    error: AbstractionExtractionValidationError | OperatorLibraryValidationError,
    output_format: str,
) -> None:
    if output_format == "json":
        print(json.dumps({"valid": False, "errors": list(error.errors)}, sort_keys=True))
    elif output_format == "markdown":
        print("# Abstraction extraction validation failed", file=sys.stderr)
        for message in error.errors:
            print(f"- {message}", file=sys.stderr)
    else:
        for message in error.errors:
            print(f"error: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", type=Path, default=project_root / "operator_library")
    parser.add_argument(
        "--extraction-root", type=Path, default=project_root / "abstraction_extraction"
    )
    parser.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    arguments = parser.parse_args(argv)
    try:
        summary = validate_abstraction_extraction(
            arguments.extraction_root, arguments.library_root
        )
    except (AbstractionExtractionValidationError, OperatorLibraryValidationError) as error:
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
