#!/usr/bin/env python3
"""Validate and summarize the offline operator source library."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlsplit


_MANIFEST_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "library_id",
        "state",
        "method_source",
        "selection_policy",
        "family_vocabulary",
        "mechanism_vocabulary",
        "sources",
        "operators",
    }
)
_SELECTION_POLICY_KEYS = frozenset(
    {"name", "inclusions", "exclusions", "non_claims"}
)
_SOURCE_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "source_id",
        "kind",
        "title",
        "url",
        "revision_kind",
        "revision",
        "tree",
        "observed_at",
        "license_spdx",
        "visibility",
        "code_copied",
        "authoring_access",
        "catalog_entries",
    }
)
_AUTHORING_ACCESS_KEYS = frozenset({"known_kernel_reproduction", "clean_start"})
_OPERATOR_KEYS = frozenset(
    {
        "$schema",
        "schema_version",
        "operator_id",
        "implementation_id",
        "source_id",
        "family",
        "variant",
        "scope",
        "semantic_role",
        "input_regime",
        "hardware_targets",
        "mechanisms",
        "locators",
        "upstream_state",
        "review_status",
        "claim_scope",
    }
)
_INPUT_REGIME_KEYS = frozenset({"dtypes", "extent_class", "shape_constraints"})
_LOCATOR_KEYS = frozenset({"kind", "value", "revision", "git_object", "symbol"})
_ROOT_FILES = frozenset({"README.md", "schema.json", "manifest.json"})
_ROOT_DIRECTORIES = frozenset({"sources", "operators"})

_SOURCE_KINDS = frozenset({"git_repository", "paper_artifact", "public_tracker"})
_REVISION_KINDS = frozenset(
    {"git_commit", "arxiv_version", "github_issue_updated_at"}
)
_REVISION_KIND_BY_SOURCE = {
    "git_repository": "git_commit",
    "paper_artifact": "arxiv_version",
    "public_tracker": "github_issue_updated_at",
}
_VISIBILITIES = frozenset({"public", "local_restricted"})
_SCOPES = frozenset(
    {
        "single_kernel",
        "kernel_family",
        "multi_kernel_program",
        "runtime_dispatch",
        "megakernel_interpreter",
    }
)
_EXTENT_CLASSES = frozenset(
    {"static", "bounded_dynamic", "ragged", "device_runtime"}
)
_LOCATOR_KINDS = frozenset({"repository_path", "pull_request", "web_document"})
_UPSTREAM_STATES = frozenset({"merged", "open", "repository_main", "local_main"})
_REVIEW_STATUSES = frozenset(
    {"collected", "source_verified", "semantics_reviewed"}
)

_ID = re.compile(r"[a-z0-9][a-z0-9._-]+")
_SAFE_PATH_CHARACTERS = re.compile(r"[A-Za-z0-9._/-]+")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_ARXIV_REVISION = re.compile(r"arxiv:[0-9]{4}\.[0-9]{4,5}v[1-9][0-9]*")


class OperatorLibraryValidationError(ValueError):
    """The complete, deterministic set of validation failures."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(sorted(set(errors)))
        super().__init__("\n".join(self.errors))


class _DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True)
class OperatorLibrarySummary:
    library_id: str
    state: str
    source_count: int
    operator_count: int
    mathematical_operator_count: int
    family_counts: dict[str, int]
    review_counts: dict[str, int]
    scope_counts: dict[str, int]
    mechanism_counts: dict[str, int]

    def report(self) -> dict[str, object]:
        return {"valid": True, **asdict(self)}


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_non_json_number(value: str) -> object:
    raise ValueError(f"non-JSON number {value}")


class _LibraryValidator:
    def __init__(self, library_root: Path) -> None:
        self.root = library_root
        self.errors: list[str] = []

    def error(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    def validate_root_closure(self) -> None:
        allowed = _ROOT_FILES | _ROOT_DIRECTORIES
        try:
            entries = {entry.name: entry for entry in self.root.iterdir()}
        except OSError as error:
            self.error("operator_library", f"could not enumerate directory ({error})")
            return
        unlisted = sorted(set(entries) - allowed)
        if unlisted:
            self.error(
                "operator_library",
                f"unlisted operator_library entries: {', '.join(unlisted)}",
            )
        for name in sorted(_ROOT_FILES):
            entry = entries.get(name)
            if entry is None or entry.is_symlink() or not entry.is_file():
                self.error(
                    f"operator_library/{name}",
                    "must be a regular non-symlink file",
                )
        for name in sorted(_ROOT_DIRECTORIES):
            entry = entries.get(name)
            if entry is None or entry.is_symlink() or not entry.is_dir():
                self.error(
                    f"operator_library/{name}",
                    "must be a regular non-symlink directory",
                )

    def read_document(self, path: Path, display: str) -> dict[str, object] | None:
        if path.is_symlink():
            self.error(display, "must be a regular file, not a symlink")
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            self.error(display, f"could not read UTF-8 JSON ({error})")
            return None
        try:
            document = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_json_number,
            )
        except (json.JSONDecodeError, _DuplicateKeyError, ValueError) as error:
            self.error(display, f"invalid JSON ({error})")
            return None
        if not isinstance(document, dict):
            self.error(display, "top level must be an object")
            return None
        return document

    def exact_object(
        self, value: object, path: str, expected: frozenset[str]
    ) -> dict[str, object] | None:
        if not isinstance(value, dict):
            self.error(path, "must be an object")
            return None
        observed = set(value)
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        if missing:
            self.error(path, f"missing fields: {', '.join(missing)}")
        if unknown:
            self.error(path, f"unknown fields: {', '.join(unknown)}")
        return value

    def literal(self, value: object, path: str, expected: object) -> bool:
        # bool is an int in Python; JSON Schema's integer and boolean domains are not.
        if type(value) is not type(expected) or value != expected:  # noqa: E721
            self.error(path, f"must equal {expected!r}")
            return False
        return True

    def string(
        self,
        value: object,
        path: str,
        *,
        pattern: re.Pattern[str] | None = None,
    ) -> str | None:
        if not isinstance(value, str) or not value:
            self.error(path, "must be a non-empty string")
            return None
        if pattern is not None and pattern.fullmatch(value) is None:
            self.error(path, f"does not match required pattern {pattern.pattern!r}")
            return None
        return value

    def enum(self, value: object, path: str, choices: Iterable[str]) -> str | None:
        admitted = frozenset(choices)
        if not isinstance(value, str) or value not in admitted:
            self.error(path, f"must be one of {', '.join(sorted(admitted))}")
            return None
        return value

    def string_list(
        self,
        value: object,
        path: str,
        *,
        nonempty: bool,
        ids: bool = False,
        sorted_required: bool = True,
    ) -> list[str]:
        if not isinstance(value, list):
            self.error(path, "must be an array")
            return []
        if nonempty and not value:
            self.error(path, "must not be empty")
        result: list[str] = []
        for index, item in enumerate(value):
            parsed = self.string(
                item,
                f"{path}[{index}]",
                pattern=_ID if ids else None,
            )
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
        unsafe = (
            _SAFE_PATH_CHARACTERS.fullmatch(parsed) is None
            or pure.is_absolute()
            or parsed != pure.as_posix()
            or any(part in {"", ".", ".."} for part in parsed.split("/"))
        )
        if unsafe:
            self.error(path, "must be a canonical safe relative path without traversal")
            return None
        return parsed

    def referenced_file(self, relative: str, path: str) -> Path | None:
        candidate = self.root / relative
        try:
            resolved_root = self.root.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
        except OSError:
            self.error(path, f"referenced file does not exist: {relative}")
            return None
        if not resolved.is_relative_to(resolved_root):
            self.error(path, f"referenced file escapes the library: {relative}")
            return None
        if candidate.is_symlink() or not candidate.is_file():
            self.error(path, f"referenced path must be a regular non-symlink file: {relative}")
            return None
        return candidate

    def https_url(self, value: object, path: str) -> str | None:
        parsed = self.string(value, path)
        if parsed is None:
            return None
        try:
            split = urlsplit(parsed)
            username = split.username
        except ValueError:
            self.error(path, "must be a well-formed absolute https URL")
            return None
        if split.scheme != "https" or not split.netloc or username is not None:
            self.error(path, "must be an absolute https URL without credentials")
            return None
        return parsed

    def hex40_or_null(self, value: object, path: str) -> str | None:
        if value is None:
            return None
        parsed = self.string(value, path, pattern=_HEX40)
        return parsed

    def validate_manifest(self, document: dict[str, object]) -> dict[str, object]:
        manifest = self.exact_object(document, "manifest.json", _MANIFEST_KEYS)
        assert manifest is not None
        self.literal(
            manifest.get("$schema"),
            "manifest.json.$schema",
            "schema.json#/$defs/manifest",
        )
        self.literal(manifest.get("schema_version"), "manifest.json.schema_version", 1)
        self.string(manifest.get("library_id"), "manifest.json.library_id", pattern=_ID)
        self.enum(manifest.get("state"), "manifest.json.state", {"draft", "collected"})

        method_source = self.safe_path(
            manifest.get("method_source"), "manifest.json.method_source"
        )
        if method_source is not None:
            self.referenced_file(method_source, "manifest.json.method_source")

        policy = self.exact_object(
            manifest.get("selection_policy"),
            "manifest.json.selection_policy",
            _SELECTION_POLICY_KEYS,
        )
        if policy is not None:
            self.string(policy.get("name"), "manifest.json.selection_policy.name")
            for name in ("inclusions", "exclusions", "non_claims"):
                self.string_list(
                    policy.get(name),
                    f"manifest.json.selection_policy.{name}",
                    nonempty=True,
                    sorted_required=False,
                )

        self.string_list(
            manifest.get("family_vocabulary"),
            "manifest.json.family_vocabulary",
            nonempty=True,
            ids=True,
        )
        self.string_list(
            manifest.get("mechanism_vocabulary"),
            "manifest.json.mechanism_vocabulary",
            nonempty=True,
            ids=True,
        )
        self._manifest_paths(manifest.get("sources"), "sources", "sources")
        self._manifest_paths(manifest.get("operators"), "operators", "operators")
        return manifest

    def _manifest_paths(self, value: object, field: str, directory: str) -> list[str]:
        path = f"manifest.json.{field}"
        paths = self.string_list(value, path, nonempty=True)
        safe: list[str] = []
        for index, relative in enumerate(paths):
            parsed = self.safe_path(relative, f"{path}[{index}]")
            if parsed is None:
                continue
            pure = PurePosixPath(parsed)
            if (
                len(pure.parts) != 2
                or pure.parts[0] != directory
                or pure.suffix != ".json"
            ):
                self.error(
                    f"{path}[{index}]",
                    f"must name one {directory}/*.json file",
                )
                continue
            safe.append(parsed)
            self.referenced_file(parsed, f"{path}[{index}]")
        self._manifest_closure(safe, directory, path)
        return safe

    def _manifest_closure(
        self, expected: Sequence[str], directory: str, path: str
    ) -> None:
        location = self.root / directory
        if not location.is_dir() or location.is_symlink():
            self.error(path, f"{directory}/ must be a regular directory")
            return
        actual = sorted(
            item.relative_to(self.root).as_posix()
            for item in location.iterdir()
        )
        unlisted = sorted(set(actual) - set(expected))
        if unlisted:
            self.error(path, f"unlisted directory entries: {', '.join(unlisted)}")

    def validate_source(
        self, document: dict[str, object], relative: str
    ) -> dict[str, object]:
        source = self.exact_object(document, relative, _SOURCE_KEYS)
        assert source is not None
        base = relative
        self.literal(
            source.get("$schema"),
            f"{base}.$schema",
            "../schema.json#/$defs/source_snapshot",
        )
        self.literal(source.get("schema_version"), f"{base}.schema_version", 1)
        self.string(source.get("source_id"), f"{base}.source_id", pattern=_ID)
        kind = self.enum(source.get("kind"), f"{base}.kind", _SOURCE_KINDS)
        self.string(source.get("title"), f"{base}.title")
        self.https_url(source.get("url"), f"{base}.url")
        revision_kind = self.enum(
            source.get("revision_kind"), f"{base}.revision_kind", _REVISION_KINDS
        )
        revision = self.string(source.get("revision"), f"{base}.revision")
        tree = self.hex40_or_null(source.get("tree"), f"{base}.tree")
        if kind is not None and revision_kind is not None:
            expected_revision_kind = _REVISION_KIND_BY_SOURCE[kind]
            if revision_kind != expected_revision_kind:
                self.error(
                    f"{base}.revision_kind",
                    f"{kind} requires {expected_revision_kind}",
                )
        if revision_kind == "git_commit":
            if revision is not None and _HEX40.fullmatch(revision) is None:
                self.error(f"{base}.revision", "git_commit must be a 40-hex object id")
            if tree is None:
                self.error(f"{base}.tree", "git_repository requires a 40-hex tree id")
        elif tree is not None:
            self.error(f"{base}.tree", "must be null for a non-repository source")
        if revision_kind == "arxiv_version" and revision is not None:
            if _ARXIV_REVISION.fullmatch(revision) is None:
                self.error(f"{base}.revision", "must be a versioned arXiv identity")
        if revision_kind == "github_issue_updated_at" and revision is not None:
            self._utc_timestamp(revision, f"{base}.revision")

        observed = self.string(source.get("observed_at"), f"{base}.observed_at")
        if observed is not None:
            try:
                parsed_date = date.fromisoformat(observed)
                if parsed_date.isoformat() != observed:
                    raise ValueError
            except ValueError:
                self.error(f"{base}.observed_at", "must be an ISO 8601 calendar date")
        self.string(source.get("license_spdx"), f"{base}.license_spdx")
        self.enum(source.get("visibility"), f"{base}.visibility", _VISIBILITIES)
        self.literal(source.get("code_copied"), f"{base}.code_copied", False)
        access = self.exact_object(
            source.get("authoring_access"),
            f"{base}.authoring_access",
            _AUTHORING_ACCESS_KEYS,
        )
        if access is not None:
            self.literal(
                access.get("known_kernel_reproduction"),
                f"{base}.authoring_access.known_kernel_reproduction",
                True,
            )
            self.literal(
                access.get("clean_start"),
                f"{base}.authoring_access.clean_start",
                False,
            )
        self.string_list(
            source.get("catalog_entries"),
            f"{base}.catalog_entries",
            nonempty=False,
        )
        return source

    def _utc_timestamp(self, value: str, path: str) -> None:
        try:
            parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
            if not value.endswith("Z") or parsed.utcoffset() is None:
                raise ValueError
        except ValueError:
            self.error(path, "must be an ISO 8601 UTC timestamp ending in Z")

    def validate_operator(
        self,
        document: dict[str, object],
        relative: str,
        sources_by_id: Mapping[str, dict[str, object]],
        family_vocabulary: frozenset[str],
        mechanism_vocabulary: frozenset[str],
    ) -> dict[str, object]:
        occurrence = self.exact_object(document, relative, _OPERATOR_KEYS)
        assert occurrence is not None
        base = relative
        self.literal(
            occurrence.get("$schema"),
            f"{base}.$schema",
            "../schema.json#/$defs/operator_occurrence",
        )
        self.literal(occurrence.get("schema_version"), f"{base}.schema_version", 1)
        self.string(occurrence.get("operator_id"), f"{base}.operator_id", pattern=_ID)
        self.string(
            occurrence.get("implementation_id"),
            f"{base}.implementation_id",
            pattern=_ID,
        )
        source_id = self.string(
            occurrence.get("source_id"), f"{base}.source_id", pattern=_ID
        )
        source = sources_by_id.get(source_id) if source_id is not None else None
        if source_id is not None and source is None:
            self.error(f"{base}.source_id", "does not reference a manifest source")
        family = self.string(occurrence.get("family"), f"{base}.family", pattern=_ID)
        if family is not None and family not in family_vocabulary:
            self.error(f"{base}.family", "is not in manifest family_vocabulary")
        self.string(occurrence.get("variant"), f"{base}.variant")
        self.enum(occurrence.get("scope"), f"{base}.scope", _SCOPES)
        self.string(occurrence.get("semantic_role"), f"{base}.semantic_role")

        input_regime = self.exact_object(
            occurrence.get("input_regime"),
            f"{base}.input_regime",
            _INPUT_REGIME_KEYS,
        )
        if input_regime is not None:
            self.string_list(
                input_regime.get("dtypes"),
                f"{base}.input_regime.dtypes",
                nonempty=False,
                ids=True,
            )
            self.enum(
                input_regime.get("extent_class"),
                f"{base}.input_regime.extent_class",
                _EXTENT_CLASSES,
            )
            self.string_list(
                input_regime.get("shape_constraints"),
                f"{base}.input_regime.shape_constraints",
                nonempty=False,
                sorted_required=False,
            )
        self.string_list(
            occurrence.get("hardware_targets"),
            f"{base}.hardware_targets",
            nonempty=True,
            ids=True,
        )
        mechanisms = self.string_list(
            occurrence.get("mechanisms"),
            f"{base}.mechanisms",
            nonempty=True,
            ids=True,
        )
        for mechanism in mechanisms:
            if mechanism not in mechanism_vocabulary:
                self.error(
                    f"{base}.mechanisms",
                    f"{mechanism!r} is not in manifest mechanism_vocabulary",
                )

        locators = self._validate_locators(occurrence.get("locators"), base, source)
        self.enum(
            occurrence.get("upstream_state"),
            f"{base}.upstream_state",
            _UPSTREAM_STATES,
        )
        review_status = self.enum(
            occurrence.get("review_status"),
            f"{base}.review_status",
            _REVIEW_STATUSES,
        )
        self.literal(
            occurrence.get("claim_scope"),
            f"{base}.claim_scope",
            "source_fact_only",
        )
        if review_status == "collected" and not any(
            locator.get("kind") in {"pull_request", "web_document"}
            for locator in locators
        ):
            self.error(
                f"{base}.locators",
                "collected occurrences require a pull_request or web_document locator",
            )
        if review_status in {"source_verified", "semantics_reviewed"}:
            repository_locators = [
                locator
                for locator in locators
                if locator.get("kind") == "repository_path"
            ]
            if not repository_locators:
                self.error(
                    f"{base}.locators",
                    f"{review_status} requires at least one repository_path locator",
                )
            else:
                source_revision = source.get("revision") if source is not None else None
                if any(
                    locator.get("revision") != source_revision
                    for locator in repository_locators
                ):
                    self.error(
                        f"{base}.locators",
                        "every reviewed repository_path revision must equal the "
                        "referenced source revision",
                    )
                if any(
                    not isinstance(locator.get("git_object"), str)
                    or _HEX40.fullmatch(str(locator["git_object"])) is None
                    for locator in repository_locators
                ):
                    self.error(
                        f"{base}.locators",
                        "every reviewed repository_path requires a 40-hex git_object",
                    )
        return occurrence

    def _validate_locators(
        self,
        value: object,
        base: str,
        source: dict[str, object] | None,
    ) -> list[dict[str, object]]:
        path = f"{base}.locators"
        if not isinstance(value, list):
            self.error(path, "must be an array")
            return []
        if not value:
            self.error(path, "must not be empty")
        result: list[dict[str, object]] = []
        keys: list[tuple[str, str]] = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            locator = self.exact_object(item, item_path, _LOCATOR_KEYS)
            if locator is None:
                continue
            kind = self.enum(locator.get("kind"), f"{item_path}.kind", _LOCATOR_KINDS)
            locator_value = self.string(locator.get("value"), f"{item_path}.value")
            revision = self.hex40_or_null(
                locator.get("revision"), f"{item_path}.revision"
            )
            git_object = self.hex40_or_null(
                locator.get("git_object"), f"{item_path}.git_object"
            )
            symbol_value = locator.get("symbol")
            symbol = None
            if symbol_value is not None:
                symbol = self.string(symbol_value, f"{item_path}.symbol")
            if kind == "repository_path" and locator_value is not None:
                self.safe_path(locator_value, f"{item_path}.value")
                if source is not None and source.get("kind") != "git_repository":
                    self.error(
                        f"{item_path}.kind",
                        "repository_path is valid only for a git_repository source",
                    )
                source_revision = source.get("revision") if source is not None else None
                if revision is not None and revision != source_revision:
                    self.error(
                        f"{item_path}.revision",
                        "must equal the referenced source revision",
                    )
            elif kind in {"pull_request", "web_document"}:
                self.https_url(locator_value, f"{item_path}.value")
            result.append(locator)
            if kind is not None and locator_value is not None:
                keys.append((kind, locator_value))
        if len(keys) == len(value):
            if len(set(keys)) != len(keys):
                self.error(path, "must contain unique locators")
            if keys != sorted(keys):
                self.error(path, "must be sorted lexicographically")
        return result


def _load_manifest_paths(manifest: Mapping[str, object], name: str) -> list[str]:
    value = manifest.get(name)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _identity(document: Mapping[str, object], name: str) -> str | None:
    value = document.get(name)
    return value if isinstance(value, str) else None


def _complete_counts(vocabulary: Iterable[str], values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {name: counts.get(name, 0) for name in sorted(set(vocabulary) | set(counts))}


def validate_operator_library(library_root: Path) -> OperatorLibrarySummary:
    """Validate ``library_root`` without third-party packages or network access."""

    try:
        root = library_root.resolve(strict=True)
    except OSError as error:
        raise OperatorLibraryValidationError(
            [f"{library_root}: operator library directory does not exist ({error})"]
        ) from error
    if not root.is_dir():
        raise OperatorLibraryValidationError([f"{root}: must be a directory"])
    validator = _LibraryValidator(root)
    validator.validate_root_closure()
    schema_path = root / "schema.json"
    if schema_path.is_symlink() or not schema_path.is_file():
        validator.error("schema.json", "must be a regular non-symlink file")
    manifest = validator.read_document(root / "manifest.json", "manifest.json")
    if manifest is None:
        raise OperatorLibraryValidationError(validator.errors)
    manifest = validator.validate_manifest(manifest)

    sources: list[dict[str, object]] = []
    for relative in _load_manifest_paths(manifest, "sources"):
        if validator.safe_path(relative, f"manifest source {relative!r}") is None:
            continue
        path = validator.referenced_file(relative, f"manifest source {relative!r}")
        if path is None:
            continue
        document = validator.read_document(path, relative)
        if document is not None:
            sources.append(validator.validate_source(document, relative))
    sources_by_id: dict[str, dict[str, object]] = {}
    for source in sources:
        source_id = _identity(source, "source_id")
        if source_id is None:
            continue
        if source_id in sources_by_id:
            validator.error(source_id, "duplicate source_id")
        else:
            sources_by_id[source_id] = source

    family_vocabulary = frozenset(
        item
        for item in manifest.get("family_vocabulary", [])
        if isinstance(item, str)
    ) if isinstance(manifest.get("family_vocabulary"), list) else frozenset()
    mechanism_vocabulary = frozenset(
        item
        for item in manifest.get("mechanism_vocabulary", [])
        if isinstance(item, str)
    ) if isinstance(manifest.get("mechanism_vocabulary"), list) else frozenset()
    operators_with_paths: list[tuple[str, dict[str, object]]] = []
    for relative in _load_manifest_paths(manifest, "operators"):
        if validator.safe_path(relative, f"manifest operator {relative!r}") is None:
            continue
        path = validator.referenced_file(relative, f"manifest operator {relative!r}")
        if path is None:
            continue
        document = validator.read_document(path, relative)
        if document is not None:
            operators_with_paths.append(
                (
                    relative,
                    validator.validate_operator(
                        document,
                        relative,
                        sources_by_id,
                        family_vocabulary,
                        mechanism_vocabulary,
                    ),
                )
            )
    implementations: dict[str, str] = {}
    for relative, occurrence in operators_with_paths:
        implementation_id = _identity(occurrence, "implementation_id")
        if implementation_id is None:
            continue
        if implementation_id in implementations:
            validator.error(
                relative,
                f"duplicate implementation_id also used by {implementations[implementation_id]}",
            )
        else:
            implementations[implementation_id] = relative

    if validator.errors:
        raise OperatorLibraryValidationError(validator.errors)

    operators = [occurrence for _, occurrence in operators_with_paths]
    library_id = _identity(manifest, "library_id")
    state = _identity(manifest, "state")
    assert library_id is not None and state is not None
    operator_ids = {
        value
        for occurrence in operators
        if (value := _identity(occurrence, "operator_id")) is not None
    }
    families = [
        value
        for occurrence in operators
        if (value := _identity(occurrence, "family")) is not None
    ]
    reviews = [
        value
        for occurrence in operators
        if (value := _identity(occurrence, "review_status")) is not None
    ]
    scopes = [
        value
        for occurrence in operators
        if (value := _identity(occurrence, "scope")) is not None
    ]
    mechanisms = [
        mechanism
        for occurrence in operators
        if isinstance(occurrence.get("mechanisms"), list)
        for mechanism in occurrence["mechanisms"]
        if isinstance(mechanism, str)
    ]
    return OperatorLibrarySummary(
        library_id=library_id,
        state=state,
        source_count=len(sources),
        operator_count=len(operators),
        mathematical_operator_count=len(operator_ids),
        family_counts=_complete_counts(family_vocabulary, families),
        review_counts=_complete_counts(_REVIEW_STATUSES, reviews),
        scope_counts=_complete_counts(_SCOPES, scopes),
        mechanism_counts=_complete_counts(mechanism_vocabulary, mechanisms),
    )


def _text_report(summary: OperatorLibrarySummary) -> str:
    lines = [
        f"operator library valid: {summary.library_id} ({summary.state})",
        f"sources: {summary.source_count}",
        f"operators: {summary.operator_count}",
        f"mathematical operators: {summary.mathematical_operator_count}",
    ]
    for title, counts in (
        ("family counts", summary.family_counts),
        ("review counts", summary.review_counts),
        ("scope counts", summary.scope_counts),
        ("mechanism counts", summary.mechanism_counts),
    ):
        lines.append(f"{title}:")
        lines.extend(f"  {name}: {count}" for name, count in counts.items())
    return "\n".join(lines)


def _markdown_report(summary: OperatorLibrarySummary) -> str:
    lines = [
        f"# Operator library: `{summary.library_id}`",
        "",
        f"State: `{summary.state}`",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Sources | {summary.source_count} |",
        f"| Operators | {summary.operator_count} |",
        f"| Mathematical operators | {summary.mathematical_operator_count} |",
    ]
    for title, counts in (
        ("Family counts", summary.family_counts),
        ("Review counts", summary.review_counts),
        ("Scope counts", summary.scope_counts),
        ("Mechanism counts", summary.mechanism_counts),
    ):
        lines.extend(("", f"## {title}", "", "| Value | Count |", "| --- | ---: |"))
        lines.extend(f"| `{name}` | {count} |" for name, count in counts.items())
    return "\n".join(lines)


def _print_failure(error: OperatorLibraryValidationError, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps({"valid": False, "errors": list(error.errors)}, indent=2))
        return
    if output_format == "markdown":
        print("# Operator library validation failed", file=sys.stderr)
        for message in error.errors:
            print(f"- {message}", file=sys.stderr)
        return
    print("operator library validation failed:", file=sys.stderr)
    for message in error.errors:
        print(f"  - {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "operator_library",
        help="operator_library directory (defaults to this checkout)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="summary format (default: text)",
    )
    arguments = parser.parse_args(argv)
    try:
        summary = validate_operator_library(arguments.library_root)
    except OperatorLibraryValidationError as error:
        _print_failure(error, arguments.format)
        return 1
    if arguments.format == "json":
        print(json.dumps(summary.report(), indent=2, sort_keys=True))
    elif arguments.format == "markdown":
        print(_markdown_report(summary))
    else:
        print(_text_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
