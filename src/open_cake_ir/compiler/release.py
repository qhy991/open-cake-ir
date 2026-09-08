"""Content-bound release gate for an Open Cake Compiler Revision."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

from .core import Compiler
from .errors import CompilerError


# Canonical model policy for independent agent reviewers (ADR 0052).
ALLOWED_REVIEW_MODELS = frozenset({"opus-5", "fable-5", "fable-5.1", "gpt-6-astra"})


def _validate_reviewer(value: object) -> None:
    reviewer = _object(value, "compiler_release_approval.reviewer")
    if reviewer.get("kind") == "human":
        if set(reviewer) != {"kind", "name"} or not _identity(reviewer.get("name")):
            raise CompilerError("Compiler release human reviewer identity differs")
        return
    if reviewer.get("kind") != "agent_session" or set(reviewer) != {
        "kind", "model", "session_id", "author_session_id"
    }:
        raise CompilerError("Compiler release reviewer fields differ")
    model = reviewer.get("model")
    if not isinstance(model, str) or model not in ALLOWED_REVIEW_MODELS:
        raise CompilerError("Compiler release reviewer model is not allowed")
    session_id = reviewer.get("session_id")
    author_session_id = reviewer.get("author_session_id")
    if not _identity(session_id) or not _identity(author_session_id):
        raise CompilerError("Compiler release reviewer session identity differs")
    if str(session_id).casefold() == str(author_session_id).casefold():
        raise CompilerError("Compiler release reviewer must use a different author session")


def _identity(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CompilerError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _safe_path(root: Path, value: object, context: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise CompilerError(f"{context} must be a non-empty string")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise CompilerError(f"{context} is unsafe")
    path = (root / value).resolve(strict=True)
    if root not in path.parents:
        raise CompilerError(f"{context} escapes project root")
    return value, path


@dataclass(frozen=True)
class CompilerRelease:
    """Released Compiler Revision document and its canonical identity."""

    document: Mapping[str, object]
    canonical_sha256: str

    def verify(self, project_root: str | Path) -> bool:
        """Rehash every bound source and corpus object."""

        root = Path(project_root).resolve(strict=True)
        sources = self.document.get("sources")
        if not isinstance(sources, list):
            return False
        for item in sources:
            if not isinstance(item, Mapping):
                return False
            try:
                _, path = _safe_path(root, item.get("path"), "release.sources.path")
            except (CompilerError, OSError):
                return False
            payload = path.read_bytes()
            if item.get("sha256") != sha256(payload).hexdigest() or item.get("size_bytes") != len(payload):
                return False
        corpus = self.document.get("corpus_manifest")
        if not isinstance(corpus, Mapping):
            return False
        try:
            _, corpus_path = _safe_path(root, corpus.get("path"), "release.corpus_manifest.path")
            parsed = json.loads(corpus_path.read_text(encoding="utf-8"))
        except (CompilerError, OSError, UnicodeError, json.JSONDecodeError):
            return False
        references = (self.document.get("corpus_gate"), self.document.get("release_approval"))
        for reference in references:
            if not isinstance(reference, Mapping):
                return False
            try:
                _, reference_path = _safe_path(root, reference.get("path"), "release.reference.path")
                reference_document = json.loads(reference_path.read_text(encoding="utf-8"))
            except (CompilerError, OSError, UnicodeError, json.JSONDecodeError):
                return False
            if reference.get("canonical_sha256") != sha256(
                _canonical_json_bytes(reference_document)
            ).hexdigest():
                return False
        return (
            corpus.get("canonical_sha256")
            == sha256(_canonical_json_bytes(parsed)).hexdigest()
            and self.canonical_sha256 == sha256(_canonical_json_bytes(self.document)).hexdigest()
        )


@dataclass(frozen=True)
class CompilerGate:
    """Persistent full-corpus observation awaiting an independent review."""

    document: Mapping[str, object]
    canonical_sha256: str


def build_gate_report(
    project_root: str | Path,
    revision_path: str | Path,
    source_set_path: str | Path,
) -> CompilerGate:
    """Build the complete review surface without promoting the draft Revision."""

    root = Path(project_root).resolve(strict=True)
    compiler = Compiler.load(root, revision_path)
    if compiler.state != "draft":
        raise CompilerError("Corpus Gate input must be a draft Compiler Revision")
    gate = compiler.check_corpus()
    proposal = _object(json.loads(Path(revision_path).read_text()), "compiler_revision")
    source_set = _object(json.loads(Path(source_set_path).read_text()), "compiler_source_set")
    paths = source_set.get("paths")
    if (
        set(source_set) != {"schema_version", "paths"}
        or source_set.get("schema_version") != 1
        or not isinstance(paths, list)
        or not paths
    ):
        raise CompilerError("compiler source-set fields differ")
    source_receipts: list[dict[str, object]] = []
    observed_paths: set[str] = set()
    for index, value in enumerate(paths):
        relative, path = _safe_path(root, value, f"compiler_source_set.paths[{index}]")
        if relative in observed_paths:
            raise CompilerError(f"compiler source {relative!r} is duplicated")
        observed_paths.add(relative)
        payload = path.read_bytes()
        source_receipts.append(
            {"path": relative, "sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}
        )
    # Everything the gate reads has to be something the Revision binds. The manifest is
    # bound, so adding a case invalidates the lock -- but the Schedule a case points at is
    # bound only if it is listed here too, and nothing connected the two lists. A gated
    # Schedule outside the source set can be edited under a released Revision without the
    # Revision noticing, which is the one thing content-binding exists to stop.
    unbound = sorted(
        {c.schedule_path for c in gate.cases if c.schedule_path not in observed_paths}
    )
    if unbound:
        raise CompilerError(
            "the Corpus Gate reads Schedules the Revision does not bind: "
            + ", ".join(unbound)
        )
    if not gate.passed:
        failed = ", ".join(case.case_id for case in gate.cases if not case.matched)
        raise CompilerError(f"compiler corpus gate failed: {failed}")
    document: dict[str, object] = {
        "schema_version": 2,
        "decision": "awaiting_review",
        "proposal": {
            "path": Path(revision_path).resolve(strict=True).relative_to(root).as_posix(),
            "canonical_sha256": sha256(_canonical_json_bytes(proposal)).hexdigest(),
        },
        "source_set": {
            "path": Path(source_set_path).resolve(strict=True).relative_to(root).as_posix(),
            "canonical_sha256": sha256(_canonical_json_bytes(source_set)).hexdigest(),
        },
        "sources": source_receipts,
        "corpus_id": gate.corpus_id,
        "compiler_revision_id": gate.compiler_revision_id,
        "compiler_revision_sha256": gate.compiler_revision_sha256,
        "case_count": gate.case_count,
        "matched_case_count": sum(case.matched for case in gate.cases),
        "cases": [asdict(case) for case in gate.cases],
    }
    return CompilerGate(document, sha256(_canonical_json_bytes(document)).hexdigest())


def _validate_review_record(root: Path, approval: Mapping[str, object], *, read_record: bool) -> None:
    if _object(approval["reviewer"], "reviewer")["kind"] != "agent_session":
        return
    locator = approval.get("review_record")
    if (not _identity(locator) or not Path(str(locator)).is_absolute()
            or ".." in Path(str(locator)).parts or "\\" in str(locator)):
        raise CompilerError("Compiler agent approval requires an absolute external review_record")
    if str(locator) not in str(approval["approval_basis"]):
        raise CompilerError("Compiler approval basis must reference its review_record")
    if not read_record:
        return
    path = Path(str(locator))
    try:
        if (path.resolve(strict=True) != path or not path.is_file()
                or root in path.parents
                or any((parent / ".git").exists() for parent in path.parents)):
            raise CompilerError("Compiler review_record must be an existing external regular file")
        record = _object(json.loads(path.read_text(encoding="utf-8")), "review_record")
    except (OSError, ValueError) as error:
        raise CompilerError("Compiler review_record is unavailable or malformed") from error
    if (set(record) != {"schema_version", "decision", "gate_report", "reviewer",
                       "reviewed_commit", "launcher_record", "review_basis"}
            or type(record.get("schema_version")) is not int or record["schema_version"] != 1
            or record.get("decision") != "approved"
            or record.get("gate_report") != approval["gate_report"]
            or record.get("reviewer") != approval["reviewer"]
            or record.get("review_basis") != approval["approval_basis"]
            or not isinstance(record.get("reviewed_commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}", record["reviewed_commit"]) is None
            or not _identity(record.get("launcher_record"))):
        raise CompilerError("Compiler review_record declarations differ from the approval")


def build_release(
    project_root: str | Path,
    revision_path: str | Path,
    source_set_path: str | Path,
    gate_report_path: str | Path,
    approval_path: str | Path,
) -> CompilerRelease:
    """Construct a new release only with its available independent review record."""
    return _build_release(project_root, revision_path, source_set_path,
                          gate_report_path, approval_path, constructing=True)


def verify_release(
    project_root: str | Path,
    revision_path: str | Path,
    source_set_path: str | Path,
    gate_report_path: str | Path,
    approval_path: str | Path,
    release_path: str | Path,
) -> bool:
    """Verify retained authorities without requiring the reviewer's private filesystem."""
    release = _build_release(project_root, revision_path, source_set_path,
                             gate_report_path, approval_path, constructing=False)
    return (Path(release_path).read_bytes() == _canonical_json_bytes(release.document) + b"\n"
            and release.verify(project_root))


def _build_release(
    project_root: str | Path,
    revision_path: str | Path,
    source_set_path: str | Path,
    gate_report_path: str | Path,
    approval_path: str | Path,
    *, constructing: bool,
) -> CompilerRelease:
    """Build a released revision only after the complete declared Corpus passes."""

    root = Path(project_root).resolve(strict=True)
    observed_gate = build_gate_report(root, revision_path, source_set_path)
    gate_report = _object(
        json.loads(Path(gate_report_path).read_text(encoding="utf-8")),
        "compiler_gate_report",
    )
    if sha256(_canonical_json_bytes(gate_report)).hexdigest() != observed_gate.canonical_sha256:
        raise CompilerError("persistent Compiler Corpus Gate report differs")
    approval = _object(
        json.loads(Path(approval_path).read_text(encoding="utf-8")),
        "compiler_release_approval",
    )
    fields = {
        "schema_version",
        "decision",
        "gate_report",
        "reviewer",
        "approval_basis",
    }
    schema = approval.get("schema_version")
    if (type(schema) is not int or schema not in ({3} if constructing else {2, 3})
            or set(approval) not in (fields, fields | {"review_record"})
            or (schema == 2 and set(approval) != fields)
            or approval.get("decision") != "approved"):
        raise CompilerError("Compiler release approval differs")
    approval_gate = _object(approval.get("gate_report"), "compiler_release_approval.gate_report")
    gate_relative, resolved_gate_path = _safe_path(
        root, approval_gate.get("path"), "compiler_release_approval.gate_report.path"
    )
    if resolved_gate_path != Path(gate_report_path).resolve(strict=True) or approval_gate.get(
        "canonical_sha256"
    ) != observed_gate.canonical_sha256:
        raise CompilerError("Compiler release approval does not bind this Gate report")
    _validate_reviewer(approval.get("reviewer"))
    approval_basis = approval.get("approval_basis")
    if not isinstance(approval_basis, str) or not approval_basis.strip():
        raise CompilerError("Compiler release approval basis differs")
    if schema == 3:
        _validate_review_record(root, approval, read_record=constructing)
    compiler = Compiler.load(root, revision_path)
    gate = compiler.check_corpus()

    proposal = _object(
        json.loads(Path(revision_path).read_text(encoding="utf-8")),
        "compiler_revision",
    )
    source_set = _object(
        json.loads(Path(source_set_path).read_text(encoding="utf-8")),
        "compiler_source_set",
    )
    if set(source_set) != {"schema_version", "paths"} or source_set.get("schema_version") != 1:
        raise CompilerError("compiler source-set fields differ")
    paths = source_set.get("paths")
    if not isinstance(paths, list) or not paths:
        raise CompilerError("compiler source-set paths must be a non-empty list")
    sources: list[dict[str, object]] = []
    observed: set[str] = set()
    for index, value in enumerate(paths):
        relative, path = _safe_path(root, value, f"compiler_source_set.paths[{index}]")
        if relative in observed:
            raise CompilerError(f"compiler source {relative!r} is duplicated")
        observed.add(relative)
        payload = path.read_bytes()
        sources.append(
            {"path": relative, "sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}
        )

    corpus_relative, corpus_path = _safe_path(
        root, proposal.get("corpus_manifest"), "compiler_revision.corpus_manifest"
    )
    corpus_document = json.loads(corpus_path.read_text(encoding="utf-8"))
    proposal_id = proposal.get("revision_id")
    if not isinstance(proposal_id, str) or not proposal_id:
        raise CompilerError("compiler revision_id is invalid")
    # The draft remains stable while it is reviewed. Final identity includes the
    # independent approval as well as the Gate, so concurrent releases of the
    # same source with different review authorities cannot reuse one identity.
    approval_identity = sha256(_canonical_json_bytes(approval)).hexdigest()
    release_identity = sha256(_canonical_json_bytes({
        "gate_report": observed_gate.canonical_sha256,
        "release_approval": approval_identity,
    })).hexdigest()
    released_id = proposal_id.removesuffix('-draft')
    if schema == 3:
        released_id += "+" + release_identity
    document: dict[str, object] = {
        "schema_version": 1,
        "revision_id": released_id,
        "state": "released",
        "target_definitions": proposal.get("target_definitions"),
        "calibration_coverage": proposal.get("calibration_coverage"),
        "corpus_manifest": {
            "path": corpus_relative,
            "canonical_sha256": sha256(_canonical_json_bytes(corpus_document)).hexdigest(),
        },
        "corpus_gate": {
            "path": gate_relative,
            "canonical_sha256": sha256(_canonical_json_bytes(gate_report)).hexdigest(),
            "case_count": gate.case_count,
            "matched_case_count": sum(case.matched for case in gate.cases),
        },
        "release_approval": {
            "path": Path(approval_path).resolve(strict=True).relative_to(root).as_posix(),
            "canonical_sha256": approval_identity,
        },
        "sources": sources,
    }
    return CompilerRelease(
        document=document,
        canonical_sha256=sha256(_canonical_json_bytes(document)).hexdigest(),
    )
