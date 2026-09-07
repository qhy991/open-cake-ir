"""Observed-versus-expected Compiler Corpus evaluation and its reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import CompilerError
from .revision import _digest, _name, _object, _objects, _strings

if TYPE_CHECKING:
    from .core import Compiler


@dataclass(frozen=True)
class CorpusCaseReport:
    """Observed-versus-expected result for one Compiler Corpus case."""

    case_id: str
    schedule_path: str
    expected_accepted: bool
    expected_lowering_eligible: bool
    expected_finding_codes: tuple[str, ...]
    expected_schedule_sha256: str
    expected_lowering_source_sha256: str | None
    observed_accepted: bool
    observed_lowering_eligible: bool
    observed_finding_codes: tuple[str, ...]
    observed_schedule_sha256: str
    observed_lowering_source_sha256: str | None
    matched: bool


@dataclass(frozen=True)
class CorpusGateReport:
    """Release-gate projection for the full declared Compiler Corpus."""

    corpus_id: str
    compiler_revision_id: str
    compiler_revision_sha256: str
    passed: bool
    cases: tuple[CorpusCaseReport, ...]

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def accepted_case_count(self) -> int:
        return sum(case.expected_accepted for case in self.cases)

    @property
    def rejected_case_count(self) -> int:
        return self.case_count - self.accepted_case_count

    @property
    def lowerable_case_count(self) -> int:
        return sum(case.expected_lowering_eligible for case in self.cases)

    @property
    def nonlowerable_case_count(self) -> int:
        return self.case_count - self.lowerable_case_count


def check_corpus(compiler: Compiler, corpus_path: str | Path) -> CorpusGateReport:
    """Assess and lower every declared case through the public Compiler methods.

    The facade owns one ``_revision`` for identity and project-root admission.
    Expected counts and ordered findings retain the Corpus manifest's semantics.
    """

    manifest = _object(
        json.loads(Path(corpus_path).read_text(encoding="utf-8")),
        "corpus",
    )
    if set(manifest) != {"schema_version", "corpus_id", "state", "cases"}:
        raise CompilerError("corpus manifest fields differ")
    if manifest.get("schema_version") != 1 or manifest.get("state") not in {"draft", "released"}:
        raise CompilerError("corpus manifest schema or state differs")
    corpus_id = _name(manifest.get("corpus_id"), "corpus.corpus_id")
    cases = _objects(manifest.get("cases"), "corpus.cases")
    if not cases:
        raise CompilerError("Compiler Corpus must contain at least one case")
    reports: list[CorpusCaseReport] = []
    observed_ids: set[str] = set()
    for index, case in enumerate(cases):
        if set(case) != {"case_id", "schedule", "expected"}:
            raise CompilerError(f"corpus.cases[{index}] fields differ")
        case_id = _name(case.get("case_id"), f"corpus.cases[{index}].case_id")
        if case_id in observed_ids:
            raise CompilerError(f"corpus case {case_id!r} is duplicated")
        observed_ids.add(case_id)
        relative = _name(case.get("schedule"), f"corpus.cases[{index}].schedule")
        schedule_path = (compiler._revision.project_root / relative).resolve(strict=True)
        if compiler._revision.project_root not in schedule_path.parents:
            raise CompilerError(f"corpus case {case_id!r} escapes project root")
        expected = _object(case.get("expected"), f"corpus.cases[{index}].expected")
        if set(expected) != {
            "accepted",
            "lowering_eligible",
            "finding_codes",
            "schedule_sha256",
            "lowering_source_sha256",
        }:
            raise CompilerError(f"corpus case {case_id!r} expected fields differ")
        expected_accepted = expected.get("accepted")
        expected_lowering = expected.get("lowering_eligible")
        if not isinstance(expected_accepted, bool) or not isinstance(expected_lowering, bool):
            raise CompilerError(f"corpus case {case_id!r} expected booleans differ")
        expected_codes = _strings(
            expected.get("finding_codes"), f"corpus.cases[{index}].expected.finding_codes"
        )
        expected_schedule_sha = _digest(
            expected.get("schedule_sha256"),
            f"corpus.cases[{index}].expected.schedule_sha256",
        )
        assert expected_schedule_sha is not None
        expected_source_sha = _digest(
            expected.get("lowering_source_sha256"),
            f"corpus.cases[{index}].expected.lowering_source_sha256",
            nullable=True,
        )
        assessment = compiler.assess_file(schedule_path)
        observed_codes = tuple(finding.code for finding in assessment.findings)
        observed_source_sha = (
            compiler.lower(assessment).source_sha256 if assessment.lowering_eligible else None
        )
        matched = (
            assessment.accepted is expected_accepted
            and assessment.lowering_eligible is expected_lowering
            and observed_codes == expected_codes
            and assessment.schedule_sha256 == expected_schedule_sha
            and observed_source_sha == expected_source_sha
        )
        reports.append(
            CorpusCaseReport(
                case_id=case_id,
                schedule_path=relative,
                expected_accepted=expected_accepted,
                expected_lowering_eligible=expected_lowering,
                expected_finding_codes=expected_codes,
                expected_schedule_sha256=expected_schedule_sha,
                expected_lowering_source_sha256=expected_source_sha,
                observed_accepted=assessment.accepted,
                observed_lowering_eligible=assessment.lowering_eligible,
                observed_finding_codes=observed_codes,
                observed_schedule_sha256=assessment.schedule_sha256,
                observed_lowering_source_sha256=observed_source_sha,
                matched=matched,
            )
        )
    return CorpusGateReport(
        corpus_id=corpus_id,
        compiler_revision_id=compiler._revision.revision_id,
        compiler_revision_sha256=compiler._revision.canonical_sha256,
        passed=all(report.matched for report in reports),
        cases=tuple(reports),
    )
