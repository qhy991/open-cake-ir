"""Observed-versus-expected Compiler Corpus evaluation and its reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from .errors import CompilerError
from .revision import _digest, _name, _object, _objects, _strings

if TYPE_CHECKING:
    from .core import Compiler


@dataclass(frozen=True)
class CorpusCaseReport:
    """Observed-versus-expected result for one Compiler Corpus case."""

    case_id: str
    schedule_path: str
    target: str
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
    # Defaulted so the Revision's declared set, not the case list, decides what
    # "unexamined" means. A Gate built without it reports no coverage rather than
    # reporting full coverage.
    declared_targets: tuple[str, ...] = ()

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def examined_targets(self) -> Mapping[str, int]:
        """Declared Targets the Corpus exercised, and how many cases carried each."""

        counts: dict[str, int] = {}
        declared = set(self.declared_targets)
        for case in self.cases:
            if case.target in declared:
                counts[case.target] = counts.get(case.target, 0) + 1
        return MappingProxyType(dict(sorted(counts.items())))

    @property
    def unexamined_targets(self) -> tuple[str, ...]:
        """Declared Targets no case names, so this Gate states nothing about them.

        A passing Gate is evidence only about the Targets it examined. Absence is
        reported and never repaired here: minting cases to empty this tuple would
        report a coverage the Corpus does not have.
        """

        examined = self.examined_targets
        return tuple(target for target in sorted(self.declared_targets) if target not in examined)

    @property
    def undeclared_case_targets(self) -> Mapping[str, int]:
        """Targets named only by cases whose Revision does not declare them.

        These exercise the refusal path. Counting them as coverage would read a
        rejection as an examination, so they are reported apart from examined.
        """

        counts: dict[str, int] = {}
        declared = set(self.declared_targets)
        for case in self.cases:
            if case.target not in declared:
                counts[case.target] = counts.get(case.target, 0) + 1
        return MappingProxyType(dict(sorted(counts.items())))

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


def assess_case(compiler: Compiler, project_root: str | Path,
                case: Mapping[str, object]):
    """Assess one declared case exactly as the Gate does.

    Any tool that recomputes an expectation must reach the same verdict as the Gate it
    is recomputing for. A second copy of the target rule here would let the refresher
    adopt a pin the Gate never produces, which is the manufactured match its own
    refusal exists to prevent.
    """

    case_id = _name(case.get("case_id"), "corpus case case_id")
    relative = _name(case.get("schedule"), f"corpus case {case_id!r} schedule")
    schedule_path = (Path(project_root) / relative).resolve(strict=True)
    return _assess_case(compiler, case, case_id, f"corpus case {case_id!r}", schedule_path)


def _assess_case(compiler: Compiler, case: Mapping[str, object], case_id: str,
                 context: str, schedule_path: Path):
    """Assess one case, on the target it names or the one its Schedule declares.

    A Schedule embeds its own target, so a second target used to mean a second copy of
    the whole document -- which is why two declared Targets carry no case at all. A case
    may instead name the exact target to assess this Schedule on. Nothing is stepped
    down: the named target replaces the declared one and each pair keeps its own
    expectation, so a Schedule legal on one target and refused on another states both.
    """

    override = case.get("target")
    if override is None:
        return compiler.assess_file(schedule_path)
    target_id = _name(override, f"{context}.target")
    if schedule_path.suffix != ".json":
        raise CompilerError(
            f"corpus case {case_id!r} names a target for a {schedule_path.suffix} "
            "Schedule; only a JSON Schedule document carries a target to replace"
        )
    document = json.loads(schedule_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise CompilerError(f"corpus case {case_id!r} Schedule must be an object")
    if document.get("target") == target_id:
        raise CompilerError(
            f"corpus case {case_id!r} names the target its Schedule already declares"
        )
    return compiler.assess({**document, "target": target_id})


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
        if not {"case_id", "schedule", "expected"} <= set(case) <= {
            "case_id", "schedule", "expected", "target",
        }:
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
        assessment = _assess_case(
            compiler, case, case_id, f"corpus.cases[{index}]", schedule_path)
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
                target=assessment.target,
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
        declared_targets=tuple(sorted(compiler._revision.targets)),
    )
