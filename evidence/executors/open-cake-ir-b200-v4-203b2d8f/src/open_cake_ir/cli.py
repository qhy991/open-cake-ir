"""Command-line composition root for Compiler and Research Lab Interfaces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from open_cake_ir.compiler import Compiler
from open_cake_ir.lab import (
    CampaignLock,
    Lab,
    execute_matched_from_config,
    execute_portfolio_from_config,
)


def _emit(value: object) -> None:
    sys.stdout.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def _compiler(args: argparse.Namespace) -> int:
    compiler = Compiler.load(args.project_root, args.revision)
    if args.compiler_command == "assess":
        assessment = compiler.assess_file(args.schedule)
        _emit(
            {
                "compiler_revision_id": assessment.compiler_revision_id,
                "compiler_revision_sha256": assessment.compiler_revision_sha256,
                "schedule_id": assessment.schedule_id,
                "schedule_sha256": assessment.schedule_sha256,
                "target": assessment.target,
                "profile": assessment.profile,
                "accepted": assessment.accepted,
                "lowering_eligible": assessment.lowering_eligible,
                "findings": [asdict(finding) for finding in assessment.findings],
                "analysis": dict(assessment.analysis),
            }
        )
        return 0 if assessment.accepted else 2
    if args.compiler_command == "check-corpus":
        report = compiler.check_corpus()
        _emit(
            {
                "corpus_id": report.corpus_id,
                "compiler_revision_id": report.compiler_revision_id,
                "compiler_revision_sha256": report.compiler_revision_sha256,
                "passed": report.passed,
                "case_count": report.case_count,
                "cases": [asdict(case) for case in report.cases],
            }
        )
        return 0 if report.passed else 2
    assessment = compiler.assess_file(args.schedule)
    lowering = compiler.lower(assessment)
    output = args.output.absolute()
    with output.open("x", encoding="utf-8") as stream:
        stream.write(lowering.source)
    _emit(
        {
            "schedule_sha256": lowering.schedule_sha256,
            "source_sha256": lowering.source_sha256,
            "entry_point": lowering.entry_point,
            "output": str(output),
        }
    )
    return 0


def _lab(args: argparse.Namespace) -> int:
    lab = Lab(args.project_root)
    if args.lab_command == "preflight":
        lock = lab.preflight(args.study)
        if args.output is not None:
            with args.output.absolute().open("x", encoding="utf-8") as stream:
                json.dump(lock.document, stream, sort_keys=True, ensure_ascii=False)
                stream.write("\n")
            args.output.absolute().chmod(0o644)
        _emit(
            {
                "study_id": lock.study_id,
                "study_kind": lock.study_kind,
                "claim_scope": lock.claim_scope,
                "workload_id": lock.workload_id,
                "compiler_revision_id": lock.compiler_revision_id,
                "run_order": list(lock.run_order),
                "experimental_unit": lock.experimental_unit,
                "estimand": lock.estimand,
                "campaign_lock_sha256": lock.canonical_sha256,
                "campaign_lock": lock.document,
            }
        )
        return 0
    lock = CampaignLock.load(args.lock)
    if args.lab_command == "execute":
        campaign = (
            execute_portfolio_from_config(
                args.project_root,
                lock,
                args.runtime_config,
                args.evidence_root,
            )
            if lock.study_kind == "portfolio"
            else execute_matched_from_config(
                args.project_root,
                lock,
                args.runtime_config,
                args.evidence_root,
            )
        )
        report = (
            lab.audit_portfolio(campaign)
            if lock.study_kind == "portfolio"
            else lab.audit(campaign)
        )
        _emit(asdict(report))
        return 0 if report.archive_integrity_passed else 2
    campaign = lab.reference_campaign(lock, args.evidence_root)
    report = (
        lab.audit_portfolio(campaign)
        if lock.study_kind == "portfolio"
        else lab.audit(campaign)
    )
    _emit(asdict(report))
    return 0 if report.archive_integrity_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="open-cake-ir")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)

    compiler = commands.add_parser("compiler")
    compiler_commands = compiler.add_subparsers(dest="compiler_command", required=True)
    for name in ("assess", "lower"):
        command = compiler_commands.add_parser(name)
        command.add_argument("--revision", type=Path, required=True)
        command.add_argument("schedule", type=Path)
        if name == "lower":
            command.add_argument("--output", type=Path, required=True)
    check = compiler_commands.add_parser("check-corpus")
    check.add_argument("--revision", type=Path, required=True)

    lab = commands.add_parser("lab")
    lab_commands = lab.add_subparsers(dest="lab_command", required=True)
    preflight = lab_commands.add_parser("preflight")
    preflight.add_argument("study", type=Path)
    preflight.add_argument("--output", type=Path)
    audit = lab_commands.add_parser("audit")
    audit.add_argument("--lock", type=Path, required=True)
    audit.add_argument("--evidence-root", type=Path, required=True)
    execute = lab_commands.add_parser("execute")
    execute.add_argument("--lock", type=Path, required=True)
    execute.add_argument("--runtime-config", type=Path, required=True)
    execute.add_argument("--evidence-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one public Compiler or Lab command."""

    args = build_parser().parse_args(argv)
    args.project_root = args.project_root.resolve(strict=True)
    if args.command == "compiler":
        return _compiler(args)
    return _lab(args)


if __name__ == "__main__":
    raise SystemExit(main())
