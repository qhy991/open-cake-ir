"""Command-line composition root for Compiler and Research Lab Interfaces."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Sequence

from open_cake_ir.compiler import Compiler, CompilerError
from open_cake_ir.compiler.frontend import FrontendError, read_schedule
from open_cake_ir.lab import (
    CampaignLock,
    Lab,
    execute_matched_from_config,
    execute_portfolio_from_config,
)
from open_cake_ir.lab.custody import admit_new_campaign_path


def _json_projection(value: object) -> object:
    """Project immutable public records without copying their implementation types."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_projection(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("CLI JSON object keys must be strings")
        return {str(key): _json_projection(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_projection(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"CLI value {type(value).__name__!r} is not JSON-compatible")


def _emit(value: object) -> None:
    sys.stdout.write(
        json.dumps(_json_projection(value), sort_keys=True, ensure_ascii=False) + "\n"
    )


def _emit_compiler(value: Mapping[str, object], args: argparse.Namespace) -> None:
    """Render the same public result for a reader or a JSON consumer."""

    if args.output_format == "json":
        _emit(value)
        return
    if "accepted" in value:
        print(f"执行计划：{value['schedule_id']}")
        print(f"编译器：{value['compiler_revision_id']}")
        print(f"目标：{value['target']}")
        print("结构检查：" + ("通过" if value['accepted'] else "未通过"))
        print("生成代码：" + ("允许" if value['lowering_eligible'] else "不允许"))
        route = value['lowering']
        if route is not None:
            print(f"生成方式：{route['backend']}")
        findings = value['findings']
        print(f"诊断：{len(findings)} 条")
        for finding in findings:
            impact = (
                "阻止接受" if finding['blocks_acceptance'] else
                "阻止生成" if finding['blocks_lowering'] else "提示"
            )
            source = finding.get("source")
            location = (f"{source['filename']}:{source['line']}:{source['column']} | "
                        if source is not None else "")
            print(f"- [{impact}] {finding['code']} | {location}{finding['path']}")
        print("这一步未运行 GPU；诊断原文和完整分析可用默认 JSON 输出查看。")
    elif args.compiler_command == "check-corpus":
        matched = sum(case['matched'] for case in value['cases'])
        print(f"编译器：{value['compiler_revision_id']}")
        print(f"语料检查：{matched}/{value['case_count']} 项符合预期")
        for case in value['cases']:
            if not case['matched']:
                print(f"- 不符合预期：{case['case_id']} | {case['schedule_path']}")
        print("预期拒绝的反例也算通过；这不是 GPU 正确性检查。")
    else:
        print(f"代码文件：{value['output']}")
        print(f"入口函数：{value['entry_point']}")
        print("来源：" + ("由执行计划生成" if value['generated'] else "已核验的固定源码"))
        print("已写出源码；尚未编译成 GPU 二进制，也未运行 GPU。")


def _compiler(args: argparse.Namespace) -> int:
    compiler = Compiler.load(args.project_root, args.revision)
    authored = (read_schedule(args.schedule)
                if args.compiler_command != "check-corpus" and args.schedule.suffix == ".py" else None)
    assessment = (compiler.assess(authored.document) if authored is not None else
                  compiler.assess_file(args.schedule) if args.compiler_command != "check-corpus" else None)
    finding_rows = []
    if assessment is not None:
        for finding in assessment.findings:
            row = asdict(finding)
            location = authored.location_for(finding.path) if authored is not None else None
            if location is not None:
                row["source"] = asdict(location)
            finding_rows.append(row)
    if (args.compiler_command == "assess"
            or (authored is not None and authored.locations and not assessment.lowering_eligible)):
        _emit_compiler(
            {
                "compiler_revision_id": assessment.compiler_revision_id,
                "compiler_revision_sha256": assessment.compiler_revision_sha256,
                "schedule_id": assessment.schedule_id,
                "schedule_sha256": assessment.schedule_sha256,
                "target": assessment.target,
                "lowering": (
                    {
                        "backend": assessment.route.backend.value,
                        "entry_point": assessment.route.entry_point,
                    }
                    if assessment.route is not None
                    else None
                ),
                "accepted": assessment.accepted,
                "lowering_eligible": assessment.lowering_eligible,
                "findings": finding_rows,
                "analysis": dict(assessment.analysis),
            },
            args,
        )
        return 0 if args.compiler_command == "assess" and assessment.accepted else 2
    if args.compiler_command == "check-corpus":
        report = compiler.check_corpus()
        _emit_compiler(
            {
                "corpus_id": report.corpus_id,
                "compiler_revision_id": report.compiler_revision_id,
                "compiler_revision_sha256": report.compiler_revision_sha256,
                "passed": report.passed,
                "case_count": report.case_count,
                "cases": [asdict(case) for case in report.cases],
            },
            args,
        )
        return 0 if report.passed else 2
    lowering = compiler.lower(assessment)
    output = args.output.absolute()
    with output.open("x", encoding="utf-8") as stream:
        stream.write(lowering.source)
    _emit_compiler(
        {
            "schedule_sha256": lowering.schedule_sha256,
            "source_sha256": lowering.source_sha256,
            "generated": lowering.generated,
            "entry_point": lowering.route.entry_point,
            "output": str(output),
        },
        args,
    )
    return 0


def _lab(args: argparse.Namespace) -> int:
    lab = Lab(args.project_root)
    if args.lab_command == "preflight":
        output_path = (
            admit_new_campaign_path(
                args.project_root,
                args.output,
                role="Campaign Lock output",
            )
            if args.output is not None
            else None
        )
        lock = lab.preflight(args.study)
        if output_path is not None:
            with output_path.open("x", encoding="utf-8") as stream:
                json.dump(lock.document, stream, sort_keys=True, ensure_ascii=False)
                stream.write("\n")
            output_path.chmod(0o644)
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
    execution_evidence_root = (
        admit_new_campaign_path(
            args.project_root,
            args.evidence_root,
            role="Campaign Evidence root",
        )
        if args.lab_command == "execute"
        else args.evidence_root
    )
    lock = CampaignLock.load(args.lock)
    if args.lab_command == "execute":
        campaign = (
            execute_portfolio_from_config(
                args.project_root,
                lock,
                args.runtime_config,
                execution_evidence_root,
            )
            if lock.study_kind == "portfolio"
            else execute_matched_from_config(
                args.project_root,
                lock,
                args.runtime_config,
                execution_evidence_root,
            )
        )
        report = (
            lab.audit_portfolio(campaign)
            if lock.study_kind == "portfolio"
            else lab.audit(campaign)
        )
        _emit(report)
        return 0 if report.archive_integrity_passed else 2
    campaign = lab.reference_campaign(lock, args.evidence_root)
    report = (
        lab.audit_portfolio(campaign)
        if lock.study_kind == "portfolio"
        else lab.audit(campaign)
    )
    _emit(report)
    return 0 if report.archive_integrity_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="open-cake-ir")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)

    compiler = commands.add_parser("compiler")
    compiler_commands = compiler.add_subparsers(dest="compiler_command", required=True)
    for name in ("assess", "lower"):
        command = compiler_commands.add_parser(name)
        command.add_argument(
            "--format", dest="output_format", choices=("json", "text"), default="json",
            help="JSON for tools (default), or a concise Chinese explanation",
        )
        command.add_argument("--revision", type=Path, required=True)
        command.add_argument("schedule", type=Path, help="Schedule JSON or restricted Python source")
        if name == "lower":
            command.add_argument("--output", type=Path, required=True)
    check = compiler_commands.add_parser("check-corpus")
    check.add_argument(
        "--format", dest="output_format", choices=("json", "text"), default="json",
        help="JSON for tools (default), or a concise Chinese explanation",
    )
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
    try:
        args.project_root = args.project_root.resolve(strict=True)
        if args.command == "compiler":
            return _compiler(args)
        return _lab(args)
    except FrontendError as error:
        if args.command != "compiler":
            raise
        if args.output_format == "text":
            print(f"命令未完成：{error}", file=sys.stderr)
        else:
            _emit({"accepted": False, "lowering_eligible": False,
                   "findings": [{"code": error.code, "path": "source", "message": str(error),
                                 "blocks_acceptance": True, "blocks_lowering": True,
                                 "source": asdict(error.location)}]})
        return 2
    except (CompilerError, OSError, json.JSONDecodeError) as error:
        if args.command != "compiler" or args.output_format != "text":
            raise
        print(f"命令未完成：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
