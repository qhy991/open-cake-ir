#!/usr/bin/env python3
"""Report an NCU-aligned static profile envelope without running a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler, EmpiricalCostModel  # noqa: E402
from open_cake_ir.compiler.toolchain import compile_triton, inspect_triton_resources  # noqa: E402
from open_cake_ir.compiler.performance.compiled_resources import load_compiled_resources  # noqa: E402


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _paths(arguments: argparse.Namespace) -> list[Path]:
    if arguments.manifest is not None:
        manifest = json.loads(arguments.manifest.resolve(strict=True).read_text())
        return [
            ROOT / case["schedule"]
            for case in manifest["cases"]
            if arguments.all or case["expected"].get("lowering_eligible") is True
        ]
    if not arguments.schedules:
        raise ValueError("provide at least one Schedule or --manifest")
    return [path.resolve(strict=True) for path in arguments.schedules]


def _metric(document: dict[str, object], name: str) -> dict[str, object]:
    return next((
        metric
        for metric in document["ncu_metrics"]
        if metric["metric"] == name
    ), {"value": None})


def _finding_lines(findings: list[dict[str, object]]) -> list[str]:
    lines = []
    for finding in findings:
        blocked = [
            stage for stage in ("acceptance", "lowering")
            if finding[f"blocks_{stage}"]
        ]
        status = "blocks " + ", ".join(blocked) if blocked else "nonblocking"
        lines.append(
            f"  {finding['code']} at {finding['path']} ({status}) "
            f"[{finding['category']}/{finding['severity']}]: {finding['message']}"
        )
    return lines


def _table(rows: list[dict[str, object]]) -> str:
    has_cost = any(row["profile"].get("empirical_cost") is not None for row in rows)
    header = (
        f"{'schedule':<34}{'regs':>8}{'CTA/SM<=':>10}{'warps%<=':>10}"
        f"{'IR read MiB<=':>14}{'barrier':>10}{'scoreboard':>12}{'source B':>12}  coverage"
    )
    if has_cost:
        header += "  est kernel us*"
    lines = [header, "-" * len(header)]
    for row in rows:
        profile = row["profile"]
        registers = _metric(profile, "launch__registers_per_thread")
        occupancy = profile["residency"] or {}
        active = _metric(
            profile, "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
        )
        barrier = _metric(
            profile,
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        )
        scoreboard = _metric(
            profile,
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        )
        source_bytes = profile["lowering"]["generated_source_bytes"]
        active_value = active["value"]
        active_text = "-" if active_value is None else f"{float(active_value):.1f}"
        payload = (profile["work"] or {}).get("scheduled_transfer_payload", {})
        read_bytes = payload.get("read_bytes_upper_bound")
        read_text = "-" if read_bytes is None else f"{read_bytes / (1 << 20):.2f}"
        lines.append(
            f"{str(row['schedule_id'])[:33]:<34}"
            f"{str(registers['value']):>8}"
            f"{str(occupancy.get('ctas_per_sm_upper_bound')):>10}"
            f"{active_text:>10}"
            f"{read_text:>14}"
            f"{str(barrier['value']):>10}"
            f"{str(scoreboard['value']):>12}"
            f"{str(source_bytes):>12}  "
            f"{len(profile['abstentions'])} abstention(s)"
        )
        if has_cost:
            estimate = (profile.get("empirical_cost") or {}).get("predicted_kernel_us")
            lines[-1] += "  " + ("-" if estimate is None else f"{estimate:.3f}")
        if occupancy:
            lines.append(
                f"  residency: binding={occupancy['binding_resource']}; "
                f"coverage={occupancy['coverage']}"
            )
        else:
            lines.append("  residency: unavailable; " + "; ".join(profile["abstentions"]))
        lines.extend(_finding_lines(row["findings"]))
        for name, metric in (("barrier", barrier), ("scoreboard", scoreboard)):
            if metric.get("reasons"):
                lines.append(
                    f"  {name} ({metric['estimate_kind']}): "
                    + "; ".join(metric["reasons"])
                )
        cost = profile.get("empirical_cost")
        if cost is not None:
            if cost["covered"]:
                low, high = cost["empirical_range_us"]
                detail = f"empirical range [{low:.3f}, {high:.3f}] us"
            else:
                detail = "uncovered: " + cost["reason"]
            lines.append(f"  conditional estimate {cost['model_id']}: {detail}")
    if has_cost:
        lines.append("* Conditional external empirical estimate; context and reported evidence are in JSON. Not a measured counter or qualified Compiler.rank.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        type=Path,
        default=ROOT / "compiler/revision.lock.json",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--cost-model", type=Path, help="explicit external empirical cost model; CPU-only, exact Revision/target/template coverage")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compile-to", type=Path, help="compile without a GPU and retain report.json plus artifacts in a new external directory")
    mode.add_argument("--compiled-report", type=Path, help="reuse an artifact-bound compiled report without CUDA tools or a GPU")
    parser.add_argument("--cuobjdump", type=Path, help="explicit NVIDIA binary inspector for --compile-to")
    parser.add_argument("schedules", nargs="*", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.manifest is not None and arguments.schedules:
        raise ValueError("--manifest and positional Schedules are exclusive")
    if bool(arguments.compile_to) != bool(arguments.cuobjdump):
        raise ValueError("--compile-to and --cuobjdump must be supplied together")
    output = None
    if arguments.compile_to is not None:
        output = arguments.compile_to.resolve()
        if output == ROOT or ROOT in output.parents:
            raise ValueError("compiled profiles and artifacts must stay outside the checkout")
        arguments.cuobjdump = arguments.cuobjdump.resolve(strict=True)
        output.mkdir(parents=True, exist_ok=False)
    observations = (
        load_compiled_resources(arguments.compiled_report)
        if arguments.compiled_report else {}
    )
    compiler = Compiler.load(ROOT, arguments.revision.resolve(strict=True))
    cost_model = EmpiricalCostModel.load(arguments.cost_model) if arguments.cost_model else None
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for path in _paths(arguments):
        assessment = compiler.assess_file(path)
        findings = [
            finding.to_dict() for finding in assessment.findings + assessment.guidance
        ]
        if not assessment.lowering_eligible:
            skipped.append(
                {
                    "schedule": _display_path(path),
                    "findings": findings,
                }
            )
            continue
        lowering = (
            compiler.lower(assessment)
            if output is not None or arguments.compiled_report is not None else None
        )
        resources = None
        uncovered = (
            lowering is not None
            and lowering.toolchain_requirements.get("compiler") != "triton"
        )
        if output is not None and not uncovered:
            compilation = compile_triton(lowering.source.encode(), lowering.toolchain_requirements)
            resources = inspect_triton_resources(compilation, arguments.cuobjdump)
            artifacts = output / f"{len(rows):04d}"
            artifacts.mkdir()
            (artifacts / "lowered.py").write_bytes(compilation.source)
            for role in ("ptx", "cubin"):
                (artifacts / f"kernel.{role}").write_bytes(compilation.artifacts[role])
        elif arguments.compiled_report is not None and not uncovered:
            resources = observations.get(lowering.source_sha256)
            if resources is None:
                raise ValueError(f"compiled report has no observation for current source {assessment.schedule_id!r}")
        profile = compiler.profile(assessment, compiled_resources=resources, cost_model=cost_model)
        if uncovered:
            profile = replace(profile, abstentions=(*profile.abstentions,
                "offline compiled resource collection does not cover this lowering backend"))
        rows.append(
            {
                "schedule": _display_path(path),
                "schedule_id": assessment.schedule_id,
                "findings": findings,
                "profile": profile.as_dict(),
            }
        )
    document = {"schema_version": 1, "rows": rows, "skipped": skipped}
    if output is not None:
        (output / "report.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    if arguments.json:
        print(json.dumps(document, indent=2, sort_keys=True))
    else:
        print(_table(rows))
        if skipped:
            print("\nskipped:")
            for row in skipped:
                print(f"  {row['schedule']}:")
                print("\n".join(_finding_lines(row["findings"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
