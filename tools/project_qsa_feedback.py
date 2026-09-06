#!/usr/bin/env python3
"""Project Compiler or GPU Infra QSA evidence into bounded provider feedback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import (  # noqa: E402
    Compiler,
    Schedule,
    Target,
    profile_envelope,
)
from open_cake_ir.compiler.compiled_resources import load_compiled_resources  # noqa: E402
from open_cake_ir.lab import (  # noqa: E402
    qsa_compiler_feedback,
    qsa_evaluation_feedback,
    qsa_next_turn_request,
)


def _plain(value: object) -> object:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _emit(value: object) -> None:
    print(json.dumps(_plain(value), indent=2, sort_keys=True, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compiler = commands.add_parser("compiler")
    compiler.add_argument("--revision", type=Path, required=True)
    compiler.add_argument("--compiled-report", type=Path)
    compiler.add_argument("schedule", type=Path)
    evaluation = commands.add_parser("evaluation")
    evaluation.add_argument("--arm", choices=("open_cake", "direct_cuda"), required=True)
    evaluation.add_argument("result", type=Path)
    turn = commands.add_parser("turn")
    turn.add_argument("--arm", choices=("open_cake", "direct_cuda"), required=True)
    turn.add_argument("--run-id", required=True)
    turn.add_argument("--turn", type=int, required=True)
    turn.add_argument("--cumulative-provider-tokens", type=int, required=True)
    turn.add_argument("--thread-id", required=True)
    turn.add_argument("--maximum-candidates-per-turn", type=int, default=1)
    turn.add_argument("result", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "compiler":
        compiler = Compiler.load(ROOT, arguments.revision.resolve(strict=True))
        schedule_path = arguments.schedule.resolve(strict=True)
        assessment = compiler.assess_file(schedule_path)
        schedule = Schedule.from_dict(json.loads(assessment.schedule_bytes))
        target = Target.load(ROOT / f"compiler/targets/{schedule.target}.json")
        lowering = compiler.lower(assessment) if assessment.lowering_eligible else None
        if lowering is not None:
            resources = None
            if arguments.compiled_report is not None:
                resources = load_compiled_resources(arguments.compiled_report).get(lowering.source_sha256)
                if resources is None:
                    raise ValueError("compiled report has no observation for this current QSA source")
            profile = compiler.profile(assessment, compiled_resources=resources).as_dict()
        else:
            profile = profile_envelope(schedule, target).as_dict()
        _emit(qsa_compiler_feedback(assessment, static_profile=profile))
        return 0
    result = json.loads(arguments.result.resolve(strict=True).read_text(encoding="utf-8"))
    if arguments.command == "evaluation":
        _emit(qsa_evaluation_feedback(result, arm=arguments.arm))
        return 0
    request = qsa_next_turn_request(
        run_id=arguments.run_id,
        arm=arguments.arm,
        turn=arguments.turn,
        cumulative_provider_tokens=arguments.cumulative_provider_tokens,
        thread_id=arguments.thread_id,
        maximum_candidates_per_turn=arguments.maximum_candidates_per_turn,
        result=result,
    )
    _emit(
        {
            "run_id": request.run_id,
            "arm": request.arm,
            "turn": request.turn,
            "cumulative_provider_tokens": request.cumulative_provider_tokens,
            "thread_id": request.thread_id,
            "maximum_candidates_per_turn": request.maximum_candidates_per_turn,
            "feedback": request.feedback,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
