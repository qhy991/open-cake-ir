#!/usr/bin/env python3
"""Create one frozen Study successor bound to the current released authorities."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.lab import (  # noqa: E402
    ExecutorRevision,
    Lab,
    ProviderQualificationReceipt,
    matched_evidence_policy_v1,
    scientific_matched_analysis_plan_v2,
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return cast(dict[str, object], value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument(
        "--enable-attribution",
        action="store_true",
        help=(
            "profile every correctness-qualified search survivor and expose the "
            "selected survivor's checked summary"
        ),
    )
    parser.add_argument("--maximum-candidates-per-turn", type=int)
    parser.add_argument("--searches-per-turn", type=int)
    parser.add_argument("--search-materiality-ratio", type=float)
    parser.add_argument("--open-cake-schedule-skeleton", type=Path)
    parser.add_argument("--direct-cuda-candidate-skeleton", type=Path)
    arguments = parser.parse_args()

    root = arguments.project_root.resolve(strict=True)
    source = arguments.source.resolve(strict=True)
    output = arguments.output.absolute()
    study_root = (root / "contracts/studies").resolve(strict=True)
    if (
        source.parent != study_root
        or output.parent.resolve(strict=True) != study_root
        or output.exists()
        or output.is_symlink()
        or not arguments.study_id
    ):
        raise ValueError("Study successor custody or identity differs")
    document = _object(json.loads(source.read_text(encoding="utf-8")), "Study")
    if document.get("state") not in {"template", "frozen"}:
        raise ValueError("Study successor source state differs")
    skeletons = (
        arguments.open_cake_schedule_skeleton,
        arguments.direct_cuda_candidate_skeleton,
    )
    if (skeletons[0] is None) != (skeletons[1] is None):
        raise ValueError("both Authoring Environment skeletons must be replaced together")

    inventory = _object(
        json.loads(
            (root / "inventory/EXECUTOR_REVISIONS.json").read_text(encoding="utf-8")
        ),
        "Executor inventory",
    )
    current_executor = _object(inventory.get("current"), "current Executor")
    executor = ExecutorRevision.load(root, root / str(current_executor["path"]))
    execution = _object(document.get("execution"), "Study.execution")
    if (
        document.get("kind") != "portfolio"
        and execution.get("broker_execution_sha256") != "c" * 64
    ):
        raise ValueError(
            "live Study successors require freeze_live_matched_study.py so the "
            "broker execution authority is refreshed"
        )
    execution["executor_revision"] = dict(executor.reference)

    compiler = Compiler.load(root, root / "compiler/revision.lock.json")
    gate = compiler.check_corpus()
    if compiler.state != "released" or not gate.passed:
        raise ValueError("current Compiler is not a gated release")
    compiler_reference = {
        "revision_id": gate.compiler_revision_id,
        "path": "compiler/revision.lock.json",
        "canonical_sha256": gate.compiler_revision_sha256,
    }
    if document.get("kind") == "portfolio":
        if arguments.enable_attribution or any(
            value is not None
            for value in (
                arguments.maximum_candidates_per_turn,
                arguments.searches_per_turn,
                arguments.search_materiality_ratio,
                *skeletons,
            )
        ):
            raise ValueError("portfolio does not use matched-search authoring options")
        document["compiler_revision"] = compiler_reference
    else:
        # A matched successor closes the semantic event vocabulary even when its
        # source predates that policy. Frozen source Studies keep their bytes.
        if document.get("schema_version") == 1:
            document["evidence"] = dict(matched_evidence_policy_v1())
        arms = _object(document.get("arms"), "Study.arms")
        open_cake = _object(arms.get("open_cake"), "Study.arms.open_cake")
        direct_cuda = _object(arms.get("direct_cuda"), "Study.arms.direct_cuda")
        open_cake["compiler_revision"] = compiler_reference
        if skeletons[0] is not None and skeletons[1] is not None:
            schedule_path = skeletons[0].resolve(strict=True)
            candidate_path = skeletons[1].resolve(strict=True)
            try:
                schedule_relative = schedule_path.relative_to(root).as_posix()
                candidate_relative = candidate_path.relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError(
                    "Authoring Environment skeletons must be inside the project"
                ) from error
            schedule = _object(
                json.loads(schedule_path.read_text(encoding="utf-8")),
                "Open Cake Schedule skeleton",
            )
            open_cake["schedule_skeleton"] = {
                "path": schedule_relative,
                "canonical_sha256": sha256(
                    _canonical_json_bytes(schedule)
                ).hexdigest(),
            }
            direct_cuda["candidate_skeleton"] = {
                "path": candidate_relative,
                "sha256": sha256(candidate_path.read_bytes()).hexdigest(),
            }
        if document.get("claim_scope") == "scientific_matched_search":
            # A successor adopts the current canonical Analysis Plan. Frozen source
            # Studies keep their bytes and remain readable through the bounded legacy
            # adapter in Lab preflight/audit.
            document["analysis_plan"] = dict(
                scientific_matched_analysis_plan_v2()
            )
        if arguments.maximum_candidates_per_turn is not None:
            if arguments.maximum_candidates_per_turn <= 0:
                raise ValueError("maximum Candidates per Turn must be positive")
            budget = _object(document.get("budget"), "Study.budget")
            budget["maximum_candidates_per_turn"] = (
                arguments.maximum_candidates_per_turn
            )
            evaluation = _object(
                document.get("evaluation_protocol"), "Study.evaluation_protocol"
            )
            if arguments.searches_per_turn is not None:
                evaluation["searches_per_turn"] = arguments.searches_per_turn
            if arguments.search_materiality_ratio is not None:
                evaluation["search_materiality_ratio"] = (
                    arguments.search_materiality_ratio
                )
            if document.get("schema_version") == 1:
                optimization = document.get("claim_scope") == "artifact_optimization_only"
                receipt_path = (
                    "contracts/providers/fixture-provider-optimization-candidate-set-v1.json"
                    if optimization
                    else "contracts/providers/fixture-provider-candidate-set-v1.json"
                )
                receipt = ProviderQualificationReceipt.load(root / receipt_path)
                prompt_paths = {
                    "open_cake": (
                        "src/open_cake_ir/lab/prompts/"
                        + (
                            "open_cake_optimization_candidate_set_turn_v1.md"
                            if optimization
                            else "open_cake_candidate_set_turn_v1.md"
                        )
                    ),
                    "direct_cuda": (
                        "src/open_cake_ir/lab/prompts/"
                        + (
                            "direct_cuda_optimization_candidate_set_turn_v1.md"
                            if optimization
                            else "direct_cuda_candidate_set_turn_v1.md"
                        )
                    ),
                }
                for arm_name, raw_environment in arms.items():
                    environment = _object(raw_environment, f"Study.arms.{arm_name}")
                    provider = _object(
                        environment.get("provider"), f"Study.arms.{arm_name}.provider"
                    )
                    provider["revision"] = receipt.provider_revision
                    provider["qualification"] = {
                        "path": receipt_path,
                        "canonical_sha256": receipt.canonical_sha256,
                    }
                    prompt_path = prompt_paths[arm_name]
                    environment["prompt_template"] = {
                        "path": prompt_path,
                        "sha256": sha256((root / prompt_path).read_bytes()).hexdigest(),
                    }
        elif any(
            value is not None
            for value in (
                arguments.searches_per_turn,
                arguments.search_materiality_ratio,
            )
        ):
            raise ValueError("search options require a Candidate-set authoring bound")
        if arguments.enable_attribution:
            evaluation = _object(
                document.get("evaluation_protocol"), "Study.evaluation_protocol"
            )
            evaluation["attribution_evaluation"] = (
                "correctness_then_profile_each_search_survivor"
            )
            for arm in arms.values():
                environment = _object(arm, "Study.arm")
                feedback = environment.get("feedback")
                if (
                    not isinstance(feedback, list)
                    or feedback.count("profile") > 1
                    or ("profile" in feedback and feedback[-1] != "profile")
                ):
                    raise ValueError("Study attribution feedback authority differs")
                if "profile" not in feedback:
                    feedback.append("profile")
    document["state"] = "frozen"
    document["study_id"] = arguments.study_id

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=study_root,
        prefix=f".{output.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(_canonical_json_bytes(document) + b"\n")
    try:
        lock = Lab(root).preflight(temporary)
        if lock.study_id != arguments.study_id:
            raise ValueError("Study successor identity did not survive preflight")
        temporary.replace(output)
        output.chmod(0o644)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(output.resolve(strict=True).relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
