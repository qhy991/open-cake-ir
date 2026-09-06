#!/usr/bin/env python3
"""Freeze one live matched Study from qualified authorities."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.lab import (  # noqa: E402
    ExecutorRevision,
    Lab,
    NvccToolchainBuilder,
    ProviderQualificationReceipt,
    broker_execution_sha256,
    matched_evidence_policy_v1,
    required_live_provider_qualification_scope,
    scientific_matched_analysis_plan_v2,
)

from open_cake_ir.lab.pairing import comparison_arm, triton_optimization_analysis_plan
from open_cake_ir.lab.triton_build import IsolatedTritonCompiler


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


def _project_relative(root: Path, path: Path, context: str) -> str:
    source = path.resolve(strict=True)
    if path.is_symlink() or not source.is_file():
        raise ValueError(f"{context} custody differs")
    try:
        return source.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{context} must be inside the project root") from error


def _refresh_raw_reference(root: Path, value: object, context: str) -> None:
    reference = _object(value, context)
    path = root / str(reference["path"])
    reference["sha256"] = sha256(path.resolve(strict=True).read_bytes()).hexdigest()


def _replace_artifact_feedback_budget(
    study: dict[str, object],
    *,
    provider_token_limit: int | None,
    maximum_turns: int | None,
) -> None:
    """Replace the live artifact budget with one bounded feedback horizon."""

    if (provider_token_limit is None) != (maximum_turns is None):
        raise ValueError(
            "provider-token-limit and maximum-turns must be declared together"
        )
    if provider_token_limit is None:
        return
    if study.get("claim_scope") != "artifact_optimization_only":
        raise ValueError("feedback budget replacement is artifact-optimization only")
    if (
        not isinstance(provider_token_limit, int)
        or isinstance(provider_token_limit, bool)
        or provider_token_limit <= 0
        or not isinstance(maximum_turns, int)
        or isinstance(maximum_turns, bool)
        or maximum_turns <= 0
    ):
        raise ValueError("feedback budget values must be positive integers")
    budget = _object(study.get("budget"), "study.budget")
    budget["limit"] = provider_token_limit
    budget["checkpoints"] = [provider_token_limit]
    budget["maximum_turns"] = maximum_turns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--qualification-anchor", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument(
        "--reasoning-effort",
        required=True,
        help="exact qualified provider reasoning effort to freeze for both arms",
    )
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--enable-attribution",
        action="store_true",
        help=(
            "profile every correctness-qualified search survivor and expose the "
            "selected survivor's checked summary"
        ),
    )
    parser.add_argument(
        "--provider-token-limit",
        type=int,
        help=(
            "replace an artifact-only live budget and use this value as its sole "
            "terminal checkpoint; requires --maximum-turns"
        ),
    )
    parser.add_argument(
        "--maximum-turns",
        type=int,
        help=(
            "hard Turn bound for an artifact-only live budget; requires "
            "--provider-token-limit"
        ),
    )
    arguments = parser.parse_args()

    root = arguments.project_root.resolve(strict=True)
    output = arguments.output.absolute()
    if output.exists() or output.is_symlink() or not arguments.study_id:
        raise ValueError("live Study output or identity differs")
    study = _object(
        json.loads(arguments.template.resolve(strict=True).read_text(encoding="utf-8")),
        "template",
    )
    if (
        study.get("kind") != "matched_search"
        or study.get("state") not in {"template", "frozen"}
    ):
        raise ValueError("live matched Study template policy differs")
    comparison = comparison_arm(_object(study["arms"], "study.arms"))
    paired_triton = comparison == "native_triton"
    if study.get("claim_scope") == "scientific_matched_search":
        expected_analysis = triton_optimization_analysis_plan() if paired_triton else dict(scientific_matched_analysis_plan_v2())
        if paired_triton and study.get("analysis_plan") != expected_analysis:
            raise ValueError("paired Triton scientific analysis differs")
        study["analysis_plan"] = expected_analysis
    if study.get("schema_version") == 1:
        study["evidence"] = dict(matched_evidence_policy_v1())
    _replace_artifact_feedback_budget(
        study,
        provider_token_limit=arguments.provider_token_limit,
        maximum_turns=arguments.maximum_turns,
    )
    qualification_path = arguments.qualification.resolve(strict=True)
    qualification_relative = _project_relative(
        root, qualification_path, "provider qualification"
    )
    anchor_path = arguments.qualification_anchor.resolve(strict=True)
    anchor_relative = _project_relative(root, anchor_path, "provider qualification anchor")
    qualification = ProviderQualificationReceipt.load(qualification_path)
    if (
        not qualification.qualified
        or qualification.scope
        != required_live_provider_qualification_scope(str(study["claim_scope"]))
    ):
        raise ValueError("live Study requires the Claim Scope's provider qualification")
    anchor = _object(json.loads(anchor_path.read_text(encoding="utf-8")), "anchor")
    if anchor.get("qualification_receipt_sha256") != qualification.canonical_sha256:
        raise ValueError("provider qualification anchor differs")

    config = _object(
        json.loads(arguments.runtime_config.resolve(strict=True).read_text(encoding="utf-8")),
        "runtime_config",
    )
    if set(config) != {"schema_version", "provider", "toolchain", "broker"} or config.get(
        "schema_version"
    ) != 1:
        raise ValueError("runtime configuration fields differ")
    provider_config = _object(config["provider"], "runtime_config.provider")
    toolchain_config = _object(config["toolchain"], "runtime_config.toolchain")
    broker_config = _object(config["broker"], "runtime_config.broker")
    if (
        set(provider_config) != {"executable", "workspace_root"}
        or set(toolchain_config) != ({"python", "bubblewrap", "runtime_roots", "triton_version", "timeout_seconds"}
                                      if paired_triton else {"nvcc", "cuobjdump"})
        or set(broker_config)
        != {
            "command",
            "cwd",
            "timeout_seconds",
            "service_user",
            "service_group",
        }
    ):
        raise ValueError("runtime configuration section fields differ")
    executable = Path(str(provider_config["executable"])).resolve(strict=True)
    executable_sha256 = sha256(executable.read_bytes()).hexdigest()
    if executable_sha256 != qualification.executable_sha256:
        raise ValueError("qualified provider executable differs")

    arms = _object(study["arms"], "study.arms")
    compiler = Compiler.load(root, root / "compiler/revision.lock.json")
    gate = compiler.check_corpus()
    if compiler.state != "released" or not gate.passed:
        raise ValueError("live Study requires the current gated Compiler release")
    _object(arms["open_cake"], "study.arms.open_cake")["compiler_revision"] = {
        "revision_id": gate.compiler_revision_id,
        "path": "compiler/revision.lock.json",
        "canonical_sha256": gate.compiler_revision_sha256,
    }
    for arm_name in arms:
        arm = _object(arms[arm_name], f"study.arms.{arm_name}")
        provider = _object(arm["provider"], f"study.arms.{arm_name}.provider")
        provider["revision"] = qualification.provider_revision
        provider["reasoning_effort"] = arguments.reasoning_effort
        provider["executable_sha256"] = executable_sha256
        provider["qualification"] = {
            "path": qualification_relative,
            "canonical_sha256": qualification.canonical_sha256,
        }
        provider["qualification_anchor"] = {
            "path": anchor_relative,
            "canonical_sha256": sha256(_canonical_json_bytes(anchor)).hexdigest(),
        }
        _refresh_raw_reference(root, provider["output_schema"], "provider.output_schema")
        _refresh_raw_reference(root, arm["scaffold"], f"{arm_name}.scaffold")
        if "prompt_template" in arm:
            _refresh_raw_reference(root, arm["prompt_template"], f"{arm_name}.prompt")
    open_arm = _object(arms["open_cake"], "study.arms.open_cake")
    skeleton = _object(open_arm["schedule_skeleton"], "open_cake.schedule_skeleton")
    skeleton_document = json.loads((root / str(skeleton["path"])).read_text(encoding="utf-8"))
    skeleton["canonical_sha256"] = sha256(
        _canonical_json_bytes(skeleton_document)
    ).hexdigest()
    direct_arm = _object(arms[comparison], f"study.arms.{comparison}")
    if paired_triton:
        isolated_toolchain = IsolatedTritonCompiler(**toolchain_config)
        identity = isolated_toolchain.canonical_sha256
        direct_arm["toolchain_sha256"] = identity
        open_arm["toolchain_sha256"] = identity
    else:
        _refresh_raw_reference(root, direct_arm["launch_contract"], "direct.launch_contract")
        _refresh_raw_reference(root, direct_arm["candidate_skeleton"], "direct.candidate_skeleton")
        direct_arm["toolchain_sha256"] = NvccToolchainBuilder(
            nvcc=str(toolchain_config["nvcc"]), cuobjdump=str(toolchain_config["cuobjdump"]),
        ).canonical_sha256

    command_value = broker_config["command"]
    command = (
        tuple(str(value) for value in command_value)
        if isinstance(command_value, list)
        else tuple(shlex.split(str(command_value)))
    )
    execution = _object(study["execution"], "study.execution")
    executor = ExecutorRevision.load(root, arguments.executor)
    if paired_triton:
        isolated_toolchain.check_executor(executor, author_workspace=str(provider_config["workspace_root"]))
    execution["executor_revision"] = dict(executor.reference)
    execution["broker_execution_sha256"] = broker_execution_sha256(
        command,
        cwd=Path(str(broker_config["cwd"])).resolve(strict=True),
        project_root=root,
        timeout_seconds=int(broker_config["timeout_seconds"]),
        service_user=str(broker_config["service_user"]),
        service_group=str(broker_config["service_group"]),
    )
    if arguments.enable_attribution:
        evaluation = _object(
            study.get("evaluation_protocol"), "study.evaluation_protocol"
        )
        if evaluation.get("attribution_evaluation") not in {
            None,
            "correctness_then_profile",
            "correctness_then_profile_each_search_survivor",
        }:
            raise ValueError("Study attribution Evaluation is already declared")
        evaluation["attribution_evaluation"] = (
            "correctness_then_profile_each_search_survivor"
        )
        for arm in arms.values():
            environment = _object(arm, "study.arm")
            feedback = environment.get("feedback")
            if (
                not isinstance(feedback, list)
                or feedback.count("profile") > 1
                or ("profile" in feedback and feedback[-1] != "profile")
            ):
                raise ValueError("Study attribution feedback authority differs")
            if "profile" not in feedback:
                feedback.append("profile")
    study["state"] = "frozen"
    study["study_id"] = arguments.study_id

    output.parent.resolve(strict=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(_canonical_json_bytes(study) + b"\n")
    try:
        lock = Lab(root).preflight(temporary)
        analysis = _object(study.get("analysis_plan"), "study.analysis_plan")
        if (
            lock.claim_scope != study.get("claim_scope")
            or lock.estimand != analysis.get("estimand")
        ):
            raise ValueError("frozen live Study data policy differs")
        temporary.replace(output)
        output.chmod(0o644)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "study_id": lock.study_id,
                "study_sha256": sha256(_canonical_json_bytes(study)).hexdigest(),
                "campaign_lock_sha256": lock.canonical_sha256,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
