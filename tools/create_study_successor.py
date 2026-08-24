#!/usr/bin/env python3
"""Create one frozen Study successor bound to the current released authorities."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.lab import ExecutorRevision, Lab  # noqa: E402


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
        help="declare correctness_then_profile and expose its checked summary",
    )
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
    if document.get("state") != "frozen":
        raise ValueError("Study successor source is not frozen")

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
        if arguments.enable_attribution:
            raise ValueError("portfolio does not use matched-search attribution")
        document["compiler_revision"] = compiler_reference
    else:
        arms = _object(document.get("arms"), "Study.arms")
        open_cake = _object(arms.get("open_cake"), "Study.arms.open_cake")
        open_cake["compiler_revision"] = compiler_reference
        if arguments.enable_attribution:
            evaluation = _object(
                document.get("evaluation_protocol"), "Study.evaluation_protocol"
            )
            evaluation["attribution_evaluation"] = "correctness_then_profile"
            for arm in arms.values():
                environment = _object(arm, "Study.arm")
                feedback = environment.get("feedback")
                if not isinstance(feedback, list) or "profile" in feedback:
                    raise ValueError("Study attribution feedback authority differs")
                feedback.append("profile")
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
    print(output.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
