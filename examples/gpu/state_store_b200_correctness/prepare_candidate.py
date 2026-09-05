#!/usr/bin/env python3
"""Create one immutable, preflighted state-store correctness bundle."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler


FIXED_SOURCE_COMMIT = "ee32b76870c4166c8143e8774c37c1212a0b6e1c"
POSITIVE = "state-store-b8-smoke.json"
NEGATIVE = {
    "state-store-b8-smoke-owner-drift.json": (
        "STATE_STORE_PROGRAM_OWNER",
        "access_maps[2].indices[0]",
    ),
    "state-store-b8-smoke-axis-drift.json": (
        "STATE_STORE_PROGRAM_AXIS_COVERAGE",
        "program_map.axes[1]",
    ),
}


def _load_frozen_compiler() -> Compiler:
    if Path(inspect.getfile(Compiler)).resolve() != ROOT / "src/open_cake_ir/compiler/core.py":
        raise RuntimeError("state-store v5 imported a Compiler from a different checkout")
    try:
        frozen_lock = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(ROOT), "show",
             f"{FIXED_SOURCE_COMMIT}:compiler/revision.lock.json"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "state-store v5 requires a Git checkout containing Compiler commit "
            f"{FIXED_SOURCE_COMMIT}; a source archive alone cannot verify its origin"
        ) from error
    lock_path = ROOT / "compiler/revision.lock.json"
    if lock_path.read_bytes() != frozen_lock:
        raise RuntimeError("current Compiler lock differs from the frozen state-store v5 source")
    # The existing released-Compiler loader verifies every bound source, Target,
    # Gate and approval. A later release needs an explicit task successor.
    return Compiler.load(ROOT, lock_path)


def _decisive(assessment) -> list[tuple[str, str]]:
    return [
        (finding.code, finding.path)
        for finding in assessment.findings
        if finding.blocks_acceptance or finding.blocks_lowering
    ]


def _static_source_checks(source: str) -> None:
    compile(source, "<released-state-store-lowering>", "exec")
    tree = ast.parse(source)
    wrappers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "cake_state_store_b8_smoke"
    ]
    if len(wrappers) != 1:
        raise RuntimeError("generated wrapper entry point differs")
    wrapper = wrappers[0]
    if [argument.arg for argument in wrapper.args.args] != ["state", "update", "out"]:
        raise RuntimeError("generated wrapper ABI differs")
    if not any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Name)
        and node.value.id == "out"
        for node in ast.walk(wrapper)
    ):
        raise RuntimeError("generated wrapper does not return its empty output tuple")
    if "tl.store(\n        state + batch * D_STATE_1 + state_d1_offsets," not in source:
        raise RuntimeError("caller-owned state pointer does not directly enter tl.store")
    if "torch.empty((8, 128)" in source:
        raise RuntimeError("generated lowering allocates a replacement [8,128] output")
    if "if out is None:\n        out = (\n        )" not in source:
        raise RuntimeError("generated wrapper does not materialize the empty output tuple")


def prepare(output_root: Path) -> dict[str, object]:
    output = output_root.expanduser().resolve()
    compiler = _load_frozen_compiler()
    schedules = ROOT / "corpus/schedules"
    accepted = compiler.assess_file(schedules / POSITIVE)
    if not accepted.accepted or not accepted.lowering_eligible or _decisive(accepted):
        raise RuntimeError("released positive schedule did not pass public assessment")
    lowering = compiler.lower(accepted)
    _static_source_checks(lowering.source)

    negative_results: dict[str, object] = {}
    for name, expected in NEGATIVE.items():
        assessment = compiler.assess_file(schedules / name)
        observed = _decisive(assessment)
        if assessment.accepted or assessment.lowering_eligible or observed != [expected]:
            raise RuntimeError(f"released negative schedule {name} did not fail closed")
        negative_results[name] = {
            "accepted": assessment.accepted,
            "lowering_eligible": assessment.lowering_eligible,
            "decisive_findings": observed,
        }

    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate"
    candidate.mkdir()
    (candidate / "kernel.py").write_text(lowering.source, encoding="utf-8")
    shutil.copyfile(HERE / "evaluate.py", candidate / "evaluate.py")
    shutil.copyfile(HERE / "task.json", output / "task.json")
    shutil.copyfile(HERE / "catalog.json", output / "catalog.json")
    provenance = {
        "schema": "open-cake.state-store-b200-candidate.v1",
        "compiler_source_commit": FIXED_SOURCE_COMMIT,
        "compiler_revision_id": lowering.compiler_revision_id,
        "schedule_id": accepted.schedule_id,
        "schedule_path": f"corpus/schedules/{POSITIVE}",
        "shape": [8, 128],
        "generated_via": "Compiler.load(revision.lock.json).assess_file/lower",
        "parent_id": "contiguous_apply2_add_fp32_u32_block256_v1",
        "parent_source_record": "data_movement_and_layout__memory_addressing__analysis__l000001_b200_v1__directderived_sol_ultra_v2",
        "parent_node_locator": "b200-local/directderived-contiguous-fp32-add-u32-v1-452f56c1e353",
        "parent_authority": "source-complete direct-derived contract without upstream repository/revision authority",
        "claim_boundary": "only the fixed [8,128] instance; no arbitrary-n or complete-parent coverage",
    }
    (candidate / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    preflight = {
        "schema": "open-cake.state-store-b200-static-preflight.v1",
        "compiler_revision_id": lowering.compiler_revision_id,
        "positive": {
            "accepted": accepted.accepted,
            "lowering_eligible": accepted.lowering_eligible,
            "decisive_findings": _decisive(accepted),
            "python_compiles": True,
            "state_pointer_direct_to_tl_store": True,
            "replacement_output_allocation": False,
            "wrapper_returns_empty_tuple": True,
        },
        "negative": negative_results,
    }
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
