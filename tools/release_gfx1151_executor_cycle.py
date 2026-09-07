#!/usr/bin/env python3
"""Release a fresh gfx1151 Executor after source and live HIP admission.

The capture command owns host discovery. This cycle requires its explicit output,
derives an unused identity from permanent released descriptors, and installs only the
admitted candidate. A failed attempt preserves every existing release.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler.target import Target  # noqa: E402
from open_cake_ir.evaluation.triton_hip import admit_exact_hip  # noqa: E402
from open_cake_ir.lab import ExecutorRevision  # noqa: E402
from tools.release_executor import _released_executor_paths  # noqa: E402


def _released_gfx1151(root: Path) -> list[tuple[Path, str, int]]:
    released = []
    for path in _released_executor_paths(root):
        document = json.loads(path.read_text(encoding="utf-8"))
        identity = document.get("executor_id")
        if document.get("state") != "released" or not isinstance(identity, str):
            continue
        if not identity.startswith("open-cake-ir-gfx1151-"):
            continue
        match = re.fullmatch(r"open-cake-ir-gfx1151-v([1-9][0-9]*)", identity)
        if match is None:
            raise ValueError(f"gfx1151 Executor identity differs at {path}")
        released.append((path, identity, int(match.group(1))))
    return released


def release_gfx1151_executor(
    root: Path = ROOT, *, host_environment: Path
) -> dict[str, object]:
    root = root.resolve(strict=True)
    if host_environment.is_symlink():
        raise ValueError("Executor host capture custody differs")
    host = json.loads(host_environment.resolve(strict=True).read_text(encoding="utf-8"))
    ExecutorRevision._validate_host_document(host, schema_version=2)

    target = Target.load(root / "compiler/targets/gfx1151.json")
    if target.target_id != "gfx1151" or target.triton_target is None:
        raise ValueError("gfx1151 Executor requires the exact Compiler Target")
    requirements = {"target": target.target_id, "triton_target": dict(target.triton_target)}
    history = max((ordinal for _, _, ordinal in _released_gfx1151(root)), default=0)
    executor_id = f"open-cake-ir-gfx1151-v{history + 1}"
    runtime = root / "runtime/executors"
    final = runtime / f"{executor_id}.json"
    candidate = runtime / f".{executor_id}.candidate.{os.getpid()}.json"
    if final.exists() or final.is_symlink() or candidate.exists() or candidate.is_symlink():
        raise FileExistsError("refusing to overwrite an Executor release or candidate")

    proposal = {
        "schema_version": 2,
        "executor_id": executor_id,
        "state": "draft",
        "sources": [],
        "host_environment": host,
    }
    with tempfile.TemporaryDirectory(prefix="open-cake-gfx1151-release-") as directory:
        proposal_path = Path(directory) / "proposal.json"
        proposal_path.write_text(json.dumps(proposal, allow_nan=False), encoding="utf-8")
        try:
            subprocess.run(
                [
                    sys.executable, str(root / "tools/release_executor.py"),
                    "--project-root", str(root), "--proposal", str(proposal_path),
                    "--output", str(candidate),
                ],
                cwd=root, check=True, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            released = ExecutorRevision.load(root, candidate)
            if released.executor_id != executor_id:
                raise ValueError("gfx1151 Executor candidate identity differs")
            released.admit_hip_host()
            admit_exact_hip(requirements)
            # Host admission can take time. Refuse changed source/candidate bytes or a
            # newly occupied identity before the create-only installation boundary.
            rechecked = ExecutorRevision.load(root, candidate)
            if rechecked.canonical_sha256 != released.canonical_sha256:
                raise ValueError("gfx1151 Executor candidate changed during admission")
            if any(
                path != candidate and identity == executor_id
                for path, identity, _ in _released_gfx1151(root)
            ):
                raise FileExistsError("gfx1151 Executor identity was released during admission")
            os.link(candidate, final)
        finally:
            if candidate.exists() or candidate.is_symlink():
                candidate.unlink()

    return {
        "executor_id": executor_id,
        "path": final.relative_to(root).as_posix(),
        "source_count": len(released.document["sources"]),
        "host_admitted": True,
        "target_admitted": target.target_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--host-environment", type=Path, required=True,
                        help="explicit JSON from capture_executor_host.py --runtime-kind hip")
    arguments = parser.parse_args(argv)
    result = release_gfx1151_executor(
        arguments.project_root, host_environment=arguments.host_environment
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
