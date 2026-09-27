#!/usr/bin/env python3
"""Compile the tested 7168-wide sibling-loop slice without a GPU or timing claim."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    output = args.output_root.resolve()
    if (not args.output_root.is_absolute() or output == root or root in output.parents
            or any((parent / ".git").exists() for parent in output.parents)):
        raise ValueError("compile output must be a new directory outside the checkout")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise ValueError("offline compilation requires a clean fixed source")
    if importlib.metadata.version("triton") != "3.7.1":
        raise ValueError("offline compilation requires Triton 3.7.1")
    import torch

    if torch.__version__ != "2.9.1+cpu":
        raise ValueError("offline compilation requires torch 2.9.1+cpu")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from open_cake_ir.compiler import Compiler
    from open_cake_ir.compiler import toolchain
    from tests.contracts.test_emit_triton import _two_pass_h7168_document

    for module in (sys.modules["open_cake_ir.compiler"], toolchain,
                   sys.modules["tests.contracts.test_emit_triton"]):
        if not Path(module.__file__).resolve(strict=True).is_relative_to(root):
            raise ValueError("offline compile imported a module outside the fixed source")
    compiler = Compiler.load(root, root / "compiler/revision.json")
    assessment = compiler.assess(_two_pass_h7168_document())
    if not assessment.accepted or not assessment.lowering_eligible:
        raise ValueError("the 7168 sibling-loop Schedule was not admitted")
    lowered = compiler.lower(assessment)
    result = toolchain.compile_triton(lowered.source.encode(), lowered.toolchain_requirements)
    if result.target != "sm_103a" or result.entry_point != "_cake_row_sum_contract_kernel":
        raise ValueError("offline compile target or entry point differs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "lowered.py").write_text(lowered.source)
    for role, payload in result.artifacts.items():
        (output / f"compiled.{role}").write_bytes(payload)
    report = {
        "kind": "offline_triton_sibling_compile",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "target": result.target,
        "entry_point": result.entry_point,
        "triton": result.compiler_version,
        "torch": torch.__version__,
        "artifact_roles": sorted(result.artifacts),
        "device_validation": False,
        "passed": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
