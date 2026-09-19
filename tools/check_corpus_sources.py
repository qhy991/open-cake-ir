#!/usr/bin/env python3
"""Check reviewable lowering snapshots against the Compiler's declared Corpus.

Run --write only when adopting reviewed expected source changes. The existing Corpus
Gate must pass first; snapshots never adopt a changed manifest or hide a failed gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.corpus import assess_case


def check_sources(root: Path, *, write: bool = False) -> bool:
    compiler = Compiler.load(root, root / "compiler/revision.json")
    gate = compiler.check_corpus()
    if not gate.passed:
        raise ValueError("Corpus Gate must pass before checking or adopting source snapshots")
    manifest = json.loads((root / "corpus/manifest.json").read_text())
    directory = root / "corpus/expected"
    expected_paths = set()
    matched = True
    for case in manifest["cases"]:
        if not case["expected"]["lowering_eligible"]:
            continue
        name = case["case_id"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"unsafe Corpus case id {name!r}")
        path = directory / f"{name}.txt"
        expected_paths.add(path)
        source = compiler.lower(assess_case(compiler, root, case)).source.encode()
        if path.is_file() and path.read_bytes() == source:
            continue
        if write:
            directory.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source)
        else:
            print(f"{path.relative_to(root)}: missing or differs from lowering")
            matched = False
    for path in sorted(set(directory.glob("*.txt")) - expected_paths):
        print(f"{path.relative_to(root)}: no lowerable Corpus case owns this snapshot")
        matched = False
    print(f"{len(expected_paths)} lowerable Corpus source snapshots checked")
    return matched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    return 0 if check_sources(arguments.project_root.resolve(), write=arguments.write) else 1


if __name__ == "__main__":
    raise SystemExit(main())
