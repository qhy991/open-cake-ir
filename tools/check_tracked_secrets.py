#!/usr/bin/env python3
"""Scan tracked source bytes; report paths and line numbers, never matched content."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.evidence.secret_detection import contains_forbidden_secret  # noqa: E402

# These immutable historical detectors and the active detector contain bare
# policy markers. Only an exact marker literal is exempt, never a file or token.
_POLICY_SOURCES = {
    "src/open_cake_ir/evidence/secret_detection.py",
    *(f"evidence/executors/{revision}/src/open_cake_ir/evidence/store.py" for revision in (
        "open-cake-ir-b200-v1-5bdc9106", "open-cake-ir-b200-v2-4be390bf",
        "open-cake-ir-b200-v3-1c18cbfb", "open-cake-ir-b200-v4-203b2d8f",
        "open-cake-ir-b200-v5-7f437598",
    )),
}
_BARE_MARKERS = {
    b"BEGIN " + b"PRIVATE KEY", b"OPENAI_" + b"API_KEY=",
    b"CODEX_" + b"ACCESS_TOKEN=", b"INFINI_" + b"API_KEY=",
}


def _policy_literal(name: str, line: bytes) -> bool:
    if name not in _POLICY_SOURCES:
        return False
    try:
        value = ast.literal_eval(line.strip().removesuffix(b",").decode("utf-8"))
        return isinstance(value, bytes) and value in _BARE_MARKERS
    except (ValueError, SyntaxError, UnicodeError):
        return False


def finding_lines(name: str, payload: bytes) -> list[int]:
    lines = [b"" if _policy_literal(name, line) else line for line in payload.splitlines()]
    if not contains_forbidden_secret(b"\n".join(lines)):
        return []
    # Zero identifies a shape crossing line boundaries without exposing its bytes.
    return [index for index, line in enumerate(lines, 1) if contains_forbidden_secret(line)] or [0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    names = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().split("\0")
    findings = []
    for name in filter(None, names):
        path = root / name
        if path.is_symlink():
            findings.append((name, 0, "tracked symlink requires separate inspection"))
        elif path.is_file():
            findings.extend((name, line, "credential shape") for line in finding_lines(name, path.read_bytes()))
    for name, line, reason in findings:
        print(f"{name}:{line}: {reason}", file=sys.stderr)
    if not findings:
        print("Tracked-file credential-shape scan passed.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
