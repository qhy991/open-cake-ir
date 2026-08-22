#!/usr/bin/env python3
"""Write or verify a content-bound Open Cake Compiler Revision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler.release import build_gate_report, build_release  # noqa: E402


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--source-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--prepare-gate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.prepare_gate:
        gate = build_gate_report(args.project_root, args.proposal, args.source_set)
        expected = _canonical_json_bytes(gate.document)
        verified = True
    else:
        if args.gate_report is None or args.approval is None:
            parser.error("release requires --gate-report and --approval")
        release = build_release(
            args.project_root,
            args.proposal,
            args.source_set,
            args.gate_report,
            args.approval,
        )
        expected = _canonical_json_bytes(release.document)
        verified = release.verify(args.project_root)
    if args.verify:
        if args.output.read_bytes() != expected or not verified:
            raise SystemExit("Compiler release artifact differs from its authority")
        return 0
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite Compiler Revision lock")
    args.output.write_bytes(expected)
    args.output.chmod(0o644)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
