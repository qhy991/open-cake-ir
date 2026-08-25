#!/usr/bin/env python3
"""Check, and only on request restore, committed Evidence filesystem custody.

Git does not preserve group/other write bits. A checkout created under a permissive
umask can therefore be unreadable by `EvidenceStore`, even though its bytes are intact.
The default mode reports and exits nonzero; `--apply` is the explicit operational repair.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY_MODE = 0o750
FILE_MODE = 0o440


def _nonconforming(root: Path) -> list[tuple[Path, int, int]]:
    findings: list[tuple[Path, int, int]] = []
    for current, directories, files in os.walk(root):
        directories.sort()
        files.sort()
        candidates = (
            [(Path(current), DIRECTORY_MODE)]
            + [(Path(current) / name, DIRECTORY_MODE) for name in directories]
            + [(Path(current) / name, FILE_MODE) for name in files]
        )
        for path, wanted in candidates:
            if path.is_symlink():
                continue
            observed = stat.S_IMODE(path.lstat().st_mode)
            if observed != wanted:
                findings.append((path, observed, wanted))
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="repository-relative evidence root; repeatable, defaults to evidence/",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="restore required modes instead of only reporting",
    )
    arguments = parser.parse_args()

    findings: list[tuple[Path, int, int]] = []
    for relative in arguments.root or ["evidence"]:
        root = ROOT / relative
        if not root.is_dir():
            raise SystemExit(f"{root} is not a directory")
        findings.extend(_nonconforming(root))

    if not findings:
        print("evidence custody conforms")
        return 0

    for path, observed, wanted in findings[:20]:
        print(f"{observed:04o} -> {wanted:04o}  {path.relative_to(ROOT)}")
    if len(findings) > 20:
        print(f"... and {len(findings) - 20} more")

    if not arguments.apply:
        print(
            f"{len(findings)} paths do not carry Evidence custody; rerun with --apply",
            file=sys.stderr,
        )
        return 1

    for path, _observed, wanted in findings:
        os.chmod(path, wanted)
    print(f"restored custody on {len(findings)} paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
