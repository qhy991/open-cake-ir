#!/usr/bin/env python3
"""Report, and on request restore, the filesystem custody committed Evidence needs.

`EvidenceStore` refuses a directory whose mode carries `0o022`, because a live Run's
tamper-evidence rests on nobody else being able to write the objects it is about to hash.
That is the right check for a Run. It is also a check git cannot satisfy: git records the
executable bit and nothing else, so every fresh checkout materialises the committed
archives under the caller's umask, and the store then refuses to read its own evidence.

The visible consequence is that two contract tests fail on a clean clone with
`evidence directory 'objects' is group/other writable`, and the re-audit gates in
`docs/ACCEPTANCE_GATES.md` cannot be executed in-repository at all.

This does not decide whether the store should audit an archive differently from a Run --
that is a Compiler/Executor-successor question recorded in `docs/AUDIT_FINDINGS_20260825.md`.
It makes the undocumented manual step explicit and runnable.

`--check` is the default and never writes. Restoring custody is `--apply`, because a tool
that repaired what it reports would report a state it had just produced.
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
    """Every path under `root` whose mode differs from the custody the store requires."""

    findings: list[tuple[Path, int, int]] = []
    for current, directories, files in os.walk(root):
        directories.sort()
        files.sort()
        for path, wanted in (
            [(Path(current), DIRECTORY_MODE)]
            + [(Path(current) / name, DIRECTORY_MODE) for name in directories]
            + [(Path(current) / name, FILE_MODE) for name in files]
        ):
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
        help="evidence root to inspect; repeatable. Defaults to evidence/.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="restore the required modes instead of only reporting them",
    )
    arguments = parser.parse_args()

    roots = [ROOT / item for item in (arguments.root or ["evidence"])]
    findings: list[tuple[Path, int, int]] = []
    for root in roots:
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
