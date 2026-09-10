#!/usr/bin/env python3
"""Refuse to mint an Executor ordinal from an incomplete view of the released set.

The release cycle derives the next id from the largest one it can see released in its own
checkout. That is correct only when the checkout has seen every released descriptor, and
nothing established that. Two checkouts that had not exchanged descriptors both minted
v95 over different bytes; it was caught because someone was watching the number, and a
released identity is reserved forever, so a committed duplicate is not something a later
merge can resolve (F-2026-09-10-012).

This checks the two ways the view can be incomplete, and refuses rather than guessing.

* A descriptor exists here but is not committed. Until it is, no other checkout can learn
  of it, so this one is the only place it exists and any peer will reuse its ordinal.
* The tracked ledger is behind its remote, or the remote cannot be consulted at all. A
  peer may have published descriptors this checkout has never seen.

The second check needs an authority to compare against. Where the remote is unreachable --
an isolated GPU host, for instance -- there is no way to establish completeness from
inside, and that is reported as the refusal it is. `--acknowledge-isolated` proceeds
anyway and prints what is being accepted, so the risk is taken deliberately and appears in
the release log rather than being discovered later in the ledger.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

EXECUTORS = "runtime/executors"
_ORDINAL = re.compile(r"open-cake-ir-b200-v([1-9][0-9]*)(?:\+[0-9a-f]{64})?\.json$")


def _git(root: Path, arguments: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *arguments],
                          capture_output=True, text=True, timeout=timeout)


def _ordinal(name: str) -> int | None:
    match = _ORDINAL.search(name)
    return int(match.group(1)) if match else None


def uncommitted_descriptors(root: Path) -> list[str]:
    """Released descriptors that exist here and nowhere else yet."""
    result = _git(root, ["status", "--porcelain=v1", "--untracked-files=all", "--", EXECUTORS])
    if result.returncode:
        raise SystemExit(f"cannot inspect {EXECUTORS}: {result.stderr.strip()}")
    names = []
    for line in result.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if _ordinal(path) is not None:
            names.append(path)
    return sorted(names)


def unseen_on_remote(root: Path, remote: str, branch: str) -> tuple[list[str], str | None]:
    """Descriptors the remote publishes that this checkout does not have.

    Returns the missing names and, when the remote could not be consulted at all, the
    reason. An unreachable remote is not evidence of completeness.
    """
    fetched = _git(root, ["fetch", "--quiet", remote, branch], timeout=180)
    if fetched.returncode:
        return [], (fetched.stderr.strip().splitlines() or ["remote unreachable"])[-1]
    listing = _git(root, ["ls-tree", "--name-only", "FETCH_HEAD", f"{EXECUTORS}/"])
    if listing.returncode:
        return [], (listing.stderr.strip() or "cannot list the remote ledger")
    here = {p.name for p in (root / EXECUTORS).iterdir()} if (root / EXECUTORS).is_dir() else set()
    missing = [Path(name).name for name in listing.stdout.splitlines()
               if _ordinal(name) is not None and Path(name).name not in here]
    return sorted(missing), None


def highest_visible(root: Path) -> int:
    directory = root / EXECUTORS
    return max((_ordinal(p.name) or 0 for p in directory.iterdir()), default=0) if directory.is_dir() else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--acknowledge-isolated", metavar="REASON",
                        help="proceed when the remote cannot be consulted, recording why")
    arguments = parser.parse_args(argv)
    root = arguments.project_root.resolve(strict=True)

    pending = uncommitted_descriptors(root)
    if pending:
        print("refusing to mint: released descriptors here are not committed, so no other "
              "checkout can learn of them and a peer will reuse their ordinal:", file=sys.stderr)
        for name in pending:
            print(f"    {name}", file=sys.stderr)
        print("commit and publish them first.", file=sys.stderr)
        return 3

    missing, unreachable = unseen_on_remote(root, arguments.remote, arguments.branch)
    if missing:
        print(f"refusing to mint: {arguments.remote}/{arguments.branch} publishes released "
              "descriptors this checkout has never seen, so the next ordinal derived here "
              "would collide:", file=sys.stderr)
        for name in missing:
            print(f"    {name}", file=sys.stderr)
        print("synchronise first.", file=sys.stderr)
        return 3
    if unreachable is not None:
        if not arguments.acknowledge_isolated:
            print(f"refusing to mint: cannot consult {arguments.remote}/{arguments.branch} "
                  f"({unreachable}), so completeness of the released set is unestablished. "
                  "Re-run with --acknowledge-isolated REASON to accept that deliberately.",
                  file=sys.stderr)
            return 3
        print(f"    ledger completeness UNVERIFIED ({unreachable}); proceeding on stated "
              f"acknowledgement: {arguments.acknowledge_isolated}")

    print(f"    ledger check passed; highest released ordinal visible here is "
          f"v{highest_visible(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
