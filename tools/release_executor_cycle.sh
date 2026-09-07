#!/usr/bin/env bash
# Release the Executor Revision that matches the current runtime sources.
#
#   release_executor_cycle.sh [--host-environment /verified/host-environment.json]
#
# An Executor Revision binds every Lab, Evaluation, and Evidence source byte, so any edit
# to that closure invalidates the released descriptor. This settles and releases the id;
# templates resolve it during preflight, while frozen Study Contracts remain unchanged.
#
# Every released descriptor reserves its id, including releases used outside this
# checkout. The next id follows the largest released ordinal; no witness scan can
# authorize deleting or reusing a released descriptor.
set -euo pipefail
source "$(dirname "$0")/release_runtime.sh"
cd "$(dirname "$0")/.."
export PYTHONPATH=src
EXECUTOR_RELEASE_TMP=$(mktemp -d)
export EXECUTOR_RELEASE_TMP
trap 'rm -r -- "$EXECUTOR_RELEASE_TMP"' EXIT

"$OPEN_CAKE_PYTHON" - "$@" <<'PY'
import argparse, hashlib, json, os, pathlib, re, subprocess

from open_cake_ir.lab.executor import ExecutorRevision
from tools.release_executor import _released_executor_paths

parser = argparse.ArgumentParser(description="Release a new B200 Executor successor")
parser.add_argument(
    "--host-environment", type=pathlib.Path,
    help="host-environment JSON already verified against the executor host",
)
arguments = parser.parse_args()

temporary = pathlib.Path(os.environ["EXECUTOR_RELEASE_TMP"])


def ordinal(value: str) -> int:
    match = re.fullmatch(r"open-cake-ir-b200-v([1-9][0-9]*)", value)
    return int(match.group(1)) if match else 0


released = []
for path in _released_executor_paths(pathlib.Path.cwd()):
    document = json.loads(path.read_text())
    if document.get("state") == "released" and ordinal(document["executor_id"]):
        released.append(document)
released.sort(key=lambda document: ordinal(document["executor_id"]))
history = max((ordinal(document["executor_id"]) for document in released), default=0)
keep = f"open-cake-ir-b200-v{history + 1}"
print(f"--- released history ends at v{history}; releasing {keep} ---")
if arguments.host_environment is not None:
    host = json.loads(arguments.host_environment.read_text())
elif released:
    host = released[-1]["host_environment"]
else:
    raise SystemExit("no released Executor provides a host environment; pass --host-environment")
# Resolve the profiler only during an explicit release. The released descriptor, not
# this discovery rule, is the authority used by every attribution assay.
candidates = [] if arguments.host_environment is not None else list(pathlib.Path("/opt/nvidia/nsight-compute").glob(
    "*/target/linux-desktop-glibc_2_11_3-x64/ncu"
))
if arguments.host_environment is not None:
    print("    using supplied verified host environment")
elif not candidates:
    if os.environ.get("OPEN_CAKE_REUSE_VERIFIED_HOST") != "1":
        raise SystemExit(
            "no x86_64 Nsight Compute executable is installed; set "
            "OPEN_CAKE_REUSE_VERIFIED_HOST=1 only after verifying the pinned host "
            "environment against the live executor host"
        )
    print("    reusing operator-verified host environment")
else:
    def profiler_version(path: pathlib.Path) -> tuple[int, ...]:
        return tuple(int(value) for value in path.parents[2].name.split("."))

    ncu = max(candidates, key=profiler_version).resolve(strict=True)
    version_output = subprocess.run(
        [str(ncu), "--version"], check=True, capture_output=True, text=True
    ).stdout
    version_match = re.search(r"^Version ([^ ]+)", version_output, re.MULTILINE)
    if version_match is None:
        raise SystemExit("Nsight Compute version output differs")
    ncu_payload = ncu.read_bytes()
    host["nsight_compute"] = {
        "path": str(ncu),
        "version": version_match.group(1),
        "sha256": hashlib.sha256(ncu_payload).hexdigest(),
        "size_bytes": len(ncu_payload),
    }
if not isinstance(host, dict):
    raise ValueError("Executor host environment must be an object")
ExecutorRevision._validate_host_document(host)
temporary.joinpath("proposal.json").write_text(json.dumps({
    "schema_version": 1,
    "executor_id": keep,
    "state": "draft",
    "sources": [],
    "host_environment": host,
}))
temporary.joinpath("keep").write_text(keep)
PY

KEEP=$(cat "$EXECUTOR_RELEASE_TMP/keep")
"$OPEN_CAKE_PYTHON" tools/release_executor.py --project-root . \
  --proposal "$EXECUTOR_RELEASE_TMP/proposal.json" \
  --output "runtime/executors/${KEEP}.json" >/dev/null

echo "--- update Executor inventory ---"
"$OPEN_CAKE_PYTHON" - "$KEEP" <<'PY'
import hashlib, json, pathlib, sys

keep = sys.argv[1]
path = pathlib.Path(f"runtime/executors/{keep}.json")
raw = path.read_bytes()
document = json.loads(raw)


def canon(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


record = {
    "executor_id": keep,
    "path": path.as_posix(),
    "canonical_sha256": hashlib.sha256(canon(document)).hexdigest(),
    "descriptor_raw_sha256": hashlib.sha256(raw).hexdigest(),
    "source_count": len(document["sources"]),
}
print(f"    {keep} -> {record['canonical_sha256'][:16]}  {record['source_count']} sources")

inventory_path = pathlib.Path("inventory/EXECUTOR_REVISIONS.json")
inventory = json.loads(inventory_path.read_text())
previous = inventory.get("current")
inventory["current"] = record
if (
    isinstance(previous, dict)
    and previous.get("path") != record["path"]
    and pathlib.Path(str(previous.get("path"))).exists()
    and all(
        entry.get("path") != previous.get("path")
        for entry in inventory["superseded"]
    )
):
    inventory["superseded"].append(previous)
inventory["superseded"].sort(key=lambda entry: entry["executor_id"])
inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
print("    Study templates resolve current; frozen contracts were unchanged")
PY
echo "--- done ---"
