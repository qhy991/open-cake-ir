#!/usr/bin/env bash
# Release the Executor Revision that matches the current runtime sources.
#
#   release_executor_cycle.sh
#
# An Executor Revision binds every Lab, Evaluation, and Evidence source byte, so any edit
# to that closure invalidates the released descriptor. This settles and releases the id;
# templates resolve it during preflight, while frozen Study Contracts remain unchanged.
#
# The id is derived, not passed, on the same rule the Compiler cycle uses: a Revision that
# any frozen artifact names is history and immutable, and one nothing names is a working
# artifact that is replaced in place rather than bumped past on every edit. For an
# Executor "names it" is broader than a sealed evidence run, because a Study Contract
# pins an Executor id directly.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
EXECUTOR_RELEASE_TMP=$(mktemp -d)
export EXECUTOR_RELEASE_TMP
EXECUTOR_RELEASE_CANDIDATE=""
cleanup_executor_release() {
  if [ -n "$EXECUTOR_RELEASE_CANDIDATE" ]; then
    rm -f -- "$EXECUTOR_RELEASE_CANDIDATE"
  fi
  rm -r -- "$EXECUTOR_RELEASE_TMP"
}
trap cleanup_executor_release EXIT

python3 - <<'PY'
import hashlib, json, os, pathlib, re, subprocess

from tools.executor_revision_witnesses import plan_executor_revision_cycle

runtime = pathlib.Path("runtime/executors")
temporary = pathlib.Path(os.environ["EXECUTOR_RELEASE_TMP"])

inventory = json.loads(pathlib.Path("inventory/EXECUTOR_REVISIONS.json").read_text())
current = inventory.get("current")
if not isinstance(current, dict):
    raise SystemExit("Executor inventory has no current B200 authority")
current_id = current.get("executor_id")
current_path = pathlib.Path(str(current.get("path")))
if not current_path.is_file():
    raise SystemExit("current Executor descriptor is unavailable")
plan = plan_executor_revision_cycle(pathlib.Path("."), current_id)
if plan.family != "b200":
    raise SystemExit("B200 release cycle received another Executor family")
keep = plan.next_revision_id
print(f"--- releasing the B200 working Executor as {keep} ---")
for relative in plan.reclaimable_descriptors:
    print(f"    reclaimable after verified replacement: {pathlib.Path(relative).name}")

document = json.loads(current_path.read_text())
host = document["host_environment"]
# Resolve the profiler only during an explicit release. The released descriptor, not
# this discovery rule, is the authority used by every attribution assay.
candidates = list(pathlib.Path("/opt/nvidia/nsight-compute").glob(
    "*/target/linux-desktop-glibc_2_11_3-x64/ncu"
))
if not candidates:
    raise SystemExit("no x86_64 Nsight Compute executable is installed")


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
temporary.joinpath("proposal.json").write_text(json.dumps({
    "schema_version": 1,
    "executor_id": keep,
    "state": "draft",
    "sources": [],
    "host_environment": host,
}))
temporary.joinpath("keep").write_text(keep)
final = runtime / f"{keep}.json"
temporary.joinpath("initial-final-sha256").write_text(
    hashlib.sha256(final.read_bytes()).hexdigest() if final.exists() else "ABSENT"
)
temporary.joinpath("reclaimable").write_text(
    "".join(f"{value}\n" for value in plan.reclaimable_descriptors)
)
PY

KEEP=$(cat "$EXECUTOR_RELEASE_TMP/keep")
FINAL_EXECUTOR="runtime/executors/${KEEP}.json"
EXECUTOR_RELEASE_CANDIDATE="runtime/executors/.${KEEP}.candidate.$$.json"
REPLACE_ARGUMENTS=()
if [ -e "$FINAL_EXECUTOR" ]; then
  REPLACE_ARGUMENTS=(--replace-unwitnessed "$FINAL_EXECUTOR")
fi
python3 tools/release_executor.py --project-root . \
  --proposal "$EXECUTOR_RELEASE_TMP/proposal.json" \
  --output "$EXECUTOR_RELEASE_CANDIDATE" \
  "${REPLACE_ARGUMENTS[@]}" >/dev/null
python3 - "$EXECUTOR_RELEASE_CANDIDATE" "$KEEP" <<'PY'
import os, pathlib, sys

from open_cake_ir.lab import ExecutorRevision

candidate = pathlib.Path(sys.argv[1])
expected_id = sys.argv[2]
released = ExecutorRevision.load(pathlib.Path("."), candidate)
if released.executor_id != expected_id:
    raise SystemExit("replacement Executor identity differs")
released.admit_host()
released.admit_profiler()
pathlib.Path(os.environ["EXECUTOR_RELEASE_TMP"]).joinpath(
    "candidate-canonical-sha256"
).write_text(released.canonical_sha256)
PY
python3 - "$EXECUTOR_RELEASE_CANDIDATE" "$KEEP" "$FINAL_EXECUTOR" <<'PY'
import os, pathlib, sys

from open_cake_ir.lab import ExecutorRevision
from tools.executor_revision_witnesses import verify_executor_revision_commit

candidate = pathlib.Path(sys.argv[1])
executor_id = sys.argv[2]
final = pathlib.Path(sys.argv[3])
temporary = pathlib.Path(os.environ["EXECUTOR_RELEASE_TMP"])
initial = temporary.joinpath("initial-final-sha256").read_text()
verify_executor_revision_commit(
    pathlib.Path("."),
    executor_id,
    final,
    None if initial == "ABSENT" else initial,
)
committed = ExecutorRevision.load(pathlib.Path("."), candidate)
if (
    committed.executor_id != executor_id
    or committed.canonical_sha256
    != temporary.joinpath("candidate-canonical-sha256").read_text()
):
    raise SystemExit("B200 Executor candidate changed during release")
PY
mv -f -- "$EXECUTOR_RELEASE_CANDIDATE" "$FINAL_EXECUTOR"
EXECUTOR_RELEASE_CANDIDATE=""
while IFS= read -r RECLAIMABLE; do
  if [ -n "$RECLAIMABLE" ] && [ "$RECLAIMABLE" != "$FINAL_EXECUTOR" ]; then
    rm -f -- "$RECLAIMABLE"
  fi
done < "$EXECUTOR_RELEASE_TMP/reclaimable"

echo "--- update Executor inventory ---"
python3 - "$KEEP" <<'PY'
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
# Reclaimed ids were never history, so they leave no supersession behind.
inventory["superseded"] = [entry for entry in inventory["superseded"]
                           if entry["path"] != record["path"]
                           and pathlib.Path(entry["path"]).exists()]
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
