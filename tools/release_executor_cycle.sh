#!/usr/bin/env bash
# Release the Executor Revision that matches the current runtime sources.
#
#   release_executor_cycle.sh
#
# An Executor Revision binds every Lab, Evaluation, and Evidence source byte, so any edit
# to that closure invalidates the released descriptor. This settles and releases the id;
# frozen Study Contracts consume it only through explicit successors.
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
trap 'rm -r -- "$EXECUTOR_RELEASE_TMP"' EXIT

python3 - <<'PY'
import json, os, pathlib, re

runtime = pathlib.Path("runtime/executors")
temporary = pathlib.Path(os.environ["EXECUTOR_RELEASE_TMP"])


def ordinal(value: str) -> int:
    match = re.search(r"-v(\d+)$", value)
    return int(match.group(1)) if match else 0


# An id is history if any frozen artifact names it: a sealed evidence run or any frozen
# Study Contract. Repository history is not the authority for whether a frozen file may
# be rewritten.
witnessed = {path.parent.parent.name.rsplit("-", 1)[0]
             for path in pathlib.Path("evidence/executors").glob("*/runtime/executor.json")}
for path in pathlib.Path("evidence").rglob("authority.json"):
    witnessed.update(re.findall(r"open-cake-ir-b200-v\d+", path.read_text()))
for path in pathlib.Path("contracts/studies").glob("*.json"):
    witnessed.update(re.findall(r"open-cake-ir-b200-v\d+", path.read_text()))

history = max((ordinal(value) for value in witnessed), default=0)
keep = f"open-cake-ir-b200-v{history + 1}"

available = sorted(
    runtime.glob("*.json"),
    key=lambda path: ordinal(json.loads(path.read_text())["executor_id"]),
)
stale = [
    path
    for path in available
    if ordinal(json.loads(path.read_text())["executor_id"]) > history
]
if not available:
    raise SystemExit("no Executor descriptor provides the host environment")
print(f"--- history ends at v{history}; releasing the working Executor as {keep} ---")
for path in stale:
    print(f"    reclaiming unwitnessed {path.name}")

document = json.loads((stale[-1] if stale else available[-1]).read_text())
temporary.joinpath("proposal.json").write_text(json.dumps({
    "schema_version": 1,
    "executor_id": keep,
    "state": "draft",
    "sources": [],
    "host_environment": document["host_environment"],
}))
for path in stale:
    path.unlink()
temporary.joinpath("keep").write_text(keep)
PY

KEEP=$(cat "$EXECUTOR_RELEASE_TMP/keep")
python3 tools/release_executor.py --project-root . \
  --proposal "$EXECUTOR_RELEASE_TMP/proposal.json" \
  --output "runtime/executors/${KEEP}.json" >/dev/null

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
print("    create successor Study Contracts explicitly; frozen contracts were unchanged")
PY
echo "--- done ---"
