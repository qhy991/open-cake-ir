#!/usr/bin/env bash
# Release the Executor Revision that matches the current runtime sources.
#
#   release_executor_cycle.sh
#
# An Executor Revision binds every Lab, Evaluation, and Evidence source byte, so any edit
# to that closure invalidates the released descriptor. This settles the id, releases, and
# re-stamps the Study Contracts bound to it.
#
# The id is derived, not passed, on the same rule the Compiler cycle uses: a Revision that
# any frozen artifact names is history and immutable, and one nothing names is a working
# artifact that is replaced in place rather than bumped past on every edit. For an
# Executor "names it" is broader than a sealed evidence run, because a Study Contract
# frozen before this branch pins an Executor id directly.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
BASE="${BASE_COMMIT:-6e4d0f1}"

python3 - "$BASE" <<'PY'
import json, pathlib, re, subprocess, sys

base = sys.argv[1]
runtime = pathlib.Path("runtime/executors")


def ordinal(value: str) -> int:
    match = re.search(r"-v(\d+)$", value)
    return int(match.group(1)) if match else 0


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True,
                          check=True).stdout


# An id is history if any frozen artifact names it: a sealed evidence run, or a Study
# Contract that existed before this branch and pins it.
witnessed = {path.parent.parent.name.rsplit("-", 1)[0]
             for path in pathlib.Path("evidence/executors").glob("*/runtime/executor.json")}
for path in pathlib.Path("evidence").rglob("authority.json"):
    witnessed.update(re.findall(r"open-cake-ir-b200-v\d+", path.read_text()))
for relative in git("ls-tree", "-r", "--name-only", base, "contracts/studies").split():
    witnessed.update(re.findall(r"open-cake-ir-b200-v\d+", git("show", f"{base}:{relative}")))

history = max((ordinal(value) for value in witnessed), default=0)
keep = f"open-cake-ir-b200-v{history + 1}"

stale = sorted(path for path in runtime.glob("*.json")
               if ordinal(json.loads(path.read_text())["executor_id"]) > history)
if not stale:
    raise SystemExit(f"no working Executor Revision above witnessed history v{history}")
print(f"--- history ends at v{history}; releasing the working Executor as {keep} ---")
for path in stale:
    print(f"    reclaiming unwitnessed {path.name}")

document = json.loads(stale[-1].read_text())
pathlib.Path("/tmp/executor-proposal.json").write_text(json.dumps({
    "schema_version": 1,
    "executor_id": keep,
    "state": "draft",
    "sources": [],
    "host_environment": document["host_environment"],
}))
for path in stale:
    path.unlink()
pathlib.Path("/tmp/executor-keep").write_text(keep)
PY

KEEP=$(cat /tmp/executor-keep)
python3 tools/release_executor.py --project-root . \
  --proposal /tmp/executor-proposal.json \
  --output "runtime/executors/${KEEP}.json" >/dev/null
rm -f /tmp/executor-proposal.json /tmp/executor-keep

echo "--- restamp Study Contracts bound to the Revision ---"
python3 - "$KEEP" "$BASE" <<'PY'
import hashlib, json, pathlib, subprocess, sys

keep, base = sys.argv[1], sys.argv[2]
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
inventory["current"] = record
# Reclaimed ids were never history, so they leave no supersession behind.
inventory["superseded"] = [entry for entry in inventory["superseded"]
                           if entry["path"] != record["path"]
                           and pathlib.Path(entry["path"]).exists()]
inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

# A Study Contract frozen before this branch is historical evidence and keeps its
# binding; the correct response to a Revision change is a successor, never an edit.
historical = set(subprocess.run(
    ["git", "ls-tree", "-r", "--name-only", base, "contracts/studies"],
    capture_output=True, text=True, check=True).stdout.split())
assert historical, "could not read the pre-branch Study Contracts"
for contract in sorted(pathlib.Path("contracts/studies").glob("*.json")):
    if contract.as_posix() in historical:
        continue
    document = json.loads(contract.read_text())
    pin = document.get("execution", {}).get("executor_revision")
    if not isinstance(pin, dict) or pin == {key: record[key] for key in pin}:
        continue
    document["execution"]["executor_revision"] = {
        key: record[key] for key in ("canonical_sha256", "executor_id", "path")
    }
    contract.write_text(canon(document).decode() + "\n")
    print(f"    restamped {contract.name}")
PY
echo "--- done ---"
