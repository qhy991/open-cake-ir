#!/usr/bin/env bash
# Release the Compiler Revision that matches the current sources.
#
#   release_compiler_cycle.sh "<approval basis>"
#
# Any edit to a Revision-bound source invalidates the released lock, which is the
# governance working as designed. This drives the documented pipeline end to end: settle
# the id, run the Corpus Gate, record the approval, release, verify, and re-stamp the
# Study Contracts bound to the Revision.
#
# The id is derived, not passed. A Revision that some sealed evidence run was produced
# under is history and its bytes are immutable, so an edit after one of those must bump.
# A Revision nothing was ever run under is a working artifact: bumping past it on every
# edit manufactures a version history that records no fact, so it is replaced in place.
# Expectations are never regenerated here — see tools/refresh_corpus_expectations.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

BASIS="${1:?usage: release_compiler_cycle.sh <approval-basis>}"

eval "$(python3 - <<'PY'
import json, pathlib, re

# A refused gate leaves no lock behind, so the draft is the fallback authority for the
# current id. Without it a single failed cycle would strand the repository with no way to
# name the Revision it was releasing.
locked = pathlib.Path("compiler/revision.lock.json")
source = locked if locked.exists() else pathlib.Path("compiler/revision.json")
current = json.loads(source.read_text())["revision_id"].removesuffix("-draft").rsplit("-", 1)[-1]

# A sealed evidence index is the only thing that makes a Revision id historical.
witnessed = set()
for path in pathlib.Path("evidence/releases").glob("*.json"):
    document = json.loads(path.read_text())
    identity = document.get("compiler_revision_id")
    if isinstance(identity, str):
        witnessed.add(identity.rsplit("-", 1)[-1])

ordinal = lambda value: int(m.group(1)) if (m := re.fullmatch(r"v(\d+)", value)) else 0
history = max((ordinal(value) for value in witnessed), default=0)

print("STALE=")
if current in witnessed:
    print(f"NEXT=v{history + 1}")
    print(f"ARCHIVE={current}")
else:
    # The working Revision's number carries no fact, so it sits directly above witnessed
    # history rather than climbing once per edit. Numbers stranded there by earlier
    # unwitnessed releases are reclaimed, so repeated edits keep releasing the same id.
    stale = sorted(p.name for p in pathlib.Path("compiler/releases").glob("v*")
                   if ordinal(p.name) > history)
    print(f"NEXT=v{history + 1}")
    print("ARCHIVE=")
    print(f"STALE='{' '.join(stale)}'")
PY
)"

if [ -n "$ARCHIVE" ]; then
  echo "--- ${ARCHIVE} is witnessed by sealed evidence; archiving and bumping to ${NEXT} ---"
  mkdir -p "compiler/releases/${ARCHIVE}"
  cp compiler/revision.lock.json compiler/corpus-gate-report.json \
     compiler/release-approval.json compiler/source_set.json "compiler/releases/${ARCHIVE}/"
else
  echo "--- no sealed evidence witnesses the working Revision; releasing it as ${NEXT} ---"
  for stale in $STALE; do
    echo "    reclaiming unwitnessed ${stale}"
    rm -rf "compiler/releases/${stale}"
  done
fi

python3 - "$NEXT" <<'PY'
import json, pathlib, sys
p = pathlib.Path("compiler/revision.json"); d = json.loads(p.read_text())
d["revision_id"] = f"open-cake-ir-sm100a-{sys.argv[1]}-draft"
p.write_text(json.dumps(d, indent=2) + "\n")
print(f"--- draft -> {d['revision_id']} ---")
PY

# Gate before retiring the released lock. A refused gate must leave the last released
# Revision standing, not strand the repository between two of them.
rm -f compiler/corpus-gate-report.json
python3 tools/release_compiler.py --project-root . \
  --proposal compiler/revision.json --source-set compiler/source_set.json \
  --output compiler/corpus-gate-report.json --prepare-gate
rm -f compiler/revision.lock.json

python3 - "$BASIS" <<'PY'
import hashlib, json, pathlib, sys

def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

gate = json.loads(pathlib.Path("compiler/corpus-gate-report.json").read_text())
assert gate["matched_case_count"] == gate["case_count"], "corpus gate did not match"
print(f"    gate {gate['compiler_revision_id']}  {gate['matched_case_count']}/{gate['case_count']}"
      f"  {len(gate['sources'])} sources")
pathlib.Path("compiler/release-approval.json").write_text(json.dumps({
    "schema_version": 1,
    "decision": "approved",
    "gate_report": {
        "path": "compiler/corpus-gate-report.json",
        "canonical_sha256": hashlib.sha256(canon(gate)).hexdigest(),
    },
    "reviewer": "repository_owner",
    "approval_basis": sys.argv[1],
}, indent=2) + "\n")
PY

echo "--- release ---"
for pass in write verify; do
  extra=""
  [ "$pass" = "verify" ] && extra="--verify"
  python3 tools/release_compiler.py --project-root . \
    --proposal compiler/revision.json --source-set compiler/source_set.json \
    --gate-report compiler/corpus-gate-report.json \
    --approval compiler/release-approval.json \
    --output compiler/revision.lock.json $extra
done

echo "--- restamp Study Contracts bound to the Revision ---"
python3 - <<'PY_INNER'
import hashlib, json, pathlib, subprocess

def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

lock = json.loads(pathlib.Path("compiler/revision.lock.json").read_text())
digest = hashlib.sha256(canon(lock)).hexdigest()
print(f"    {lock['revision_id']} -> {digest[:16]}")

# A Study Contract frozen against an earlier Revision is historical evidence and keeps
# its binding; the correct response to a Revision bump is a successor, never an edit.
# Only contracts introduced on this branch are re-stamped.
historical = set(subprocess.run(
    ["git", "ls-tree", "-r", "--name-only", "6e4d0f1", "contracts/studies"],
    capture_output=True, text=True, check=True).stdout.split())
assert historical, "could not read the pre-branch Study Contracts"

for path in sorted(pathlib.Path("contracts/studies").glob("*.json")):
    if str(path) in historical:
        continue
    document = json.loads(path.read_text())
    holders = [document, document.get("arms", {}).get("open_cake", {})]
    changed = False
    for holder in holders:
        pin = holder.get("compiler_revision") if isinstance(holder, dict) else None
        if isinstance(pin, dict) and pin.get("path") == "compiler/revision.lock.json":
            if pin["canonical_sha256"] != digest:
                pin["canonical_sha256"] = digest
                changed = True
    if changed:
        path.write_text(canon(document).decode() + "\n")
        print(f"    restamped {path.name}")
PY_INNER
echo "--- done ---"
