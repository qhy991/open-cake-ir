#!/usr/bin/env bash
# Release the next Compiler Revision after a compiler source edit.
#
#   release_compiler.sh <next-version> "<approval basis>"
#
# Any edit to a Revision-bound source invalidates the released lock, which is the
# governance working as designed. This drives the documented pipeline end to end:
# archive the current release, bump the draft, regenerate the Corpus Gate, record the
# approval, release, verify, and re-stamp the Study Contracts bound to the Revision.
# A released ID is never reused for different bytes.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

NEXT="${1:?usage: release_compiler.sh <next-version> <approval-basis>}"
BASIS="${2:?approval basis is required}"

CURRENT=$(python3 -c "import json;print(json.load(open('compiler/revision.lock.json'))['revision_id'].rsplit('-',1)[-1])")
if [ "$CURRENT" = "$NEXT" ]; then
  echo "refusing to reuse released id ${NEXT}: a released ID is never reused for different bytes" >&2
  exit 1
fi
if [ -d "compiler/releases/${NEXT}" ]; then
  echo "refusing to release ${NEXT}: compiler/releases/${NEXT} already exists" >&2
  exit 1
fi
echo "--- archive ${CURRENT} ---"
mkdir -p "compiler/releases/${CURRENT}"
cp compiler/revision.lock.json compiler/corpus-gate-report.json \
   compiler/release-approval.json compiler/source_set.json "compiler/releases/${CURRENT}/"

python3 - "$NEXT" <<'PY'
import json, pathlib, sys
p = pathlib.Path("compiler/revision.json"); d = json.loads(p.read_text())
d["revision_id"] = f"open-cake-ir-sm100a-{sys.argv[1]}-draft"
p.write_text(json.dumps(d, indent=2) + "\n")
print(f"--- draft -> {d['revision_id']} ---")
PY

rm -f compiler/revision.lock.json compiler/corpus-gate-report.json
python3 tools/release_compiler.py --project-root . \
  --proposal compiler/revision.json --source-set compiler/source_set.json \
  --output compiler/corpus-gate-report.json --prepare-gate

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
