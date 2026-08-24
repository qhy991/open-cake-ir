#!/usr/bin/env bash
# Release the Compiler Revision that matches the current sources.
#
#   release_compiler_cycle.sh "<approval basis>"
#
# Any edit to a Revision-bound source invalidates the released lock, which is the
# governance working as designed. This drives the documented pipeline end to end: settle
# the id, run the Corpus Gate, record the approval, release, and verify. Frozen Study
# Contracts are never re-stamped; a new release is consumed by a successor contract.
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

from tools.compiler_revision_witnesses import compiler_revision_witnesses

# A refused gate leaves no lock behind, so the draft is the fallback authority for the
# current id. Without it a single failed cycle would strand the repository with no way to
# name the Revision it was releasing.
locked = pathlib.Path("compiler/revision.lock.json")
source = locked if locked.exists() else pathlib.Path("compiler/revision.json")
current = json.loads(source.read_text())["revision_id"].removesuffix("-draft").rsplit("-", 1)[-1]

# Study Contracts, evidence, and inventory observations are all frozen witnesses. The
# helper is the sole owner of that discovery rule, so release and verification cannot
# silently disagree about what makes an id historical.
witnessed = {
    item.revision_id.rsplit("-", 1)[-1]
    for item in compiler_revision_witnesses(pathlib.Path("."))
}

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
  if [ -e "compiler/releases/${ARCHIVE}" ]; then
    echo "refusing to overwrite frozen compiler/releases/${ARCHIVE}" >&2
    exit 1
  fi
  mkdir "compiler/releases/${ARCHIVE}"
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

python3 - <<'PY'
import hashlib, json, pathlib

lock = json.loads(pathlib.Path("compiler/revision.lock.json").read_text())
canonical = json.dumps(lock, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode()
print(f"    {lock['revision_id']} -> {hashlib.sha256(canonical).hexdigest()[:16]}")
print("    create successor Study Contracts explicitly; frozen contracts were unchanged")
PY
echo "--- done ---"
