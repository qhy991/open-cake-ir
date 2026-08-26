#!/usr/bin/env bash
# Release the Compiler Revision that matches the current sources.
#
#   release_compiler_cycle.sh
#
# Any edit to a Revision-bound source invalidates the released lock, which is the
# governance working as designed. This settles the id and runs the Corpus Gate, but it
# never writes the approval it consumes. A reviewer outside this automation must inspect
# the gate and write `compiler/release-approval.json`; rerunning the cycle then validates
# that exact artifact, releases, and verifies. Frozen Study Contracts are never
# re-stamped; a new release is consumed by a successor contract.
#
# The id is derived, not passed. A Revision that some sealed evidence run was produced
# under is history and its bytes are immutable, so an edit after one of those must bump.
# A Revision nothing was ever run under is a working artifact: bumping past it on every
# edit manufactures a version history that records no fact, so it is replaced in place.
# Expectations are never regenerated here — see tools/refresh_corpus_expectations.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
if [ "$#" -ne 0 ]; then
  echo "usage: release_compiler_cycle.sh" >&2
  exit 2
fi
COMPILER_RELEASE_TMP=$(mktemp -d compiler/.release-cycle.XXXXXX)
export COMPILER_RELEASE_TMP
trap 'rm -r -- "$COMPILER_RELEASE_TMP"' EXIT

eval "$(python3 - <<'PY'
import json, pathlib

from tools.compiler_revision_witnesses import plan_compiler_revision_cycle

# A legacy or interrupted cycle may leave only the draft. Use it as the current-id
# fallback when no released lock exists.
locked = pathlib.Path("compiler/revision.lock.json")
source = locked if locked.exists() else pathlib.Path("compiler/revision.json")
current = json.loads(source.read_text())["revision_id"].removesuffix("-draft")

# Study Contracts, evidence, and inventory observations are all frozen witnesses. The
# helper is the sole owner of that discovery rule, so release and verification cannot
# silently disagree about what makes an id historical.
plan = plan_compiler_revision_cycle(pathlib.Path("."), current)

print(f"NEXT={plan.next_label}")
print(f"ARCHIVE={plan.archive_label or ''}")
print(f"COLLISION={plan.collision_incident or ''}")
print(f"STALE='{' '.join(plan.stale_labels)}'")
PY
)"

if [ -n "$ARCHIVE" ]; then
  echo "--- ${ARCHIVE} is witnessed by sealed evidence; archiving and bumping to ${NEXT} ---"
  if [ -e "compiler/releases/${ARCHIVE}" ]; then
    python3 - "$ARCHIVE" "$COLLISION" <<'PY'
import hashlib, json, pathlib, sys

from tools.compiler_revision_witnesses import _registered_archive_collision

archive = pathlib.Path("compiler/releases") / sys.argv[1]
collision = sys.argv[2]
lock_path = archive / "revision.lock.json"
current_path = pathlib.Path("compiler/revision.lock.json")
lock = json.loads(lock_path.read_text())
current = json.loads(current_path.read_text())
if lock_path.read_bytes() != current_path.read_bytes():
    registered = _registered_archive_collision(
        pathlib.Path("."),
        revision_id=current.get("revision_id"),
        current=current,
        archived=lock,
    )
    if not collision or registered != collision:
        raise SystemExit(f"refusing non-identical frozen {lock_path}")


def canonical(path):
    return hashlib.sha256(json.dumps(
        json.loads(path.read_text()), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode()).hexdigest()


for field, name in (("corpus_gate", "corpus-gate-report.json"),
                    ("release_approval", "release-approval.json")):
    path = archive / name
    if canonical(path) != lock[field]["canonical_sha256"]:
        raise SystemExit(f"refusing non-identical frozen {path}")
source_paths = json.loads((archive / "source_set.json").read_text())["paths"]
if source_paths != [source["path"] for source in lock["sources"]]:
    raise SystemExit(f"refusing non-identical frozen {archive / 'source_set.json'}")
PY
    if [ -n "$COLLISION" ]; then
      echo "    registered identity collision preserved by ${COLLISION}; existing archive unchanged"
    else
      echo "    complete archive already exists; resuming the interrupted release"
    fi
  else
    mkdir "compiler/releases/${ARCHIVE}"
    cp compiler/revision.lock.json compiler/corpus-gate-report.json \
       compiler/release-approval.json "compiler/releases/${ARCHIVE}/"
    python3 - "$ARCHIVE" <<'PY'
import json, pathlib, sys

archive = pathlib.Path("compiler/releases") / sys.argv[1]
lock = json.loads(pathlib.Path("compiler/revision.lock.json").read_text())
archive.joinpath("source_set.json").write_text(json.dumps({
    "paths": [source["path"] for source in lock["sources"]],
    "schema_version": 1,
}, indent=2) + "\n")
PY
  fi
else
  echo "--- no sealed evidence witnesses the working Revision; releasing it as ${NEXT} ---"
  for stale in $STALE; do
    echo "    reclaiming unwitnessed ${stale}"
    rm -rf "compiler/releases/${stale}"
  done
fi

python3 - "$NEXT" <<'PY'
import json, pathlib, sys

from tools.compiler_revision_witnesses import generic_compiler_revision_id

p = pathlib.Path("compiler/revision.json"); d = json.loads(p.read_text())
d["revision_id"] = generic_compiler_revision_id(sys.argv[1], draft=True)
p.write_text(json.dumps(d, indent=2) + "\n")
print(f"--- draft -> {d['revision_id']} ---")
PY

# Preparing a Gate cannot manufacture a successor lock. The old lock bytes stay in place
# until a separately written approval validates and a verified replacement is ready; a
# source edit may of course already make that old lock unloadable.
rm -f compiler/corpus-gate-report.json
python3 tools/release_compiler.py --project-root . \
  --proposal compiler/revision.json --source-set compiler/source_set.json \
  --output compiler/corpus-gate-report.json --prepare-gate
python3 - <<'PY'
import json, pathlib
gate = json.loads(pathlib.Path("compiler/corpus-gate-report.json").read_text())
assert gate["matched_case_count"] == gate["case_count"], "corpus gate did not match"
print(f"    gate {gate['compiler_revision_id']}  {gate['matched_case_count']}/{gate['case_count']}"
      f"  {len(gate['sources'])} sources")
PY

if [ ! -f compiler/release-approval.json ]; then
  echo "--- external approval required: review the Gate and write compiler/release-approval.json ---" >&2
  exit 3
fi

echo "--- validate external approval and build release ---"
if ! python3 tools/release_compiler.py --project-root . \
    --proposal compiler/revision.json --source-set compiler/source_set.json \
    --gate-report compiler/corpus-gate-report.json \
    --approval compiler/release-approval.json \
    --output "$COMPILER_RELEASE_TMP/revision.lock.json"; then
  echo "--- external approval required: it must bind this exact Gate digest ---" >&2
  exit 3
fi
python3 tools/release_compiler.py --project-root . \
    --proposal compiler/revision.json --source-set compiler/source_set.json \
    --gate-report compiler/corpus-gate-report.json \
    --approval compiler/release-approval.json \
    --output "$COMPILER_RELEASE_TMP/revision.lock.json" --verify
mv -f "$COMPILER_RELEASE_TMP/revision.lock.json" compiler/revision.lock.json

python3 - <<'PY'
import hashlib, json, pathlib

lock = json.loads(pathlib.Path("compiler/revision.lock.json").read_text())
canonical = json.dumps(lock, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode()
print(f"    {lock['revision_id']} -> {hashlib.sha256(canonical).hexdigest()[:16]}")
print("    Study templates resolve current; frozen contracts were unchanged")
PY
echo "--- done ---"
