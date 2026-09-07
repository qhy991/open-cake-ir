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
# The id is derived, not passed. A released lock reserves its identity even when its
# consumers live outside this checkout. A changed release gets a successor; repeated
# preparation keeps that draft id, and verifying an unchanged release does not bump it.
# Expectations are never regenerated here — see tools/refresh_corpus_expectations.py.
set -euo pipefail
source "$(dirname "$0")/release_runtime.sh"
cd "$(dirname "$0")/.."
export PYTHONPATH=src
if [ "$#" -ne 0 ]; then
  echo "usage: release_compiler_cycle.sh" >&2
  exit 2
fi
if [ -f compiler/revision.lock.json ] && "$OPEN_CAKE_PYTHON" tools/release_compiler.py --project-root . \
    --proposal compiler/revision.json --source-set compiler/source_set.json \
    --gate-report compiler/corpus-gate-report.json --approval compiler/release-approval.json \
    --output compiler/revision.lock.json --verify >/dev/null 2>&1; then
  echo "--- released Compiler already matches; no successor is needed ---"
  exit 0
fi
COMPILER_RELEASE_TMP=$(mktemp -d compiler/.release-cycle.XXXXXX)
export COMPILER_RELEASE_TMP
trap 'rm -r -- "$COMPILER_RELEASE_TMP"' EXIT

REVISION_ASSIGNMENTS=$("$OPEN_CAKE_PYTHON" - <<'PY'
import json, pathlib, re

from tools.compiler_revision_witnesses import compiler_revision_witnesses

# A legacy or interrupted cycle may leave only the draft. Use it as the current-id
# fallback when no released lock exists.
locked = pathlib.Path("compiler/revision.lock.json")
source = locked if locked.exists() else pathlib.Path("compiler/revision.json")
document = json.loads(source.read_text())
current = document["revision_id"].removesuffix("-draft").rsplit("-", 1)[-1]

# Frozen references also reserve historical ids whose locks are no longer available.
# Missing local references cannot authorize reclaiming the current released lock.
witnessed = {
    item.revision_id.rsplit("-", 1)[-1]
    for item in compiler_revision_witnesses(pathlib.Path("."))
}

ordinal = lambda value: int(m.group(1)) if (m := re.fullmatch(r"v(\d+)", value)) else 0
history = max((ordinal(value) for value in witnessed), default=0)

if locked.exists():
    if document.get("state") != "released" or not ordinal(current):
        raise SystemExit("the current Compiler lock is not a valid released identity")
    print(f"NEXT=v{max(history, ordinal(current)) + 1}")
    print(f"ARCHIVE={current}")
else:
    print(f"NEXT=v{history + 1}")
    print("ARCHIVE=")
PY
)
eval "$REVISION_ASSIGNMENTS"

if [ -n "$ARCHIVE" ]; then
  echo "--- preserve released ${ARCHIVE}; prepare successor ${NEXT} ---"
  if [ -e "compiler/releases/${ARCHIVE}" ]; then
    "$OPEN_CAKE_PYTHON" - "$ARCHIVE" <<'PY'
import hashlib, json, pathlib, sys

archive = pathlib.Path("compiler/releases") / sys.argv[1]
lock_path = archive / "revision.lock.json"
if lock_path.read_bytes() != pathlib.Path("compiler/revision.lock.json").read_bytes():
    raise SystemExit(f"refusing non-identical frozen {lock_path}")
lock = json.loads(lock_path.read_text())


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
    echo "    complete archive already exists; resuming the interrupted release"
  else
    mkdir "compiler/releases/${ARCHIVE}"
    cp compiler/revision.lock.json compiler/corpus-gate-report.json \
       compiler/release-approval.json "compiler/releases/${ARCHIVE}/"
    "$OPEN_CAKE_PYTHON" - "$ARCHIVE" <<'PY'
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
  echo "--- no current released lock; prepare working draft ${NEXT} ---"
fi

"$OPEN_CAKE_PYTHON" - "$NEXT" <<'PY'
import json, pathlib, sys
p = pathlib.Path("compiler/revision.json"); d = json.loads(p.read_text())
d["revision_id"] = f"open-cake-ir-sm100a-{sys.argv[1]}-draft"
p.write_text(json.dumps(d, indent=2) + "\n")
print(f"--- draft -> {d['revision_id']} ---")
PY

# Preparing a Gate cannot manufacture a successor lock. The old lock bytes stay in place
# until a separately written approval validates and a verified replacement is ready; a
# source edit may of course already make that old lock unloadable.
rm -f compiler/corpus-gate-report.json
"$OPEN_CAKE_PYTHON" tools/release_compiler.py --project-root . \
  --proposal compiler/revision.json --source-set compiler/source_set.json \
  --output compiler/corpus-gate-report.json --prepare-gate
"$OPEN_CAKE_PYTHON" - <<'PY'
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
if ! "$OPEN_CAKE_PYTHON" tools/release_compiler.py --project-root . \
    --proposal compiler/revision.json --source-set compiler/source_set.json \
    --gate-report compiler/corpus-gate-report.json \
    --approval compiler/release-approval.json \
    --output "$COMPILER_RELEASE_TMP/revision.lock.json"; then
  echo "--- external approval required: it must bind this exact Gate digest ---" >&2
  exit 3
fi
"$OPEN_CAKE_PYTHON" tools/release_compiler.py --project-root . \
    --proposal compiler/revision.json --source-set compiler/source_set.json \
    --gate-report compiler/corpus-gate-report.json \
    --approval compiler/release-approval.json \
    --output "$COMPILER_RELEASE_TMP/revision.lock.json" --verify
mv -f "$COMPILER_RELEASE_TMP/revision.lock.json" compiler/revision.lock.json

"$OPEN_CAKE_PYTHON" - <<'PY'
import hashlib, json, pathlib

lock = json.loads(pathlib.Path("compiler/revision.lock.json").read_text())
canonical = json.dumps(lock, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode()
print(f"    {lock['revision_id']} -> {hashlib.sha256(canonical).hexdigest()[:16]}")
print("    Study templates resolve current; frozen contracts were unchanged")
PY
echo "--- done ---"
