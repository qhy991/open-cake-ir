# ADR 0030: Compiler release approval is external to the release cycle

Status: accepted for successor releases, 2026-08-25.

## Outcome and non-goals

The release cycle may derive the next Compiler id, archive witnessed history, and run
the full Corpus Gate. It must not create or modify `compiler/release-approval.json`.
A reviewer outside that automation writes the sole approval artifact after inspecting
the gate diff; rerunning the same cycle consumes it and releases only if it binds the
exact gate digest.

This does not add signatures, reviewer accounts, a second release implementation, or a
claim that the current v24 approval was independently reviewed. Identity assurance is a
repository process fact; the code enforces the smaller fact it can prove: preparing a
release cannot approve it.

## Authority and state transition

- `compiler/corpus-gate-report.json` is the review surface produced by the cycle.
- `compiler/release-approval.json` is the only approval authority and has an external
  writer. Its reviewer, basis, gate path, and gate digest are required by the existing
  release builder.
- `tools/release_compiler.py` remains the single release validator and builder. The
  cycle invokes it once into a temporary lock, verifies that lock, then atomically
  replaces the current lock.

```text
draft sources -> cycle prepares failing-capable Corpus Gate
             -> external reviewer writes exact approval
             -> same cycle validates gate + approval
             -> verified release lock
```

A missing, stale, or malformed approval produces no successor lock and leaves the
approval bytes untouched. A Corpus mismatch still stops before approval. Frozen prior
approvals remain immutable in their release archives.

## Acceptance evidence

An executable contract mutates one Revision-bound source in a temporary checkout. The
first cycle must prepare a new passing gate, refuse the stale approval, preserve both the
approval and prior lock bytes, and produce no successor. Only after the test acts as an
external reviewer and writes an approval for that exact digest may the second cycle
release and verify the successor.
