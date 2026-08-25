# ADR 0027: Backend preconditions are Assessment Findings

Status: accepted, 2026-08-25.

## Outcome and non-goals

Every condition that a selected source emitter checks before it starts emitting is also
visible as a lowering-blocking `Finding` during `Compiler.assess`.  In particular, a
CuTe-DSL epilogue formula that the backend does not implement is refused before source
generation.  TinyGEMM2's retained closed asset separately verifies that its Schedule
declares the bias-add/BF16-round epilogue implemented by that asset.

This does not make profile conformance generic, generate the retained TinyGEMM2 CUDA
asset from its Schedule, add an emitter capability schema, or promise that arbitrary
well-typed operations lower on every backend.  Deep derivation checks that the common
Verifier already owns remain there.

## Authority and minimal primitives

`BackendPrecondition(code, path, message)` is the one shared value shape.  Each emitter
module owns one `preflight(schedule, target)` function because the backend owns what it
can lower.  Its constructor consumes that function directly, while `Compiler.assess`
projects the same results into non-acceptance-blocking, lowering-blocking Findings.  The
Compiler does not restate backend predicates.

The initial CuTe-DSL formula domain is the singleton
`centroid_sq_minus_two_dot`.  That closed set is read by preflight; the body is therefore
unreachable for `bias_add_bf16_round` instead of silently emitting centroid arithmetic.
The TinyGEMM2 asset profile conformance reads the other formula explicitly.  The two
backends implement different formulas and neither invents an interpretation for the
other.

## Dataflow and failure semantics

```text
typed Schedule + Target
        -> backend.preflight
        -> BackendPrecondition[]
        -> Assessment Findings (accepted, not lowering_eligible)
        -> lower is unreachable
```

Direct emitter callers run the identical preflight and receive `EmitError` for the first
failure.  Existing accepted schedules have an empty preflight result and retain their
source bytes.  Asset profiles do not call an operation emitter and remain governed by
their explicit conformance rule plus the closed semantic digest.

## Smallest complete slice and evidence

1. Flipping the full Flash-KMeans epilogue to `bias_add_bf16_round` produces a localized
   formula Finding and cannot lower.
2. Violating the Triton role/loop/store shape or the CuTe pipeline/MMA/epilogue/load
   shape produces a preflight Finding rather than a late `CompilerError`.
3. Flipping TinyGEMM2 to the centroid formula produces an explicit TinyGEMM epilogue
   conformance Finding.
4. All previously accepted Corpus cases retain their lowering source digests.
