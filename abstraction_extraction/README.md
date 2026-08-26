# Source-grounded abstraction extraction

This directory is the first tracer for stage 2 of the method described in
[CAKE v1 Appendix A](https://arxiv.org/html/2608.12629v1#A1): extract recurring
abstractions from concrete operator implementations after corpus collection.
The stage remains `in_progress`; this tracer records one recurring source
pattern and does not claim that extraction over the operator library is
complete.

## Contract

- `manifest.json` binds the exact `operator_library` input with a canonical
  SHA-256 digest and lists every admitted observation and candidate.
- `observations/*.json` bind reviewed prose observations to an operator-library
  occurrence, immutable Git revision and blob, upstream symbol, inclusive line
  span, and span hash. A `source_verified` catalog entry is not semantic
  evidence by itself; an observation is admitted only after the cited source
  span has been read and summarized.
- `candidates/*.json` normalize only structure repeated by at least two
  observations from distinct implementation IDs and distinct Git blobs. The
  threshold of two is a local repository policy, not a CAKE paper requirement.
- Upstream source code, generated kernels, binaries, benchmark outputs, and
  tuning tables are not vendored here.

The canonical operator-library digest is SHA-256 over UTF-8 JSON encoded with
sorted object keys and compact separators for this object:

```text
{
  "manifest": <parsed operator_library/manifest.json>,
  "sources": {<manifest source path>: <parsed source document>, ...},
  "operators": {<manifest operator path>: <parsed operator document>, ...}
}
```

A source-span hash is SHA-256 over the UTF-8 bytes of the cited inclusive line
range, retaining LF line terminators (`sha256-utf8-lf-inclusive-lines-v1`).
The source itself remains in the pinned upstream repository.

The default offline validator proves metadata closure, operator/source locator
consistency, and exact derivation of candidate breadth and counts. Because
upstream bytes are intentionally not vendored, it does not fetch a repository or
recompute a source-span hash. `semantics_reviewed` is therefore a reviewer
attestation, not an automated source-content proof. The authoring/review workflow
must independently recompute each Git blob and span hash from the pinned checkout;
the two observations in this tracer were checked that way before admission.
Free-text semantics likewise remain reviewer-attested: the validator enforces the
closed `source_structure_only` and `source_pattern_only` claim scopes, but does
not pretend that a lexical filter can prove the meaning of prose.

## Scope boundary

The CAKE study normalized collected implementations to CUDA before extracting
its abstractions. This Apple-focused tracer deliberately retains native Metal
structure (and may retain native Triton structure in later observations) so it
does not erase target-relevant facts. This is an explicit local deviation, not
a claim about the paper's unpublished corpus.

These files make source-pattern claims only. They do not admit an IR operation,
change a Target contract, establish Schedule expressibility, or report
correctness, calibration, or performance. The compiler-evolution outer loop may
inspect these pinned sources; the clean-start implementation lane may not use
low-level source as authoring input.

The next gates remain in paper order: hardware-informed design, P1-P8
principle-driven iteration, and only then port-driven expansion. The richer
historical survey in `docs/IR_SURVEY_SYNTHESIS.md` covers a different source set
and is not silently relabeled as evidence in this extraction.
