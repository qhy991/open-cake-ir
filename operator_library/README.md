# Operator source library

This directory is the first, bottom-up input to future Compiler evolution. It records
production operator implementations and the hardware mechanisms they exercise before any
new Schedule vocabulary is proposed.

It is deliberately **not** the Compiler `corpus/`, a Workload Contract, an operator-to-
backend registry, a correctness result, Calibration, or performance Evidence. Low-level
sources collected here are permitted only for known-kernel reproduction and design review;
the whole library is forbidden as clean-start authoring input.

## Three layers

1. `sources/` freezes the public identity and access policy of each external source.
2. `operators/` separates a mathematical `operator_id` from one concrete
   `implementation_id`, and records only source facts: location, input regime, target and
   mechanisms.
3. `assessments/` is intentionally absent in v1. A later assessment must bind one exact
   Compiler Revision before it may say `full`, `partial` or `not_expressible`.

The initial seed is a representative tracer corpus, not a claim to reproduce the paper's
unpublished full corpus or its reported count of roughly 28 families. It covers every
top-level family visible in the frozen FlashInfer CAKE tracker, a focused cross-section of
MLX's production Metal kernels, and four current ApxInf/M4 decode hotspots. No upstream
source code is vendored.

## Validate and summarize

```console
python tools/validate_operator_library.py
python tools/validate_operator_library.py --format markdown
python tools/validate_operator_library.py --format json
```

Validation is offline and fail-closed. It checks the closed schemas, path safety, ordering,
unique identities, source references, clean-start prohibition, canonical mechanism tags,
and the presence of exact Git blob identities for source-reviewed repository paths.
One reviewed blob may enumerate multiple named symbols; locator identity and ordering use
the `(kind, value, symbol)` tuple, while a null symbol is reserved for a file-level locator.

Promotion is a separate change: first create a complete Workload Contract and oracle for
one vertical slice. Only a recurring, reusable primitive gap justifies a Compiler proposal,
positive and falsifying Schedules, a full Corpus Gate, and external release approval.
