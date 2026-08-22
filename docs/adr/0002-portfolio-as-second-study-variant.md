# ADR 0002: Portfolio is the second closed Study variant

Status: accepted, 2026-08-22.

## Context

The final legacy Stage 6 run supplied a real second use case after fixed-shape matched search: one pre-held-out
KernelSeed was specialized for an anchor and two held-out Workload cases, then evaluated through three persistent
CUBINs and an exact-key dispatcher. r43 exposed a source-layer custody bug, r44 exposed a headline-only host tensor
gate, and r45 completed correctness while leaving held-out dispatcher timing unstable.

Keeping Portfolio deferred would omit stable remote functionality. Copying r43/r44/r45 runners would reintroduce
versioned paths, duplicated shape tables and success/failure archive schemas.

## Decision

Add `StudyContract.kind=portfolio` on the existing Lab, Evaluation and Evidence path. The Lab owns the frozen seed,
case roles, no-retune policy and execution sequence. The Compiler continues to accept complete Schedules only; the
Lab projects each Workload case into a Schedule and invokes the same `assess → lower` interface. A
`PortfolioArtifact` is the sole semantic-key-to-LaunchableCandidate map. Evaluation retains all 30 raw cohorts,
correctness receipts and route counters, then derives measurement quality. Evidence archives the artifacts and raw
receipt; offline audit replays medians, CV, route accounting and the terminal endpoint.

Serving is not added. The exact dispatcher rejects unsupported shapes before launch and has no fallback.

## Consequences

- Compiler and kernel evolve on different timescales: a Campaign freezes one Compiler Revision while kernels vary;
  Compiler evolution remains an outer Corpus-Gated process between Campaigns.
- Correctness, measurement quality, protocol adherence and claim support remain orthogonal. Unstable dispatcher
  timing cannot erase r45 correctness or create a performance claim.
- r43/r44 failures survive as regression evidence; their wrappers and special verifiers do not enter active source.
- The current claim scope is exactly three B200 shapes. Arbitrary-shape, causal, serving and paper-result claims are
  structurally false.
