# ADR 0039: Logical register pressure is not a physical bound

## Status

Proposed. It authorizes implementation and Corpus expectation review, not a Compiler
Revision, Executor Revision, provider run, GPU campaign, or Candidate promotion. Those
remain behind their existing human gates.

## Irreducible goal

Keep pre-GPU analysis useful without rejecting a lowerable Schedule on a relationship the
Compiler cannot prove. Static hard gates may use exact declared quantities and Target
capacity; backend register allocation remains compiled-artifact evidence.

## Observation that changes the decision

The QSA tile search retained two correct B200 profiles under the same Workload Contract.
For tile128, logical register pressure was 72 per CTA thread and NCU measured 156 physical
registers per thread. For tile256, logical pressure was 140 while NCU measured 116. The
second observation refutes the claimed lower-bound direction. Both observations are in
`inventory/QSA_TILE_SEARCH_R8_20260828.json`; the raw route mirrors remain outside the
checkout.

The divergence is permitted by the lowering boundary: a backend may introduce temporaries
and allocation granularity, but it may also distribute logical values across lanes, alias
them more aggressively, recompute them, or realize them in another storage class. A live
logical Buffer extent therefore has no sound ordering against physical registers.

## Decision

- Retain the live logical register-Buffer calculation as
  `logical_register_pressure_per_thread`, an uncalibrated structural feature.
- Remove it from physical-register residency, `maxnreg` fit, impossible-residency, and
  declared-CTA hard gates.
- Continue deriving safe residency upper bounds from CTA threads and explicit shared and
  tensor-memory allocations. State that backend implicit storage can make those bounds
  loose.
- Report NCU physical-register metrics as unknown until a compiled artifact supplies them.
- Retain `residency.registers_per_thread` because it reaches the backend as `maxnreg` and
  role register redistribution consumes it.
- Remove `allow_spill`: no lowering or evaluator consumed it, and the only verifier rule
  that read it depended on the refuted lower bound.

No QSA-named Compiler branch, calibration-derived hard gate, or compatibility alias for
the false lower-bound name is added.

## Failure semantics

- Exact thread/shared/tensor capacity violations remain blocking hardware-conformance
  Findings.
- A large logical register-pressure proxy is non-blocking feedback.
- A backend compile failure or measured spill is candidate/toolchain evidence, not a fact
  reconstructed by the Schedule verifier.
- Missing compiled allocation evidence is unknown, not zero and not acceptance.

## Acceptance evidence

1. Focused analysis tests protect the separation between exact bounds and the proxy.
2. A former register-only rejection becomes lowerable; exact-resource rejection remains
   covered separately.
3. Corpus expectation changes are displayed and adopted only after repository-owner
   review.
4. The full Corpus Gate matches after that adoption.
5. A repository-owner approval binds the exact new Gate before release.
6. A successor Executor profiles the newly admitted QSA tile before any promotion.
