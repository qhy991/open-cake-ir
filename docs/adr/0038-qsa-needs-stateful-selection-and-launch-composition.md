# ADR 0038: QSA needs stateful selection and launch composition

## Status

Proposed.  It authorizes implementation and corpus-gate preparation, not a Compiler Revision,
provider run, GPU campaign, or scientific comparison.  Those remain behind the existing human
release gate and a separately frozen matched Study Contract.

## Irreducible goal

Evaluate one implementation-hidden, fixed-shape QSA prefill workload with matched Cake IR and
direct CUDA/PTX authoring environments.  The treatment is the authored representation and its
structured compiler feedback.  The task, oracle, target, timing, provider, scaffold, reasoning,
reference visibility, budget, and stopping rule must otherwise be identical.

The first target is `target_t32768` in
`contracts/workloads/qsa-prefill-t32768-v1.json`.  It begins after Q/K/V and index-Q/K projection
and uses task-declared geometry.  It is not a checkpoint reproduction or serving result.

## Why the current model is insufficient

The current Compiler can express the arithmetic pieces separately, but not their necessary state
and composition:

1. ReLU is absent from the canonical elementwise vocabulary.
2. `top_k` selects one resident tile; QSA must merge top-512 state across key-block tiles.
3. Sparse GQA needs numerically stable online-softmax state across selected-token tiles.
4. One `LaunchableCandidate` owns one kernel, while full QSA needs an indexer/selection program
   and a selected-attention program behind one logical ABI unless an independently validated fused
   Schedule is discovered.

Treating a hand-written Triton kernel as Cake output, precomputing indices outside the candidate,
or adding one opaque `qsa` operation would erase the representation treatment or move task work
outside the measured boundary.

## Decision

Add only the following general capabilities.

### Compiler-owned Schedule vocabulary

- Extend the existing canonical `elementwise` operation with `relu`; do not add another operation
  spelling.
- Extend the existing canonical `top_k` operation with loop-carried selection state.  Its contract
  fixes descending order, lowest-index tie handling, NaN policy, input/index dtypes, state shape,
  initialization, and finalization.  A tile-local `top_k` remains the same operation without carried
  state.
- Add one `online_softmax` operation for the stable recurrence over a logits tile and a value tile.
  It declares the reduction axis and FP32 running maximum, normalizer, and weighted-value
  accumulator.  The verifier owns shape, dtype, state-lifetime, and loop-placement legality; the
  backend owns the mechanical update formula.

Each addition requires parser/schema support, target admission, localized verifier findings,
work-accounting behavior, Triton lowering, one accepted corpus Schedule, and one failure Schedule
that proves the new boundary.  No QSA-named compiler branch is allowed.

### Evaluation-owned launch composition

Add one immutable launch plan for a logical candidate composed from independently compiled
Schedules.  The plan owns ordered kernel nodes, explicit intermediate buffers, data dependencies,
the public input/output ABI, and per-node launch manifests.  Compiler Revision identity remains on
each Schedule artifact; Evaluation owns allocation, ordered launch, whole-operator correctness,
timing, profiling, and lifecycle.  There is no mutable program database, implicit fallback, or
second experiment history.

The direct CUDA/PTX arm uses the same logical ABI and launch-plan evaluator.  It may author CUDA
C++ and inline PTX, including multiple kernels, but receives no Cake IR findings.

## Deliberate exclusions

- No model projection, query normalization, RoPE, KV cache, continuous batching, serving endpoint,
  or end-to-end token trajectory.
- No Qwen checkpoint configuration claim and no comparison to the reported 10.2x result.
- No dispatcher or multi-shape portfolio before the fixed-shape inner loop produces strong seeds.
- No inspection of the retained direct Triton seed by either clean-start authoring arm.
- No automatic release approval, expectation refresh, campaign retry, or replacement run.

## Failure semantics

- A parser, verifier, lowering, compile, sanitizer, or oracle rejection is candidate evidence.
- Missing analysis coverage or unsupported composition is a Compiler finding, not candidate
  correctness.
- SSH, broker, evaluator, timeout, custody, or missing-result failure is unknown.
- A valid but slower candidate is a negative result.  It is not hidden behind routing.
- A green corpus gate without independent release approval does not authorize provider or GPU
  experiment work.

## Acceptance evidence

Implementation may proceed only toward these gates, in order:

1. The frozen QSA Workload Contract and oracle-source binding validate.
2. Positive and failure corpus cases protect every new semantic boundary.  Expectation adoption is
   a separate reviewed act, never a regeneration performed to make the same gate pass; afterward
   the full corpus gate passes.
3. A distinct reviewer approves the exact gate and a successor Compiler Revision is released.
4. The QSA launch-plan evaluator proves both representations cross the same ABI, oracle,
   sanitizer, cold-L2 CUPTI timing, and profiler path on B200.
5. A matched Study freezes model, scaffold, reasoning effort, provider revision, run count, token
   budget, stopping rule, reference visibility, task statement, oracle, evaluator, and target shape
   before either arm starts.
6. Every reported candidate is correct at the target shape; conclusions retain provider-token and
   active-time accounting and distinguish lifecycle, validity, eligibility, and promotion.

This follows the compiler-evolution and matched clean-start contracts in Sections 3.2, 4, and 5 and
the IR principles in Appendix B of CAKE, arXiv:2608.12629v1.
