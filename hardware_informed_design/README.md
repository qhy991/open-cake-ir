# Hardware-informed design

This directory is the first tracer for stage 3 of the method described in
[CAKE v1 Appendix A](https://arxiv.org/html/2608.12629v1#A1): use target
hardware knowledge to shape an abstraction extracted from concrete operator
sources. It maps one width-neutral hierarchical-reduction candidate to a
reviewable Apple Metal design. It does not yet apply the paper's P1-P8
principle gate and does not implement the design.

## Stage contract

- `manifest.json` admits one candidate closure from `abstraction_extraction/`
  and lists the complete Stage-3 evidence and proposal closure.
- `evidence/*.json` separates vendor-documented facts, sanitized local
  engineering observations, assumptions, and non-evidentiary repository
  context. The bound Target snapshot is context only; it is not treated as
  proof of Apple hardware behaviour. The newer hierarchical-reduction
  observation is retained separately from the earlier graph-pipeline metadata
  probe so their scopes cannot be conflated.
- `proposals/*.json` separates mapping decisions from facts and assumptions,
  exposes resource arithmetic and synchronization commitments, records every
  unresolved hypothesis with a falsifier and retained-result contract, and
  names conditions that invalidate this particular design.
- `review_requests/*.json` asks an external human hardware reviewer to judge
  the complete live manifest/evidence/proposal closure. A request is not a
  decision: it keeps the decision artifact absent, every authorization false,
  and the Stage-3 state awaiting human review.
- `schema.json` is closed: every object sets `additionalProperties: false`.

The only digest used for the Stage-2-to-Stage-3 candidate handoff is
`candidate_closure_sha256`. It binds the selected candidate and exactly its two
observations. Parse the three input files as JSON and form this object, using
the repository-relative observation paths as keys:

```text
{
  "candidate": <parsed candidate document>,
  "observations": {
    <observation path>: <parsed observation document>,
    ...
  }
}
```

Encode it as JSON with sorted object keys, compact separators, UTF-8,
`ensure_ascii=false`, and non-finite numbers refused; then take SHA-256. The
digest is an input-identity boundary only. It is not evidence that the source
pattern or hardware mapping is semantically correct.

Two official Apple PDFs also carry previously verified content digests because
they are pinned external evidence inputs. Vendor webpages have no stored
content digest and explicitly report `offline_content_verification` as
`not_performed`; their cited prose remains subject to human review.

The local probe and non-evidentiary implementation context are bound to Git
parent revision `2238610a4e8923330d73d125e79fd374fc7d2397`. The probe binding
names `tools/probe_metal_operators.py`; the implementation-context binding
names `compiler/targets/apple_gpu_family9.json` and
`src/open_cake_ir/compiler/emit_metal.py`. This revision identifies only the
historical repository inputs used for the observation and context inspection.
Raw output from that earlier probe was not retained, so the binding does not
preserve device state or make the observation automatically reproducible. It
is not a per-file hash inventory or hardware-evidence claim.

## Local FP32 RMS-style semantics probe

A later standalone Metal probe exercises the common two-level choreography and
the RMS-style single-owner path on the local Apple M4. Its six dispatches cover
1, 2, 4, 8, 16, and 32 SIMDgroups at an observed width of 32, with two distinct
FP32 dyadic-fixture epochs per dispatch. The normalized record retains the
expected and observed owner sums, broadcast-consumer counts, dirty-sentinel
reuse checks, and runtime pipeline resource fields. All finite retained cases
passed their internal exact-FP32-bits fixture rule.

This scope is deliberately narrow: it is FP32 and RMS-style only. It does not
exercise BF16-to-FP32 conversion, the Layer-style all-SIMDgroup final reducer,
an external Workload Contract oracle, framework correctness, timing, profiling,
or the current Compiler lowering. Matching finite outputs do not prove uniform
barrier participation or race freedom. The evidence therefore records a local
outcome for every existing proposal falsifier, including out-of-scope and
still-blocking outcomes, without resolving any hypothesis.

The source role and artifact role are separate:

- Git commit `f858612c7ee695b57d72115c76723c7bacc22a9b` binds the complete
  tracked source closure for the Python driver, MSL kernel, and Swift runner.
  There are no per-file hashes.
- The generated probe artifact remains outside the checkout under a
  caller-managed engineering-observation root, at relative path
  `f858612c7ee695b57d72115c76723c7bacc22a9b/metal-hierarchical-reduction.json`.
  This is an unverified external reference: the offline validator has no
  artifact root and neither locates nor reads those bytes.

The source-controlled evidence document is the only normalized record reviewed
by this stage. Its relationship to the caller-managed external artifact is not
byte-verified here. Raw Swift-runner stdout is not retained, and no stable
device identifier is recorded. The commit does not bind device state. Neither
the normalized record nor the external reference grants Evaluation, scientific,
performance, Compiler-change, human-review, or promotion authority.

## Proposed mapping

The extracted candidate remains width-neutral. The first concrete design is a
fail-closed width-32 specialization: it may be selected only when the actual
compute pipeline reports `threadExecutionWidth == 32`. It uses `simd_sum`, one
lane-zero partial store per SIMDgroup and accumulator, a uniform
`threadgroup_barrier(mem_threadgroup)`, and an explicit final-reducer owner.

At the Apple GPU family 9 theoretical ceiling, `1024 / 32 = 32` SIMDgroups fit
in a threadgroup, so one width-32 final SIMDgroup can consume the partial set.
Thirty-two FP32 partials require `32 * 4 = 128` bytes per accumulator. Each
current operator variant receives one combined threadgroup allocation governed
by `raw_dynamic_bytes = accumulator_count * 128 + broadcast_scalar_bytes` and
`aligned_dynamic_bytes = align_up(raw_dynamic_bytes, 16)`:

- RMS supplies one current accumulator and a 4-byte broadcast scalar: bytes
  `[0,128)` hold its partials, `[128,132)` the scalar, and `[132,144)` padding,
  for 132 raw and 144 aligned bytes.
- Layer supplies two current accumulators and no broadcast scalar: bytes
  `[0,128)` and `[128,256)` hold the two partial arrays, with no padding, for
  256 raw and aligned bytes.

The Layer count of two is an input of this current variant, not an invariant of
the width-neutral candidate. This arithmetic is only a ceiling derivation: the
selected pipeline's actual maximum thread count remains a blocking gate, and
`pipeline.staticThreadgroupMemoryLength` plus the selected variant's 144- or
256-byte aligned dynamic allocation must fit both
`device.maxThreadgroupMemoryLength` and the 32768-byte family ceiling. Scratch
reuse remains governed by the proposal's explicit barrier-and-lifetime rule.

The proposal keeps RMS and Layer normalization ownership distinct. The RMS
variant has SIMDgroup zero produce and publish one final scalar; a second
uniform barrier is required before other SIMDgroups consume it. The Layer
variant permits every SIMDgroup to repeat the final reduction and retain the
result in thread-local state. BF16 contributions are converted to FP32 before
`simd_sum`, and no bitwise reduction order is promised.

## Human review request, not decision

`review_requests/hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json`
binds the live Stage-3 manifest, evidence, and proposal by their repository
paths and document IDs. It lists the complete review closure: 21 fact IDs,
five decision IDs, five unresolved hypothesis IDs, four observed-scope finding
IDs, and twelve falsifier IDs. Nine review items organize that closure around
vendor applicability, width policy, partial cardinality, barriers, scratch
lifetime, final-reducer ownership, resource accounting, numeric scope, and the
observation/authority boundary.

The request deliberately retains one phase-order contradiction. The
`resource-accounting-fits-selected-pipelines` hypothesis says it is required
before implementation, while its falsifier requires compiled future pipelines
and their compiler-created static threadgroup-memory usage. The current finite
Compiler lowering cannot create the proposed scratch or barriers. An external
human reviewer must select and justify exactly one of two options:

1. keep the gate literal and collect resource observations from separately
   reviewed standalone RMS and Layer Metal prototypes before Compiler
   implementation; or
2. require a successor proposal that moves the gate to port acceptance and
   permits only the bounded post-P1-P8 implementation needed to materialize
   those pipelines, without accepting or releasing the port first.

Automation may not choose between these options or reinterpret the existing
proposal. The future external decision contract permits only `approved`,
`changes_requested`, or `rejected`; it requires one verdict and localized
reason for every review item, the exact reviewed Git revision, and one
localized ambiguity resolution. The global disposition is the maximum of the
nine review-item severities under the fixed order `approved <
changes_requested < rejected`; it is not an independent override. Every
review-item verdict must therefore be `approved` to clear Stage 3.

The two ambiguity choices are not approval-equivalent:

| Selected option | Required item/global result | Result for this Stage-3 closure |
| --- | --- | --- |
| `preimplementation-standalone-metal-prototypes` | All nine items and the derived global disposition may be `approved` | May clear only the hardware-review gate and proceed to the separate P1-P8 review |
| `amend-gate-to-port-acceptance-after-bounded-implementation` | `resource-accounting-and-runtime-gates` must be non-approved; global disposition must be `changes_requested` or `rejected` | Cannot clear this closure; a successor Stage-3 proposal and review request are required |

The contradiction, the two ordered option texts, aggregation rule, option
matrix, and stage-transition rule are literal public schema contracts rather
than reviewer-request prose that automation may rewrite.
`automation_may_write_decision` is false.
No decision artifact exists in this closure, and all human-review, next-gate,
Compiler-change, implementation, Evaluation, performance-claim,
scientific-claim, and promotion authorizations remain false.

An eventual approved external hardware decision can clear only this Stage-3
gate. The next action would still be the separate P1-P8 principle-driven
iteration, not implementation.

## External decision intake boundary

The public schema defines the closed structural shape of
`hardware_review_decision` so an external human can write a decision after
reviewing an exact Git revision. The offline intake validator additionally
enforces the ordered nine-item closure, derived global disposition, and option
matrix; generic JSON Schema validation alone is not semantic intake. This
repository does not contain a decision instance or a `decisions/` membership
directory, and the Stage-3 review request continues to record the decision as
absent. The decision remains in caller-managed storage and may be supplied
read-only to the offline validator:

```sh
python tools/validate_hardware_informed_design.py \
  --review-decision /absolute/caller-managed/hardware-review-decision.json \
  --format json
```

Omitting `--review-decision` preserves the current review-pending validation.
Supplying one does not copy, normalize, rewrite, or admit it into the checkout.
A well-formed `changes_requested` or `rejected` document is a valid completed
human review but does not clear the gate; malformed, stale, or contradictory
bytes are an intake validation failure.

The decision binds a complete 40-character lowercase Git commit and exactly
these five reviewed paths, in order:

1. `hardware_informed_design/schema.json`
2. `hardware_informed_design/manifest.json`
3. `hardware_informed_design/evidence/apple-metal-hierarchical-reduction-v1.json`
4. `hardware_informed_design/proposals/hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json`
5. `hardware_informed_design/review_requests/hierarchical-simdgroup-threadgroup-reduction-apple-family9-v1.json`

The intake verifies that commit and the reviewed path closure at this external
handoff boundary without adding a digest field or requiring the current `HEAD`
to equal the reviewed commit. The existing Stage-1-to-Stage-3 validators still
validate the semantic and predecessor closure independently.

`reviewer_identity`, `decision_basis`, every per-item verdict and localized
reason, the global disposition, the reviewed revision, and the ambiguity
choice and reason are human-authored fields. Automation may only read them,
validate their closed contract, and report derived in-memory status. It may
not create a template that purports to be a decision, fill any human field,
change the request or manifest, or materialize the next stage.

`authorship_attestation = external-human-outside-automation` records a process
attestation. This JSON protocol has no signature or reviewer-account system,
so structural validation does not prove the person's identity, independence,
or custody. Those remain repository-process facts. The decision's fixed scope
acknowledgements also retain the unresolved resource hypothesis and keep
Compiler change, implementation, Evaluation, performance, scientific, and
promotion authority false. Even a fully approved intake means only that the
separate P1-P8 principle-driven review may begin.

## Validation is not readiness

The offline validator can establish closed artifact membership, JSON shape,
candidate identity, source-reference closure, normalized local-observation
shape, falsifier-outcome coverage, resource arithmetic, and fixed
review-boundary fields. The external generated artifact is intentionally not a
checkout member. Validation does not fetch the web, interpret vendor prose,
compile or run Metal, reproduce device state, prove barrier safety, check an
external workload oracle, or measure performance.

Accordingly the manifest and proposal remain:

```text
state = review_pending
claim_scope = hardware_mapping_only
disposition = await_human_hardware_review
next_gate = principle_driven_iteration
readiness.ready = false
```

The evidence file carries the narrower `claim_scope = hardware_evidence_only`;
it records facts and assumptions but makes no mapping decision. It otherwise
has the same review-pending state, disposition, next gate, and false readiness.

Only a human hardware review may clear this stage. Even then, the next action is
the separate P1-P8 principle-driven iteration, not implementation. The current
finite Metal lowering has no threadgroup allocation or barrier support, so this
directory neither modifies it nor claims that it can materialize the proposal.

No correctness, performance, calibration, occupancy, Evaluation, or scientific
claim is made here. Raw runner output and stable device identifiers are not
retained in this Stage-3 closure.
