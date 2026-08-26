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
- `evidence/*.json` separates vendor-documented facts, a sanitized local
  engineering observation, assumptions, and non-evidentiary repository
  context. The bound Target snapshot is context only; it is not treated as
  proof of Apple hardware behaviour.
- `proposals/*.json` separates mapping decisions from facts and assumptions,
  exposes resource arithmetic and synchronization commitments, records every
  unresolved hypothesis with a falsifier and retained-result contract, and
  names conditions that invalidate this particular design.
- `schema.json` is closed: every object sets `additionalProperties: false`.

The only digest introduced for the Stage-2-to-Stage-3 handoff is
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
Raw probe output was not retained, so the binding does not preserve device
state or make the observation automatically reproducible. It is not a new
per-file hash inventory or hardware-evidence claim.

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

## Validation is not readiness

The offline validator can establish closed artifact membership, JSON shape,
candidate identity, source-reference closure, resource arithmetic, and fixed
review-boundary fields. It does not fetch the web, interpret vendor prose,
compile or run Metal, prove barrier safety, check an external oracle, or measure
performance.

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
claim is made here. Raw local probe output and device identifiers are not
retained.
