# BW1100 Compiler expression and rewrite round — 2026-10-06

This round repairs three supported authoring boundaries. Shared implementation is
[PR349](https://github.com/qhy991/open-cake-ir/pull/349); the hardware branch owns
Target admission, hardware Corpus cases and this bounded qualification evidence.

| Observation | Owning improvement | Acceptance scope |
| --- | --- | --- |
| RoPE cannot name FP32 sine/cosine | Explicit OCML contracts, FP32 typing, HSACO-only direct library lowering | Each function agrees with the external Torch oracle on8,388,620 finite arguments in FP32 and BF16. Original multiply/rounding order remains the author's responsibility. |
| Sorting segment starts need maximum prefix propagation | INT32 resident inclusive max scan with a narrow JIT combine helper | Forward/reverse each agree with CPU oracle on17,408 signed integers; scalar identity and floating refusal are tested. |
| Legal reduction loops require manual width edits | Explicit width API accepts fixed pure CTA-reduction loops |199 retained artifacts replayed,29 API rewrites produce the same complete source as retained choices; original decoder160/160 and caller4/4 pass. |

The decoder's fresh paired full-call conservative ratio over its unchanged
community baseline is1.25445698. It is a replay of retained choices, not a
Compiler-induced improvement or a new3h authoring outcome. Trig and max scan
qualification is at component level; complete fused RoPE and complete Cake sorting
are not delivered by these additions. No approximate hardware trig, fast sigmoid,
transpose primitive or unconstrained inline-assembly escape is enabled.

Evidence owner: `/data3/testuser01/experiments/bw1100-compiler-trig-scan-qualification-20261006/`.
Under `bench/campaign/results/`, `component-summary.json`, `component-isa.json`,
`api-replay.json`, `full-task-summary.json` and each admission/terminal retain
source, correctness, timing and release evidence. The initial device carrier is
`82b79910`; reviewed successor metadata/code changes require source replay at the
final handoff boundary. All four device phases completed and released HCU2.
The HCU0 historical load was preserved.

## Disposition of the other leads

| Lead | Current judgement | Next meaningful action |
| --- | --- | --- |
| Power-of-two resident padding for96-element rows | Existing `program_tile` with tile128 and grid17x1 already emits masked loads. A single-trip loop is unnecessary. | Provide the existing-IR example; masks must neutralize padded lanes at every affected reduction. This does not establish full layernorm performance. |
| Exp2 sigmoid instead of the tanh chain | Existing mul/exp2/add/reciprocal operations can express the alternative. It changes the numerical formula and the frozen Task contract excludes it. | Keep separate numerical-contract research; do not add a duplicate sigmoid primitive or silently substitute approximate math. Native full-call gain was below1% materiality. |
| Cyclic head modulo gathering | Capacity lead with parity/worse component results; no full-call benefit demonstrated. | Investigate composition using existing coordinate/remainder/indirect access before extending AccessMap. |
| MMA-side transpose or shifted convolution stencil | Native prototypes did not beat the strong community implementations. | Establish a useful implementation and concrete semantics before adding a new commitment. |
| BF16 pair packing, stable sort/scatter and disjoint multiple stores | Capacity research remains; trig/scan alone do not complete these algorithms. | Stabilize dtype, address/effect and disjointness contracts with external oracles before a successor implementation. |

The new pass preserves math, precision sites, ABI and access graphs; launch width
can still select a different floating reduction tree and requires the unchanged
external correctness gate. Future fixed-budget comparisons should preserve the
same starting seed, strong baseline, Task, budgets and provider. Historic3h scores
are useful observational context; they are not a concurrent randomized control.
