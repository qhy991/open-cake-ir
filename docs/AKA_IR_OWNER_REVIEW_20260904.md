# AKA v6 IR owner design review — 2026-09-04

## Decision and authority

Accept three bounded additions to the existing `elementwise` vocabulary for draft
implementation: FP32 `fma`, `log`, and `cos`. Do not add an OperationKind, general
expression language, layout algebra, runtime dispatcher, or whole-operator opcode.
The first implementation slice is FMA; log and cosine follow as separate attributable
changes. No Compiler source is changed by this review.

This is Codex's semantic-design decision under the user's explicit authorization to
review and approve new IR on 2026-09-04. It is **not Compiler release approval**.
[ADR 0030](adr/0030-compiler-release-approval-is-external.md) leaves
`compiler/release-approval.json` as the sole, externally written release authority.
This document cannot be consumed in its place. An implementation, passing exact Corpus
Gate, independently reviewed expectation changes, external approval, and release cycle
remain necessary before any successor becomes released.

Review scope: all 51 `repeated_candidate` clusters and the one conflict cluster receive
a proposal/ownership disposition below. Original snippet, derived baseline, and reference
inspection is deeper for the seven witness cases discussed below, not an assertion that
all 176 underlying cases have received full source-level equivalence review. The 150
singleton clusters are outside this review and remain unapproved, not permanently rejected.

| Disposition | Clusters | Meaning |
| --- | ---: | --- |
| accept | 3 | Minimum design is justified; only the explicitly bounded draft scope is approved. |
| defer | 39 | A real concern remains, but typing, ownership, ordering, ABI, or proof obligations are not closed. |
| reject | 9 | Reject the proposed composite opcode or wrong-layer solution, not the underlying workload. |
| conflict | 1 | Conflicting source semantics must be resolved before design. |

The three accepted clusters contain 20 corpus rows (12 + 6 + 2). This is **not 20 enabled
tasks**, 20 independent upstream sources, or a coverage estimate. Each full task can retain
other blockers such as runtime ABI, indexing, reduction order, state, or broadcast.

## Evidence and current vocabulary

- Review input: Open-Cake commit `f0bca700f3010a3e33e906e1835d1569dcfa0e2e`,
  [published cluster proposal](data/aka-qualified-ir-v6-review-20260904/ir-gap-clusters.json)
  and [677-row review](data/aka-qualified-ir-v6-review-20260904/phase-a-review.jsonl).
- Source input: AKA commit `387aa7faf521a0b72c994ff15a7638cd7e6a8583`,
  `datasets/curated/cuda_kernel_parent_completions_v6/`. Paths beginning with `bundles/`
  below are relative to that dataset. No test split or unrelated corpus was consumed.
- `src/open_cake_ir/compiler/ir.py` owns `ElementwiseOp`, `ElementwiseParameters`,
  `DType`, `AccessMap`, and `AtomicRmwParameters`. Existing elementwise arithmetic is
  square/rsqrt/exp/relu/tanh/add/sub/mul/div. An instruction contract currently has
  defined elementwise meaning only for tanh. Constants are not runtime scalar arguments.
- `verifier.py` owns promotion, shape/broadcast checks, INT32 immediate-valued atomic
  add, and mechanical direct-state-store ownership. A Workload assertion of uniqueness
  cannot replace its ownership proof. `emit_triton.py`, `emit_cutedsl.py`, `work.py`,
  and `analysis.py` must evolve with any admitted vocabulary.
- Existing FP32 `reduce` does not promise an arbitrary serial order or a specified XOR
  tree. Existing inclusive sum `scan` does not make seeded/exclusive/segmented scans
  immediately expressible. Existing elementwise arithmetic does not admit INT32 operands.

The published corpus qualification, 56 fixed-instance dynamic-valid Lab results, and one
authoring rejection retain their original meanings. None is a GPU result for these new
designs, and no historical cluster or verifier receipt is rewritten by this decision.

## Per-cluster disposition

IDs and membership remain owned by the linked cluster proposal. An `accept` applies only
to the minimum contract in the following section. Other rows state the next missing
obligation; they do not establish source independence or final expressibility.

| Cluster | Decision | Owner / reason and next obligation |
| --- | --- | --- |
| c001-int32-coordinate-divmod-transform | defer | AccessMap/index arithmetic: bound signedness, overflow, div/mod domain and mask-before-address rules; do not introduce an unrestricted expression tree. |
| c002-stable-adjacent-unique | defer | Compaction: equality, run-head flags, prefix positions, count and tail effects need a compositional proof; two payload/tail contracts are not yet one primitive. |
| c003-affine-index-i32 | defer | AccessMap: positive contiguous subranges already exist, but affine scale/reversal is not that feature; prove coordinate range and ownership before choosing one representation. |
| c005-masked-affine-gather-i32 | defer | Compose bounded index arithmetic with the existing load/AccessMap owner; a separate gather opcode would duplicate address and mask authority. |
| c007-masked-affine-gather-i64 | defer | Same address-owner issue plus an unadmitted INT64 type/overflow contract; no implicit index widening. |
| c009-fp32-computed-atomic-add-state | defer | Extend existing atomic_rmw, not a new operation: settle computed update edge, FP32 result/old value, masked result, collisions and nondeterministic sum acceptance. |
| c010-initialized-fp32-atomic-accumulate | reject | Initialization before cross-CTA accumulation is a Program dependency; an atomic opcode cannot establish a grid-wide initialization barrier. |
| c012-fp32-atomic-scatter-i32 | defer | Existing AccessMap plus the c009 atomic extension should own this; prove update/index/result alignment and collision semantics first. |
| c013-zero-initialized-fp32-scatter-reduce-i32 | reject | Split Program-owned reset from FP32 indexed atomic effects; do not hide both in a scatter-reduce opcode. |
| c020-coordinate-predicated-select-fp32 | defer | Comparison/select and logical-coordinate production need one typed design; do not encode coordinate predicates as operator-specific arithmetic. |
| c023-segmented-gather-mean-fp32 | defer | Runtime bounded ordered reduction, gather and empty-segment finalization are separate obligations; ordinary associative reduce is not a proof of serial equality. |
| c025-capped-grid-stride-program-map | defer | ProgramMap/TileLoop: exact physical grid, logical coverage, reuse order and declared-work analysis must agree; preserve one mapping authority. |
| c026-clamped-affine-index | defer | Consolidate with bounded AccessMap arithmetic; exact signed clamp and output-tail masks must not be conflated. |
| c029-length-predicated-select-fp32 | defer | ValidExtent already owns some length masks, but arbitrary false-arm values require typed selection; prove the supported relation rather than duplicate it. |
| c038-segmented-gather-sum-fp32 | defer | Share the bounded-domain/order design with c023/c173; no separate whole segmented-gather operator yet. |
| c042-elementwise-cos-fp32 | accept | Add unary FP32 cos to elementwise with an explicit libdevice target contract; bounded scope below. |
| c046-clamped-natural-log-fp32 | reject | Clamp and log are distinct primitives; c052 does not authorize a fused clamp-log opcode or an inexact relu-based clamp rewrite. |
| c048-ordered-fp32-greater-select | defer | A minimal comparison/select design must fix ordered comparison, NaN and signed-zero selection, operand roles and output typing. |
| c049-fp32-equal-select-zero | defer | Consolidate with typed comparison/select, retaining exact equality and bit-preserving selected values; do not approximate equality with arithmetic. |
| c050-elementwise-fma-fp32 | accept | Add ternary FP32 fma to elementwise with one RN-even result; bounded scope below. |
| c052-elementwise-natural-log-fp32 | accept | Add unary FP32 log to elementwise with an explicit libdevice target contract; bounded scope below. |
| c062-uint8-add-mod256 | defer | DType plus modular integer arithmetic needs a common signedness/promotion rule; a special u8-add opcode would bypass that owner. |
| c071-fused-softmax-cross-entropy-forward-backward | reject | This is a Workload-level composition of reductions, exp/log, selection and arithmetic, not a new whole-loss primitive. Missing components remain real. |
| c078-inclusive-segmented-scan-by-adjacent-key | defer | Extend scan only after exact key equality, segment-head ownership, identity/order and cross-tile carry are specified; no generic tuple-body escape hatch. |
| c079-int64-index-result-store | defer | DType/cast/store: exact INT32-to-INT64 widening and eight-byte ABI need shared support, not a special argmin/argmax store. |
| c080-indexed-atomic-add-fp32-i64-state | defer | c009 plus INT64 coordinate support; neither an assertion of index validity nor a new scatter name closes address safety. |
| c081-zero-initialized-indexed-scatter-add-fp32-i64 | reject | Program reset plus the deferred typed atomic/indexing components; no hidden initialization in Schedule atomic semantics. |
| c082-unique-i64-scatter-store-fp32-state | defer | Indirect state-store ownership is not established by a caller promise; require a mechanical proof or explicit external runtime validation design. |
| c084-int32-equal-select-fp32 | defer | Keep INT32 comparison distinct from FP32 selected payload; one typed compare/select relation must cover this without implicit casting. |
| c096-masked-load-with-fill-fp32 | defer | Load owns no-access semantics. Eager load followed by select is unsafe; bind predicate, address and false-arm value without a second mask authority. |
| c098-windowed-equal-gather-sum-fp32 | reject | Reject the whole pooling-backward opcode; derive it from bounded window indexing, exact equality selection and an explicitly ordered fold. |
| c111-ordered-grid-stride-cta-sum-fp32 | defer | The fixed 1024-lane/two-XOR-tree schedule is not ordinary reduce semantics; represent lane ownership/tree ordering transparently or narrow the Workload. |
| c112-ordered-indexed-state-add-fp32-i64 | defer | Ascending-source-order colliding state updates are not unordered atomic add; requires ordered iteration and state-effect proof. |
| c129-segmented-broadcast-expand | defer | Runtime segment ranges and complete disjoint stores must compose; do not accept a scatter assertion as the coverage proof. |
| c134-unique-indirect-state-scatter-i32 | defer | Preserve existing mechanical state-store gate; metadata-derived uniqueness and untouched state require proof before widening it. |
| c140-uint32-modular-sum | defer | Extend DType and reduction numeric semantics together; modulo-2^32 sum is not FP32 reduce or undefined signed overflow. |
| c142-endpoint-excluding-reflect-index | defer | Treat reflect as a bounded integer-map use case; extent-one, period, modulo and negative coordinates need one exact index design. |
| c144-roi-max-pool-forward | reject | ROI geometry, coordinate conversion, windows, strict max and output indices form an operator, not one minimal Schedule primitive. |
| c149-runtime-dense-nd-transpose | reject | Reject a transpose opcode/layout algebra. First choose frozen-rank specialization versus Program dispatch; runtime rank/shape ABI is not an AccessMap flag. |
| c150-runtime-scalar-parameter | defer | High-priority ABI design: by-value typed scalar bindings must propagate through signature, validation, lowering and invocation; a length-one pointer is not ABI equivalent. |
| c154-runtime-segmented-reduce | reject | Runtime opcode selection plus five reducers, seeds and special-value rules is not a minimum primitive; split Program selection from typed ordered reduction. |
| c159-unique-i64-scatter-add-fp32-state | defer | Read/add/store could compose only after indirect uniqueness and alias proof; non-atomic updates cannot inherit atomic safety. |
| c161-seeded-exclusive-scan-i32-with-total | defer | Existing inclusive scan is insufficient: typed INT32 arithmetic, overflow, seed ABI and total extraction must close before a composite scan variant is admitted. |
| c163-segment-owner-from-boundaries | defer | Specify monotone boundaries, duplicate/empty segments, total coverage and bounded search/index results; connect to one index owner. |
| c168-segmented-serial-foreach | defer | A generic serial body is an unbounded control-flow expansion. Find the finite recurrence/effect subset and zero-trip semantics first. |
| c173-segmented-ordered-sum-fp32 | defer | Share bounded-domain and ordered-fold design with c023/c038; a serial rounded sum cannot silently become a parallel tree. |
| c176-segmented-scan-conflicted | conflict | Two records disagree about marker/boundary inclusion in reverse scan. Resolve source contracts separately; do not merge contradictory results. |
| c181-signed-int8-dtype | defer | Pure storage and truncating FP32 conversion are different obligations; define DType/cast range and rounding without widening every numeric operation. |
| c190-uint32-dtype | defer | Storage alone does not grant unsigned arithmetic/order/scan/indexing. Specify admitted consumers and exact ABI together. |
| c194-uint8-buffer-dtype | defer | Storage/load/store is a plausible small future slice, but runtime extents and constant generation in member tasks are not implied by dtype admission. |
| c197-welford-chan-reduce | defer | Statistics state and merge are real recurrence candidates, but one fixed XOR tree and a flexible partition are not yet one numerical contract. |
| c199-windowed-strict-max-reduce-fp32 | defer | Exact -FLT_MAX seed, strict greater update, NaNs, signed-zero ties and empty windows differ from generic max; close ordered-fold semantics first. |

## Accepted minimum contracts

P1–P8 design check: these are familiar elementwise operations (P1); the selected hardware
realization stays explicit (P2/P8); one existing operation family and one contract per
initial realization avoid alternate spellings (P3); exact arity/dtype/shape rules reject
bad construction (P4); dataflow, register pressure and work accounting evolve together
(P5/P7); implementation acceptance requires both positive/near-miss cases and the full
Corpus Gate (P6). This is design acceptance of those obligations, not a claim that the
future implementation has already satisfied them.

Common restrictions: reuse `OperationKind.ELEMENTWISE`, typed Buffer reads/writes,
dependency ordering, Target instruction contracts, and the existing assessment/lowering
path. Initial scope is same-shaped FP32 register operands and one same-shaped FP32 register
result. No scalar literal, by-value runtime scalar, broadcast, mixed dtype, additional
memory effect, implicit cast, or new ABI is granted by this first slice. Those exclusions
are deliberate; the 20 full parent tasks are not thereby declared expressible.

Missing/wrong instruction contracts, bad arity, unknown edges, non-register operands,
wrong dtype, mismatched shapes and unsupported backend must fail at construction or
assessment with localized Findings, not at emission. No unmodeled backend fallback.
The proposed contract names below are not claims that the current Target already admits them.

### c050: FP32 fused multiply-add — accept

Canonical design: `op: fma`, three reads ordered as `a, b, c`, and
`instruction.contract: ptx.fma.rn.f32`. Evaluate the exact product-plus-addend and round
once to binary32 RN-even. Keep subnormal handling consistent with non-FTZ PTX FMA,
preserve dependency/operand order, and make no NaN-payload identity promise. Do not make
the existing `mul`/`add` spelling an implicit synonym or reassociate a previously rounded
producer into this FMA. NVIDIA documents the single-rounding operation and PTX's explicit
rounding/subnormal distinction in the [libdevice FMA contract](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_rn.html)
and [PTX floating-point instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma).

Source-complete recurrence witnesses, at the AKA commit above:

- `bundles/data_movement_and_layout__copy_vectorized__analysis__l000002_b200_v1/`:
  `source/input.json` contains Momentum SGD's explicit `__fmaf_rn` and `__fmul_rn`;
  `sources/baseline/kernel.cu:38` and `sources/reference/reference.cpp:19` retain the
  fused operation and separate rounded producers.
- `bundles/normalization__normalization_group__analysis__l000002_b200_v1/`:
  the original GroupNorm snippet contains nested `fmaf`;
  `sources/baseline/grouped_nchw_affine_combine.cu:31` and
  `sources/reference/grouped_nchw_affine_combine_reference.cpp:49` preserve its nesting.
- The LAMB witness at
  `bundles/elementwise_and_activation__elementwise__optimization_neutral__l000001_b200_v1/`
  was inspected but **not counted as an independent original explicit-FMA witness**:
  its original snippet is `p - ratio * update`, while its narrowed derived contract
  deliberately fixes FMA. Also do not count NCHW/NHWC or repeated normalization shards
  automatically as independent sources. Distinct operator consumers are observed;
  independent upstream projects and complete original-framework provenance are not claimed.

Executed host witness (not a GPU test): for binary32 `a=1+2^-23`, `b=1-2^-23`, `c=-1`,
exact rational arithmetic gives `-1/70368744177664 = -2^-46`. Host `fmaf` returns
bits `0xa8800000`; separately rounded FP32 multiply then add returns `0x00000000`.
All three inputs and the nonzero result are normal finite FP32 values. This proves a
rounding distinction, not correct new Compiler lowering.
It does not claim that the current backend always leaves a mul/add chain unfused:
incidental optimizer contraction is precisely not a declarative single-rounding guarantee.

Lowering acceptance: emit explicit non-FTZ RN-even FMA in the Triton backend and inspect
generated PTX. An explicit inline PTX instruction is a bounded lowering mechanism, not
new author-visible inline assembly. Test an already-rounded producer followed by FMA and
nested FMA so compiler contraction cannot erase declared rounding boundaries. Other
backends must refuse until they implement this exact contract.

Analysis acceptance: charge two arithmetic operations per result element in `work.py`,
retain all three input lifetimes in existing dependency/pressure analysis, and add no
invented latency or occupancy claim. Required tests after implementation: same-shape
positive plus nested/rounded-producer positives; malformed arity/dtype/shape/space/contract
near misses; the cancellation witness; signed-zero/subnormal/special-value semantics;
unsupported-backend refusal; unchanged historical Corpus expectations and complete new Gate.

### c052: FP32 natural logarithm — accept

Canonical design: `op: log`, one read, `instruction.contract: libdevice.log.f32`.
Use the CUDA `__nv_logf` realization and its documented special-value behavior, not
`log2(x)*ln(2)`, an inverse-exp search, or an automatically selected fast approximation.
The [NVIDIA log contract](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_logf.html)
defines the realization; exact numerical acceptance still belongs to each unchanged
external Workload oracle/tolerance. Do not promise correctly rounded mathematical log
or bitwise agreement with every host math library. Pin the actual toolchain at Executor
handoff rather than adding a second version field to each operation.

Distinct original consumers inspected:

- categorical probability NLL,
  `bundles/data_movement_and_layout__gather_scatter__analysis__l000115_b200_v1/`:
  original and `sources/baseline/operator.cu:22` use `-logf(probability)`;
  `sources/reference/reference.cpp:21` evaluates the host double logarithm.
- sigmoid cross-entropy row mean,
  `bundles/reduction_and_selection__reduction__analysis__l000070_b200_v1/`:
  original helpers use log of an exponential expression;
  `sources/baseline/derived_sigmoid_xent_rowmean.cuh:21` uses `logf`, while
  `sources/reference/derived_sigmoid_xent_rowmean_reference.hpp:11` independently
  evaluates stable long-double softplus. The full task's mode and reduction behavior
  are not approved by the log primitive.

Lowering uses typed FP32 `libdevice.log`; Triton's
[external-library interface](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html)
selects device functions by operand type. The implementation must verify its pinned
toolchain mapping and must not silently use `tl.log` as a different realization.
`work.py` must mark log as transcendental/unknown arithmetic work, as it does for tanh;
register/dataflow analysis remains shared. Tests must cover positive inputs near one and
across scales, zero/negative/infinite/NaN contract behavior, same-shape parse/assessment,
wrong dtype/arity/contract, emitted libdevice mapping and explicit unsupported backends.
No clamp-log, log1p, log-sum-exp, softplus or complete loss opcode is granted.

### c042: FP32 cosine — accept

Canonical design: `op: cos`, one read, `instruction.contract: libdevice.cos.f32`.
The input is in radians and lowering uses CUDA `__nv_cosf`, not an approximate PTX cosine
with an unproven range reduction. Use the [NVIDIA cosine contract](https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cosf.html)
and unchanged Workload tolerances; no claim of universal exact mathematical rounding.

Two different derivative consumers are visible in original snippets and complete derived
closures: `TanGradientCUDAKernel` in
`bundles/data_movement_and_layout__gather_scatter__analysis__l000158_b200_v1/`
uses cosine followed by square/division (`sources/baseline/kernel.cu:20`); its reference
uses the independent `1+tan(x)^2` identity. `SinGradientCUDAKernel` in
`bundles/elementwise_and_activation__elementwise__analysis__l000014_b200_v1/`
uses cosine followed by multiplication (`sources/baseline/sin_gradient.cu:28`); its
reference computes double cosine then narrows to FP32. These are distinct consumers
within the same framework family, not two independently sourced framework projects.

Initial GPU witnesses should retain their finite frozen input domains (including the
tan-gradient interval [-1,1]); no expansion toward poles is authorized. Test cosine
itself around zero and selected finite range boundaries, negative inputs, and documented
special values; separately test derivative compositions against their original oracles.
Type/shape/contract/backend negative tests match log. Classify cosine as transcendental
unknown arithmetic work, keep ordinary dataflow/pressure analysis, and make no latency claim.

## Acceptance boundary and next action

Executed during this review: current IR/Target/verifier/emitter/analysis inspection;
original/derived/reference inspection for the seven named witnesses; an exact-rational
and host-FP32 FMA counterexample. Existing f0bca700 publication verification is unchanged
evidence, not rebranded as new-IR verification. New-IR parse tests, lowering, Corpus Gate,
GPU compile, correctness, sanitizer, performance and release are all **not run**.

The existing task `Review published AKA IR results` was resumed against this exact corpus
for independent read-only advisory. Its result is not presumed from task activity; only
a substantive returned review can be used as independent corroboration.

Next action: implement the bounded FMA design in a fresh Compiler successor, with its
typed/analysis/lowering changes together and focused regression tests added afterward.
Then prepare the full Gate without regenerating historical expectations. Stop at the
external approval boundary; do not write `compiler/release-approval.json` from the release
automation. Log and cosine remain approved design slices, not concurrent unreviewed
Compiler mutations. No `main` merge, new GPU submission, or historical artifact rewrite
is performed by this review.
