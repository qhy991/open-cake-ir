# What the IR must express, derived from real kernels

The paper derives Cake IR bottom-up from production kernels (CUTLASS, FlashInfer,
FlashAttention-4, TensorRT-LLM, DeepGEMM, Alpha-MoE, TileLang) and names what that survey
motivated: barrier choreography, warp-role partitioning, pipeline staging, TMA descriptors
and TMEM lifecycles, cluster-scoped operations (arXiv:2608.12629v1 Appendix A).

This repository skipped that step and grew its vocabulary one operator at a time. This note
is the missing survey, run against a second and independent source: kernel work actually
carried out on this machine on B200 -- the SGLang CUTLASS FP8 grouped-MoE sweep (16 SGLang worktrees, each paired with a
private CUTLASS fork bound per build -- most of the knobs live on the CUTLASS side), the SGLang MoE serving integration, and nine hand-written PTX
phases on the Llama-3.2-1B QKV projection.

The two sources disagree in an informative way, and that disagreement is the point.

## Coverage

`YES` the Schedule can state it today, `~` partially, `NO` there is no field or enum for it.
Ordered by how often the surveyed work exercised each axis.

| # | Axis | Cake IR | Where it stands |
| --- | --- | --- | --- |
| 1 | Shape/route-driven config selection, opt-in contracts | NO | one Schedule is one kernel; no dispatch predicate |
| 2 | CTA-to-tile rasterization / scheduler order | YES | `ProgramMap.traversal`, meaningful only under persistence |
| 3 | CTAs per SM, persistent grid size | YES | `residency.ctas_per_multiprocessor`, and `ProgramMap.persistent` sized from it |
| 4 | Software-pipeline depth | ~ | one `Pipeline.stages`; real work tunes mainloop, accumulator and scheduler pipelines separately |
| 5 | Register / TMEM / shared budget | YES | `residency.registers_per_thread` with `allow_spill`; reaches ptxas as `maxnreg`, while the verifier uses only the declared logical-storage lower bound |
| 6 | Tile geometry, thread-to-output mapping | YES | `MmaInstruction.shape`, `cta_group`, `tile`, `Role.warps` |
| 7 | Global load width and cache policy | ~ | `LoadParameters.reuse` states the intent; width left out, Triton derives it |
| 8 | Buffer ownership, intermediate-copy elision | NO | buffers are input/output/scratch; no caller-provided destination |
| 9 | On-chip staging and bank swizzle | ~ | `Buffer.swizzle` covers four CUTLASS modes, not a hand-derived layout |
| 10 | Scheduler-metadata precomputation | NO | no work-tile table |
| 11 | ILP, independent accumulator count | NO | `RangeOptions` carries Triton loop knobs, not an accumulator count |
| 12 | Split-K reduction decomposition | YES | `reduce_sum(axis, scope)` plus roles |
| 13 | Two kernel configs in one dispatch | NO | one Schedule is one kernel |
| 14 | Weight precision and scale granularity | ~ | v17 relates FP32 scales to FP8 E4M3 data by per-axis granularity and physical grouped-axis order; padded/packed, dynamic and generated scales remain absent |
| 15 | GEMM orientation (transposed decode) | ~ | buffers can be declared transposed; scale-config major mode cannot |

Five of fifteen are fully expressible and five are partial. Of the five most-exercised axes, three are now
`YES`, one `~`, and one belongs to the portfolio stage rather than to a Schedule.

Closed since this note was written, each with a hardware check that the declaration reaches
the backend rather than stopping at the verifier:

* Axis 3, then 2. The order in this note was wrong and implementing it said so: for a
  non-persistent grid the axis-to-program-id assignment already decides which axis varies
  fastest, so traversal is only a distinct decision once a scheduler walks more tiles than
  there are CTAs. Persistence had to come first, and the persistent CTA count is derived
  from the residency commitment rather than declared again.
* Axis 5's register half. `maxnreg` appears in the compiled kernel's metadata where
  declared and nowhere else. `RESIDENCY_UNMET` is a static proof only when the optimistic
  logical-storage lower bound already makes the commitment impossible; actual register
  allocation and spill counts remain ptxas evidence.
* Axis 7's cache half. The first emission did not compile -- ptxas rejects `.cg` combined
  with `.evict_first` -- which only surfaced on hardware.

## The constraint neither survey would have found alone

Seven or more of the MoE variants are shaped by one fact:

> Problem sizes are device-resident during CUDA-graph replay, so validating them on the
> host would introduce a synchronization.

That single constraint is why the work reaches for device-side masking of the problem list,
a device flag with both kernels' parameters resident, an environment-variable contract
pushed to the deployer, and a precomputed work-tile table -- rather than the host-side
branch each of those replaces.

Cake IR v19 can now say one narrow version of that fact: a global INT32 input owns the
runtime-valid prefix of one padded Buffer axis, and lowering consumes it on-device without
a host read. It still cannot express a general device-resident scheduler flag, problem
list or work-tile table, so it cannot yet reject every Schedule or program composition
that would force host synchronization. That broader property remains verifier work and
does not appear in the paper's own list because the paper's corpus is standalone kernels
rather than a captured serving path.

## Where the two surveys diverge

The paper's motivators are about **what a kernel is**: roles, barriers, pipelines, TMA,
TMEM, clusters. This repository has four of those five; only cluster-scoped operation is
absent, and `cta_group` covers the MMA case.

The axes above are about **what a kernel commits to**: how CTAs traverse work, how many
stay resident, how wide a load is and what it does to cache, how many registers may be
held. Those are exactly the "performance-relevant hardware decisions" P2 asks to keep
visible, and they are where the vocabulary is thin.

Growing the IR from toy operators found the arithmetic gaps -- a reduction that could not
sum an arbitrary axis, no way to square or rescale, broadcasting that could not be stated.
Real kernels find placement and resource gaps instead. Both halves are needed and only one
had been exercised.

## The surveyed work already writes the commitments the IR cannot hold

Every PTX phase states its acceptance criteria as a resource budget: at most 48 registers
with zero spill, at most 4096 bytes of shared memory, at most one barrier, no global
workspace, no atomics, no second kernel. The CUTLASS side does the same through
`static_assert` -- shared storage under 116736 bytes so two CTAs fit, exactly 64 TMEM
columns so two CTAs own disjoint regions, `MinBlocksPerMultiprocessor == 2`.

These are commitments a Schedule should carry and a verifier should gate. They are written
as prose in a task document, or as an assertion inside a template, because there is nowhere
in the IR to put them. That is the strongest argument for the residency and register work
below: the demand already exists and is being met outside the system.

It also changes what the analysis is for. It derives per-resource upper bounds on
residency, not measured occupancy. Once a Schedule declares the residency it needs, a
bound below that commitment is a gate; actual register allocation still belongs to the
toolchain rather than the static Schedule model.

## How the surveyed work runs its loop

Worth recording because it is closer to the paper's four-stage loop than this repository's
own campaigns are. Each PTX phase freezes the geometry and changes exactly one axis, and
each carries a citation of the previous phase's measured outcome. Four of the nine did not
cross their bar, and those negative results are retained with the falsifier that produced
them -- two rows per warp regressed because active warps per scheduler fell from 4.466 to
2.675, wider loads halved request count while leaving sectors and DRAM bytes unchanged.

One phase shipped no kernel at all, because its own noise rule made a clean pre-edit
measurement a prerequisite and the measurement was noisy. That is the discipline the
paper's harness is supposed to enforce, arrived at independently.

## Ordering

Sorted by exercised frequency against implementation cost, and split by whether the axis
belongs to one Schedule at all.

**Belongs in the IR, and is cheap**

1. Traversal order on `ProgramMap` (axis 2). A closed enum, verifiable against the declared
   axes, and directly a P2 commitment. Six of the surveyed forks exist only to change it.
2. Residency commitment (axis 3, and the register half of 5). `grid` gains a persistent
   form, and a Schedule may *declare* the CTAs-per-SM and register budget it needs instead
   of only having them derived. The analysis then checks the declaration rather than
   reporting a number nobody can act on -- which also closes the loop on the report the
   harness already returns.

**Belongs in the IR, larger**

3. Load width and cache policy on `LoadParameters` (axis 7). Five of the nine PTX phases
   turn on this and nothing else.
4. Separate the pipeline kinds (axis 4). One `stages` field conflates three distinct
   pipelines that the real sweeps tune independently.
5. General device residency as a declared, verifiable property. v19 closes only the
   valid-prefix Buffer relation; scheduler flags and work tables remain separate.

**Does not belong to one Schedule**

Axes 1, 10 and 13 are dispatch and portfolio concerns, which the paper places in its
generalization stage and gates behind strong per-shape seeds. This repository's portfolio
is exact-shape specialists only. They should not be pushed into the Schedule vocabulary.

**Needs a decision before it is designed**

Axis 8 sits against P1, which asks to avoid unnecessary destination-passing. The MoE
integration uses destination-passing deliberately, to write into a symmetric collective
buffer and delete a copy per rank. "Unnecessary" is doing real work in that principle and
the boundary should be drawn explicitly rather than assumed.

Axis 14 now has one deliberately narrow model: `Buffer.scale_of` owns the FP8 data buffer,
per-data-axis granularity and physical grouped-axis order. The verifier derives the scale
shape, preserves the relation across loads and proves the four MMA reads are associated.
It is not a general quantization model: padding, packing, dynamic extents, generated scales
and sub-rate pipelines remain separate missing mechanisms.

## Corpus coverage

The current Compiler Corpus has 34 cases across fifteen program slices, against the
paper's roughly four hundred cases across twenty-eight. Attention and MoE, which are what
the surveyed work is actually about, still have no complete representation here.

The independent 57-version KDA MoE evolution sharpens that statement. Cake can name
several isolated deltas, but it cannot describe even the complete v1 baseline, so the
complete-version coverage is 0/57 rather than a count of matching knobs. The snapshot,
ownership boundary and first minimal slice are recorded in
[`ADR 0020`](adr/0020-kda-deltas-are-an-external-expressibility-corpus.md); the external
history remains the evidence authority instead of being copied into this repository.
ADR 0021 closes the standalone deterministic indexed-selection primitive and validates it
on B200, but grouping and mask/update composition are still needed before the three-stage
v1 routing chain can be represented as one Schedule.

ADR 0022 closes a subtler KDA-derived gap: the mathematical word `tanh` did not say which
target implementation realized it. The standalone positive now declares
`libdevice.tanh.f32`; KDA's actual `tanh.approx.f32` spelling is structurally expressible
but rejected by the current Target. A brokered diagnostic exceeded the unchanged standalone
`1e-5` tolerance, so approximate math remains an explicit unsupported contract rather than
an implicit emitter choice or a conveniently widened oracle. Complete KDA coverage remains
0/57.

ADR 0023 closes the first KDA v1 quantization relation without claiming the complete
kernel. The static smoke slice uses KDA's FP8 E4M3 data, FP32 scales, activation storage
`[K/128, M]`, weight storage `[N/128, K/128]` and two independently scaled K contractions.
Its released v17 source compiled and matched 2,048/2,048 B200 outputs at maximum deviation
`1.788e-7` under the preregistered `1e-5` gate. General K-block counts, grouped routing and
scatter are still outside the lowering domain, so complete-version coverage stays 0/57.

ADR 0024 closes the independent valid-row prerequisite. A padded Buffer owns one
device-resident INT32 length relation; lowering reuses AccessMap coordinates to load the
length and mask invalid rows. Its v19 generated source matched 512/512 B200 outputs with
zero deviation. This does not schedule only-valid tiles, perform grouped GEMM or scatter
results, so it does not change complete-version coverage.

ADR 0025 then tests composition instead of adding vocabulary. Ordinary `ProgramMap`
group/M/N axes, group-indexed `AccessMap`s, the existing K `TileLoop`, `mma` and
`valid_extent` lower one four-group ragged BF16 contraction without a schema, IR,
verifier or emitter change. Compiler v20 matched 1,024/1,024 B200 outputs with maximum
deviation `1.9073486328125e-06` under the fixed `1e-5` gate. This closes the arithmetic
core of one KDA v1 grouped GEMM, not routing/group formation, valid-tile work acquisition,
the second contraction, scatter or their program DAG; complete-version coverage remains
0/57 and no performance claim is made. Review of its drift case exposed one generic
address-safety gap: Compiler v21 now rejects a scalar program coordinate when its derived
program range exceeds the indexed dimension of another Buffer. No lowering digest changed.

ADR 0026 adds runtime INT32 Buffer coordinates to the existing `AccessMap`, not a gather
operation. Its v22 generated source selected KDA-style `(expert, row)` tuples, masked the
`-1` sentinel and matched 1,024/1,024 B200 outputs exactly. ADR 0034 then composes that
load with existing `mul`, `sum` and `store` primitives to express the arithmetic body of
KDA v1's non-fused weighted combine. Compiler v26 makes the implied numeric contract
explicit: loads preserve dtype, BF16 multiplied by FP32 produces FP32, the admitted
reduction produces FP32, and the final store may narrow to BF16. No `moe`, `combine`,
`scatter`, `cast` or program-DAG vocabulary was added. The first B200 attempt stopped
before compilation because the broker worker could not import Torch.
The first successor executed but lost its result to a late Cutlass metadata import; both
failures are retained and neither supports correctness. The metadata import now precedes
GPU work; the fully preflighted second successor compiled and launched once on B200 and
matched 128/128 BF16 outputs with zero deviation. The check remains correctness-only and
cannot establish a performance or complete-MoE claim. Compiler v27 then adds caller-owned
global `state` and one returned-old-value `atomic_rmw`, rather than a route or slot mode.
Its first admitted form is masked INT32 add with relaxed device scope and an AccessMap-owned
rank-one runtime index. The frozen zero-retry B200 check compiled and launched once; all 64
outputs formed the exact per-expert old-counter permutations, invalid routes were zero and
final counters were exact. Route/group formation, runtime-indexed store and dispatch, both
grouped contractions, workspace reset, fused scatter and the multi-kernel DAG remain absent,
so complete-version coverage stays 0/57.
