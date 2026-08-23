# What the IR must express, derived from real kernels

The paper derives Cake IR bottom-up from production kernels (CUTLASS, FlashInfer,
FlashAttention-4, TensorRT-LLM, DeepGEMM, Alpha-MoE, TileLang) and names what that survey
motivated: barrier choreography, warp-role partitioning, pipeline staging, TMA descriptors
and TMEM lifecycles, cluster-scoped operations (arXiv:2608.12629v1 Appendix A).

This repository skipped that step and grew its vocabulary one operator at a time. This note
is the missing survey, run against a second and independent source: kernel work actually
carried out on this machine on B200 -- the SGLang CUTLASS FP8 grouped-MoE sweep (16 paired
SGLang/CUTLASS worktrees), the SGLang MoE serving integration, and nine hand-written PTX
phases on the Llama-3.2-1B QKV projection.

The two sources disagree in an informative way, and that disagreement is the point.

## Coverage

`YES` the Schedule can state it today, `~` partially, `NO` there is no field or enum for it.
Ordered by how often the surveyed work exercised each axis.

| # | Axis | Cake IR | Where it stands |
| --- | --- | --- | --- |
| 1 | Shape/route-driven config selection, opt-in contracts | NO | one Schedule is one kernel; no dispatch predicate |
| 2 | CTA-to-tile rasterization / scheduler order | NO | `ProgramAxis` has `tile`, no traversal order |
| 3 | CTAs per SM, persistent grid size | NO | `grid` is a static triple; occupancy is derived, never committed |
| 4 | Software-pipeline depth | ~ | one `Pipeline.stages`; real work tunes mainloop, accumulator and scheduler pipelines separately |
| 5 | Register / TMEM / shared budget | ~ | TMEM and shared yes; registers derived only; no spill contract |
| 6 | Tile geometry, thread-to-output mapping | YES | `MmaInstruction.shape`, `cta_group`, `tile`, `Role.warps` |
| 7 | Global load width and cache policy | NO | `LoadParameters` is `movement` + `descriptor_box` |
| 8 | Buffer ownership, intermediate-copy elision | NO | buffers are input/output/scratch; no caller-provided destination |
| 9 | On-chip staging and bank swizzle | ~ | `Buffer.swizzle` covers four CUTLASS modes, not a hand-derived layout |
| 10 | Scheduler-metadata precomputation | NO | no work-tile table |
| 11 | ILP, independent accumulator count | NO | `RangeOptions` carries Triton loop knobs, not an accumulator count |
| 12 | Split-K reduction decomposition | YES | `reduce_sum(axis, scope)` plus roles |
| 13 | Two kernel configs in one dispatch | NO | one Schedule is one kernel |
| 14 | Weight precision and scale granularity | NO | `fp8_e4m3` exists; scale tensors and group size do not |
| 15 | GEMM orientation (transposed decode) | ~ | buffers can be declared transposed; scale-config major mode cannot |

Two of fifteen fully expressible. Of the five most-exercised axes, four are `NO` or `~`.

## The constraint neither survey would have found alone

Seven or more of the MoE variants are shaped by one fact:

> Problem sizes are device-resident during CUDA-graph replay, so validating them on the
> host would introduce a synchronization.

That single constraint is why the work reaches for device-side masking of the problem list,
a device flag with both kernels' parameters resident, an environment-variable contract
pushed to the deployer, and a precomputed work-tile table -- rather than the host-side
branch each of those replaces.

Cake IR cannot say that a fact is device-resident, and therefore cannot reject a Schedule
that would require reading it on the host. This is a verifiable property and squarely the
verifier's business, and it does not appear in the paper's own list because the paper's
corpus is standalone kernels rather than a captured serving path.

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
5. Device residency as a declared, verifiable property.

**Does not belong to one Schedule**

Axes 1, 10 and 13 are dispatch and portfolio concerns, which the paper places in its
generalization stage and gates behind strong per-shape seeds. This repository's portfolio
is exact-shape specialists only. They should not be pushed into the Schedule vocabulary.

**Needs a decision before it is designed**

Axis 8 sits against P1, which asks to avoid unnecessary destination-passing. The MoE
integration uses destination-passing deliberately, to write into a symmetric collective
buffer and delete a copy per rank. "Unnecessary" is doing real work in that principle and
the boundary should be drawn explicitly rather than assumed.

Axis 14 needs a quantization model -- scale tensors, group granularity, where scales live
relative to the weights -- which is a family the corpus does not cover at all.

## Corpus coverage

Eight cases across three families, against the paper's roughly four hundred cases across
twenty-eight. Attention and MoE, which are what the surveyed work is actually about, have
no representation here at all.
