# Explicit execution-group selection on gfx938

The existing `specialize_triton_warps` pass now admits the exact gfx938 route,
ordinary fixed LOAD/MMA loops, unchanged FP16/BF16-to-FP32 widening, and retained
runtime output-validity masks. It changes only width and result identity. It
does not choose a width or promise bitwise dot/reduction equivalence. The exact
contract and refusals remain in [CTA width documentation](TRITON_CTA_WIDTH.md).

The clean shared implementation passed independent review, 2667 full contracts
(26 skips), Python 3.10/3.11/3.12 CI and both offline compilation jobs. Original
193 DCU Corpus cases and 106 source snapshots remain unchanged.

## Bounded original-Task qualification

The owning external root is
`/data3/testuser01/experiments/bw1100-compiler-version-study-20261005` on
`bw1100-1`. No old frozen Run was changed.

| qualification directory | measured Compiler | actual mapping | original correctness | caller checks |
|---|---|---|---|---|
| `qualification/groups2` | `7f6ab973` | GateUp four small-M dual kernels: 4→2 groups | 160/160 | 4/4 |
| `qualification/groups8` | `7f6ab973` | Same four kernels: 4→8 groups | 160/160 | 4/4 |
| `qualification/rms4` | `7f6ab973` | RMSNorm original nine emissions: explicit width4 where different | 160/160 | 4/4 |
| `qualification/moe4` | `9187a3b0` | Strict FP32 MoE 48 emissions: width4, widening and runtime prefix preserved | 160/160 | 4/4 |

The two pins differ at the pass's validity-mask admission boundary, not at
implicit lowering. Earlier GateUp/RMS evidence does not stand in for the later
MoE domain: the latter has its own new-pin full device result. Original
Workloads, wrappers, baseline adapters, oracle, tolerances, dtype/rounding and
fixed DTK image remain unchanged. Device admissions ended normally with HCU
VRAM0%, no task KFD context and no remaining container.

## Null and negative mapping evidence

Separate complete-call, both-order/A-A GateUp timing covers all16 original cells.
Against the same fixed FlagGems GELU denominator, width2 has conservative
geometric ratio **0.98203×**, minimum **0.26831×**; width8 has **1.99602×**,
minimum **1.37521×**. Inputs remain unchanged. These were separate allocations;
there is no fresh same-allocation width4 head-to-head, so do not attribute a
precise marginal effect by subtracting an earlier width4 score.

Direct M128 emission profiling validates five actual target rows per mapping.
Width2 reports workgroup128, LDS16384, VGPR256, SGPR32 and `scr=1808`. Width8
reports workgroup512, LDS32768, VGPR140, SGPR32 and `scr=0`. Each dispatch has
192 CTAs and wave64. Installed skill `load_rows` validates grid/wave counts.
The resource/performance association motivates mapping exploration. It does not
establish exact occupancy, total traffic, bandwidth or a single bottleneck.
No WRITE_SIZE was collected, and profile durations are not scores.

Promotion: retain the guarded explicit tool and negative/valid mapping evidence.
Do not promote width2, width8 or another fixed width as a default optimizer.

## Same-time Compiler–kernel pilot

`COMPARISON.json` assigns three Tasks (RMSNorm, GateUp/GELU and strict-FP32 MoE)
to base `76a937be` and successor `9187a3b0`: six independent Ralph contexts,
Claude/GLM-5.3 high, each3h with 165min search and15min handoff. Both versions
start from the exact same strong seed, community baseline, precision/caller
checks, image, reference access and engine. All71 initial emitted seed sources
are byte-identical across versions; this comparison measures explicit-tool
access and subsequent search, not an initial implicit-codegen improvement.

Checkpoints30/60/120/165/180 and first robust improvement over the seed derive
from immutable qualified outcomes. One final nomination per arm is confirmed
on common HCU1 with the same original full gate/caller/paired-A-A protocol;
no further search occurs during confirmation. Missing, failed and no-win
results remain in the report. Source semantics and real profile attribution
still require owner review before promotion.

This is a one-run-per-cell engineering pilot on a shared host. OS author
isolation and external physical GPU exclusivity are not established. It is not
a replicated causal Study or an E/P ablation. The shared initializer, separate
contexts, trace review and common-card confirmation reduce known confounds;
they do not remove model randomness or prove general Compiler improvement.

`HISTORY.json` and `HISTORY.md` summarize50 retained benchmark search endpoints,
six prior Compiler versions, inherited/fresh starts and known exclusions.
They keep historical search scores separate from later independent confirmation.
Five launchers failed at a missing fixed plan input before intake; the immutable
input projection was supplied and those unstarted entries recovered. Their
failure records remain in `PRE-INTAKE-RECOVERIES.json`. The already-running
GateUp base was preserved. No rolling batch or heartbeat was started.
