# NVIDIA CAKE reproduction support

Development belongs to `nvidia`, through `task/nvidia-cake-support`. The collection
and its four preparation specifications remain in
[`experiments/flashinfer_rewrites`](../experiments/flashinfer_rewrites/README.md).
No task is promoted by the presence of a reference or a successful component probe.

## TinyGEMM task boundary

`tasks/tinygemm/reproduction.py` owns the new fixed-shape BF16+bias Workload, five
deterministic distributions, independent FP64 CPU mathematical oracle, fixed external
TinyGEMM2 peer, and complete typed candidates. It registers through `tasks/workloads.py`.
Common Evaluation receives the peer's BF16 outputs only after every element passes
the independent mathematical check. Candidate comparison is `bitwise_bf16`, with zero
tolerance. The historical materialized TinyGEMM contracts are unchanged.

The peer uses the packaged upstream CUDA and FFI header, never the installed package's
current TinyGEMM kernel. FlashInfer supplies its recorded JIT SDK and dependencies.
This is an external reference, not a candidate backend or opaque Compiler escape.
The three published shape fixtures are owned by the collection specification. The
unpublished 35/239-row matrix, PDL, original dispatch policy, faithful warp/TMA scheduling,
broader-shape performance and model serving require separate evidence.

`tools/test_cake_tinygemm_infra.py` is a development correctness judge, run only as a
frozen GPU Infra task with one exclusive B300 allocation. It checks three shapes, five
input distributions and two generated schedules, preserving complete outputs and
reporting numeric failure as invalid rather than infrastructure unknown. Explicit
`benchmark: true` enables the matched timing described below after correctness passes.
It starts no provider and issues no Campaign promotion. Collection task 029 is ready
for the integrated authoring path after the 30/30 checks; this status does not establish
mechanism equivalence or performance parity.

## Observed reduction-order gap and bounded lowering proposal

The first B300-M2 run at `9aaace45`,
`cake-tinygemm-paper-b300-9aaace45-e020ac34c970`, passed 22/30 strict peer checks.
The K3072 ordinary-input case differed in 341 BF16 elements for both pipeline depths.
Changing `num_stages` did not fix this. The source reference maintains four independent
K256 accumulators within every K1024 cycle, then combines them in a fixed order;
the initial Cake candidate had one accumulator. This is a falsifiable attribution
hypothesis, not a claim that every remaining difference has this cause.

The successor uses existing `MMA.k_ranges` and contraction-loop carry. Triton lowering
admits one aligned K256 quarter of a K1024 tile, carried across a single K loop, on
exact `sm_103a`. It passes the previous partial as the `tl.dot` accumulator. The author
explicitly writes four partials and their left-associated combination. Existing K128/
selected-K64 noncarried behavior and refusals remain intact. No operator name dispatch,
new IR primitive, new layout language or automatic performance choice is introduced.

P1–P3 retain the existing Python and canonical selected-contribution representation;
P4–P5 retain typed ranges, carry and explicit dataflow; P6 requires the unchanged full
Corpus Gate plus selected-coordinate/carry tests and wrong-target/range controls;
P7 uses the existing selected-K work accounting without changing the data model;
P8 keeps the bounded NVIDIA lowering domain explicit. GPU peer equality is a separate
gate, and no performance benefit is inferred from source generation.

Results, complete outputs, source commits, broker receipts and terminal states remain
in the external GPU Infra run directories. This document describes the task and the
proposal; it is not an alternative acceptance ledger.

## Matched official comparison

The development judge accepts explicit `benchmark: true` only on its B300 input.
Before timing, all 30 candidate checks and all 15 official-export checks must pass the
same original-peer bitwise gate, with the independent CPU mathematical check retained.
The comparison includes Open-Cake stage requests 2/4, the pinned CAKE-generated
stage4 export (selected by the original dispatcher on these three fixtures), and the
pinned TensorRT-LLM-derived reference. Short K candidates declare no loop and report
`applied_loop_stages: null`; duplicate requests there are not distinct pipelines.

Timing uses five rounds in alternating forward/reverse arm order, 25 samples per arm
per round, strict CUPTI, cold L2, no CUDA Graph and no PDL. Inputs are stateless and the
output is fully overwritten; complete output and input preservation are rechecked
after each block. Allocation, compilation, oracle work and CPU wrapper cost are outside
the GPU interval. A separate profiler observation must see exactly one CUDA kernel per
arm. Raw samples, coefficients of variation, kernel symbols and paired ratios remain
in the report. There is no predeclared parity threshold, no fixed-clock claim and no
extrapolation to the paper's 35/239-shape or model-serving results.

The K720/K1024 successor uses a loop-free masked K tile; the original single-trip
loop was refused before launch. Run `cake-tinygemm-paper-b300-8fd0d5c9-8a88723e6d54`
then passed its first 21 strict checks, including the large-shape stage4 ordinary
input. Its large-shape stage8 launch was refused for 475136 bytes of shared memory
against a 232448-byte device limit. Stage8 remains a retained failed candidate;
the next comparison uses 2/4 stages under the unchanged shapes and numerical gate.

## Measured B300 gap, 2026-09-20

Run `cake-tinygemm-compare-b300-539c6c81-a98903082f5c` completed on B300-M2 under
broker job `gpuq-1b17d8d753e4`, source `539c6c81`. All **30/30** Open-Cake output
checks and **15/15** official-export checks passed the same bitwise reference gate;
every input remained unchanged. A separate profiler observation recorded one CUDA
kernel for every timed arm. The subsequent public-build admission repair admits only
the emitter's exact pure FP32 `mov` identity, with negative checks against arbitrary
assembly; it does not change the measured kernel source.

Times below are medians of five round medians, in microseconds. Slowdown is the median
paired Open-Cake/official ratio, so it need not equal a ratio of rounded table entries.

| B / N / K | TRT-derived reference | Official CAKE stage4 | Open-Cake s2 | Open-Cake s4 | s2 slowdown vs CAKE | s4 slowdown vs CAKE |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 128 / 720 | 2.912 | 2.720 | 99.969 | 100.192 | 36.72x | 36.77x |
| 16 / 1024 / 1024 | 3.137 | 3.040 | 106.016 | 105.825 | 34.87x | 34.81x |
| 64 / 4096 / 3072 | 38.016 | 21.216 | 239.969 | 289.186 | 11.31x | 13.63x |

The official-export round-median ranges are 2.688–2.848, 3.008–3.072 and
21.088–21.248 us. Its largest within-round CV is 7.78% on the smallest shape;
the comparisons are descriptive measurements, not a declared performance-parity
acceptance. The TRT-derived arm also has cross-round outliers (up to 7.603 and
91.240 us for the latter shapes); the full raw samples remain retained, and its
ratios are not substituted for the official CAKE comparison.

This establishes a correct **numeric rewrite**, with a large measured performance gap.
The candidate expresses the four-way arithmetic but has not recreated the original
compact TMA/warp pipeline. The earlier 475136-byte stage8 refusal is a concrete resource
gap. Register spills, occupancy and memory-traffic attribution have not been measured,
so those are profiling hypotheses rather than an asserted cause of the slowdown.

The authoritative evidence is on B300-M2 under
`/mnt/b300-shared/home/qinhaiyan/workspace/aka-mechanism-remaining605-b300-20260907/state/runs/`
and the run id above: `stages/comparison/checks/report.json` contains samples and kernel
symbols, the per-shape files contain complete outputs, and `stages/comparison/result.json`
contains the judge disposition. The local summary mirror is
`/Users/haiyan-infiniai/open-cake-assessments/nvidia-539c6c81/final-report.json`.

## What is still different from the paper's system

| Family | Current Open-Cake evidence | Missing complete capability | Performance gap |
|---|---|---|---|
| TinyGEMM2 | B300 complete numeric candidate; strict peer gate; normal authoring/build entry integrated | Faithful TMA/warp roles, PDL, full dispatcher and original 35/239-shape coverage; framework validation | Measured above: roughly 11–37x slower for tested candidates |
| KDA prefill | Component probes and pinned six-shape preparation specification | Complete recurrent Workload/oracle/authoring chain; mutable-state CuTe/native lowering and cross-chunk residency | Not measured; no full candidate |
| KDA decode | Component probes and pinned 30-shape preparation specification | Complete public state/checkpoint/GQA contract and authoring/evaluation path; no equivalence from simple state-store | Not measured; no full candidate |
| Alpha-MoE | Pinned later export and four-fixture specification | Complete FP8 routed fused computation, quantization/scales and BF16 accumulation; native FP8/atomic probes are refused | Not measured; no full candidate |

These are four task families, not a percentage of the private original compiler. The
remaining three have not been rewritten and qualified; a standalone probe cannot stand
in for them. This run does not compare agent token efficiency, cost-model quality or
compiler-evolution effectiveness under a matched original-CAKE experiment.
