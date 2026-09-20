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
performance and model serving require separate evidence.

`tools/test_cake_tinygemm_infra.py` is a development correctness judge, run only as a
frozen GPU Infra task with one exclusive B300 allocation. It checks three shapes, five
input distributions and two generated schedules, preserving complete outputs and
reporting numeric failure as invalid rather than infrastructure unknown. It starts no
provider, measures no latency and issues no Campaign promotion. The collection keeps
029 blocked until external correctness and mechanism qualification are complete.

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
The comparison includes Open-Cake stage requests 4/8, the pinned CAKE-generated
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
