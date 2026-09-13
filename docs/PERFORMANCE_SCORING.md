# Task efficiency scoring

Status: implemented in the prospective successor, 2026-09-13; release acceptance
is recorded by the Compiler/Executor authorities. The owner selected useful-work
efficiency as the primary performance score, with thresholds calibrated per task.
The first version supplies a fixed task-IO reference score. Actual memory traffic,
arithmetic efficiency and complete roofline coverage remain explicitly unavailable.
Historical Studies and experiments do not acquire this policy retroactively.

## Use

New tasks created by `tools/launch_task.py` declare
`analysis_plan.performance_reporting = "task_efficiency_v1"` and write their complete
audited result to the external task workspace's `report.json`. The console displays
the primary score and coverage. All existing task/backend/model/harness/effort and
Python-frontend authoring arguments remain supported.

The report's `descriptive.performance.rows` contains stable, correctness-passing
confirmations from adhered, custody-verified, semantically replayed Runs. Candidate
and baseline rows use identical task bytes; rows are ordered by increasing latency,
equivalently decreasing reference efficiency within that exact task and assay.
The primary metric is `logical_task_bandwidth_reference_pct`; its status is
`reference_ratio`, or `unavailable` when the target lacks a rate. An unqualified Run
does not receive a score. Threshold status is `not_calibrated_no_hard_efficiency_threshold`.

Recompute an opted-in workspace's report with its exact frozen source checkout:

```bash
python tools/report_task_efficiency.py \
  --project-root /absolute/path/to/exact/released/checkout \
  --workspace /absolute/external/task-workspace
```

The command prints JSON. Optional `--output /new/external/report.json` refuses an
existing path. It reuses `TaskLab.audit`, verifies the declared policy and does not
rewrite old evidence, infer a policy for an older Campaign, or execute GPU/provider
work. A complete arithmetic score and per-task calibrated thresholds are not claimed
by this version. In particular, a compute-heavy task's small bandwidth reference ratio
is not an overall hardware-efficiency grade.

## Current behavior

At source commit `d7d7bc7b`, Lab promotion chooses the lowest eligible confirmed
latency in `lab/reporting.py:_promoted_artifact`. Correctness, measurement quality,
custody and semantic replay remain prerequisites. Fixed-baseline speedup describes
improvement, but does not establish proximity to hardware capability.

The Compiler already exposes `performance.utilization.utilization` and
`roofline_seconds`, and `tools/profile_lowered_kernel.py` reports their results.
They use Schedule-declared work, not a candidate-independent Workload work count.
All five current Target files omit `peak`; arithmetic and bandwidth fractions are
therefore unavailable. CUDA profiler throughput counters are diagnostic observations,
not the common promotion score. Metal currently collects compute-stage timestamps,
explicitly omits traffic/instruction counters, and times repeated dispatches under
`warm_no_explicit_flush`.

## Meaning of the score

Fix the task, input case, precision, semantics and timing boundary before comparing
Candidates. Let `F_task` be the justified useful arithmetic under a preregistered
algorithmic convention, `B_task,L` the justified necessary bytes at memory level L,
`t` the qualified measured time, `P` the matching arithmetic rate, and `BW_L` the
matching memory-level rate. Candidate source must not choose these numerators.

- Compute efficiency: `F_task / (t * P)`.
- Effective bandwidth: `B_task,L / t`; bandwidth efficiency:
  `B_task,L / (t * BW_L)`.
- When both lower-bound terms apply to the same task and timing scope, roofline
  efficiency: `max(F_task / P, B_task,L / BW_L) / t`. Report the included resource
  domains and missing terms. A memory-only bound is not complete roofline coverage
  of a compute-heavy task.

Under fixed task counts and peaks these scores are proportional to `1/t`; selecting
the fastest qualified Candidate already selects the highest score. The improvement
is an absolute reference and honest coverage, not a different mathematical ranking
of the same fixed task. Compute-bound tasks should not be penalized for low bandwidth
use, nor bandwidth-bound tasks for low arithmetic use. Do not average the two ratios.
Small, cache-resident, launch-limited, transcendental and dependency-limited cases
may need additional terms or a different declared regime; report partial coverage.

Keep raw latency and fixed-baseline speedup alongside the primary score. Keep
correctness, side-effect checks, measurement stability, profiler evidence and
common confirmatory Evaluation as mandatory acceptance conditions. No universal
utilization threshold is introduced.

## Traffic and busy counters are separate diagnostics

`B_task,L / B_actual,L` is a traffic-efficiency diagnostic only when both quantities
refer to the same memory level, interval, read/write convention and cache protocol.
Its denominator requires actual retained traffic counters. Static counts of issued
loads, logical tensor bytes, cache transactions and DRAM bytes are different facts.
An implementation can issue fewer loads yet move similar DRAM bytes due to caching.

If a well-defined minimum and observed traffic are available, report redundancy
`B_actual,L / B_task,L`. If counters are absent, report unavailable, not zero and not
a static estimate labelled as a measurement. This diagnostic alone cannot score
kernel quality: a very slow kernel can move only the necessary bytes. Similarly,
high hardware busy percentages can reflect redundant work.

For warm repeated dispatches, task input bytes need not cross DRAM on every dispatch.
`logical task bytes / t / advertised DRAM bandwidth` may be shown as a labelled
reference ratio, but is not observed DRAM utilization or a certified roofline floor.
Do not cap a suspicious ratio at 100%; first check memory level, timing, work scope
and denominator applicability. `1 - efficiency` is not a fraction of elapsed time
spent outside memory transfers, because resource work overlaps and dependencies,
latency and other bottlenecks are not represented by that subtraction.

## Verified hardware references

Rates below use decimal bytes per second. They are advertised specification
references, not measured sustainable throughput or GPU-exclusive allocations.
Keep the exact device/variant and source qualification attached to each reference.

| Exact target | Reference bandwidth | Primary source and qualification |
| --- | --- | --- |
| Apple M1 Pro / family7 | up to 200 GB/s | [Apple M1 Pro announcement](https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/), SoC unified memory shared by clients |
| Apple M4 / family9 | 120 GB/s | [M4 MacBook Pro announcement](https://www.apple.com/newsroom/2024/10/new-macbook-pro-features-m4-family-of-chips-and-apple-intelligence/), base M4 configuration; do not apply to Pro/Max |
| NVIDIA B200 SXM / sm_100a | up to 8 TB/s | [NVIDIA HGX AI Factory components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html), per-GPU row; [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) corroborates 64 TB/s across eight GPUs |
| NVIDIA B300 SXM / sm_103a | up to 8 TB/s | [NVIDIA HGX AI Factory components](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html), explicitly B300 SXM per-GPU bandwidth, not an inherited B200 value |

The checked Apple announcements do not establish an FP32 FLOP/s ceiling. GPU core
count is not sufficient to infer one. The existing Target peak model admits measured
microbenchmark references as well as specifications, but arithmetic rates are keyed
by declared instruction contracts. Adding an arbitrary contract string merely to
attach a number does not establish the corresponding hardware behavior. A future
Metal arithmetic reference needs a modeled execution-unit/precision contract and
matched on-device calibration. Keep arithmetic coverage absent until then.

The [HGX specification table](https://www.nvidia.com/en-us/data-center/hgx/) describes
eight-GPU systems and marks Tensor Core sparse versus dense values. Divide aggregate
rates by eight and apply the row's own dense/sparse footnote. FP32 scalar/SIMT,
TF32, BF16 and Tensor Core modes are not interchangeable. Do not copy ambiguous
INT8 or combined FP64/FP64 Tensor Core rows into a differently named metric.

## GEMM-SiLU example and scope

[F-2026-09-11-004](../findings/2026-09-11-004-metal-output-column-specialization.json)
retains a confirmation with baseline 1.878596029 ms and Candidate 0.079588216 ms.
For the stated FP32 `M=128, K=256, N=32` task, reading A, B and bias once and writing
the output once gives `4 * (M*K + K*N + N + M*N) = 180352` logical bytes.
Using that same convention for both implementations gives:

| Implementation | Logical effective GB/s | Ratio to 200 GB/s |
| --- | ---: | ---: |
| Baseline | about 0.0960 | about 0.0480% |
| Candidate | about 2.2661 | about 1.1330% |

This is a re-expression of the retained timing, not a new measurement, not physical
traffic, and not a measured DRAM utilization. Their ratio is still about 23.604.
Using a different implementation-dependent byte count for the baseline would erase
the fixed-task interpretation. The 0.902 microsecond reference memory term does not
explain the remaining time or prove the shape is only useful as a smoke test. The
Finding supports a private-state/scheduling mechanism; it does not contain a measured
24.07-fold DRAM-traffic reduction. Such a causal claim needs separate profiling.

## Implementation and calibration sequence

1. Attach the verified bandwidth references through a reviewed Compiler successor,
   using existing Target citations and peak provenance. Preserve missing arithmetic
   coverage. Inspect all existing utilization consumers before exposing these peaks:
   their Schedule work bounds do not prove per-dispatch DRAM traffic under warm-cache
   timing. A new hard refutation must not arise just from plugging a peak into an
   inapplicable byte model.
2. Put useful-work conventions in the existing task/Workload owner, reference rates
   in the Target/calibration owner, scoring policy in Study, and qualified measurement
   projections in Lab reporting. Do not add a second campaign engine or experience
   ledger. Reports show metric status, units, numerator scope, denominator source,
   cache/timing scope and missing coverage before a numerical rank.
3. Report qualified Candidates by primary applicable efficiency within a fixed task;
   keep uncalibrated or partially covered tasks explicitly unranked across tasks.
   Display useful arithmetic, effective bandwidth, roofline coverage, latency and
   speedup. Add hardware throughput/traffic efficiency only when actually observed.
4. Use Python-front-end tasks and the existing Lab matrix for a preregistered shape
   sweep. Include real deployment-small cases, cache-resident throughput cases,
   sufficiently large working sets for memory-throughput evaluation, and tails.
   Sweep concurrency separately from reduction depth/width. Preserve current small
   cases as controls; do not discard them because their utilization is low. Freeze
   each task baseline and distinguish Compiler-floor movement from provider headroom.
5. Calibrate task-specific score thresholds only after the relevant regime reaches
   stable measurements. Validate positive cases, null/negative results, cache effects,
   redundant-compute/traffic counterexamples, wrong precision/device references and
   unmeasured denominators. On-device evaluation and independent release review are
   required before claiming the new scoring is active or calibrated.

The main checkout had unrelated uncommitted Compiler pass changes when this analysis
began. This proposal lives in a separate worktree; those changes and historical
evidence were not modified. No new provider or GPU experiment has been run here.
