# Open-CAKE-IR Weekly Progress — 2026-W35

> **Attribution correction (2026-09-06):** the historical result labeled “RoPE” below is Flash-KMeans. Its frozen Workload is `flash-kmeans-assign-independent-v2`, case `headline_b32`, and the winning entry points are `cake_flash_kmeans_assign`. The 3.533× figure must not be cited as a RoPE result. The original identifiers and evidence are preserved.

**Period:** 2026-08-24 to 2026-08-28, Asia/Shanghai

**Scope:** Compiler, Lab/evaluation, B200 evidence, AMD gfx1151 enablement, and operator-level results.

**Evidence rule:** correctness, timing, profiler observations, scientific estimates, and serving claims are reported separately.

## Executive summary

This week Open-CAKE-IR moved from primarily extending IR coverage to running a complete,
falsifiable optimization loop:

```text
Schedule
  -> Compiler and Verifier
  -> B200 correctness and timing
  -> NCU attribution
  -> lowering revision
  -> fresh confirmation
  -> correction of the static analysis itself
```

The strongest new QSA result is a correctness-valid `2.34692x` speedup for a frozen
`T=32768` operator Program relative to its per-run fixed Direct CUDA baseline. The strongest
scientific arm comparison remains the completed RoPE matched-search campaign, where the
ratio of arm medians is `3.53343x` in favor of Open-CAKE. Neither result is a model checkpoint
or serving end-to-end claim.

## Status at a glance

| Area | Result | Status |
|---|---|---|
| QSA compact top-k | `399.121 ms` vs `676.870 ms`, `1.69590x` | B200 correctness and timing passed in Compiler v30 |
| QSA tile search | tile256 `288.293 ms`, `2.34717x` | Search-valid; all three candidates passed correctness |
| QSA fresh confirmation | tile256 `288.210 ms` vs `676.407 ms`, `2.34692x` | Fresh run passed; canonical Program promotion withheld |
| RoPE matched search | arm medians `1.511927 ms` vs `5.342292 ms`, ratio `3.53343x` | 6/6 prescheduled runs qualified and archived |
| B200 semantic coverage | TinyGEMM2, KDA weighted combine, atomic reservation | Correctness passed; no performance claim |
| AMD gfx1151 | 47-source Executor v2, 47/47 Gate, 160/160 focused tests | Executor published on AMD branch; Compiler approval pending |
| NCU-aligned analysis | 65 focused tests, 54/54 Corpus Gate, 73-source closure | Compiler v31 published in this feature history; ranking calibration not authorized |

## 1. QSA: from a valid negative seed to a confirmed `2.34692x`

### 1.1 Initial qualification and attribution

The frozen QSA task uses a `T=32768` Program and a complete FP32 oracle. The first Open-CAKE
seed was correct but substantially slower than the fixed Direct CUDA baseline:

| Arm | Candidate median | Fixed baseline | Speedup | Correctness match |
|---|---:|---:|---:|---:|
| Direct CUDA negative control | `676.876 ms` | `676.902 ms` | `1.00004x` | `1.0` |
| Open-CAKE v29 seed | `7001.505 ms` | `676.396 ms` | `0.09661x` | `1.0` |

NCU localized the regression to the old `score_topk` lowering:

- 224 registers/thread;
- 1 resident CTA/SM;
- `12.46%` active warps;
- `35.33%` barrier stall;
- `39.29%` long-scoreboard stall;
- approximately 1.12 MiB generated source, 7.99 MiB PTX, and 5.10 MiB CUBIN.

This was retained as a valid negative seed rather than hidden behind a dispatcher. Evidence:
[QSA seed qualification](../inventory/QSA_GPU_INFRA_SEED_R4_20260828.json) and
[QSA NCU profile](../inventory/QSA_GPU_INFRA_PROFILE_R5_20260828.json).

### 1.2 Compiler v30 compact top-k

Compiler v30 replaced the serially expanded loop-carried top-k with a compact lowering.
The generated `score_topk` source fell to 7,310 bytes. The B200 result was:

- candidate: `399.120929 ms`;
- fixed Direct CUDA baseline: `676.870211 ms`;
- speedup: `1.6959026x`;
- correctness match fraction: `1.0`;
- baseline CV: `0.000972`;
- same-workload candidate improvement over the v29 seed: `17.54x`.

Evidence: [Compiler v30 QSA validation](../inventory/QSA_GPU_INFRA_V30_R6_20260828.json).

### 1.3 Compiler v31 tile search and fresh confirmation

The next experiment changed only the score/top-k block-loop tile. Program ABI, Workload,
oracle, warps, maxnreg, fixed Direct CUDA baseline, and CUPTI protocol remained fixed.

| Candidate | Candidate median | Fixed baseline | Speedup | Correctness match |
|---|---:|---:|---:|---:|
| tile64 | `523.540744 ms` | `676.402811 ms` | `1.291977x` | `1.0` |
| tile128 | `399.234581 ms` | `676.395184 ms` | `1.694230x` | `1.0` |
| tile256 | `288.292671 ms` | `676.670564 ms` | `2.347165x` | `1.0` |
| tile256 fresh confirmation | `288.209971 ms` | `676.407137 ms` | `2.346925x` | `1.0` |

The search/confirmation candidate-latency ratio is `1.000287`. tile256 is `1.38522x`
faster than tile128 under this task. Evidence:
[QSA tile search and confirmation](../inventory/QSA_TILE_SEARCH_R8_20260828.json).

### 1.4 NCU refuted a static-analysis assumption

The confirmed tile256 profile measured:

- 116 registers/thread;
- 1 resident CTA/SM, bound by shared memory;
- `12.44%` active warps;
- `1.88%` barrier stall;
- `2.29%` long-scoreboard stall;
- `18.85%` SM throughput.

The static model had reported 140 logical register elements as a physical hard lower bound.
The measured 116 registers/thread refutes that interpretation: backend lowering may alias or
realize logical tensors through hidden shared-memory temporaries. Static residency ordering
would also have preferred the wrong candidate; reducing loop trips dominated the tested
occupancy proxy.

Therefore tile256 is the confirmed experimental winner, but canonical Program promotion is
withheld until the field is downgraded to an uncalibrated logical-pressure proxy and every
Verifier/ranking consumer is audited. This is a product correction, not an inconvenient
measurement to ignore.

### 1.5 QSA claim boundary

The `2.34692x` result is limited to the frozen hand-authored QSA operator Program. It is not:

- a result for a pinned Qwen checkpoint;
- the Qwen model-card `10.2x` comparison;
- a matched-agent arm estimate;
- an SGLang or vLLM serving result;
- a strict matched-physical-GPU estimand.

Each run used a same-GPU fixed baseline, but search candidates were not all evaluated on the
same physical B200. Model promotion requires a pinned checkpoint/configuration, a framework
adapter, matched serving workloads, and complete request/token trajectories.

## 2. RoPE: completed matched-search evidence

The v28 RoPE campaign prescheduled three Open-CAKE and three Direct CUDA runs. All six runs
qualified, preserved protocol adherence, passed semantic replay, and retained archive and
filesystem custody.

| Repetition | Open-CAKE | Direct CUDA | Open-CAKE speedup |
|---|---:|---:|---:|
| 1 | `1.191845 ms` | `3.898164 ms` | `3.27070x` |
| 2 | `1.511927 ms` | `7.130795 ms` | `4.71636x` |
| 3 | `1.512163 ms` | `5.342292 ms` | `3.53288x` |

The arm medians are `1.511927 ms` for Open-CAKE and `5.342292 ms` for Direct CUDA; the ratio
of arm medians is `3.5334325x`. Both qualification rates are 1.0. The scientific estimand is
available, but system qualification remains unset, so this is operator-level matched-search
evidence rather than serving evidence.

Evidence: [v28 RoPE matched-search report](../evidence/campaigns/v28-rope-r1-report.json).

## 3. Compiler and B200 semantic coverage

Real workload failures drove the following minimal Compiler capabilities this week:

- deterministic top-k and explicit `tanh` implementation contracts;
- block-scale relations and runtime-valid padded extents;
- runtime-indexed access and backend preflight;
- KDA weighted-combine composition;
- stateful atomic reservation and reservation-owned indexed stores;
- RoPE dimension sub-ranges and multi-output host wrappers;
- declared-work analysis and compact loop-carried top-k;
- typed NCU-aligned static profiles with explicit `exact`, `lower_bound`, `upper_bound`,
  `uncalibrated_risk`, and `unknown` epistemic classes.

Representative B200 correctness results are intentionally not reported as performance wins:

- **TinyGEMM2 v2:** bitwise parent equality, tolerance pass, maximum absolute error
  `0.00024414`, zero fallback calls.
  [Evidence](../inventory/V25_TINYGEMM2_V2_B200_CORRECTNESS_RESULT_20260825.json)
- **KDA weighted combine:** 128 elements, zero mismatches, zero maximum deviation; no timing.
  [Evidence](../inventory/KDA_WEIGHTED_COMBINE_B200_OBSERVATION_20260825.json)
- **Atomic reservation:** 64 elements; unique old values, masked-zero behavior, and final counts
  all passed; no timing.
  [Evidence](../inventory/ATOMIC_RESERVATION_B200_OBSERVATION_20260825.json)

Compiler v31 added NCU-aligned static profiles and projected bounded diagnostics into the QSA
agent feedback path. It passed 65 focused tests and a 54/54 Corpus Gate with a 73-source
closure. These estimates do not affect candidate ranking until held-out B200 calibration
supports that use.

## 4. AMD gfx1151: evidence-safe Executor publication

The AMD line resolved an Executor identity problem before authorizing performance work. A
release planner had not seen frozen evidence outside the checkout and incorrectly considered
`open-cake-ir-gfx1151-v1` reusable. Historical v1 incarnations were registered, the colliding
47-source v1 was retained as invalid and not published, and an immutable successor was created.

The canonical AMD branch now contains `open-cake-ir-gfx1151-v2`:

- 47-source closure;
- exact gfx1151 target and wave32;
- ROCm 7.2.1 Torch/Triton environment;
- explicit build/runtime dependencies;
- rocprof and rocprofv3 admission;
- 47/47 Compiler Gate;
- 160/160 AMD/Executor focused tests;
- two expected skips awaiting independent Compiler approval;
- live-host admission with no residual GPU process.

The release is registered so future pinned-source changes must create v3; v1 and v2 cannot be
overwritten. Canonical AMD branch:
[`codex/amd-gfx1151-consolidated@c1a849a`](https://github.com/qhy991/open-cake-ir/tree/codex/amd-gfx1151-consolidated).

Formal AMD performance work remains blocked: the independent Compiler v29 review timed out,
the current approval binds an older Gate, and the RMSNorm Search Contract is not frozen. No AMD
performance result is claimed this week.

## 5. Publication state

At report freeze:

- GitHub `origin/main` is `415a1f0` and contains the v28 RoPE campaign and its published
  Compiler history.
- The AMD branch is published at `c1a849a` and remains separate from `main`.
- This weekly-report branch is based on local `e2e2865`, 17 commits ahead of `origin/main`, so
  the v30/v31/QSA code and inventories cited above travel with this reviewable feature history.
- No `main` ref is modified or merged by publishing this report branch.

## 6. Decisions and next actions

1. **Repair the QSA register-pressure semantics before further search.** Reclassify logical
   register storage as an uncalibrated pressure proxy and audit Verifier/ranking consumers.
2. **Promote only a frozen tile256 Program successor.** Rerun the exact correctness, CUPTI,
   and NCU contract after the analysis repair; do not expand into an unbounded tile sweep.
3. **Separate review surfaces.** Compiler v30/v31, QSA experimental evidence, and the analysis
   correction should remain independently reviewable even though this report branch preserves
   their coherent history.
4. **Add a real model boundary only after the operator gate.** Pin the Qwen checkpoint and
   configuration, then add the framework adapter and matched serving trajectory.
5. **Keep AMD expansion stopped at the approval gate.** Do not start DSA, AsmEvo, new AMD
   primitives, or formal RMSNorm timing before the 47-case Compiler Gate is independently
   approved and the Search Contract is frozen.

## Bottom line

Open-CAKE-IR now has a demonstrated evidence loop that can both produce operator wins and
invalidate its own analytical assumptions. This week delivered a completed `3.53343x` RoPE
matched-search result, a freshly confirmed `2.34692x` QSA Program result, several new B200
correctness closures, and an evidence-safe AMD Executor. The remaining work is not another
round of broad feature expansion: it is to repair the refuted analysis boundary, promote the
frozen QSA winner, obtain AMD approval, and qualify a pinned model-serving integration.
