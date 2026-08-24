# The static analysis, measured

The paper's harness reports a performance analysis as a *report*, and says plainly that
on-device measurement remains the ground truth. This repository had the report and none of
the measurement, so nothing established whether the bounds it prints are true.

Measured with Nsight Compute on a B200, exclusive, by `tools/profile_lowered_kernel.py`.
The raw records are in `evidence/calibration/residency-b200-*.json`.

| | predicted | measured | |
| --- | --- | --- | --- |
| **rmsnorm**, Triton | | | |
| registers per thread | >= 66 | 96 | lower bound holds |
| resident CTAs | <= 7 | 5 | upper bound holds, loose by 2 |
| binding resource | registers | registers | correct, and measured uniquely |
| **softmax**, Triton | | | |
| registers per thread | >= 65 | 96 | lower bound holds |
| resident CTAs | <= 7 | 5 | upper bound holds, loose by 2 |
| binding resource | registers | registers | correct, and measured uniquely |
| **layernorm**, Triton | | | |
| registers per thread | >= 66 | 96 | lower bound holds |
| resident CTAs | <= 7 | 5 | upper bound holds, loose by 2 |
| binding resource | registers | registers | correct, and measured uniquely |
| **gemm + bias**, Triton | | | |
| registers per thread | >= 64 | 81 | lower bound holds |
| resident CTAs | <= 8 | 5 | upper bound holds, loose by 3 |
| binding resource | registers | registers *and* shared memory | correct, but the measurement ties |
| **flash-kmeans assignment**, CuTe-DSL | | | |
| registers per thread | >= 1 | 99 | holds, but says nothing |
| resident CTAs | <= 2 | 2 | upper bound holds, and is exact |
| binding resource | shared memory | registers *and* shared memory | correct, but the measurement ties |

## What this establishes

**Both bounds are sound in the direction claimed, on five kernels across both backends.**
The register figure is a lower bound on storage and was below the measured allocation every
time; the residency figure is an upper bound on resident CTAs and was at or above the
measured limit every time. Nothing here contradicts the analysis, which is not a given --
the analysis was renamed to say "bound" only after it was written.

**The exact quantities are exact and the estimated one is loose, as claimed.** Shared memory
is an explicit allocation the Schedule declares, and its bound came out equal to the
measurement. Registers are inferred from declared buffers with liveness and aliasing, and
that bound runs 21% to 31% low on the four Triton kernels and is vacuous on the fifth.

**The attribution is weaker than it looked.** This is the half the paper calls attribution
and the half an author can act on -- being told that registers rather than shared memory
limits residency points at which declaration to change. On three of the five kernels the measurement
singles out one resource and the prediction names it. On the other two it does not:
flash-kmeans is limited to 2 CTAs by registers and shared memory alike, and the GEMM to 5
by both, so naming one is correct in the sense of naming a member and discriminates nothing.
An earlier run of the flash-kmeans row, made by a script outside the repository, recorded it
as a clean win; it was a tie, and the instrument is how that came to light. The ties are not
a property of one backend -- one is CuTe-DSL and one is Triton.

**The GEMM tie is the sharper finding, because the analysis could not have seen it.** That
Schedule declares no shared memory at all. Triton allocates it anyway, for the `tl.dot`
operands, and it bounds residency exactly as tightly as the registers the analysis does
model. So the prediction named a binding resource while being structurally blind to the
one beside it. This is the register limit again -- a bound derived from declarations cannot
see what a backend needs on its own behalf -- and it is worse here, because registers are
under-counted by about 31% while this shared memory is under-counted by all of it.

That earlier run also read the register figures differently -- 95 rather than 96 on rmsnorm,
and 80 rather than 99 on flash-kmeans. Which metric it sampled cannot be recovered, because
the script is gone. The figures above are `launch__registers_per_thread` and can be
re-derived by anyone with the repository and a B200.

## What it exposes

**The register bound says nothing about a schedule that keeps its data off-register.**
`flash-kmeans-assignment-full` declares almost no register buffers -- its staging lives in
shared and tensor memory -- so the bound is 1 against a measured 99. It did not mislead here
because shared memory bounds residency anyway and the report said so. But a schedule that is
genuinely register-bound while declaring few register buffers would get a residency ceiling
far above the truth, and the report would name the wrong resource.

That is a real limit of an analysis that reasons only over declarations: it cannot see the
registers a backend needs for addressing, predication and staging, and those were 30 and 98
registers per thread across these five kernels. Worth recording rather than filing as a defect,
because closing it means either measuring or modelling the backend, and the paper puts
measurement at the end of the loop for exactly this reason.

## The ranking, measured

The ranking hypothesis was first tested the same way: nine tilings
of the Flash-KMeans schedule, all nine correct, timed as a median of twenty runs with the
cache dropped between them.

Twenty-nine of thirty-six pairs came out in the predicted order -- meaningfully better than
the eighteen a coin would give, and well short of an oracle. The shape of the error matters
more than the number. The model put the true best second and the true worst last, and its
first and second choices were the measurement's second and first. It separates good from
bad and does not resolve fine distinctions.

That appeared to be the right capability for the stage it serves. A pre-GPU filter has to keep the
winner in the surviving set, not name it; the paper leaves naming to measurement. Here the
true best survives a cut at k=2. What must not be done with this number is to treat the
order as a result -- reporting a ranked list as though position three were meaningfully
worse than position two would be reading precision the model does not have.

One thing the run exposed about the model rather than the kernels: every candidate at this
shape fits in a single wave, so the wave term did no work and the whole order came from how
much of the device the grid fills. A workload with more tiles than the device holds would
exercise the term that is actually about tail waste. That measurement is below.

## The wave term, measured -- and removed

`Cost.order` sorted on wave count first, on the reasoning that a wave is a round of the
whole device and a partial final round is a round of dead time. Nothing had tested it.

`tools/calibrate_wave_term.py` sweeps one Schedule's batch extent with the tile fixed, so
per-CTA work is constant and the only thing that changes is how the grid quantises against
the device. 171 points, 60 to 400 batches, median of 41 samples with L2 flushed between
them, exclusive B200, compiler `open-cake-ir-sm100a-v4`.

Under the wave model, latency across four predicted wave bands should be four flat levels.
Measured, it is a straight line:

```
latency_us = 21.73 + 9.514 * (ctas / 1000)     R^2 = 0.9728, residual rms 1.26us
```

The staircase is not there. At the model's own boundary -- seven resident CTAs, 1036 per
round -- a wave step would have to be 9.86us; the measured jump across it is +0.14us,
well under the 0.61us typical jump between any two adjacent points. And it is not that the
boundary was merely in the wrong place: every wave size from one to twelve resident CTAs
was tested against its own required step height, and none came close.

| resident | per round | step the model requires | step measured |
| --- | --- | --- | --- |
| 1 | 148 | 1.41us | +0.26us |
| 4 | 592 | 5.63us | +0.13us |
| 5 | 740 | 7.04us | +0.66us |
| 7 | 1036 | 9.86us | +0.14us |
| 8 | 1184 | 11.27us | -0.05us |
| 10 | 1480 | 14.08us | +0.18us |

The largest mean jump at any wave size in the sweep is +0.70us, where that size's own step
would have to be 16.90us. Two of the twelve gave a *negative* mean jump at their
boundaries. This is noise, which is the point: there is no signal to find.

**What changed.** `waves` and `last_wave_occupancy` are gone from `Cost`, and `cost()`
returns None for a grid that overfills the device. Wave count was the only term that
separated candidates past one round, so past one round the model now declines rather than
ordering on a refuted basis. What is left is device fill, which the nine-tiling run above
did test. The model got smaller and its domain got explicit.

Declining costs real capability, and the sweep shows how much: of its 171 grids the model
now ranks 35 and declines the rest, the boundary falling exactly where capacity does. That is the honest state. A confident wrong order is worse than
none, and it is worse now than it was before: the loop routes a ranking that disagrees
with measurement to the cost model, so a refuted term would have generated a steady stream
of diagnoses that were really this term firing.

**What this does not establish.** Wave quantisation is a real effect on kernels whose CTAs
run long enough that a partial final round is a full round of wall time. This kernel's CTAs
are short and the hardware streams them, and one kernel on one device is the whole sample.
The claim here is narrow and it is about the model, not the hardware: the term was in the
sort key without evidence, and the one direct test of it refuted it. Restoring it takes a
measurement showing the staircase, not an argument that it should be there -- timing-model
coverage is per-target evidence and is never inherited.

## The ranking on a second kernel, which disagrees

The nine-tiling run above is Flash-KMeans. Everything the model claims rests on it, so the
same instrument was pointed at RMSNorm: `tools/calibrate_ranking_at_scale.py` builds one
workload's candidate set the way the ranking's caller does -- five row tiles, five register
budgets, two role widths -- checks every candidate against a float32 reference, times it,
and scores each order key by what a top-k cut would cost. 37 correct candidates, 13 refused
by the gates, at three workload scales.

Concordance is the wrong measure here and the run showed why: 24 of the 37 candidates at
batch 512 are within 6% of the best. The optimum is a plateau with a few cliffs, so a
ranking's job is to miss the cliffs, not to find the peak. What follows is the penalty of
the best candidate a top-k cut keeps, against the penalty of picking blind.

| workload | device fill | in the model's domain | shipped key at k=1 | blind pick |
| --- | --- | --- | --- | --- |
| batch 16 | 0.12 .. 0.43 | 37 of 37 | **+1.6%** | +3.7% |
| batch 64 | 0.46 .. 1.73 | 32 of 37 | **+4.5%** | +3.7% |
| batch 512 | 3.69 .. 13.84 | 0 of 37 | declines | +9.2% |

**On this kernel the ranking is not a filter.** It beats a blind pick at one scale by two
points and loses to one at the next by one, on a kernel whose whole in-domain spread is
under 22%. The Flash-KMeans result -- twenty-nine of thirty-six pairs, true best surviving
k=2 -- does not carry to RMSNorm. At this point the honest reading was that the model had
one narrow kernel result rather than a general capability; the declared-domain successor
below withdraws even that profile-level interpretation.

Nothing here says which key to use instead, and that is deliberate. `+fill` -- preferring
the emptier device -- wins at batch 64 and 512 and loses at batch 16. Adopting it would fit
this kernel and break the one the model was built on, which is how the wave term got in.

**The decline was load-bearing.** At batch 512 the model refuses to rank, on the wave-term
evidence alone. Had it instead extended its own key past saturation, `-device_fill` would
have selected the single worst candidate of the 37, at +40.9%. That was not the argument
for declining and it is not why the boundary is there, but it is the strongest evidence for
it: the term stops working at exactly the point the model stops claiming.

**What this changes about running the loop.** A pre-GPU filter this weak makes the
`searches_per_turn` above one worth its GPU time rather than a luxury, and it means a
`cost_model` diagnosis is the expected outcome on kernels like this rather than a signal
that something broke.

It also sets a bar. With 24 of 37 candidates inside 6% of the best, routing every inversion
to the cost model would report mostly measurement error, so a Study that searches more than
one candidate must declare `search_materiality_ratio` -- how much faster the measurement has
to be before the order counts as wrong. The faster candidate is carried forward either way;
what the ratio decides is whether a claim gets made about the model.

## The declared-domain check withdraws the remaining coverage claim

The original Flash-KMeans result varied nine tilings. It did not cover the register budget
and role width choices the actual candidate vocabulary exposes. The successor instrument
therefore declares one fixed domain per profile before measurement: three tiles, five
register budgets and two warp counts at global extent 512. Every one of the 30 candidates
is either compiler-refused or externally checked against the float32 oracle; every correct
survivor retains 41 cold-L2 timing samples.

| profile | correct measured | compiler-refused | device-fill top-1 regret | top-4 regret |
| --- | ---: | ---: | ---: | ---: |
| GEMM + bias | 25 | 5 | **1.72%** | 0.46% |
| Flash-KMeans | 16 | 14 | **19.43%** | 0.00% |

Pairwise concordance is not the release criterion; top-k survivor quality is. Device fill
is coarse enough that many candidates tie, and the deterministic `schedule_id` tie-break
does not carry performance meaning. A k=4 search recovers near-best candidates, but that is
evidence for spending four GPU searches, not for pruning to one candidate before
measurement.

Flash-KMeans is direct negative evidence for a k=1 filter. The GEMM sweep is encouraging,
but it is only one non-preregistered run: there was no acceptance threshold or independent
repeat fixed before seeing it. It is insufficient evidence for promotion, not evidence
that the hypothesis fails on GEMM. A successor calibration must declare those two facts
before measuring rather than deriving a pass rule from this result.

Therefore Compiler v8 keeps `calibration_coverage` empty. `Compiler.rank` returns every
otherwise eligible candidate as withheld, preserving provider order in the Lab rather than
publishing an uncalibrated order. The structural cost function remains as an explicitly
dormant hypothesis so this instrument can test it; only a later reviewed Revision with a
predeclared profile domain and useful top-k behavior may expose it publicly.

## The inner-loop profile is a different observation

`profile_lowered_kernel.py` above calibrates a Compiler report. The matched Lab's
`attribution_evaluation` instead explains the exact sealed Candidate that just passed
confirmatory Evaluation. It uses the same canonical worker and external oracle, but a
separate NCU launch with timing structurally absent. The raw long-form CSV is the
observation; the 11 retained metrics and feedback summary are projections that replay
recomputes and can reject.

The bounded B200 mechanism check used the frozen, previously correctness-qualified Triton
`b32_smoke` CUBIN. Its profiled launch again produced 16,384/16,384 exact assignments,
one target-kernel call, zero fallback calls and zero timing samples. The targeted profile
reported 163 registers/thread and a two-CTA residency limit bound by shared memory; the
separate full report measured 83,208 bytes/CTA. SM, DRAM and L2 throughput were 7.22%,
3.37% and 2.25%; active warps were 2.45%; long-scoreboard and barrier stalls were 37.36%
and 1.41%.

The full report also shows only 64 CTAs for 148 SMs (`launch__waves_per_multiprocessor =
0.216`). The two-CTA shared-memory limit is therefore a theoretical occupancy bound, not
the realized concurrency limit at this shape: reducing shared memory alone cannot fill an
SM that has no CTA. The first actionable hypothesis is a finer token grid, measured against
the extra centroid traffic it creates. The second is improving load/compute overlap: source
sampling placed
426 of 502 long-scoreboard samples at `lowered.py:62`; six unavailable CTC metrics and four
requested PM throughput series are reported as missing, not passed. These observations do
not prove either change is faster.

The complete external diagnostic root is
`/home/qinhaiyan/open-cake-ir-evidence/ncu-attribution-20260824-v1`. This establishes the
evaluator/profile/replay seam for one frozen Candidate and shape. It does not measure
profiler-free latency, current Compiler performance, every search survivor, or a paper
Campaign.

The first mechanism observation used working-v9 precursor digest `52fb920a…`. After the
profiled-launch correctness invariant was made fail-closed, canonical attempt 10 repeated
the assay under released v9 digest `b1c30a16…`: 16,384/16,384 exact assignments, one
target launch, zero fallback and zero timing, with the raw CSV and complete Evaluation
Receipt replayed under the released source closure. Its signals were 7.17% SM, 3.35% DRAM,
2.24% L2, 2.44% active warps, 37.46% long scoreboard and 1.36% barrier stalls; the small
difference from the precursor profile does not change the bounded diagnosis.

Executor v10 then repaired only the Claim Scope → provider-qualification composition
boundary. Canonical attempt 12 nevertheless repeated the assay rather than inheriting v9's
result: current digest `23a2c79f…` again produced 16,384/16,384 exact assignments, one
target launch, zero fallback and zero timing, and its complete receipt replayed. The
signals were 7.17% SM, 3.36% DRAM, 2.23% L2, 2.44% active warps, 37.44% long scoreboard
and 1.37% barrier stalls. Attempt 11 never started the worker because the broker service
could not traverse the temporary checkout chosen as its cwd; that is retained as a
zero-work harness-staging failure, not retried under the same evidence identity.

Earlier attempts 8/9 were manual invocations that omitted the runtime configuration's
frozen `GPUQ_JOB_ID` environment argument. They rejected before inspecting card processes,
so their all-zero counters establish fail-closed broker-environment handling, not dirty
cards. Attempts 6/7 also did zero work, but their retained results do not support a narrower
cause. The correction and exact artifact hashes are in the external diagnostic report.


## Reproducing

`docs/RUNBOOK.md` covers the broker. The profiling job runs exclusive per the shared-GPU
policy, and Nsight Compute writes its `==PROF==` preamble into stdout ahead of the CSV, so
a parser has to seek the header rather than reading from line one.
