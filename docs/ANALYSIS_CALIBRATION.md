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
| **flash-kmeans assignment**, CuTe-DSL | | | |
| registers per thread | >= 1 | 99 | holds, but says nothing |
| resident CTAs | <= 2 | 2 | upper bound holds, and is exact |
| binding resource | shared memory | registers *and* shared memory | correct, but the measurement ties |

## What this establishes

**Both bounds are sound in the direction claimed, on three kernels across both backends.**
The register figure is a lower bound on storage and was below the measured allocation every
time; the residency figure is an upper bound on resident CTAs and was at or above the
measured limit every time. Nothing here contradicts the analysis, which is not a given --
the analysis was renamed to say "bound" only after it was written.

**The exact quantities are exact and the estimated one is loose, as claimed.** Shared memory
is an explicit allocation the Schedule declares, and its bound came out equal to the
measurement. Registers are inferred from declared buffers with liveness and aliasing, and
that bound is 31% low on the two Triton kernels and vacuous on the third.

**The attribution is weaker than it looked.** This is the half the paper calls attribution
and the half an author can act on -- being told that registers rather than shared memory
limits residency points at which declaration to change. On the two Triton kernels the
measurement singles out one resource and the prediction names it. On flash-kmeans it does
not: registers and shared memory both limit to 2, so "shared memory" is correct in the sense
that it is one of the two, and it discriminated nothing. An earlier run of this measurement,
made by a script outside the repository, recorded that row as a clean win; it was a tie, and
the instrument is how that came to light.

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
registers per thread in these kernels. Worth recording rather than filing as a defect,
because closing it means either measuring or modelling the backend, and the paper puts
measurement at the end of the loop for exactly this reason.

## The ranking, measured

The ranking that filters candidates before GPU time was tested the same way: nine tilings
of the Flash-KMeans schedule, all nine correct, timed as a median of twenty runs with the
cache dropped between them.

Twenty-nine of thirty-six pairs came out in the predicted order -- meaningfully better than
the eighteen a coin would give, and well short of an oracle. The shape of the error matters
more than the number. The model put the true best second and the true worst last, and its
first and second choices were the measurement's second and first. It separates good from
bad and does not resolve fine distinctions.

That is the right capability for the stage it serves. A pre-GPU filter has to keep the
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
k=2 -- does not carry to RMSNorm, and the honest reading is that the model has one kernel
of support rather than a general capability.

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


## Reproducing

`docs/RUNBOOK.md` covers the broker. The profiling job runs exclusive per the shared-GPU
policy, and Nsight Compute writes its `==PROF==` preamble into stdout ahead of the CSV, so
a parser has to seek the header rather than reading from line one.
