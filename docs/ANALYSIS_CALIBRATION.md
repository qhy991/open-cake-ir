# The static analysis, measured

The paper's harness reports a performance analysis as a *report*, and says plainly that
on-device measurement remains the ground truth. This repository had the report and none of
the measurement, so nothing established whether the bounds it prints are true.

Measured with Nsight Compute on a B200, exclusive, on the two emitted backends.

| | predicted | measured | |
| --- | --- | --- | --- |
| **rmsnorm**, Triton | | | |
| registers per thread | >= 66 | 95 | lower bound holds, understates by 31% |
| resident CTAs | <= 7 | 5 | upper bound holds, loose by 2 |
| binding resource | registers | registers | correct |
| **flash-kmeans assignment**, CuTe-DSL | | | |
| registers per thread | >= 1 | 80 | holds, but says nothing |
| resident CTAs | <= 2 | 2 | upper bound holds, and is exact |
| binding resource | shared memory | shared memory | correct |

## What this establishes

**Both bounds are sound in the direction claimed.** The register figure is a lower bound
on storage and was below the measured allocation in both cases; the residency figure is an
upper bound on resident CTAs and was at or above the measured limit in both cases. Nothing
here contradicts the analysis, which is not a given -- the analysis was renamed to say
"bound" only after it was written, and this is the first check that the renaming was
accurate rather than merely more cautious.

**Both binding-resource predictions were correct.** This is the half the paper calls
attribution, and it is the half an author can act on: being told that registers rather than
shared memory is what limits residency points at which declaration to change. Two for two
is not a large sample, but the two cases bind on *different* resources, which is the
interesting pair.

**The exact quantities are exact and the estimated one is loose, exactly as claimed.**
Shared memory is an explicit allocation the Schedule declares, and its bound came out equal
to the measurement. Registers are inferred from declared buffers with liveness and
aliasing, and that bound is 31% low on one kernel and vacuous on the other. The distinction
between exact Schedule quantities and the estimated one is not a hedge; it is visible in the
numbers.

## What it exposes

**The register bound says nothing about a schedule that keeps its data off-register.**
`flash-kmeans-assignment-full` declares almost no register buffers -- its staging lives in
shared and tensor memory -- so the bound is 1 against a measured 80. It did not mislead
here because shared memory bound residency anyway and the report said so. But a schedule
that is genuinely register-bound while declaring few register buffers would get a residency
ceiling far above the truth, and the report would name the wrong resource.

That is a real limit of an analysis that reasons only over declarations: it cannot see the
registers a backend needs for addressing, predication and staging, and those were 29 and 79
registers per thread in these two kernels. Worth recording rather than filing as a defect,
because closing it means either measuring or modelling the backend, and the paper puts
measurement at the end of the loop for exactly this reason.

## Reproducing

`docs/RUNBOOK.md` covers the broker. The profiling job runs exclusive per the shared-GPU
policy, and Nsight Compute writes its `==PROF==` preamble into stdout ahead of the CSV, so
a parser has to seek the header rather than reading from line one.
