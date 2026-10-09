# MetaX register transpose admission

Related issues: #418 and #427. The source change adds the existing `transpose`
operation to exact Target `xcore1002`. It changes no IR syntax, typing, effects,
analysis or emission. The existing operation exchanges the two axes of a rank-two
register tile while preserving its dtype. FP8 remains refused by
`VALUE_OPERATION_TYPE` and `MACA_FP8_OPERATION_UNQUALIFIED`.

## Device evidence

Public probe `346d4934`, in job `maca-2c5cc31ae97a`, passed the following bounded
native storage checks on MetaX C550:

| Dtypes | Original shapes | Checks per shape |
|---|---|---|
| FP32, FP16, BF16, INT32, INT64 | 16×32, 35×67, 1×3 | Four for each dtype and shape |
| BOOL | 16×32, 35×67, 1×3 | Nine, twelve and two coordinate bit planes |

The 18 cases completed 83 native calls and 83 exact-byte output checks. All
261,554 checked output bytes matched the independent CPU byte permutation.
Every input retained its bytes, all 18 modules closed, and the device lock was
released after process exit. The relevant probe, Compiler, Evaluation, Lab build
and exact Target sources match public `346d4934`. This relation does not assert
whole-source or host identity. Private infrastructure and raw observations remain
outside the repository.

These observations qualify bounded register storage movement. They do not
establish transpose feeding MMA, the complete Decoder backward Workload, arbitrary
shapes, timing or performance. Those compositions require their own native and
original-Workload checks. Issues #418 and #427 remain open during source review.

The pre-admission probe and its closed-Target tests stay at their original commit.
This successor does not import that probe or change its frozen results.

## P1–P8 review and software gate

| Principles | Scope of this change |
|---|---|
| P1, P3, P4, P5, P7 | Reuse the existing canonical operation, type rule, effects and analysis. No new representation or layout algebra. |
| P2, P8 | The author declares a concrete register-tile axis exchange. Lowering stays visible as `tl.trans`; the Target declaration cites its own device observations. |
| P6 | The new real-Target CPU contracts check all six dtypes and three original shapes, unique-coordinate movement and one store per output. Type/shape/rank/storage/arity negatives name the existing owning rule. FP8 and another unqualified Target stay refused. The existing 204-case Corpus must pass without changing expectations. |

Independent source review and the fixed-commit CPU/Corpus results are required
before using this successor for the Decoder starter. This admission contains no
new GPU or provider Run and changes no frozen C1 source.

At `aed4d35a`, 54 of 55 focused CPU contracts passed and the 204-case Corpus
matched unchanged expectations. The remaining new test incorrectly expected
backend findings after value typing had already stopped Compiler assessment.
Commit `5ad04d6c` checks both existing FP8 refusals at their own boundaries.
All four new admission contracts pass there with no skips. The 51 unaffected
contracts and Corpus were not repeated after that test-only correction.
