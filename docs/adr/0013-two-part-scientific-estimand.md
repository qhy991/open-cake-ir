# ADR 0013: Keep candidate failure separate from missing scientific data

Status: accepted, 2026-08-24.

## Outcome and non-goals

The current scientific matched-search Analysis Plan estimates a two-part Run endpoint:
the arm qualification rate at the terminal provider-token budget, followed by confirmed
latency conditional on qualification. A Run that adheres but produces no qualified
Candidate is an observed negative outcome. A provider, harness, custody or broker fault
is missing data. Reports never turn the latter into the former.

This does not define the paper's unavailable token-accounting or plateau rules, add an
imputation method, authorize replacement Runs, introduce inferential statistics, or
rewrite any frozen Study, Campaign, event or report.

## Canonical plan and authorities

`budget.limit` is the sole owner of the terminal budget; the Estimand name does not copy
its numeric value. `allocation.order` owns the target population, so the Analysis Plan
names the prescheduled Runs rather than restating their count. `run_inclusion` owns which
Run contributes to each endpoint.

The current Analysis Plan requires all prescheduled endpoint observations to be present
and at least one qualified Run in each arm. The first condition makes qualification rates
estimable without treating missing Runs as failures. The second makes both conditional
latency medians and their ratio defined. One or more observed candidate failures are
therefore allowed; an arm with zero qualified Runs leaves the complete two-part Estimand
unavailable while its observed qualification count remains descriptive evidence.

The estimate contains the preregistered qualification contrast
`open_cake_rate - direct_cuda_rate` as well as conditional arm medians and
`direct_cuda_median / open_cake_median`. The earlier plan remains a bounded compatibility
adapter for frozen authorities; it keeps its original all-Runs-qualified availability
rule and output spelling.

## Failure and acceptance evidence

A current report states, per arm, prescheduled, observed, qualified and missing endpoint
counts. Its descriptive qualification rate divides by observed endpoints and is named as
such. The scientific estimate is emitted only when archive integrity, filesystem custody,
semantic replay, zero missing endpoints and conditional-latency availability all hold
(the custody distinction is owned by the later ADR 0031).

Acceptance requires three contract cases: all Runs qualified; one adhered Run with no
qualified Candidate while both arms retain conditional latency; and one external fault.
The second must still produce the two-part estimate and qualification-rate contrast. The
third must make the Estimand unavailable and must not count the missing Run as a failed
qualification.
