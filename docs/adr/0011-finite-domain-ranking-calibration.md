# ADR 0011: calibrate the decision the Lab actually makes

## Outcome and non-goals

Ranking calibration decides whether the current Lab may reduce three launchable
candidates to two without discarding a candidate that is more than the Study's 5%
materiality boundary faster. It does not fit a new term, predict latency, or grant a
profile-wide capability from one shape.

## Authority and minimal primitives

`contracts/calibrations/gemm-b200-ranking-m512-v6.json` freezes the Compiler and evaluator
closure, complete 30-candidate domain, two independent measurements, and pass/fail rule
before either measurement. The existing calibration instrument remains the sole producer
of raw timing rows. `tools/check_ranking_calibration.py` is a read-only projection: for
each repetition it checks custody and domain completeness, then evaluates every
three-candidate subset of correct lowering-eligible rows under the shipped ordering.

This is the smallest decision domain matching `maximum_candidates_per_turn=3` and
`searches_per_turn=2`. Sets of zero, one, or two need no ranking cut. Enumerating all
three-member subsets avoids a convenient partition and makes the claim independent of
provider order within this finite measured domain.

## Failure and promotion semantics

An incorrect candidate, missing domain member, source mismatch, unranked eligible row,
duplicate repetition, or any survivor regret above 5% fails the calibration. Failure
retains the raw records and changes no released coverage. Passing permits only a later
human-reviewed, exact-domain release; it does not justify the current profile-name-wide
`calibration_coverage` spelling. A mixed calibrated/unmeasured set must retain provider
order as a whole, because ranking the measured members first would silently prune an
unmeasured candidate.

## Acceptance evidence

Both raw measurements must be produced in separate exclusive one-GPU broker allocations,
with 41 cold-L2 samples per correct candidate. The checked decision record is a derived
view and must replay byte-for-byte from the frozen plan and raw records.

## Validation

Both repetitions contain 25 correct measured candidates, 5 compiler refusals and all
2,300 possible three-candidate decisions. They fail the fixed 5% boundary: maximum
survivor regret is 35.95% and 8.16%. The Compiler Revision remains v8 with empty
`calibration_coverage`; this negative result authorizes no implementation change.
