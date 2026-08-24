# ADR 0018: ranking is a preorder and calibration controls drift

## Outcome and non-goals

The pre-GPU model may remove a candidate only when its declaration-derived key strictly
separates the retained and discarded groups. The successor experiment decides whether
that selective 3-to-2 action stays within the Study's fixed 5% materiality boundary on
the exact GEMM M=512 domain. It does not choose within a tie, fit another term, predict
latency, grant profile-wide coverage, or reinterpret the failed v6 result.

## Minimal primitives and authorities

`Cost.order` owns the performance-semantic preorder: device fill, then the residency
ceiling. `schedule_id` remains only a stable serialization key. `rank_for_cut` is the one
decision primitive: when the requested boundary splits an equal-key group it returns an
abstention instead of manufacturing an order.

The v7 plan misspelled the display `schedule_id` already present in its pinned Schedule
bytes. Its checker rejected both raw records at authority validation, before performance
evaluation; the plan and records remain frozen as a non-evaluable attempt. The v8
successor corrects only that spelling and collects new repetitions. Its contract owns the
same finite domain, source closure, sampling protocol and pass/fail rule. The checker classifies every eligible
three-row subset before looking at performance: decisive subsets are evaluated and tied
boundary subsets are reported as abstentions. This partitions the declared domain; it
does not select convenient rows or add new ones.

## Why the measurement protocol changes

The two frozen v6 repetitions remain valid negative evidence for the old total-order
decision. They also expose a measurement limitation: candidates were timed one at a time,
all 41 samples consecutively. One candidate's retained median changes from 39.84 us to
26.98 us between repetitions, and its first repetition spans 26.72--73.28 us. The protocol
cannot distinguish candidate performance from the time window in which that candidate was
visited.

The v7 instrument therefore builds and correctness-checks the complete domain first, then
collects 50 rounds. Each round visits every eligible candidate once; cyclic rotation gives
each of the expected 25 candidates every measurement position exactly twice. The input,
cold-L2 flush, median statistic and two independent exclusive-B200 allocations remain
fixed. The checker is a read-only mode of the same pinned source and never regenerates a
measurement.

## Failure, promotion and compatibility

Any incorrect candidate, missing domain member, source mismatch, duplicate repetition,
or decisive survivor regret above 5% fails. Passing can authorize only a successor whose
coverage identifies the exact measured Schedule semantics and whose Lab preserves provider
order when `rank_for_cut` abstains. Compiler v11 retains empty coverage, so committing this
plan or collecting its measurements cannot change live behavior.

Compiler v10 and earlier retain their frozen ranking bytes. Calibration v6, its two raw
records and its failed decision remain unchanged and replay through their original source
closure.
