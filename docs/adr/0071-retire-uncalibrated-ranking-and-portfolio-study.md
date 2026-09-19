# Retire structural ranking and the Portfolio Study path

Status: accepted

The owner approved D4 (no further structural-ranking calibration) and D13 (close the
Portfolio Study kind) in the 2026-09-18 reform, and requested completion of the remaining
cleanup on 2026-09-19.

The structural device-fill model never acquired released calibration coverage. Remove
Compiler.rank, its Cost primitive, calibration_coverage, the assessment availability
flag and the wave-calibration runner. The negative measurements remain in
ANALYSIS_CALIBRATION.md and at their original source commits. Compiler manifest v3 has
only schema_version and corpus_manifest; the retired field is refused rather than ignored.

Lab preserves provider order among launchable candidates unless the Study explicitly
binds an EmpiricalCostModel. That separate model, its context checks, complete-coverage
and tie abstention, replay and diagnosis stay supported. The existing filter-event
vocabulary retains cost:null for new events; replay can still read the historical field.

The Portfolio Study lifecycle is retired. Matched search remains the one live Study kind;
old Portfolio documents are refused by name before dependency resolution or execution.
Their sources and template remain accessible on history and at ba7c48b1. Generic KernelSeed,
exact-shape specialization, portfolio artifact validation and its standalone timing/oracle
code remain available where current tools or tests consume them.

This removal changes no IR operation, legality rule, Target fact, emitted source or oracle
(P1–P8). Corpus expectations and source snapshots must stay unchanged. CPU contracts cover
remaining selection behavior and the retired document refusals; historical evidence is
neither rewritten nor requalified. The currently used utilization helpers and CuTe register
route are retained: both have live consumers, unlike the discarded structural scorer.
