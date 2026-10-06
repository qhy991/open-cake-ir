# Declared timing statistic

New paired policies may explicitly set `statistic: mean`. Their raw samples retain
the same correctness, count, finite-positive, command timing and replay checks.
Speedup, search selection, feedback, fresh confirmation, report thresholds and
incumbent promotion use that statistic. Mean receipts retain correctly labeled
median diagnostics; no mean is written into a median field. Historical policies
without a statistic keep their median behavior and receipt shape.

For an explicitly declared mean policy, `maximum_cv: null` with no IQR threshold
makes dispersion diagnostic-only. Such feedback reports `valid_samples`, never
`stable`; passing the declared sample contract does not establish repeatability. No samples are discarded or retried to obtain a
pass. `required_pair_wins: 0` makes the pooled materiality threshold decide direction;
cohort wins remain descriptive. Finite-positive samples and external correctness
remain required. These settings do not prove repeatability or authorize a study claim.

The Metal owner requested 30 samples per participant, each comprising 64 dispatches,
normalized to single-dispatch milliseconds before taking the arithmetic mean.
The platform policy owns those counts and warmups. This shared change introduces
no device default and does not reinterpret old experiments.
