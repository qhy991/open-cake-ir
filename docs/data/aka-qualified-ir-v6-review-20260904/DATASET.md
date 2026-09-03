# AKA qualified-parent Open-Cake review export

This directory is a publishable projection of external, append-only evidence. It does
not contain raw model events, credentials, complete GPU tensors, or mutable campaign
state.

- `phase-a-review.jsonl`: 677 canonical qualified-parent rows. Exactly 439 have a
  deterministic accepted static result, 237 retain reviewer/schema rejection, and one
  retains a pre-model infrastructure failure.
- `lab-terminal-results.jsonl`: the 57 static schedule/expressible/lowered admissions.
  Exactly 56 are fixed-instance dynamic-valid after B200 correctness, memcheck,
  racecheck, and independent complete-output recomputation; one is an authoring
  rejection with GPU not run.
- `ir-gap-clusters.json`: remote Sol/max semantic clustering of all 326 accepted
  `ir_gap` rows. The accompanying verifier proves exact 326-case/296-name partition.
- `l000214-authoring-rejection.json`: the sole Lab terminal non-dynamic record.

The cluster output is proposal-only. It approves, implements, releases, GPU-tests, and
performance-qualifies no IR primitive. The 56 Lab successes prove only the frozen
fixed instances. All rows have `performance_measured=false`; training eligibility is
false for the Lab set.

Source identities are recorded in `manifest.json`. Compact admission, progress,
independent-verification, node-result, environment, and clustering-treatment summaries
are under `evidence/`; raw model events and complete tensors remain in the external
append-only roots. SHA-256 values in the manifest exist only for this explicit external
handoff and the user-requested corpus identity, not as semantic correctness evidence.
