# B300 paired cost calibration boundary

The B300 GEMM Study's search measurement is the common evaluator's fixed-baseline
FlashInfer CUPTI assay. It uses the Study's pair order, fresh arguments, 25 samples
per cohort, the Workload's oracle, and an Executor-bound runtime. A Kineto/CUPTI
single-candidate measurement with explicit L2 zeroing is a different assay, even
when its Schedule, Target and nominal case shape match.

`lab.selection._paired_empirical_context` defines the context for a future model
derived from that paired assay. It keeps the existing single-candidate context
untouched and additionally binds the exact paired policy and fixed baseline record.
`lab.paired_cost_calibration.observed_paired_cost` replays a common Evaluation
receipt against that policy, baseline and the worker's exclusive broker counters
before projecting the candidate and baseline medians. It accepts only correct,
stable paired CUPTI observations. `derive_paired_cost_model` fits one exact-case
point per candidate from separate fit observations, sets each descriptive range
from calibration observations, then checks held-out prediction error, all provider
orders of the three-to-two cut, and fixed-baseline drift. This is a pure
derivation; a future collector must still establish the Schedule, compiled
artifact, broker and split-freeze custody before publishing its output.
The context is a comparison boundary, not evidence that any model has been measured
or qualified. Existing `EmpiricalCostModel` instances and the Lab policy remain as
they are; the B300 scientific Study does not admit empirical candidate selection.

| Model context field | Paired owner |
| --- | --- |
| `timer` | FlashInfer CUPTI helper plus `fixed_baseline_paired_cupti_v1` |
| `cache_protocol` | Evaluator's `cold_l2_cache=true` |
| `runtime` | Exact Executor identity and declared Triton version |
| `input_scope` | Workload and case, complete Evaluation protocol identity, sealed baseline record |

A calibration successor must prepare complete Schedules under a clean Compiler
commit, compile and seal the corresponding candidates, and collect paired Evaluation
receipts under GPU Infra. It must replay raw timing, oracle, baseline, broker,
source and split-freeze bindings before using the pure fitter. Only after an actual pre-GPU candidate cut and
its separately frozen full audit can a Study policy be considered for this Workload;
that decision must preserve the scientific comparison's treatment and acceptance
contract. The current context helper alone authorizes no device work or Lab policy
change.
