# AKA qualified-parent review and Lab plan

[中文原始报告](../AKA_QUALIFIED_IR_REVIEW_AND_LAB_PLAN_20260903.md) · [English home](README.md)

This is an English reading companion to the historical report updated on 2026-09-04 JST. The original owns the detailed batch ledger and evidence locators. Phase A is terminal and all 57 Lab admissions have terminal outcomes.

## Results and meaning

Of 677 qualified AKA-v6 parents, 676 produced model output plus deterministic verifier records; one failed before the model. The verifier accepted 439 classifications, rejected 211 reviewer submissions, and rejected 26 schema outputs. Classification acceptance does not mean executable GPU code.

| Accepted class | Count | Responsibility |
| --- | ---: | --- |
| schedule / expressible / lower passed | 57 | Fixed-instance Lab admission |
| ir_gap / not_expressible | 326 | Review proposed semantic gaps |
| program_composition | 27 | Program/Executable composition |
| insufficient_evidence | 21 | Obtain missing semantic/source evidence |
| workload_evidence | 8 | Complete Workload definitions |

The five rows total 439. The 237 reviewer/schema refusals and one infrastructure failure remain 238 failures, not hidden exclusions. Of 57 static admissions, 33 were marked optimization-eligible and 24 correctness-only; these were queue priorities, not performance qualifications.

Final Lab outcomes are 56 fixed instances valid after complete-output B200 correctness, memcheck, racecheck, and independent recomputation; one authoring/schema/custody/evidence-surface rejection remains GPU not run. No final unknown remains in this Lab set. All have performance_measured=false; none supplies training qualification or full dynamic-parent coverage.

## Gap clustering

The 326 accepted gap cases used 296 exact names. A proposal partition contains 202 clusters: 51 repeated clusters covering 174 cases, 150 singleton/distinct clusters covering 150, and one conflict covering two. The reverse segmented_scan conflict disagrees about including a boundary element. The verifier checks exact membership without duplicates or omissions; it does not prove source independence or authorize primitives.

Large repeated proposals include FP32 FMA (12 cases), typed runtime scalars (9), computed atomic state add (7), indexed atomic scatter (7), segmented indirect sum (7), and natural log (6). [The later owner review](../AKA_IR_OWNER_REVIEW_20260904.md) owns subsequent design dispositions and does not rewrite this proposal.

## What execution taught

Admission bound 253 complete parent/reference/harness files, approximately 7.2 MiB, to the 57 cases. Fixed GPU Infra was independently checked with 76/76 tests, and task-owned daemon/socket/state were separated from shared control.

Early canaries failed at working-directory, home traversal, guard path, NumPy, and environment-link boundaries. They remain infrastructure unknowns. Copy4 successor v6 passed three stages and independent recomputation of 12,288 elements. A second sigmoid canary passed with 306 outputs checked. Missing standalone verification JSON was transparently reconstructed later from retained arrays with gpu_rerun=false and run_modified=false, rather than backdated.

Eleven batches of at most five in-flight items covered copies, elementwise functions, broadcasts, gradients, reductions, and mutable state. Failures included bootstrap directory policy, oracle field aliases, racecheck summary parsing, evaluator cache leaks, missing evidence fields, non-power-of-two Triton spans, and store/outer-product authoring errors. Each remained under its own identity; narrowly repaired successors retained the oracle and numerical rules. A broker run already submitted was not resubmitted just because a local receipt parser failed.

The final l000214 submission lacked required task fields, used incompatible runtime/old aliases, and encoded a huge evidence surface unsuitable for the contract. It terminated as authoring/schema/evidence rejection before GPU. It establishes no IR gap and requires a fresh authoring attempt plus disk admission to recover.

The source report retains every batch, array count, and predecessor. Published compact evidence is in the [dataset directory](../data/aka-qualified-ir-v6-review-20260904/). Its export checks uniqueness, partition, and sensitive-pattern exclusion. The combined 678-test suite retained one historical replay/custody failure; it was not repaired or hidden by this work.

## Remaining work and limits

The 57 executions exposed authoring/schema, evaluator/artifact hygiene, or backend compile boundaries, with no remaining dynamic failure requiring new Schedule vocabulary. They therefore do not justify new IR by themselves. The 326 gap proposals still need independent source/ownership/typing/effect/analysis/lowering review and positive/near-miss Corpus evidence.

Five in-flight items means preparation, queueing, or execution combined, not five GPUs. Broker-owned exclusive stages use actual available cards. Historical disk figures in the original are not current capacity; future work must obtain new resource admission and preserve formal evidence.

Performance work additionally requires eligible scope, repeatability, frozen baselines, paired timing, and profiler evidence. Program cases go to their owner, evidence cases need contracts, and authoring failures remain useful negatives. Completion here means unique terminal results for all 57, not a performance or scientific comparison.
