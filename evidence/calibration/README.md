# Calibration measurements

Raw output of the instruments in `tools/` that test the static analysis and the ranking
against a device. `docs/ANALYSIS_CALIBRATION.md` is where these are read; this is what it
is reading, so a claim there can be checked rather than taken.

| file | instrument | what it settled |
| --- | --- | --- |
| `wave-term-b200-rmsnorm.json` | `tools/calibrate_wave_term.py` | The ranking's wave term describes no measurable step. It was removed and the model's domain now stops at one round of the device. |
| `rmsnorm-b200-ranking-b16.json` | `tools/calibrate_ranking_at_scale.py` | The ranking beats a blind pick on a deeply under-filled workload, by two points. |
| `rmsnorm-b200-ranking-b64.json` | same | And loses to one at the next scale up. The model has one kernel of support, not a capability. |
| `rmsnorm-b200-ranking-b512.json` | same | Past saturation the model declines; extending its key there would have picked the worst of 37 candidates. |
| `gemm-b200-ranking-m512-v5.json` | same | In the declared 30-candidate domain, 25 pass the gates and correctness; the single sweep has 1.72% top-1 regret but cannot set its own promotion criterion. |
| `flash-kmeans-b200-ranking-n512-v5.json` | same | In the declared 30-candidate domain, 16 pass the gates and correctness; device fill misses the best by 19.43% at k=1. |
| `gemm-b200-ranking-m512-v6-repeat-1.json` | same, under the frozen v6 calibration plan | The first preregistered exclusive-B200 repeat measures 25 correct candidates and retains 5 compiler refusals. |
| `gemm-b200-ranking-m512-v6-repeat-2.json` | same, under the frozen v6 calibration plan | The independent repeat has the same complete disposition: 25 correct candidates and 5 compiler refusals. |
| `gemm-b200-ranking-m512-v6-decision.json` | `tools/check_ranking_calibration.py` | Exhausting all 2,300 three-candidate subsets fails the 3-to-2, 5% survivor criterion in both repeats: worst regret is 35.95% and 8.16%. |
| `gemm-b200-ranking-m512-v7-repeat-1.json` | `tools/calibrate_gemm_ranking_interleaved.py` | Complete interleaved measurement retained but not evaluated because the frozen v7 plan's display `schedule_id` disagrees with its own pinned Schedule bytes. |
| `gemm-b200-ranking-m512-v7-repeat-2.json` | same | The second complete raw repetition has the same non-evaluable authority mismatch. |
| `gemm-b200-ranking-m512-v7-harness-fault.json` | frozen-checker observation | Records the authority mismatch and forbids repairing or interpreting v7 after measurement. |
| `gemm-b200-ranking-m512-v8-repeat-1.json` | `tools/calibrate_gemm_ranking_interleaved.py`, under the frozen v8 successor | Measures 25 correct candidates in 50 cyclically interleaved rounds and retains 5 compiler refusals. |
| `gemm-b200-ranking-m512-v8-repeat-2.json` | same | The independent repeat retains the same complete disposition and sampling protocol. |
| `gemm-b200-ranking-m512-v8-decision.json` | same source, read-only `check` mode | Each repeat classifies all 2,300 subsets as 1,450 decisive cuts and 850 tie abstentions. Worst decisive regret is 5.73% and 2.79%; the fixed every-repeat 5% rule fails. |

Each ranking row carries the candidate's own `max_deviation`, every declared timing sample,
and the evaluator/oracle source hashes. Refused and incorrect candidates remain in the
declared domain as explicit exclusions. An incorrect candidate has no place in a ranking,
and a file that does not record the check cannot be read as though one happened.

| `residency-b200-rmsnorm-b8-smoke.json` | `tools/profile_lowered_kernel.py` | Both bounds sound; binding resource correct and measured uniquely. |
| `residency-b200-softmax-b8-smoke.json` | same | Same, on the second Triton operator. |
| `residency-b200-layernorm-b8-smoke.json` | same | Same again on the third, which is the operator that needed no new vocabulary. |
| `residency-b200-flash-kmeans-assignment-full.json` | same | Bounds sound; the binding resource is a tie between registers and shared memory, so the prediction discriminated nothing. |
| `residency-b200-gemm-bias-b1-smoke.json` | same | Bounds sound, and a tie again -- on a Schedule declaring no shared memory at all, which Triton allocates for the dot. |

These are measurements, not Study Contract evidence: no hash chain, no Executor Revision.
They inform the model rather than witnessing a run, and the honest way to read one is to
re-run the instrument, not to trust the file.
