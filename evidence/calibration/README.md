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

Each row of a ranking file carries the candidate's own `max_deviation` against a float32
reference. An incorrect candidate has no place in a ranking calibration, and a file that
does not record the check cannot be read as though one happened.

| `residency-b200-rmsnorm-b8-smoke.json` | `tools/profile_lowered_kernel.py` | Both bounds sound; binding resource correct and measured uniquely. |
| `residency-b200-softmax-b8-smoke.json` | same | Same, on the second Triton operator. |
| `residency-b200-layernorm-b8-smoke.json` | same | Same again on the third, which is the operator that needed no new vocabulary. |
| `residency-b200-flash-kmeans-assignment-full.json` | same | Bounds sound; the binding resource is a tie between registers and shared memory, so the prediction discriminated nothing. |
| `residency-b200-gemm-bias-b1-smoke.json` | same | Bounds sound, and a tie again -- on a Schedule declaring no shared memory at all, which Triton allocates for the dot. |

These are measurements, not Study Contract evidence: no hash chain, no Executor Revision.
They inform the model rather than witnessing a run, and the honest way to read one is to
re-run the instrument, not to trust the file.
