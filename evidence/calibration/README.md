# Calibration measurements

Raw output of the instruments in `tools/` that test the static analysis and the ranking
against a device. `docs/ANALYSIS_CALIBRATION.md` is where these are read; this is what it
is reading, so a claim there can be checked rather than taken.

| file | instrument | what it settled |
| --- | --- | --- |
| `wave-term-b200-rmsnorm.json` | `tools/calibrate_wave_term.py` | The ranking's wave term describes no measurable step. It was removed and the model's domain now stops at one round of the device. |

These are measurements, not Study Contract evidence: no hash chain, no Executor Revision.
They inform the model rather than witnessing a run, and the honest way to read one is to
re-run the instrument, not to trust the file.
