# Direct reference comparison for the existing h2048 task

`tools/compare_flashinfer_reference.py` is a development judge for GPU Infra. It
uses `fib_rmsnorm_h2048`, exact `sm_103a`, BF16 and epsilon 1e-6. No provider or
new Workload is needed. The immutable input snapshot contains the unchanged
`submission.py` and `kernel.py` plus `comparison.json`:

```json
{"task":"fib_rmsnorm_h2048","rows":79,"columns":2048,"target":"sm_103a"}
```

An optional `candidate.py` in the same snapshot replaces the generated Cake
starter with an authored Cake program. The existing `bind_baseline` owner
checks its exact target, ordered tensor ABI and lowering route; the Compiler
then assesses and lowers it. Both the author source and bound Schedule are
retained. This permits reference-derived row-packing candidates to be tested
without modifying the external implementation or introducing a new Workload.

Run the tool as an exclusive `judge` stage with its pinned checkout as cwd and
the admitted Executor Python. The existing task oracle checks both outputs and
input preservation for all five distributions before and after timing. The
original reference performs its autotune during preflight; its selected
configuration is recorded, and its internal timing scores are not results here.
The reference's code and host wrapper are not transformed.

Timing uses the existing paired policy, fresh argument sets, the common
`_fresh_tile_cohort` checker and strict CUPTI helper: 25 samples per cohort,
alternating AB/BA order, cold L2, no CUDA Event/graph fallback. Every timed output
is checked. The interval is device kernel time, not Python allocation overhead,
autotune overhead or application latency. `candidate` means Cake and `baseline`
means the external implementation. Ratios, raw samples and coverage are retained
in `comparison-report.json`.

This is not a sealed Campaign, provider comparison or promotion receipt. The
Infra result projects validity without populating its timing frontier. It
records `No promotion`; changing a kernel or tolerance after a failure is not
part of this test. A missing environment or timing facility is an unknown result,
not an incorrect candidate or a reason to substitute a timer.
