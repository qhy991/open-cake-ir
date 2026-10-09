# MetaX complete Program event measurements

This successor makes the existing complete Program event and MCPTI attribution
adapters available for exact target `xcore1002`. The source proposal requires
review of the retained component evidence and its implementation relation before
adoption. The registry owns adapter availability, not per-baseline readiness.

Formal authors in the controlled workflow still require an external
`performance_qualified` result. That result requires a fresh common-evaluator
A/A comparison and independent profiling on the frozen successor. `TaskLab`
does not itself enforce a prior-A/A requirement. Do not invoke an author Run
directly before the external readiness gate passes.

## Measured interval

`MacaProgramEventBenchmark` uses MACA PyTorch events on the default stream. Each
sample includes every stage in the sealed `Program`, the loader's tensor checks,
and host submission gaps. Allocation, module loading, input construction, and
output checks occur outside that interval. The timer does not claim kernel-only
time or an activity trace.

The existing `fixed_baseline_paired_maca_event_v1` policy supplies two cohorts of
five samples per participant and reports their mean. Each cohort retains eleven
warmup output sets and five measured output sets. The reset fills an FP32 buffer
whose byte size is four times the target's declared L2 capacity. It is ordered
before each start event. The timer checks the exact physical stage count for
every Program call. The common worker checks all retained outputs and unchanged
inputs against the Workload oracle.

The native, Torch-reset-native, and gated single-dispatch timers keep their old
scope. They do not accept an ordered Program. An aligned Program is also refused
by this first Program event adapter.

## Separate attribution

The existing `MACA_PROGRAM_PROFILE` observes one complete invocation after an
independent correctness preflight. It verifies every stage's identity, order,
stream, launch configuration, and reported resources. The instrumented output
also passes the Workload oracle. All modules close before the worker publishes
an attribution receipt. A capture, numerical, or teardown failure retains the
available observation and produces no successful receipt.

Profiling never runs inside the event intervals. Its stage durations and gaps
describe attribution only and do not enter the paired latency statistic.

## Device qualification

Use a clean successor checkout, the admitted CPU build environment, the original
Workload oracle, and the existing physical-device MACA lock. Preserve each output
directory and stop on an environment or qualification failure. Do not retry a
failed A/A comparison to obtain a passing sample.

The existing tool can build and check a complete Program. It now also exposes a
bounded event observation command:

```text
python tools/qualify_tensor_program.py build --workload <workload.json> \
  --program <program.json> --output <new-build-directory>
python tools/qualify_tensor_program.py evaluate --built <build-directory> \
  --case primary --output <new-correctness-directory>
python tools/qualify_tensor_program.py events --built <build-directory> \
  --case primary --output <new-event-directory>
python tools/qualify_tensor_program.py profile --built <build-directory> \
  --case primary --output <new-profile-directory>
```

The device commands start outside an allocation, then obtain the repository's
existing local lease. Physical index, runtime index, and expected PCI address
must be bound through that lease's normal configuration.

`events` checks a complete preflight and all 32 fresh warmup/sample outputs,
retains ten event values and their mean, verifies all stage calls, and closes the
modules. Its `event-observation.json` is a device qualification observation. It
is not a common Run receipt, an A/A result, or authorization to promote a kernel.
Review complete component correctness, event and profile evidence to decide
exact-target adapter availability in a successor source commit. Then use the
existing `CommandBrokerSubmitter` and `BoundedBrokerEvaluator` to perform one
confirmatory A/A with the same sealed Program on both arms and
`evaluation_policy(workload, metax_mean10=True)`. Require the original
correctness checks, measurement-quality checks and `close_null` classification
at materiality 1.05. Ten event samples per arm remain the mean statistic;
dispersion remains diagnostic. A failed qualification stays failed.

After A/A passes, require one separate attribution Evaluation with a valid
`MACA_PROGRAM_PROFILE`, all ordered stages, unchanged inputs, correct outputs
and successful teardown. The controlled workflow may then mark that baseline
`performance_qualified`. Repeat the baseline readiness gates for each actual
Bench contract. An ordinary component fixture does not qualify Bench cases.
Retain the exact successor and observations under the existing Finding workflow.

## Retained component evidence and open gates

The retained `solx_fib.attention` component observation covers four stages and
five correctness input cases. It reports unchanged inputs, two five-sample event
cohorts with fresh-output checks, and a separate four-stage MCPTI profile.
The existing cohort and profile readers replay those records successfully.
This is an ordinary repository task, not an original Bench result or an A/A pass.

The component was recorded at `d9b92681`. Its implementation relation to this
source proposal must be confirmed before adoption. Finding F-2026-10-09-010
remains open. A real common-worker A/A, attribution and each Bench baseline's
readiness remain pending; the adapter declaration does not assert their success.

## Scope and resource limits

This change does not modify IR syntax, semantics, analyses, or target hardware
facts; the P1–P8 IR-change gate therefore has no new primitive to assess. It
implements an Executor measurement boundary and retains the original Program,
Workload, and oracle. It does not modify frozen C1 Runs.

Inputs are shared across a cohort, but all fresh outputs and intermediates are
retained until their checks finish. Account for sixteen sets of each Program's
workspace when admitting a large task. Do not shrink a benchmark shape or reuse
outputs to make the measurement fit. A later memory-lifetime or sampling change
needs its own successor and verification.
