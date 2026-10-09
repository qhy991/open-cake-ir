# MetaX complete Program event measurements

This successor implements complete Program event capture and separate MCPTI
attribution. **No physical target is qualified by this software change.**
`evaluation/program.py::_MACA_PROGRAM_MEASUREMENT_EVIDENCE` remains empty.
Production Ralph Runs still refuse ordered Program timing and attribution until
a reviewed successor commit records device qualification for the exact target.

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
Complete the fixed-baseline A/A and full worker qualification before populating
the production evidence set. Record the precise successor commit and original
device observations under the existing Finding workflow.

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
