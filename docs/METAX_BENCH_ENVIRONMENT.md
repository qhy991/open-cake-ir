# MetaX benchmark environment qualification

Tracking issue: [#406](https://github.com/qhy991/open-cake-ir/issues/406).

The original benchmark input factories and references need a coherent software
stack that includes binaries for the exact declared target. Passing a small
operator or a compiler check does not qualify every framework library used by
the benchmark.

## Successor boundary

Preserve the failed environment and its partial observations. Prepare a separate
successor from one complete distribution. Do not mix libraries across releases,
substitute another architecture, or alter benchmark shapes, references or
numerical tolerances.

The CPU phase uses no device or provider credentials and no network. Capture the
host through the existing Executor capture command. Commit that capture in an
isolated private checkout of the fixed public source. The resulting clean commit
owns the Executor identity. Keep the public-to-private source mapping and replay
bundle with the private qualification evidence; do not publish machine inventory,
local paths, images or raw environment diagnostics.

## Acceptance sequence

1. Check the complete package and compiler provenance against the declared target.
2. Capture and admit the host, then run software ABI contracts at that commit.
3. Check the existing isolated compiler with representative emitted sources,
   including integer and Boolean storage, numeric conversion and Program stages.
4. Review the CPU evidence before allocating a device through the existing broker.
5. Qualify original benchmark input factories and references on all declared cases.
6. Qualify sealed candidate baselines and the actual measurement route separately.

Use the existing capture, isolated compiler, test and Evaluation owners. A failed
environment check stops that attempt. Any correction belongs to a separately
recorded successor. CPU compilation is not device correctness or performance
qualification. Partial input-layout observations do not qualify a complete task.

## Current scope

CPU host admission, isolated native compilation of a small two-stage Program,
and discrete-storage/conversion compilation have passed. A software coverage
contract failure was retained and resolved by its owner in a separate reviewed
tick; the affected contracts passed at the successor commit.

The successor passed exact-target admission on eight C550 devices. The unmodified
benchmark at `ababa4c0` then passed its original reference self-check for all ten
tasks: 16 workloads per task, ten fresh-input rounds per workload, and 1,600
passing checks. Each task ran in a separate process through the existing device
lease. All processes exited successfully, and the final device observation
showed no remaining GPU process from these checks.

The checked public entry is `95fbbcf5`, with the separate coverage-contract fix
`583e6cc6`. The private, clean Executor source records their ancestry together
with the canonical host capture. Its source mapping, replay bundle, commands and
original reports remain in private evidence.

This closes the environment and original-reference scope of #406. The reports
retain `mode=reference_selfcheck`, `full_device_correctness=false`, and
`performance=not_measured`. Candidate correctness, paired timing, provider
qualification and optimization Runs remain separate gates under #400. The
draft PR is not an integration or merge claim.
