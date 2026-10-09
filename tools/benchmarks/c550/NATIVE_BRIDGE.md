# Original Bench sealed native bridge

Related issue: #438. This adapter connects sealed CAKE artifacts to the fixed
original C550 Bench `run(*inputs)` interface. The original Bench owns input
generation, continuous RNG progression, reference execution and comparison.

## Interface and owners

```text
BoundCase(uuid, workload, candidate, manifest)
bind_all(prepared, built, problem, source_commit) -> tuple[BoundCase, ...]
NativeBench(bound_cases, original_input_names, rounds, admission, trace).run(*inputs)
```

The preparation plan supplies all sixteen original Workloads and authored sources.
CPU build uses the existing Compiler and isolated Triton builder. The retained
canonical Schedule/Program, Workload and existing candidate identity bind every
case. Before allocation, evaluation rechecks all original UUIDs in order, exact
source/target/ABI, physical-view authority, sealed payloads and each leaf's emitted
source and launch record. Artifact paths must stay inside their declared root.

The runtime callback chooses the next expected UUID from the original
`all / rounds=10` callback order, then checks every original tensor and scalar.
This supports repeated shapes without guessing which workload is active. It is
only a qualification adapter, not a production portfolio dispatcher or a
generalization guard. Callback trace and original report must agree exactly.

The adapter may allocate output/scratch storage, create validated zero-copy input
views, copy bytes for unchanged-input checks, and call existing native loaders.
It performs no Torch candidate arithmetic, list conversion, JIT or oracle work.
Every callback closes its modules, checks exact stage calls and returns original
ordered Tensor outputs. Failure retains the trace and propagates to the Bench.

The CLI exposes only CPU build and original-device qualification. It does not
create a Run, provider session or another evaluator. The pre-admission math probe
is not imported; production Compiler admission applies to every source here.

## Gates

CPU fake-loader contracts must cover complete binding, changed seals/source/ABI,
escaping paths, missing cases, unqualified views, wrong call arguments/output
structure, missing dispatches and cleanup failures. Native/device execution
requires independent review and the existing MACA device lease. This source unit
contains no GPU/provider result and does not change Compiler or Target code.
