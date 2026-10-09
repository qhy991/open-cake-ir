# Original Bench sealed native bridge

Related issue: #438. This adapter connects sealed CAKE artifacts to the fixed
original C550 Bench `run(*inputs)` interface. The original Bench owns input
generation, continuous RNG progression, reference execution and comparison.

## Interface and owners

```text
BoundCase(uuid, workload, candidate, manifest)
bind_all(problem, index_path, input_views=...) -> tuple[BoundCase, ...]
NativeBench(bound_cases, rounds=10, admission=..., trace=...).run(*inputs)
```

The bridge consumes the existing `launch_task.py --baseline-only` output through
`load_prepared_baseline`. It does not compile kernels, copy baseline artifacts or
introduce another candidate format. Each baseline workspace retains
`prepared-baseline.json`, `workload.json`, `starter.py`, `compiler-gate.json` and
the `baseline/candidate.json` artifact bundle.

Before allocation, evaluation checks all original UUIDs in order, exact
source/target/public ABI, qualified physical views, sealed payloads, emitted source
and native launch records. The retained starter must reproduce the selected
candidate's original Python envelope. A different selected source is refused.
Artifact paths must stay inside their owning workspace and bundle. Build and
current Compiler commits must match; no cross-commit compatibility mode exists.

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

The CLI exposes CPU inspection and original-device qualification. It does not
create a Run, provider session or another evaluator. The pre-admission math probe
is not imported; production Compiler admission applies to every source here.

## Existing baseline locators

An external JSON file identifies the sixteen existing handoffs, in original suite
order. It contains `bench_commit`, `task` and `cases`. Each case contains only its
original `uuid` and an absolute canonical `prepared_baseline` path. These are
locators, not copied candidates or a second evidence store. Missing or reordered
UUIDs, escaping paths and changed seals are refused before allocation.

```sh
python tools/qualify_c550_bench_native.py inspect \
  --bench-root "$BENCH" --task "$TASK" --baselines "$LOCATORS" \
  --input-views "$QUALIFIED_VIEWS" --output "$NEW_INSPECTION"

python tools/qualify_c550_bench_native.py evaluate \
  --bench-root "$BENCH" --task "$TASK" --baselines "$LOCATORS" \
  --input-views "$QUALIFIED_VIEWS" --physical-device "$PHYSICAL" \
  --runtime-device "$RUNTIME" --expected-pci "$PCI" --output "$NEW_EVIDENCE"
```

The view observation is mandatory for nonidentity permutations. Identity layouts
still undergo the runtime zero-copy and contiguity check. The original Bench's
`check_problem` owns all sixteen workloads, ten rounds and the continuously
advancing RNG. The callback receives and returns tensors without converting them
to lists. Every successful callback executes exactly its sealed stage count and
closes all modules before returning. The final trace must match every UUID/round
in the original report; a failed original verdict cannot become bridge success.

## Oracle numeric policy

Every Workload must carry the complete policy owned by
`tasks.c550_bench.binding`; missing old fields are refused, never backfilled.
All sixteen cases must name the same policy because this loop changes no numeric
settings. `require_oracle_numerics` checks after device observation, immediately
before and after the original Bench loop, and at each callback entry after the
original reference has run. A successful 160-call loop performs 163 such checks.
The bridge never sets precision or removes the TF32 initialization override.

This boundary coverage differs from the common Workload preparation path, which
checks before/after each factory and reference call. The bridge does not hook or
monkeypatch those original functions. It does not claim to detect a setting changed
and restored inside one unmodified original function.

## Gates

CPU fake-loader contracts cover complete binding, changed seals/source/ABI,
escaping paths, missing cases, unqualified views, wrong call arguments/output
structure, missing dispatches and cleanup failures. Native/device execution
requires independent review and the existing MACA device lease. This source unit
contains no GPU/provider result and does not change Compiler or Target code.

At fixed `3ebb782a`, eight CPU contracts pass with no skips. They exercise both
Schedule and complete Program callbacks, duplicate shapes, the original loop and
report delegation, numeric drift, partial module construction, launch/teardown
failures and self-consistently changed manifests. Tests use an explicit frozen
source `PYTHONPATH` and assert the imported Compiler path. Independent source review
found no blocker. These results qualify source/CPU behavior only; the actual
native builds and original 160 device checks remain pending.
