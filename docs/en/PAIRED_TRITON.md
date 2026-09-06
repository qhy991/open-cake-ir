# Matched Triton baseline optimization

This extends `matched_search` through its existing AuthoringEnvironment, sealed
LaunchableCandidate, common Evaluation, budget and append-only Evidence path. The
`treatment` is agent search using Cake IR (JSON or restricted Python) versus the declared
**kernel-only native Triton subset**. Both start from one known Compiler-generated kernel.
The study does not compare unrestricted Triton, clean-start invention, or syntax alone.
The Compiler's static diagnostics and native source restrictions are treatment conditions.

## Contract and owners

`contracts/studies/matched-search-triton-optimization-template.json` is a stable scientific
optimization template. Its `scientific_matched_search` scope declares qualification-rate
and conditional confirmed-latency endpoints, three independent Runs per arm, no replacement
Runs, and observed candidate failures versus missing external faults. Provider, scaffold,
case, Workload/oracle, backend, exact toolchain, ABI, budget and common Evaluation remain
matched. Search timing never promotes an artifact: a selected fixed candidate needs fresh
confirmatory correctness and stable paired-CUPTI timing. Profiling remains a separate assay.
Neither artifact-only optimization nor system qualification is relabeled as science.

The template selects the primary RMSNorm case and existing Corpus Schedule. To bind another
Workload, explicitly select its contract, case, already-shaped baseline Schedule and matching
lowering route. `WorkloadContract.tensor_abi(case_id)` owns the ordered input/output names,
shapes, dtypes and modes; the Lab does not dispatch on an operator name to guess them.
`bind_baseline` checks these against the visible Schedule and binds Workload metadata. Use
`examples/paired_triton/prepare.py` for a different case's Schedule specialization first.
Historical Workloads and direct-CUDA studies retain their old fixed input/replay boundary.
They acquire no inferred tensor ABI or native-Triton treatment.

Compiler and Executor references resolve at CampaignLock creation. No new revision-specific
Study is needed. The new provider output schema requires its own live qualification; the
pending provider reference in the template deliberately does not claim an old fixture is
qualified for the native arm. `tools/freeze_live_matched_study.py` binds the qualified
provider, the common toolchain and the existing broker/executor authorities. No provider
campaign is authorized by this source template.

## Candidate submission

Both arms use the existing canonical `candidate-set.json` envelope:

```json
{"arm":"open_cake","candidates":[{"python_source":"..."}],"schema_version":1}
```

An Open Cake member is either the full canonical Schedule object or an object containing
only `python_source`. The existing Compiler Python frontend parses source without executing
it, elaborates once into Schedule, and retains file/line/column locations in assessment
feedback. Findings point back to the author's Python source. The frozen environment permits
only its explicit `triton` route and entry point. Source has no authority to choose a backend.

A native member has exactly four fields:

```json
{"kernel_source":"import triton\nimport triton.language as tl\n\n@triton.jit\ndef kernel(...):\n    ...\n","compile_constants":{},"compile_options":{"num_warps":4,"maxnreg":96},"grid":[1,1,1]}
```

The example shows the envelope shape, not a runnable kernel. The real starting member comes
from `native_baseline(lowering)`: it explicitly extracts the single kernel from trusted
Compiler lowering, dropping the generated Torch host wrapper, while preserving the exact
kernel body and compile/launch choices. C's full `baseline.triton.py` is never imported to
perform this preparation. The supplied kernel's actual name, pointer arguments, constexpr
names, options and grid come from that lowering. Native candidates cannot add a launcher,
change the pointer ABI, rename parameters or add compile knobs outside that declared interface.
Constants, existing option values and grid sizes may be optimized within the admitted shape.
Correctness is still decided by the external oracle.

The entire submitted native module is validated. It must contain exactly `import triton`,
`import triton.language as tl`, and one function decorated exactly `@triton.jit`. Pointer
arguments follow the Workload's input-then-output order. Compile constants are named values;
their declaration order is not a runtime pointer ABI. Only `tl.constexpr` annotations are
admitted, with no defaults, extra decorators, helpers, imports, classes, closures, host
callbacks, comprehensions, exception handling or `while` loops. Kernel arithmetic,
assignments, conditionals, `for`/`range`/`tl.range`/`tl.static_range`, tensor `.to`, and the
closed call/type sets in `compiler/toolchain.py` are supported. This includes the three
baseline kernels' loads, stores, reductions, dot products and gather arithmetic. Dunder
access and arbitrary calls/attributes are refused. Unsupported host edits are rejected;
they are never silently removed from native candidate source.

## Build and common Evaluation

`NativeTritonEnvironment` and `OpenCakeEnvironment` use the same `TritonToolchainBuilder`
bound to Workload/case and `IsolatedTritonCompiler`. The latter requires Linux bubblewrap,
an explicit Python runtime, a pinned Triton version and explicit read-only runtime mounts.
Its configuration keys are `python`, `bubblewrap`, `runtime_roots`, `triton_version`, and
`timeout_seconds`. The common toolchain binding includes bubblewrap bytes; the frozen
Executor owns Python/package and source identity. Runtime mounts must contain the Python
invocation and its dependencies (typically the selected environment and system runtime
folders), never `/` or the user's home directory. The worker receives the Compiler package
read-only, an isolated temporary/cache/home filesystem, no network namespace, and only its
build directory writable. No author's workspace is mounted. The existing process supervisor
owns timeout, group termination and bounded retained output. A subprocess or filtered
environment alone is not this isolation boundary. Missing bubblewrap, namespace permission,
runtime dependencies or a version mismatch is an explicit harness fault, with no fallback.

The coordinator never imports native source. The isolated worker validates it again before
compilation. Triton compiles for exact `sm_100a`, retains expanded source/TTIR/TTGIR/LLVM/PTX/
CUBIN, checks target and auxiliary-scratch limits, and seals the same LaunchableCandidate
boundary. A `workload_tensors_v1` launch manifest projects the Workload ABI and actual compiled
launch resources. The candidate envelope and kernel source retain separate custody.

`evaluate_tile_workload` uses C's `materialize_case` and `reference_outputs`, then compares
every output and verifies inputs remain unchanged. `TorchTensorLauncher` allocates from
the ABI and calls the existing admitted CUDA Driver module/launch/unload lifecycle. It never
executes candidate host wrappers. Launch observations become the ordinary EvaluationReceipt;
search, fresh confirmation, timing and attribution still use the existing RunEvaluator and
broker protocol. A source-only export or a test-double receipt is not a real sealed GPU
qualification. The correctness helper itself has no timing or promotion authority.

## Derived results

`Lab.audit` retains every prescheduled repetition in its paired view, including missing,
failed and unqualified partners. Conditional latency ratios are unavailable unless both
partners have eligible confirmed results. The scientific estimate additionally requires
archive integrity, filesystem custody, semantic replay and its declared missingness gate.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m open_cake_ir.cli lab audit \
  --lock /absolute/external/campaign-lock.json \
  --evidence-root /absolute/external/evidence --threshold-ms 0.1
```

`Lab.threshold_view` audits those same records and reports the first stable, correct, fresh
confirmation at or below the caller's latency threshold within the provider-token budget.
It returns confirmation turn, cumulative provider tokens and recorded elapsed wall time;
failures and non-crossings remain rows. The threshold is a descriptive query, not a newly
preregistered scientific endpoint. New paired Runs record wall time from Run start through
archived confirmation, including authoring, compilation and Evaluation. Historical records
without that observation report unknown wall time; neither search timing nor terminal time
fills the gap. No observed Runs are created by the report.

## R1 evidence and R2 limits

R1 verifies CPU admission, canonicalization, baseline projection, command construction,
replay and common-boundary contracts using explicit test fixtures. It does not run Triton
compilation, a Linux isolation canary, a provider or a GPU. Real bubblewrap filesystem/process
confinement and runtime closure, target compilation, all-case on-device oracle checks,
matched CUPTI timing with the frozen cache policy, profiler evidence, live provider
qualification and target-framework acceptance remain R2 pending. A new native syntax scope
would change the treatment and require a matched successor experiment. Historical releases,
Executor descriptors and approvals remain immutable; these sources require reviewed
Compiler/Executor successors before live use.
