# Cake and native CuTeDSL on B300

[中文说明](../zh-CN/PAIRED_CUTE.md)

Use `contracts/studies/matched-search-cute-b300-gemm-optimization-template.json`
for the existing `matched_search` / Ralph path. Its arms are `open_cake` and
`native_cute_dsl`. Both begin from the same register GEMM+bias kernel and use
`cutlass_cute_dsl` on exact `sm_103a`. The Study is stable: Compiler, Executor,
provider qualification, runtime and sealed fixed baseline resolve into an external
CampaignLock. Creating the template does not authorize its six-run scientific study.

The Workload is `gemm-bias-bf16-fp32-v2`: BF16 A and B, FP32 bias/output, and
`C = A @ B.T + bias`. It owns the oracle and `atol = rtol = 0.001`. Its cases are
primary 512×256×256, tail 3×5×65, tiny 2×3×65, zeros 1×2×65 and cancellation
2×3×66 (M×N×K). The primary Schedule is
`corpus/schedules/b300-cute-register-primary.json`; Lab binds the real Workload
identity into its metadata. The Compiler fixture carries no fabricated Workload hash.

## Authoring

`examples/python/b300_cute_gemm_bias.py` is a complete restricted Python Schedule.
Assess it through the ordinary Compiler CLI:

```sh
PYTHONPATH=src python -m open_cake_ir.cli compiler assess --revision compiler/revision.lock.json examples/python/b300_cute_gemm_bias.py
```

The Cake arm submits the existing JSON Schedule member or an object containing
`python_source`. Python parsing preserves source locations and never executes authored
host code. Keep `lowering.backend = cutlass_cute_dsl` and the supplied entry point.
This first route supports one warp, BM=16, BN divisible by 8, BK divisible by 16,
K>BK, one K-loop stage, explicit register operands and the BF16 warp MMA. It masks
M/N/K tails and zeroes invalid operand lanes before collective MMA. Register/residency
caps are unsupported; `coalesced=False` makes no unproved store-coalescing promise.

The native arm edits `candidate-baseline.cute.json` from its TASK. It contains exactly
`kernel_source`, `grid`, `block` and `dynamic_shared_memory_bytes`; the accompanying
`candidate.schema.json` states the structural contract. For both arms, write the
existing `candidate-set.json` envelope with the assigned arm and candidate members.
Generated-source comments record the initial baseline's origin; they do not prohibit
optimizing that kernel in the native arm.

The native source is one kernel-only Python module: the fixed `cutlass`, `cute` and
`warp` imports, followed by exactly one `@cute.kernel`. Its ordered `cute.Pointer`
parameters are the Workload's A, B, bias and output pointers. No helpers, extra imports,
host compilation, host launches or arbitrary Python calls are admitted. The whole
module is checked before compilation. The launch has `block=[32,1,1]` and zero dynamic
shared memory. Positive integer grid changes are allowed; correctness still decides
whether the resulting kernel covers every output. Extra compile options are rejected.

## Runtime and qualification

The `toolchain` section of the ordinary runtime configuration has these exact fields:

```json
{
  "python": "/opt/cute-runtime/bin/python",
  "bubblewrap": "/usr/bin/bwrap",
  "runtime_roots": ["/opt/cute-runtime", "/usr/local/cuda-13.1", "/usr/lib", "/lib", "/lib64"],
  "cuobjdump": "/usr/local/cuda-13.1/bin/cuobjdump",
  "cutlass_version": "4.5.2",
  "timeout_seconds": 600
}
```

These are example deployment paths; bind the actual provisioned paths and the existing
provider/broker sections in the external runtime file. The released Executor must bind
that Python and all three SDK packages (`nvidia-cutlass-dsl`,
`nvidia-cutlass-dsl-libs-base`, `nvidia-cutlass-dsl-libs-cu13`) at 4.5.2. Runtime mounts
include the interpreter and tool dependencies, retain loader aliases, and exclude the
author's workspace and home. Linux bubblewrap is required. Compilation denies driver
initialization, exposes no NVIDIA device nodes and retains that CPU boundary evidence.
There is no unsandboxed fallback or automatic target downgrade.

Use `tools/qualify_codex_provider.py` with
`contracts/providers/codex-cute-optimization-output-schema-v1.json` to qualify the two
actual arm envelopes in a zero-GPU two-turn provider run. This proves the provider
contract, not kernel correctness. Pass the external qualification receipt, anchor,
runtime configuration and sealed fixed-baseline bundle through the ordinary
`lab preflight --execution-bindings` path. Keep every new lock and evidence root outside
the source checkout. The stable template is not rewritten by the frozen-Study helper.

`OpenCakeEnvironment` and `NativeCuTeEnvironment` use the same `CuTeToolchainBuilder`
and `IsolatedCuTeCompiler` from `open_cake_ir.lab.cute_build`. The compiled SDK symbol
may differ from the authored Python name; the sealed TensorLaunchManifest uses the
observed symbol and, for this GEMM study, exactly four pointers with no hidden parameters. Raw submissions,
source, compiler diagnostics and artifacts remain in the ordinary append-only archive.
Common Evaluation performs oracle correctness, paired cold-L2 CUPTI timing and separate
profiler attribution. Source tests and successful CPU compilation do not establish GPU
correctness, performance, or an IR advantage. Those claims require their own retained
common Evaluation evidence and independent release gates.


## General FP32 SIMT lowering

Schedules without MMA may select `cutlass_cute_dsl` and enter a structural single-warp
SIMT route. It supports FP32 loads/stores, add/sub/mul/div, square/relu/rsqrt/exp/exp2/
reciprocal/tanh, explicit broadcasts and single-tile sum/max reductions, with multiple
inputs and outputs on exact `sm_100a` or `sm_103a`. The BF16 register-MMA study above
continues to use its own Workload and Study contracts.

Declare one role with `warps=[0]`, nonpersistent scalar (tile=1) ProgramMap axes, global
inputs/outputs, register intermediates and PROGRAM/DIMENSION AccessMaps. Element i is
owned by lane i%32, slot i//32. Reduction folds local values then a full-warp collective;
broadcast shuffles source slots uniformly. Padded lanes still participate. Global
copies are scalar, with `coalesced=false` stores. Unsupported cache/reuse, residency,
pipeline, barrier, loop and storage declarations are refused. The logical live-slot
limit is an implementation bound, not physical register allocation or a spill estimate.

Compilation accepts the complete ordered FP32 pointer signature. PTX and CUBIN checks
bind parameter count, offsets, pointee alignment, address space and exact target.
Source admission remains one kernel and fixed imports. CPU execution of generated source
checks broadcasts, reduction axes, lane slots and output writes. It does not simulate
CuTe compilation, GPU math approximations or physical allocation. Actual SDK compilation,
GPU correctness, timing and profiling require separate evidence and remain unqualified
by these source/CPU checks.
