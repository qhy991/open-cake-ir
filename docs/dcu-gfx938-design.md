# Hygon DCU (gfx938) target design

The exact device, what admitting it cost, and what is still not admitted. The Compiler
draft now binds `gfx938` alongside `sm_100a`, `sm_103a` and Apple families 7/8/9, and the
Triton route emits for it. Nothing here is released: a released successor needs an
independent [ADR 0052](adr/0052-independent-agent-release-review.md) review, and no DCU
Executor exists, so no campaign, claim or calibration is in scope.

Every hardware fact below was read inside the development container on `bw1100`, from
`rocminfo`, `hipGetDeviceProperties`, `hipcc` and Triton itself. Retained captures and
probe scripts: `~/.local/share/open-cake-ir/dcu-gfx938-survey-20260914/probes/`; the
emitted lowering and its on-device check are under `.../lowering/`. Cited by
[F-2026-09-15-002](../findings/2026-09-15-002-dcu-gfx938-closes-the-amd-naming-sequence.json).

## The device

One node, eight agents, each reporting ISA `amdgcn-amd-amdhsa--gfx938:sramecc+:xnack-`.

| Fact | Value | Where it lands |
| --- | --- | --- |
| Marketing name / vendor | `BW1101` / `C-3000` | `device_names`, `architecture: c3000`, `vendor: amd` |
| Wavefront size | **64** | `warp_size`, and every thread count derived from it |
| Compiled object | `hsaco` | `code_object`, read by the Evaluation layer |
| Workgroup max size | 1024 | `maximum_threads_per_cta` |
| Workgroup max slots | 1024 / 64 = 16 | `maximum_warps_per_cta` |
| GROUP segment | 64 KiB | `maximum_shared_memory_bytes` |
| Tensor memory | none | `maximum_tensor_memory_bytes: 0` |
| Grid max per dimension | 2147483647 (HIP) | `resource_limits.grid` |
| Compute units | 64, 4 SIMD each | `occupancy.multiprocessor_count` |
| Registers per CU | 131072 | `occupancy.registers_per_multiprocessor` |
| LDS per CU | 64 KiB | `occupancy.shared_memory_per_multiprocessor_bytes` |
| Threads per CU | 2560 (40 waves x 64) | `occupancy.maximum_threads_per_multiprocessor` |
| L1 / L2 | 32 KiB / 8 MiB | L2 size sets the flush obligation for timing |
| Device memory | ~144 GiB | not a Target field |

`rocminfo` reports 4294967295 per grid dimension and HIP reports 2147483647. HIP is the
launch API a kernel actually goes through, so its limit is the one declared.

Two sources disagreed about LDS per compute unit. `torch.cuda.get_device_properties`
reports `shared_memory_per_multiprocessor` 4194304, while `hipGetDeviceProperties`
reports `maxSharedMemoryPerMultiProcessor` 65536 and `rocminfo` reports a 64 KiB GROUP
segment. The compiler's own occupancy model settles it: for one 256-thread workgroup,
`hipcc -Rpass-analysis=kernel-resource-usage` reports 10, 8, 4, 2 and 1 waves/SIMD as
declared LDS rises 4, 8, 16, 32 and 64 KiB, which is 64 KiB per CU exactly. The PyTorch
figure is not used.

No `peak` is declared. `memory_bus_width` 256 at `memory_clock_rate` 875 MHz implies
~56 GB/s, two orders below the ~1.76 TB/s measured below, so neither field is quotable;
and the measurement below is an exploratory probe, not a Target peak observation.
Timing-model coverage for gfx938 is **unavailable**.

## Two routes

**Triton on the `hcu` backend** is the route this change implements. Triton 3.6.0 in the
container registers `amd`, `hcu` and `nvidia`; the active driver is
`triton.backends.hcu.driver` and reports `GPUTarget(backend='hip', arch='gfx938',
warp_size=64)`. Ahead-of-time compilation through `ASTSource` + `triton.compile` with
that explicit `GPUTarget` succeeds **in a container with no `/dev/kfd` and no
`/dev/dri`**, where Triton reports "0 active drivers", so the offline-compilation
property survives the port unchanged.

**HIP C++ as the peer of `native_cuda`** is surveyed and deliberately not proposed.
`hipcc` (dcc 25.10.0 / clang 17) compiles for `--offload-arch=gfx938`, and
`-Rpass-analysis=kernel-resource-usage` reports SGPRs, VGPRs, ScratchSize bytes/lane, LDS
bytes/block, occupancy waves/SIMD and both spill counts at compile time -- the exact peer
of the `-Xptxas=-v` contract `native_cuda` relies on. Per AGENTS.md a new
`LoweringBackend` member is justified by a Schedule that cannot express its work through
an existing route. A recurring expressibility failure on the Triton route would be that
evidence; its absence is not.

## What this change implements

### The lane width and the vendor are declared facts

This work first carried its own answer to the lane width: an optional `wavefront_size`,
required only for the architectures whose ISA does not fix 32. A pull retired it. Compiler
v81 had already made `warp_size` a **required field on every Target document**, which is
[F-2026-09-13-006](../findings/2026-09-13-006-amd-naming-needs-paired-successors.json)'s
step 1 and the better answer, because it leaves no document able to omit a fact the shared
code would then invent. gfx938 declares `warp_size: 64` on that schema.

Steps 2 and 3 of that sequence land here, because a gfx938 document cannot parse without
them. `vendor` becomes a required declared field over a closed set, so a well-formed AMD
document is refused for the reason that applies to it rather than with
`target.compute_capability is required for CUDA targets` -- and the fabricated-capability
path, which made the same document parse and then silently carry a 32-lane width and a
four-slot warpgroup rule against a 64-lane wavefront, is closed. `warps_per_warpgroup`
stops being `4 if compute_capability is not None else None` and becomes declared: present
with its citation on the two NVIDIA documents, absent -- not zero, not a borrowed four --
on the others. `executable_role` and the Executor host validator stop refusing every other
vendor in CUDA's words. A synthetic third-vendor fixture holds all of it, and the
exact-target pin stays what AGENTS.md keeps it for.

Nine shared computations multiply by `warp_size` and every one of them was already
parameterized on it; what changes is the answer. Corpus case
`gfx938-wave64-thread-extent` is one Schedule declaring 32 slots: lowering-eligible on
`sm_100a` at 1024 threads, refused on `gfx938` at 2048.

### The Triton compile path is routed, not assumed

`toolchain.compile_triton` carried four CUDA commitments. Each is now a field of a
`TritonRoute` selected by the declared target:

| | CUDA | gfx938 |
| --- | --- | --- |
| `GPUTarget` | `("cuda", 100\|103, 32)` | `("hip", "gfx938", 64)` |
| artifact roles | source, ttir, ttgir, llir, **ptx, cubin** | source, ttir, ttgir, llir, **amdgcn, hsaco** |
| exact-target text | `.target sm_100a` in the PTX | `amdhsa.target:` in the AMDGCN |
| scratch fields read | `global_scratch_size`, `profile_scratch_size` | `profile_scratch_size` only |

The last row is not a shortcut. `HIPOptions` defines no global scratch field at all, so
the previous code would have raised `AttributeError` rather than refused; reading an
absent field through a default would instead have reported "no auxiliary scratch" about a
backend that was never asked. The route also checks `metadata.warp_size` against the
declared lane width, because a backend that compiled at another width would invalidate
every thread count the analyses derive.

### Register budgets are refused rather than silently dropped

`HIPOptions` has no `maxnreg` field and Triton's option parser drops an unknown key
without raising, so the cap the emitter attaches for CUDA is accepted here and never
applied. A Schedule that declares `residency.registers_per_thread` on gfx938 is therefore
refused with `TRITON_AMDGCN_REGISTER_BUDGET_UNENFORCEABLE` (corpus case
`gfx938-register-budget-unenforceable`). This is the reason the accepted gfx938 corpus
case declares no residency block.

### The CUBIN inspector has an AMDGCN peer that needs no binary

`inspect_triton_resources` shells out to `cuobjdump`, which is absent from the DTK image,
as are `roc-obj` and `llvm-objdump`. It needs no replacement binary:
`inspect_amdgcn_resources` reads `.vgpr_count`, `.group_segment_fixed_size` and
`.private_segment_fixed_size` out of the `.amdgpu_metadata` note the AMDGPU backend
already wrote into the assembly Triton returned.

Two counts do not map one-to-one, and neither is reshaped to look as though they do.
`.sgpr_count` is per wavefront, not per thread, so it is not folded into the per-thread
register number. `.private_segment_fixed_size` is one per-lane scratch allocation
covering what CUDA reports separately as LOCAL and STACK; the whole figure is reported as
local bytes and the stack figure is zero because it is not observable here, not because
no stack frame exists.

Static LDS is a related trap, and the same one AGENTS.md records for
`gemm-bias-b1-smoke`: the lowered rmsnorm reports `.group_segment_fixed_size` 0 while
`metadata.shared` is 512, because the shared memory is allocated dynamically at launch. A
residency report reading only the static directive would say "unconstrained" and mean
"not looked at", so the inspector carries both and `shared_bytes` sums them.

### What the target admits

`gfx938` declares `load`, `elementwise`, `reduce` and `store`, the `global`, `shared` and
`register` spaces, and **no instruction contracts**. That is deliberate. The FP16
`tl.dot` this device supports lowers to `v_mmac_f32_16x16x16_f16` -- not `v_mfma_*` --
and the `hcu` backend metadata carries `enable_v_mmac_cluster`, `mmac_layout_force` and
`empty_arrive_after_mmac`. This is not a relabelled CDNA part, so its matrix contracts
cannot be borrowed from an MFMA or NVIDIA vocabulary, and P8 wants each admitted contract
to carry documented hardware behaviour. Declaring none is what keeps the Schedules that
name `ptx.fma.rn.f32` or `libdevice.tanh.f32` refused on this target rather than lowered
through an NVIDIA inline-asm body.

## What the task registry admits, and why it used to admit less

The 29 tasks the launcher exposes reach gfx938 through one device registry row:

```
"triton-dcu": {"target": "gfx938", "device_name": "BW1101",
               "provenance_token": "BW1101", "route": "triton",
               "tanh_contract": None, "power_of_two_width": True}
```

Adding the row was not enough, because five of the 29 were not portable to begin with.
`normalization` (rmsnorm, layernorm, residual_rmsnorm, softmax) and `gemm` (gemm_bias)
read a second registry, `tasks/apple.py`, that held only the three Metal rows, and their
starters wrote `backend="metal"` as a literal. `--task rmsnorm --backend triton-b200`
therefore refused with `unsupported normalization task/backend` before this change --
on B200, not only on a DCU. `add_rmsnorm` had the mirror-image problem: a literal
`BACKENDS_SUPPORTED = ("triton-b200", "triton-b300")` and `backend="triton"`.

Those five now derive their route and Target from the device row the way the other five
families already did, `tasks/apple.py` is a read-only Metal view of the one registry
rather than a second table, and `add_rmsnorm` asks what its ABI needs instead of naming
devices:

| check | question it asks | example refusal |
| --- | --- | --- |
| `admit_width` | can this route tile this row width? | `triton-b200 tiles a row with tl.arange, which requires a positive power-of-two span` |
| `tanh_contract` | does this Target admit a tanh instruction contract? | `triton-dcu has no admitted tanh instruction contract` |
| `admit_dtype` | can this route name this dtype? | `metal-m4 lowers through metal, which cannot name dtype 'bf16'` |
| `admit_operations` | does this Target admit this body's operation kinds? | `triton-dcu targets gfx938, which does not admit operation kind 'cast'` |

Every existing frozen Workload and starter is unchanged: all 280 (task, backend, case)
combinations that produced a document before produce byte-identical bytes after, and the
20 that changed were refusals becoming available -- the five formerly Metal-only tasks on
the two CUDA Triton backends.

**27 of the 29 launcher tasks reach a lowering-eligible Schedule on gfx938.** The two
that do not are `gelu_tanh` and `gelu_tanh_backward`, refused by name because gfx938
declares no tanh contract. On ROCm, Triton's `libdevice` resolves to ocml; reusing the
CUDA spelling `libdevice.tanh.f32` would claim NVIDIA libdevice numerics for a different
function, so the row declares `None` and the task is refused rather than lowered against
numerics nobody measured here. Admitting one is a Target change with its own evidence.

Lowering is stage one of four. Nothing above evaluates anything -- see
[F-2026-09-15-003](../findings/2026-09-15-003-evaluation-layer-has-no-amdgcn-peer.json) for
the Evaluation half that does not exist yet. `--backend triton-dcu` reaches
`no current Executor is published for exact target 'gfx938'`, which is the correct place
to stop.

## Measurement: per-dispatch timing and L2

**`torch.cuda.Event` is not the CUPTI peer.** Its empty-interval overhead measures min
8.48 us / median 9.28 us, against kernels of interest at 3-7 us. The same 1 M-element
`mul` measured 11.5-31.2 us by events and 6.72 us by the profiler.

**`rocprofv2 --kernel-trace`** (from `/opt/dtk/rocprofiler/bin`, which `env.sh` does not
put on PATH) writes one CSV row per dispatch with nanosecond `Start_Timestamp` and
`End_Timestamp`, and alongside them `GRD`, `WGR`, `LDS`, `SCR`, `Arch_VGPR`,
`ACCUM_VGPR`, `SGPR` and `Wave_Size`. One trace therefore serves both the timing and the
runtime resource report. `hipprof --hip-trace` is a second source with per-dispatch
durations at 10 ns granularity plus a per-kernel `TotalDurationNs`/`AverageNs` summary.
`rocprofv3` does not exist in this image.

**L2 flush** has no HIP API; the mechanism is a device buffer larger than L2, and a
256 MiB `zero_()` measures 243.8 us. Its effect, `torch.mul` warm versus flushed, from
`rocprofv2` per-dispatch durations:

| buffer | traffic | warm | cold | cold/warm | warm GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 MiB | 2 MiB | 3.36 us | 4.16 us | 1.24 | 624 |
| 2 MiB | 4 MiB | 3.84 us | 4.32 us | 1.12 | 1092 |
| 4 MiB | 8 MiB | 6.32 us | 6.24 us | 0.99 | 1327 |
| 16 MiB | 32 MiB | 20.32 us | 20.32 us | 1.00 | 1651 |
| 128 MiB | 256 MiB | 152.16 us | 152.24 us | 1.00 | 1764 |

The residency effect is real and confined to buffers below about 4 MiB. At and above
that, input and output together meet or exceed the 8 MiB L2, the kernel evicts its own
input while running, and warm and cold are indistinguishable. The flush obligation stands
for exactly the small shapes where it changes the answer, and it costs 244 us a sample.

Two negative results are retained rather than dropped: a 2 MiB `mul` pair measured 1.06x
and a 6 MiB `sum` reduction 0.94x at ~400 GB/s, both latency-bound rather than
bandwidth-bound. An earlier run reported ~1.96x by matching kernel names on
`vectorized_elementwise`, which also matches PyTorch's fill kernel; that number is wrong
and is superseded by the `MulFunctor`/`FillFunctor` split.

## On-device evidence

The Compiler's own path lowered `gfx938-rmsnorm-b8-smoke` and compiled it in the
container to an hsaco: 256 threads per CTA (4 slots x 64, not 128), 512 dynamic shared
bytes, 134 VGPRs, no scratch. Running the emitted host wrapper against a PyTorch
reference on one BW1101 matched at `rtol=2e-5, atol=2e-6` across five input
distributions -- primary, zeros, near-zero, alternating signs and mixed magnitudes -- at
a maximum absolute error of 3.8e-6.

That is a correctness check, not an evaluation. It has no Workload Contract, no frozen
oracle, no tolerance contract and no Executor, and it establishes nothing about
performance.

All 27 task baselines that gfx938 admits also compile. Retained log and lowered sources:
`~/.local/share/open-cake-ir/dcu-gfx938-survey-20260914/task-baselines/`. Every one of
them launches 64 threads, because each starter declares one role slot and a slot is a
wavefront here; the same starters are 32 threads on CUDA, which is the lane width change
showing up in the launch rather than in an analysis.

Five of the 27 reach the architected 256-VGPR ceiling and spill:

| baseline | VGPR | dynamic LDS | scratch bytes/lane |
| --- | ---: | ---: | ---: |
| `gemm` | 256 | 256 | 3288 |
| `gemm_bias` | 256 | 256 | 3140 |
| `gemm_silu` | 256 | 256 | 3288 |
| `pairwise_sqdist` | 256 | 1024 | 6656 |
| `attention_decode` | 256 | 0 | 28 |

These are the naive starters -- `gemm` keeps the whole B tile and its products resident
in one role slot, which its own authoring docstring calls the plain reading of the
definition and not a claim that it is cheapest -- so spilling is what the inner loop
exists to fix, not a defect in the target. What is not yet known is whether these
baselines spill on CUDA too at their shapes: no comparison was run, so this is one
target's observation and nothing more. It is recorded as a case, not as evidence for a
rule.

Nothing above is a correctness or performance result for those 27. Checking them is
exactly what [F-2026-09-15-003](../findings/2026-09-15-003-evaluation-layer-has-no-amdgcn-peer.json)
says there is no code for.

## The second AMD target, and what it is evidence for

`gfx1151` is admitted beside `gfx938` from the retained `rocminfo` and
`hipGetDeviceProperties` observations of the earlier delivery on
`codex/main-native-amd-followup-20260909`. This session never reached that device, and
the Target document says so in its own citations.

It earns its place by being **wave32**. It shares a vendor with gfx938 and not a lane
width, so nothing on the AMD route can be a wave64 constant wearing a vendor's name. The
same Schedule lowered for both launches 256 threads on gfx938 and 128 on gfx1151, from
four role slots either way, and the corpus reports a residency bound for gfx938 and none
for gfx1151 because only one of them declares occupancy.

Adding it also found a real bug in the exact-target check. The emitter quotes the
`amdhsa.target` line only when YAML makes it: `'amdgcn-amd-amdhsa--gfx938:xnack-'`
contains a colon and is quoted, while a bare `amdgcn-amd-amdhsa--gfx1151` with no feature
flags is not. The pattern required the closing quote, so it would have refused any
feature-flag-free build of *either* target. The quotes are now optional and the line end
is anchored, which is what keeps `gfx938` from matching a `gfx9380`.

Both targets compile through the route in the DCU container. That is evidence for gfx938,
which is the device in the room; **it is not evidence for gfx1151**. This toolchain's
`llc -march=amdgcn -mcpu=help` lists 51 AMDGPU CPUs including the Hygon gfx926/928/936/938
and RDNA3's gfx1100-gfx1103, and does not list gfx1151. The emitted object names gfx1151
and the compile returns cleanly, and neither fact says the code would run on one. A
gfx1151 result needs a gfx1151 toolchain and device.

## Open decisions

- **Exact-target string.** The device reports `gfx938:sramecc+:xnack-`; the toolchain
  emits `gfx938:xnack-`. The route pins the ISA exactly and admits only well-formed
  feature suffixes after it. What the `sramecc` difference means, and whether an exact
  check should pin the features too, is undecided rather than resolved by the pattern.
- **Target id spelling.** `gfx938` is an ISA shared by all eight agents, not a device;
  `BW1101` is the product. The choice follows `sm_100a` (an ISA) over
  `apple_gpu_family9` (a family), and decides what a second Hygon part collides with.
- **`maximum_registers_per_thread`.** Not declared, because it was not measured. The
  verifier reports `TARGET_REGISTER_CAP_UNMODELED` rather than assuming a CDNA number.
- **Matrix instructions.** `v_mmac_f32_16x16x16_f16` exists and is unadmitted. Adding it
  means a contract name, its numerical behaviour, and the verifier and cost-model rules
  that go with it -- a second change, with its own evidence.
- **An Executor.** One exists: `open-cake-ir-gfx938-v5`, captured from the container's
  exact Python 3.10.12, PyTorch 2.11.0 (HIP 6.3.26113), Triton 3.6.0 and dcc 25.10.0. It
  is a separate gate from the Compiler, and a B200/B300 Executor was never a fallback for
  it. Its successor is pending: see gate 7 below.

## Environment notes

- `docker exec` into a long-lived container is refused for this user by the Hygon device
  plugin (`/var/lib/opt-container/device-plugins/hcu.sock: permission denied`), while
  `docker run --rm` works. Every probe above therefore ran in a transient container.
- `source /opt/dtk/env.sh` does not put `/opt/dtk/rocprofiler/bin` on PATH, so `rocprof`
  and `rocprofv2` appear missing until it is added.

## Gate sequence

1. F-2026-09-13-006 steps 1 to 3: `warp_size` declared (upstream, Compiler v81), then
   vendor identity, warpgroup width and the neutrality fixture. *(done)*
2. The gfx938 Target document, the AMDGCN Triton route and the `.amdgpu_metadata`
   inspector, with three corpus cases adopted as a separate act. *(done: Gate 150/150)*
3. The task registry: one device registry, every family deriving route and Target from
   it, and four capability checks in place of device lists. *(done)*
4. Independent ADR 0052 review of the exact source and Gate diff, then a released
   Compiler successor. *(done: reviewed by haiyan, Compiler v83 then v84, 7 targets,
   210 sources, Gate 151/151)*
5. An Executor successor for every target whose pinned closure this change touched --
   `evaluation/artifacts.py`, `lab/executor.py` and the task registry files -- the paired
   successor F-2026-09-13-006 predicted. *(done: `apple_gpu_family8` v124 on this M2,
   `sm_103a` v123 on B300-M2, `gfx938` v5 on bw1100. `apple_gpu_family7` has none; its
   M1 Pro host is not reachable and no other host may stand in for it.)*
6. A DCU Executor and the rest of the AMDGCN Evaluation half, under
   [F-2026-09-15-003](../findings/2026-09-15-003-evaluation-layer-has-no-amdgcn-peer.json).
   An earlier version of this document said `capture_executor_host.py` could not produce a
   DCU Executor because its `--kind` admitted only `cuda` and `metal`; that was read from a
   stale ref and is wrong. It admits `hip` and `amd`, `HipHostAdmission` verifies the host
   without touching a device, `evaluation/triton_hip.py` owns exact-HIP runtime custody and
   `evaluation/rocprofv3.py` projects kernel-trace, kernel-stats and results evidence
   fail-closed. The executable role, the AMD-parameterized Executor identity and the single
   host `kind` field are in. *(partly done)*
7. Actually launching a Lab task on the DCU. This gate was not on the list until
   `launch_task.py --baseline-only` was run against `triton-dcu` for real. **Done**: the
   DCU now builds and seals a baseline candidate through the real entry point --
   `/runs/.../baseline/candidate.json`, a 5800-byte HSACO beside 19 KB of AMDGCN assembly,
   a launch manifest declaring block `[64, 1, 1]` (one wave64), grid `[128, 1, 1]` and one
   hidden scratch pointer. Seven separate defects stood between the checklist's 7/8 and
   that file, and only the first two were in the compile chain the checklist covered:

   1. **Allocation.** The launcher read the allocator off the lowering route, so a Triton
      route demanded the `gpu-run` CUDA cluster allocator and refused a device with no use
      for one. Route and allocation are now separate declared columns of
      `tasks/devices.py`; `triton-dcu` is `triton` + `local_broker`.
   2. **The jail binary.** `/usr/bin/bwrap` was a constant -- a fact about the
      distributions this route was built against, not about bubblewrap, which the DTK
      image does not ship at all. Discovered on the host now, with a refusal naming
      bubblewrap instead of a `FileNotFoundError` naming a path nobody chose.
   3. **The provider.** `--baseline-only` says it stops before provider qualification and
      then resolved the harness executable three statements before creating the
      workspace, refusing a DCU baseline for not having `claude` in a compile container.
   4. **The jail environment.** The jail runs `--clearenv`, correctly. Two DTK facts live
      only in `/opt/dtk/env.sh`: without `LD_LIBRARY_PATH` the build died with
      `libgalaxyhip.so.5: cannot open shared object file` while the file sat mounted under
      `/opt`, because ldconfig does not know `/opt/dtk`; with it, clang-18 reported
      `cannot find ROCm device library` because `ROCM_PATH` was gone too. A HIP host now
      declares `runtime.build_environment`, one fact rather than one field per variable.
   5. **Artifact roles.** `TritonToolchainBuilder` named `ptx` and `cubin` as literals
      while `triton_route` has carried `artifact_roles` per target all along, and wrote
      `hidden_null_pointer_parameters: 2` -- Triton's two CUDA scratch pointers. HIPOptions
      has no global scratch field, so the literal would have sealed an ABI with a
      parameter the kernel does not take. Both come from the route now, and the measured
      manifest says 1.
   6. **The launch target.** `CudaKernelSpec.from_dict` resolved its target through
      `cuda_target`, which decodes an sm_1xxa capability first, so a gfx938 manifest came
      back as `unsupported exact CUDA target`. Grid, block and shared-memory limits are
      declared by every Target document and none of them is CUDA's; the Evaluation layer
      reads the document directly now, as `executable_role` in that layer already does.
   7. **Allowed roles.** `allowed_artifact_roles` returned the CUDA set for anything that
      was not Metal, so a candidate carrying the assembly and HSACO its own route produced
      was refused for not being PTX and a CUBIN.

   Five of the seven are the same defect in different files: a per-target question decided
   by "is it Metal? otherwise CUDA". They are recorded in
   [F-2026-09-15-004](../findings/2026-09-15-004-three-axes-collapsed-into-target-identity.json)
   with the process lesson -- the eight-gate readiness checklist reached 7/8 and had no
   gate for how a sealed candidate reaches a device, because every gate in it was
   assembled from a compile-chain failure already seen.
8. A named AMD timing source. **The measurement exists now; the name does not yet.**

   What blocked this was never that the device cannot be timed. It was that the timer
   CUDA uses is an in-process API -- CUPTI wraps one Python callable and hands back
   per-dispatch nanoseconds -- and the DCU's obvious profiler, `rocprofv2`, wraps a whole
   process. Naming it would have changed the assay's shape, not just its source. Three
   candidates were measured instead of argued:

   | candidate | result |
   | --- | --- |
   | `torch.cuda.Event` | 8.48 us empty-interval floor against kernels of 3-7 us. Unusable. |
   | raw HIP event pair through ctypes | 5.60 us min / 6.08 us median empty interval; a 1 MiB `mul` reads 15.68 us where the profiler says 3.36 us. Still unusable, and now measured at the C level rather than inferred from torch's wrapper. |
   | `torch.profiler` (roctracer underneath) | Per-kernel device time. A 1 MiB `mul` reads 3.071 us against `rocprofv2`'s retained 3.36 us for the same kernel and shape -- the same measurement through an interface already inside the admitted runtime. |

   roctracer's in-process activity API is present on this host (`roctracer_open_pool_expl`
   and friends resolve, and `/opt/dtk/lib/libroctracer64.so` is on disk), which is what
   makes `torch.profiler` CUPTI's structural peer here rather than a convenience. Reading
   roctracer's records directly would mean pinning a record layout that varies by version;
   torch is a package the Executor host already pins.

   **It attributes the harness's own dispatch.** That was the question a torch-operator
   measurement could not answer, since the evaluation launches through
   `hipModuleLaunchKernel` from its own driver. Measured: `_cake_rmsnorm_fp32_kernel`
   n=1 3.359 us, n=30 2.964 us, launched through `evaluation/hip_driver.py`.

   So all three of what AGENTS.md requires a target to declare can now be stated: the
   timer is `torch.profiler`'s CUDA activity, the interval is the dispatch as roctracer
   reports it, and the device-state reset is the 256 MiB `zero_()` L2 flush, which costs
   244-251 us a sample and changes the answer only below about 4 MiB of working set.

   **The source is wired, and the quality gate refuses the cohorts it produces.** Measured
   on a 128x1024 fp32 rmsnorm, which touches 1.00 MiB against an 8 MiB L2 -- inside the
   region where the flush changes the answer:

   | cohorts | CV | median | distinct values |
   | --- | --- | --- | --- |
   | warm, no reset | 0.031-0.043 | 2.879-3.039 us | 3-4 of 25 |
   | cold, 256 MiB zeroed first | 0.044-0.063 | 5.279-5.439 us | 5-7 of 25 |

   So the spread is the reset's, not the kernel's, and the reset is not optional at this
   size: without it the kernel reads its input from L2 and the number is not a cold
   dispatch. The timer's own quantum is exactly 0.160 us -- confirmed as the smallest
   non-zero step between samples -- which is 3% of a cold median, while the observed range
   spans about sixteen quanta. Quantisation contributes and does not explain it.

   A full evaluation therefore returns `correctness_passed` true with
   `measurement_quality_passed` **false** at the declared 0.05 bound, which is the gate
   doing its job: these cohorts are not tight enough to declare a winner on.

   **The bound is not being widened to make that go away.** A per-target CV bound is a
   calibration, and AGENTS.md gates calibration on that target's own evidence: ten cohorts
   at one shape on one device is a measurement, not a calibration, and a bound tuned until
   the run it judges passes is the state it then reports. What the ten cohorts do settle is
   that the number to calibrate against is the cold one, and that a bound for this source
   has to be stated per reset mechanism rather than inherited.

   One defect found on the way, in shared code rather than here:
   `measurement_quality_passed` compared against the literal `0.05` while the Study
   declared its own `maximum_cv`, so a Study that widened or tightened the bound was
   judged against a number it never named. It reads the declared bound now, and reports
   both that bound and the worst CV observed.

   What is left is the wiring, not the evidence: a benchmark with `StrictCuptiBenchmark`'s
   call shape, a paired policy kind beside `cupti` and `metal`, and a `timing_source` on
   the DCU's registry row. *(next)*

   The paragraph below records what was true before those measurements.

   > **This was where the DCU stopped, and it stopped honestly.**
   The Study template chose its measurement source with `"metal" if metal else "cupti"`,
   so the first DCU study declared `correctness_then_paired_cupti` and
   `fixed_baseline_paired_cupti_v1` -- a profiler's name on evidence that profiler never
   produced, on a machine where CUPTI is not installed. `timing_source` is now a declared
   column of the device registry, `triton-dcu` declares None, and its Study states the
   coverage limitation instead: no `paired_timing`, and an explicit `measurement_coverage`
   saying no latency is reported for this target and why. So the DCU builds, seals and can
   be checked for correctness, and reports that nothing has measured a latency on it.

   Minting the source needs a measurement, not a row. The open question is whether
   `MinNs`, `MaxNs` and `StdDev` -- which `hipprof`'s stats CSV does not carry, verified by
   reading all eight of its columns rather than the five a truncated `sed` first showed --
   can be derived from rocprofv2's per-dispatch `Start`/`End` timestamps. *(blocked on the
   measurement)*
9. The AMDGCN evaluation driver. **Done, and verified on the device.**
   `evaluation/hip_driver.py` is the AMDGCN peer of `cuda_driver.py` -- smaller on
   purpose, since CUDA's cluster attributes, binary-version check and dynamic-shared
   opt-in threshold are not facts about an `amdgcn-amd-amdhsa--` object -- and it names
   no soname (the admitted ROCm PyTorch has already loaded whichever fork this host
   installs) and calls no `hipFuncGetAttribute` (the kernel's own `.amdgpu_metadata`
   already carries its register and scratch counts). `observe_local_hip` is the allocation
   half, reading the device contract off `triton_route` rather than a new request field,
   and the local broker now serializes per device family so a DCU run is not recorded
   under a `metal-` job id.

   Four more not-Metal-so-CUDA branches were on the way there: the worker's own dispatch,
   `allowed_artifact_roles`, an unconditional `collect_timing=True`, and attribution
   handing a DCU candidate to Nsight Compute.

   The device run then named three more that no host test could have. `Compiler.load`
   returns an object with no identity of its own, so a request carrying
   `compiler.revision_id` is refused -- the released Revision is the authority.
   `load_hip_runtime` resolved through `ctypes.CDLL(None)`, on the reasoning that the
   admitted ROCm PyTorch has already loaded whichever fork this host installs; torch loads
   its extensions with RTLD_LOCAL, so on the DCU all five entry points were mapped and
   none globally visible, and the lookup now reads the process's own memory map (still
   naming no soname). And `close()` took no arguments where the shared tensor-tile
   lifecycle closes both drivers through one call with `synchronize=torch.cuda.synchronize`
   -- which surfaced only *after* the kernel had launched, at `module_loads` 1,
   `preflight_calls` 1, `kernel_calls` 1.

   **The passing run**: local-broker job `hip-ecbdb752ace2`, `correctness_passed` true,
   `output_mismatches` 0, `max_abs_error` 4.76837158203125e-07 against the external CPU
   oracle, `inputs_unchanged` true, `timing` null -- the target declares no timing source
   and the receipt says so rather than reporting a latency.
10. A DCU optimization campaign. **Correctly blocked, and it is the same block as gate 8.**
   A single-arm optimization campaign selects on getting faster, so a Study with no timed
   assay has nothing to select on; `validate_evaluation` refuses it and now gives the
   reason that applies -- this target has no named timer -- rather than reporting a
   misconfiguration. So the order is: correctness evaluation on the device (gate 9), then
   a measurement (gate 8), then a campaign. *(blocked on gate 8)*

### Running it in the container

The DTK image needs two things the B200/B300 hosts do not, both established by probing:

- `bubblewrap`, which the image does not ship. It installs from the configured aliyun
  jammy mirror (0.6.1) once `TMPDIR` points somewhere writable -- `/tmp` in this image is
  `drwxr-xr-t`, and apt fails to create its own temporary files there.
- `--security-opt systempaths=unconfined` on `docker run`. Measured, rather than assumed:
  user namespaces work in this container, mount namespaces work, and `bwrap` without
  `--proc` works; the single refused operation is mounting a fresh procfs. On this 4.19
  kernel that is `mount_too_revealing()` -- Docker bind-mounts over `/proc/kcore`,
  `/proc/keys` and friends, and the kernel then refuses any new procfs mount inside a
  nested user namespace. Relaxing seccomp and AppArmor, tried first, changes nothing:
  it is not a filtered syscall, it is a kernel visibility check. Authorized by the
  repository owner on 2026-09-15 for this development container.

### What still blocks a full campaign

`bw1100` is intermittently unreachable from the development host -- several multi-minute
outages over one session, each with no ICMP reply and `ssh` timing out at connect, while
the B300 hosts on the same tunnel stayed up throughout. Work continues across them; runs
are started detached so an outage costs the wait rather than the run.
