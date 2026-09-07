# Exact gfx1151 support

[中文](zh-CN/GETTING_STARTED_AMD.md) · [English home](en/README.md) · [Documentation catalog](README.md)

The AMD path targets **gfx1151 with HIP and 32-lane waves**. A Schedule still selects
`lowering.backend=triton`; the Compiler Target owns the hardware facts. Generated
code, GPU compilation, correct outputs, timing and profiler evidence are separate
results. The [generated status](../reports/current/STATUS.md) names the released
Compiler; this guide does not assign a revision number.

The active development branch is `codex/amd-gfx1151-consolidated`. It carries the
accepted AMD migration rebased onto main's B300 support and task-owned runtime.
[`tasks/amd`](../src/open_cake_ir/tasks/amd) owns AMD Workload validators, numerical
references and search decisions; [`tasks/workloads.py`](../src/open_cake_ir/tasks/workloads.py)
selects their exact frozen contracts. Common Evaluation retains reusable HIP artifact
and profiler handling and does not dispatch AMD operators. These standalone contracts
do not acquire a generic TaskLab or native-tensor ABI through registration.

## What the path covers

| Path | Contract and current boundary |
| --- | --- |
| FP32 SwiGLU | Two separate inputs; CPU FP64 oracle and a correctness-only runner. |
| FP32 RMSNorm plus Mul | Input normalization followed by a separate weight multiply; input storage and bytes must remain unchanged. |
| Raw Q4_0/Q8_1 records | UINT8/INT8 storage with explicit record size and byte access. |
| Q8_1 producer | Typed scale, sum and quantized values; the live check compares all 576 output bytes, including 15 zero padding records. |
| RMSNorm one-row search | Four one-row wave-count candidates, fixed 64-row/four-wave baseline, B/B ABBA noise control and paired replay. The formal Search Contract remains pending. |
| AITER baseline and rocprofv3 | Separate baseline preparation and profiler handoff. Their presence supplies no new matched comparison or performance result. |

The Q4_0/Q8_1 MMVQ consumer is outside this migration. The producer needs its own real
gfx1151 byte comparison before work advances to that consumer. No full llama.cpp,
model-layer, token-generation or serving result follows from these leaf paths.

## Check and generate without a GPU

Use the existing project Python environment described in the
[first-use guide](en/GETTING_STARTED.md). From the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m open_cake_ir.cli compiler assess --format text \
  --revision compiler/revision.lock.json \
  corpus/schedules/swiglu-b8-smoke-gfx1151.json

AMD_SOURCE_DIR=$(mktemp -d)
PYTHONPATH=src .venv/bin/python -m open_cake_ir.cli compiler lower --format text \
  --revision compiler/revision.lock.json \
  corpus/schedules/swiglu-b8-smoke-gfx1151.json \
  --output "$AMD_SOURCE_DIR/swiglu.py"
```

This checks the complete Schedule and writes source outside the checkout. It does not
load HIP or launch a kernel. Lowering carries these exact requirements:

```json
{
  "target": "gfx1151",
  "triton_target": {"backend": "hip", "arch": "gfx1151", "warp_size": 32},
  "binary_role": "hsaco",
  "assembly_role": "amdgcn"
}
```

An unsupported architecture, instruction, dtype or access pattern produces a localized
Finding. The Compiler cannot silently select another architecture. gfx1151 has no
calibrated residency or ranking model; `RESIDENCY_TARGET_UNMODELED` describes that
coverage gap. The search preparation records `ranking_applied=false` and withholds a
cost estimate instead of borrowing NVIDIA calibration.

## How packed values remain explicit

[`ir/resources.py`](../src/open_cake_ir/compiler/ir/resources.py) owns the packed-record
relation. `ggml_q4_0_v1` is 18 bytes and `ggml_q8_1_v1` is 36 bytes. The relation connects
the logical record axis to physical bytes; it is not a layout algebra.

The Q8 producer composes existing generic operations with the migrated primitives:
absolute value, zero-safe division, explicit half-away rounding, register-value
reshaping, a wave32 XOR reduction and typed stores. The sole conversion operation
is `cast`. FP32-to-FP16 uses `rounding=nearest_even, overflow=ieee`; FP32-to-INT8 uses
`rounding=toward_zero, overflow=forbid`. Here `forbid` requires the external input
contract to keep values finite and within INT8 range; the Compiler neither proves
that range nor inserts saturation. The typed Q8 store takes `d:fp16`, `s:fp16` and
`qs:int8[32]` in the record registry's order. See the
[primitive guide](en/wiki/primitives.md) and the
[Q8 producer Schedule](../corpus/schedules/packed-q8_1-producer-gfx1151.json).

## Before a live run

The 2026-09-07 successor delivery is a static and CPU contract migration. A new live
host capture and an admitted gfx1151 Executor for these runtime sources remain pending.
The preserved v1/v2/v3 descriptors belong to historical source commits; loading them
against changed sources must fail.

Host discovery has one entry point:
[`capture_executor_host.py`](../tools/capture_executor_host.py). Its `--runtime-kind hip`
path requires explicit Python packages, ROCm/build-tool paths, `amd-smi`, runtime
libraries and any profiler paths. Capture validates consumed host facts and writes a
new external JSON file. It does not establish GPU correctness or timing. The command's
`--help` lists the required arguments without performing capture.

After the applicable acceptance gates pass and a live-host action is authorized, use
that new capture with
[`release_gfx1151_executor_cycle.py`](../tools/release_gfx1151_executor_cycle.py):

```bash
PYTHONPATH=src "$AMD_PYTHON" tools/release_gfx1151_executor_cycle.py \
  --project-root . --host-environment "$AMD_HOST_CAPTURE"
```

`AMD_PYTHON` denotes the captured ROCm Python executable and `AMD_HOST_CAPTURE` its
external capture file. The cycle derives the next unused identity, assembles a hidden
candidate through the common Executor writer, checks its sources, admits the declared
host and exact Compiler Target, then installs it without replacing a released file.
A missing capture or failed admission preserves every existing release. No version is
chosen by hand. The cycle itself launches no correctness or timing workload.

The correctness entry points are
[`swiglu_amd_quickstart.py`](../examples/gpu/swiglu_amd_quickstart.py),
[`rmsnorm_amd_quickstart.py`](../examples/gpu/rmsnorm_amd_quickstart.py), and
[`llama_q8_1_amd_quickstart.py`](../examples/gpu/llama_q8_1_amd_quickstart.py).
Their preparation paths are CPU-only. A live attempt also requires the released
Compiler, a clean source checkout, its exact Executor and a new external evidence root.
Retained failures remain part of the attempt's evidence.

## Search and profiling boundaries

The one-row hypothesis uses `row_tile=1`, `num_warps` in `{1,2,4,8}`, and a fixed
`row_tile=64, num_warps=4` baseline. Without a frozen formal Search Contract, the
search cannot claim an accepted optimization result. The checked decision path keeps:

- Both Workload cases, unchanged input pointers/bytes and zero fallback.
- A same-artifact B/B ABBA control before candidate screening; unstable cohorts or a
  material false direction make measurement quality inconclusive.
- Raw paired confirmation samples with the declared 0.05 CV and 1.05x materiality
  rules, so the decision can be replayed independently.
- A separate rocprofv3 evidence root after a sealed leaf timing win. Profiling replays
  the parent decision and matches the exact artifact and kernel dispatch; it does not
  use profiler durations as substitute timing samples.

AMDHSA resource fields from generated assembly are observations. They do not establish
occupancy; the projection keeps `occupancy_derived=false`. A timing win still needs its
profiler evidence before a matched external-baseline comparison is considered.

## Historical identity and replay

The accepted refresh at `b2d0a42409afcdef05e66da4723df44a6a09dfef` started from
`ee233f48824a7d1955e449db9636da974d93253f`; `codex/amd-gfx1151-refresh-20260907`
retains that historical accepted tree. Its earlier AMD lineage is
`d32b92f886e21adb51aa5e6648a0a3fbc4562565`, preserved on
`codex/amd-consolidated-before-sync-20260907`. That complete historical tree remains the replay
source for its old descriptors. The unchanged
[v1 external bindings](../inventory/AMD_GFX1151_EXECUTOR_V1_BINDINGS_20260826.json),
[v2 release](../inventory/AMD_GFX1151_EXECUTOR_V2_RELEASE_20260826.json), and
[v3 release](../inventory/AMD_GFX1151_EXECUTOR_V3_RELEASE_20260826.json) retain their
original dates, source commits and evidence meaning. Their observations do not qualify
the successor source tree.

The accepted refresh's B200 v53 preserved the baseline v51 host
declaration explicitly. Updating that source binding did not repeat live host admission;
actual execution must still admit the recorded host. It does not inherit the concurrent
B300 branch's host merely because that descriptor reserves a higher historical id.

The subsequent main synchronization uses pinned main
`6d9a198e69098cd51d427b9bffbe967a6f62de90`. Main and the accepted refresh independently
released different B200 v53 descriptors. Main's descriptor keeps its original path;
the unchanged AMD variant is reserved as
[`open-cake-ir-b200-v53-amd-refresh.json`](../runtime/executors/open-cake-ir-b200-v53-amd-refresh.json).
Replay that variant only at `b2d0a424`, using its original
`runtime/executors/open-cake-ir-b200-v53.json` path. This alias reserves history and
does not describe the rebased runtime. [Compiler v57](../compiler/releases/v57/README.md)
is likewise preserved with its original approval; changed combined sources require
coordinated successor release and independent review.

The earlier refresh reserved Compiler v56 and B200 Executor v52 from the concurrent
B300 branch at `bfc446e384b87634daf5a289a35bc1e36bbad9ff` through
[release metadata](../compiler/releases/v56/README.md). That first refresh imported no
B300 implementation; the later synchronization preserves the B300 implementation now
present on main. The old AMD branch's colliding
Compiler v28 archive also stays at its original Git revision. Existing main archives
are preserved. [ADR 0049](adr/0049-released-executor-descriptors-reserve-their-identities.md),
[ADR 0050](adr/0050-released-compiler-locks-reserve-their-identities.md), and
[ADR 0052](adr/0052-independent-agent-release-review.md) govern successors and approval.
