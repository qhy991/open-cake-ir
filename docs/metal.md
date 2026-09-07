# Authoring and measuring Apple Metal programs

Write a Python Schedule, read localized compiler findings, and inspect the emitted Metal
before testing it. The exact targets are `apple_gpu_family7` on `Apple M1 Pro` and
`apple_gpu_family8` on `Apple M2`.
Python and JSON use the same canonical Schedule; another GPU is never an implicit fallback.

The [elementwise example](../examples/python/metal_elementwise.py),
[row reduction](../examples/python/metal_row_sum.py), and
[weighted RMSNorm](../examples/python/metal_rmsnorm.py) use the existing tensor frontend.
RMSNorm composes square, sum, scalar arithmetic, rsqrt and multiplication:

```python
from open_cake_ir.compiler import Compiler, frontend

compiler = Compiler.load(".", "compiler/revision.lock.json")
source = frontend.read_schedule("examples/python/metal_rmsnorm.py")
assessment = compiler.assess(source.document)
for finding in assessment.findings:
    print(finding.code, finding.path, finding.message)
    print(source.location_for(finding.path))
if assessment.lowering_eligible:
    lowered = compiler.lower(assessment)
    print(lowered.source)
    print(dict(lowered.toolchain_requirements))
```

An agent changes a formula or concrete scheduling decision, reads the returned findings,
fixes the relevant declaration, and inspects the next lowering. Source maps connect emitted
operations to the Schedule. Static acceptance, compilation, output correctness and a
qualified measurement remain distinct results.

## Execution and numerical scope

Current lowering assigns each flattened value to lane `index % 32` and private slot
`index // 32`. All 32 lanes participate in supported SIMD collectives, including tails;
program coordinates retain ownership of separate output regions. Scalar `(1,)` results can
feed tensor arithmetic. Odd widths need no caller padding. The compiler checks supported
access, shape, storage, broadcast and reduction declarations; unsupported commitments
produce localized refusals. Per-lane storage analysis is a modeled view, not a measurement
of physical registers, spills, residency or bandwidth.

The [generic Swift runner](../tools/metal/runner.swift) accepts the current SIMD launch
metadata and the explicit serial reference/replay seam. It derives no behavior from an
operator name. The [Python boundary](../tools/metal/adapter.py) projects buffer shape, dtype,
size and launch information from the assessed Schedule. The runner verifies exact device
and pipeline limits, reuses device/queue/pipelines/buffers, poisons outputs outside timing,
uses serial dispatch ordering, and checks command completion and input immutability.

Runtime source compilation uses `MTLDevice.makeLibrary`, MSL 2.3, safe math, precise math
functions and contraction disabled in source. An existing `xcrun swiftc`, Apple Silicon
and macOS 15+ are required; standalone `xcrun metal` is unnecessary. This does not establish
IEEE/PTX bit equivalence: Metal permits denormal flushing and different FP32 rounding
behavior. See Apple's [compile options](https://developer.apple.com/documentation/metal/mtlcompileoptions)
and [MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
The host executable is built once with `swiftc -O` and shared by all arms;
`swift-build-command.json` records the invoked build argv. Metal math options remain separate.

## Correctness and measurement

After independent release review and the applicable local GPU authorization, run from the
checkout with absolute output roots outside every project worktree:

```sh
env PYTHONPATH=src python3 tools/metal/check_correctness.py \
  --target apple_gpu_family7 --output-root /absolute/external/metal-correctness
env PYTHONPATH=src python3 tools/metal/benchmark.py \
  --target apple_gpu_family7 --output-root /absolute/external/metal-measurements
```

Each command verifies a reviewed released Compiler and committed, clean runtime sources,
then creates a fresh external receipt. Correctness retains the 60 elementwise/sum/max cases
and adds 35 RMSNorm cases. The [RMSNorm contract](../tools/metal/rmsnorm.py) owns the equation,
input ranges, seeds, epsilon, tolerances and shapes. It includes zeros, normal bounded inputs,
epsilon-dominated inputs, negative/zero weights, and widths 1, 7, 32, 65, 257, 1024 and 4096.
The independent high-precision CPU oracle uses fixed `atol=rtol=2e-5` for RMSNorm.

The [benchmark protocol](../tools/metal/benchmark.py) compares four equivalent RMSNorm
formula DAGs on the fixed primary `(128, 1024)` shape. Handwritten serial and SIMD references
have explicit source provenance and the same input bytes and oracle. They are not forged
Compiler-generated artifacts or a previous-Compiler RMSNorm result: the older serial
Compiler could not lower rsqrt. This is known-kernel reproduction/optimization.

One process constructs every treatment before comparison. A reference-only pilot chooses
a fixed batch count from bounded powers; randomized matched sweeps then perform one search
round and two independent confirmation rounds for the selected candidate. An independent
identical-reference slot supplies the A/A noise control through the same binding/dispatch
path. Buffers stay warm; there is no cache flush or inherited NVIDIA CUPTI protocol.

Receipts keep these intervals separate:

| Field | Actual interval |
| --- | --- |
| Cold construction | Swift host build; device/queue creation; host preparation and library/pipeline construction, with possible system caches |
| Warmed host call | Encode, submit and wait for completion; excludes poison, oracle checks and file I/O |
| GPU command buffer | `GPUEndTime - GPUStartTime` after command completion |
| Amortized dispatch | Command-buffer interval divided by the recorded dispatch count; not pure kernel latency |

The predeclared engineering rule requires A/A paired median ratio in `[0.95, 1.05]`,
relative IQR at most 10% in each relevant arm, and a gain exceeding 5% in search and both
confirmations. Otherwise the result is inconclusive or has no material gain. Raw samples,
orders, warmups and batch counts are retained without trimming; invalid timers, execution
or correctness fail with a nonzero exit. See Apple's [GPU command-buffer timestamps](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime).
Each pilot and ordinary batch carries its own output/input validation, completed outside
timing before another dispatch can overwrite its buffers; profile validation is separate.

After ordinary timing, a separate instrumented observation requests compute-stage
`GPUTimestamp` samples when supported. It records actual capability enumeration, resolved
raw values and absence/failure reasons. Counter values are not converted to host time or
called kernel cycles. Occupancy, bandwidth and instruction counters are not inferred.
See Apple's [counter sampling](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers).

`feedback.json` gives candidate disposition, localized findings, search/confirmation
results and rejection ownership through the existing Lab routing vocabulary. There is no
calibrated Apple ranker or inferred ranking inversion. This local evaluation is not a
qualified Study/provider campaign, framework integration, serving or end-to-end result.
Portable tests dispatch no GPU work:

```sh
env PYTHONPATH=src python3 -m unittest \
  tests.contracts.test_metal_runtime tests.contracts.test_metal_benchmark
```

For M2 pass `--target apple_gpu_family8` (the backward-compatible default).
The selected target is retained in every Schedule, manifest and receipt. The fourth
RMSNorm DAG scales weights by inverse RMS before multiplying the input. All four
are hypotheses under the same oracle; no improvement is assumed. Cost-model
abstention is retained before dispatch, without borrowing another GPU calibration.
See the [M1 Pro successor design](metal-m1-pro-design.md) for scope and future work.
