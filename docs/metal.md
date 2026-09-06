# Authoring for Apple Metal

Write a Python Schedule with the existing tensor frontend, assess it, then inspect the
generated Metal source. The first target is `apple_gpu_family8` on the exact device
`Apple M2`; another Apple GPU is not an implicit fallback.

The [elementwise example](../examples/python/metal_elementwise.py) computes
`(x + y) * 2` on `(3, 37)` tensors. The [row-sum example](../examples/python/metal_row_sum.py)
reduces `(5, 65)` to `(5,)`. Both use normal Python tensor expressions and the same
canonical Schedule as JSON authoring. No hand-written Metal is needed.

```python
from open_cake_ir.compiler import Compiler, frontend

compiler = Compiler.load(".", "compiler/revision.lock.json")
source = frontend.read_schedule("examples/python/metal_row_sum.py")
assessment = compiler.assess(source.document)
for finding in assessment.findings:
    print(finding.code, finding.path, finding.message)
    print(source.location_for(finding.path))
if assessment.lowering_eligible:
    lowered = compiler.lower(assessment)
    print(lowered.source)
    print(dict(lowered.toolchain_requirements))
```

An agent follows the same loop: change the Python expression or a concrete scheduling
decision, read localized Findings, fix the relevant declaration, and inspect lowering
before requesting an on-device check. Assessment is a static result; it does not establish
GPU correctness or speed. Unsupported commitments are refused instead of silently erased.

## Current execution boundary

This is a correctness prototype with **one active lane per program tile**. A threadgroup
has 32 threads; lane 0 walks the private FP32 values and performs reductions in increasing
index order. It is not an optimized SIMD reduction. The emitter admits composable
FP32 arithmetic and finite sum/max reductions within its modeled subset, subject to
private-storage and buffer-count limits. The examples use one row per program tile,
`coalesced=False` stores, and `across_loop=False` reductions. Odd row lengths are supported
without caller-provided padding. Wider program tiles, loop-carried reductions, shared
collectives, tensor operations and unsupported memory/access commitments are refused.

The generic [Swift adapter](../tools/metal/runner.swift) consumes buffer order and launch
metadata from lowering; the [Python boundary](../tools/metal/adapter.py) derives dtype,
shape and byte sizes from the assessed Schedule. It checks exact device/family and pipeline
limits, compiles with `MTLDevice.makeLibrary`, initializes outputs to NaN, waits for command
completion and preserves input/output bytes. It requires macOS 15+, Apple Silicon and an
existing `xcrun swiftc`. A standalone `xcrun metal` executable is not required.

Compilation explicitly selects MSL 2.3, safe math and precise math functions; generated
source disables contraction. Metal still permits behavior such as denormal flushing and
different FP32 rounding modes. The current oracle covers finite normal-range inputs and
zeros with explicit tolerances, not IEEE/PTX bit equivalence. See Apple's
[compile options](https://developer.apple.com/documentation/metal/mtlcompileoptions) and
[MSL specification, sections 1.6.3 and 8](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

## Correctness evidence

After the applicable release gates pass and local GPU execution is authorized, run from
the checkout with an absolute output directory outside every project worktree:

```sh
env PYTHONPATH=src python3 tools/metal/check_correctness.py \
  --output-root /absolute/external/metal-checks
```

Each invocation creates a fresh receipt directory. The harness verifies the reviewed
released Compiler, records the committed and clean runtime source identity, uses public
`assess`/`lower`, and checks 60 combinations: elementwise, row sum, and a row-max variant
of the same Python reduction template; five shapes (including row lengths 1, 7, 32, 65
and 257); and four deterministic
input distributions. The external CPU oracle checks output lengths, finite values and
numerical error, while byte comparisons check input immutability. A failure exits nonzero
and retains the receipt. Static findings, compilation/completion and output correctness
remain separate fields. These checks establish only the listed local output cases;
no timing, profiler, framework integration or end-to-end performance result is implied.

Portable host tests dispatch no GPU work:

```sh
env PYTHONPATH=src python3 -m unittest tests.contracts.test_metal_runtime
```

There is no calibrated Apple timing model in this slice. Inspect any emitted coverage
limitations rather than interpreting missing estimates as free resources.
