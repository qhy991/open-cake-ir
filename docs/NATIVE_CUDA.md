# Native CUDA/PTX lowering

[中文](zh-CN/NATIVE_CUDA.md) · [Design and limits](NATIVE_CUDA_DESIGN.md)

`native_cuda` generates standalone CUDA C++ and inline PTX from the operation DAG.
It implements exact B200 (`sm_100a`) and B300 (`sm_103a`) targets independently. It uses
NVIDIA nvcc/ptxas, without invoking a Triton/CuTe compiler or selecting a kernel template.

The examples under `examples/schedules/native/` are complete Schedules:

| Example | Explicit composition |
| --- | --- |
| `gemm-bias.json` | TMA → shared → tcgen05 → TMEM load → FP32 bias add → store |
| `gemm-bias-k65-tail.json` | Masked global → swizzled shared, preserving the same MMA/epilogue path for K65 |
| `two-mma.json` | Two independent MMA accumulators, two TMEM loads and an explicit FP32 add |
| `kmeans.json` | Batched contraction, multiply/add distance, streaming argmin with global indices |
| `kmeans-partials.json` | Complementary K contributions in two explicit TMEM results, two readouts and FP32 ADD |

The native half/BF16 placement contracts are written explicitly. Changing a Triton
backend name does not construct these declarations. KMeans consumes `centroid_sq` at the
existing core-kernel preparation boundary; its preprocessing must be accounted for by
the evaluator. The original Workload requires exact INT32 assignments and lowest-index
ties. A distance tolerance cannot substitute for that oracle.

`kmeans-partials.json` keeps each physical input stage at K64. Its two MMA operations
select `k_ranges=[[0,16],[32,48]]` and `[[16,32],[48,64]]`, respectively. Intervals are
relative to the input tile and end-exclusive. The typed owner rejects invalid ranges,
merges legal adjacency and canonicalizes full coverage to omission. Each endpoint must
align to the declared instruction's K step. An accumulator initializes at its first
selected contribution, even when that first coordinate is nonzero. Both results own
separate TMEM regions, completion barriers and readouts; stage reuse waits all readers.
The compiler creates no additional accumulator implicitly.

The companion `examples/schedules/triton/kmeans-partials.json` keeps full K128 resident
operands and a single centroid loop. It gathers the four even/odd intervals into two
K64 BF16 dots, materializes each FP32 result with an opaque NVIDIA register move, then
adds them. This boundary prevents the downstream compiler from absorbing the first
partial into the second dot's accumulator. Initial selected-dot support is restricted
to NVIDIA `sm_100a`/`sm_103a`, BF16, K128 inputs, 64 selected elements and power-of-two
M/N >=16. Contraction-loop carry, other selected lengths/contracts and CuTe/Metal are
refused. Existing generic output/candidate-domain rules still apply.

Two complementary partials count the same multiply/add products as the original
contraction, plus the declared FP32 ADD. Full TMA traffic stays unchanged; the second
TMEM region and live register result remain visible in storage analysis. These examples
are source contracts. Separate diagnostic success does not qualify their newly emitted
issue order, completions or numerical behavior; the unchanged Workload oracle must be
run on their own compiled artifacts before timing/profiler qualification.

When all candidates fit in one tile, the Schedule omits the candidate loop and declares
`across_loop=false`. Local tile indices then match global candidate indices because the
supported resident domain is complete and starts at zero. Both native and Triton
lowering derive that domain from the access maps and exclude padding from selection.

## Generate and inspect

From this checkout, keep generated files outside it:

```sh
PYTHONPATH=src python3 -m open_cake_ir.cli compiler assess \
  --revision compiler/revision.json examples/schedules/native/gemm-bias.json
PYTHONPATH=src python3 -m open_cake_ir.cli compiler lower \
  --revision compiler/revision.json examples/schedules/native/gemm-bias.json \
  --output /absolute/external/output/gemm-bias.cu
```

These commands explicitly use a development draft. After independent release approval,
use `compiler/revision.lock.json` for a released Compiler. Lowering never compiles or
launches the kernel. The Python API also exposes the launch/compile metadata:

```python
from pathlib import Path
from open_cake_ir.compiler import Compiler
root = Path.cwd()
compiler = Compiler.load(root, root / "compiler/revision.json")
assessment = compiler.assess_file(root / "examples/schedules/native/gemm-bias.json")
lowered = compiler.lower(assessment)
print(dict(lowered.toolchain_requirements))
print(dict(lowered.source_map))
```

Use the emitted `nvcc_flags` verbatim: the exact architecture/code pair matters. Append
`-shared -Xcompiler=-fPIC` and link `cuda`/`cudart` for the host ABI, or compile the same
source to PTX/CUBIN for resource inspection. Each source contains its exact target guard.
Source markers locate every operation and allocation/barrier/pipeline declaration.

## Host interface

The metadata owns `argument_order`, `arguments`, `signature`, `kernel_entry_point`,
`grid`, `block`, `threads_per_cta`, `dynamic_shared_bytes`, `nvcc_flags` and `host_abi`.
All are derived from the Schedule/Target. No operator-name lookup is involved.

- `int <entry>_create(void** device_buffers, void** handle_out)` encodes tensor maps,
  checks the exact device and configures dynamic shared memory. Pointers follow
  `argument_order`. It performs no input copies, padding, norm computation or GPU work.
- `int <entry>_launch(void* handle, void* cuda_stream)` launches on the supplied stream.
  The current CUDA device must equal the device recorded at create. The caller owns
  synchronization and output validation.
- `int <entry>_destroy(void* handle)` frees host metadata. Complete queued launches
  before destroying the handle or its caller-owned device buffers.

Zero is success. Nonzero CUDA runtime errors are returned unchanged; tensor-map Driver
API errors return `10000 + CUresult`. No errors are converted into a fallback kernel.

## Verification boundary

Run `PYTHONPATH=src python3 -m unittest tests.contracts.test_emit_cuda -v` for the native
public boundary. It covers the two uses, multiple MMA nodes, tail staging, target/role/
stage/tile changes and localized refusals. The full Corpus Gate also covers native
positive and deliberately invalid Schedules while retaining prior backend expectations.

The initial store mapping is row-owned and explicitly declares `coalesced=false`.
A pipeline producer can get up to the declared stage count ahead. Its consumed barrier
is signalled only after all same-thread MMA readers finish. A different issuing role
cannot share that implicit completion coverage. Unsupported role register budgets,
residency demands, range knobs and placements are refused during assessment.

Static/source success, exact compilation, device correctness, matched timing and profiler
are separate evidence. No native speedup, calibration or framework acceptance follows
from code generation. The external delivery ledger owns current acceptance; this guide
is not a status report.
