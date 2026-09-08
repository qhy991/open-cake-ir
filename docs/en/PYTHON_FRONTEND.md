# Write a Schedule in Python

[中文入门](../zh-CN/PYTHON_FRONTEND.md) · [Canonical interface contract in Chinese](../PYTHON_FRONTEND.md) · [English home](README.md)

The Compiler accepts Python variables, expressions, role scopes, and symbolic tile loops as an authoring frontend for the existing Schedule model. JSON remains its canonical serialization. This does not extend operator semantics, backend coverage, GPU correctness, or a frozen Study's authoring environment.

## One complete example

The [FMA example](../../examples/python/fma.py) computes matching positions of a*b+c:

```python
from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="fma-b8-smoke", target="sm_100a", backend="triton",
               entry_point="cake_fma_b8_smoke",
               residency={"ctas_per_multiprocessor": 4, "registers_per_thread": 64})
def fma(lm, a: cake.Tensor((8, 128), "fp32"), b: cake.Tensor((8, 128), "fp32"),
        c: cake.Tensor((8, 128), "fp32"), y: cake.Tensor((8, 128), "fp32", mode="output")):
    compute = lm.role(warps=[0, 1, 2, 3])
    batch = lm.program(a, axis=0, dimension=0, tile=1)
    with compute:
        a_tile = lm.load(a[batch, :], reuse="streamed", id="load_a")
        b_tile = lm.load(b[batch, :], reuse="streamed", id="load_b")
        c_tile = lm.load(c[batch, :], reuse="streamed", id="load_c")
        y_tile = lm.fma(a_tile, b_tile, c_tile, id="fma")
        lm.store(y[batch, :], y_tile, id="store_y")
```

Tensor parameters declare fixed shapes, dtype, and mode. A role owns the operations in its with block; program selects row work. Loads establish data dependencies, fma computes the result, and store writes output. Intermediate register shapes and types are inferred.

`lm.fma` preserves the existing single RN-even rounding contract. `a*b+c` constructs two independent operations. `lm.broadcast(value,axis=...)` uses the existing broadcast_axis relation, not a new splat or reshape.

## Check and generate without a GPU

Use the project environment from [Getting started](GETTING_STARTED.md), running at the repository root. These commands use the released lock and a new external output:

```bash
PYTHONPATH=src .venv/bin/python -m open_cake_ir.cli compiler assess \
  --revision compiler/revision.lock.json examples/python/fma.py --format text

CAKE_PYTHON_OUTPUT=$(mktemp -d)
PYTHONPATH=src .venv/bin/python -m open_cake_ir.cli compiler lower \
  --revision compiler/revision.lock.json examples/python/fma.py \
  --output "$CAKE_PYTHON_OUTPUT/fma.py" --format text
```

Text output currently reports Chinese structural acceptance and generation eligibility. The commands produce source, not GPU binaries, numerical checks, or performance measurements. Existing output files are refused. The canonical contract also describes using revision.json for an unreleased development draft; do not mix changed bound sources with an old released lock.

## Supported authoring contract

- One source file defines one cake.schedule function. Tensor parameters become global input, output, or caller-owned state Buffers.
- lm.buffer names required results or scratch. lm.smem/lm.tmem declare storage, and views preserve concrete offsets, stages, and swizzle.
- Each address has one coordinate per dimension: a program coordinate, loop tile, or continuous static slice. The initial frontend refuses integer coordinates, strided slices, dynamic indirect indices, and omitted dimensions.
- Indexed views are operation addresses only. They cannot be indexed again or stand in for Buffer identities required by lm.program, lm.range, or lm.broadcast. Their ranges must not be silently discarded.
- Arithmetic supports existing expressions and primitive calls. Broadcasting is explicit on the second operand or through an admitted broadcast_axis parameter. Reductions name the operation, axis, and scope.
- Operations with explicit out=buffer or out=(buffer1,buffer2) are not assigned to another variable. lm.store writes back; output modes determine the result collection without a separate return instruction.
- Variables are single-assignment. Read/write dependencies include reads before overwrites; depends_on may add operation ids but cannot replace cross-role waits, signals, or pipeline synchronization.
- lm.range builds one symbolic TileLoop rather than executing its body repeatedly on the host. Defaults are num_stages=1, loop_unroll_factor=1, and false for remaining boolean options. Arbitrary Python range, while, and conditionals are not admitted. Expressible nesting/roles remain subject to each backend.

## Parsing, APIs, and diagnostics

The CLI parses the AST without executing imports, the function body, or arbitrary host effects. Unsupported syntax receives a source location. Compiler.assess_file accepts .py and yields the same Assessment as equivalent JSON. read_schedule returns the canonical document plus a companion location map; the [source contract](../PYTHON_FRONTEND.md) contains the complete API example.

CLI JSON diagnostics add source filename and start/end line/column while preserving Finding code/path, category, severity, and both blocking dispositions. `findings` retains blocking diagnostics and reports; `guidance` carries nonblocking hints. Iterate over `assessment.findings + assessment.guidance` to inspect both. Hints do not change structural acceptance, generation eligibility, or Corpus Gate expectations, and are not GPU measurements. FrontendError carries PYTHON_SYNTAX or the existing SCHEDULE_STRUCTURE code. Invalid candidates cannot create output. Locations are display projections, not part of Schedule, Assessment, or lowering identity.

Ordinary Python import can still execute module-level code and argument expressions even though the decorator does not call the function body. Candidate-reading tools must use read_schedule to maintain the nonexecuting boundary.

## Acceptance and further examples

FMA, [softmax](../../examples/python/softmax.py), and the [CuTe pipeline](../../examples/python/kmeans_pipeline.py) must construct canonically equivalent existing plans and use the same lowering. Checks preserve shape, dtype, synchronization, and backend refusals, source localization, symbolic loops, distinct FMA/mul-add, no host effects, and JSON compatibility. Compiler release still requires its full Corpus and independent approval. Example names and optional operation ids do not inherit historical result identities.

## Casts, promotion and source diagnostics

Use `lm.cast(x, to="fp32")` or `lm.cast(x, to="bf16")`. The canonical `to` parameter
also determines the inferred result dtype; explicit `out=` remains available. The
[cast example](../../examples/python/cast.py) matches the existing JSON cast Schedule.
Arithmetic shares the IR promotion rule: matching floating dtypes stay unchanged;
FP32 with one 16-bit floating format produces FP32. BF16 plus FP16 requires an explicit
cast. Both operand orders in the [mixed-dtype example](../../examples/python/mixed_dtype.py)
produce FP32. A reduction's sole supported scope defaults to `cta`.

Output, target/backend/entry-point and program-axis diagnostics point to their parameter,
decorator keyword or latest relevant axis declaration. Construction errors use Python
variable/call names; `FrontendError.canonical_path` retains the underlying IR path for
tool correlation. Source locations do not enter Schedule semantics.
