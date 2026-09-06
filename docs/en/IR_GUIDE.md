# The current IR: structure, semantics and implementation

[中文完整说明](../IR_GUIDE.md) · [English home](README.md) · [Bilingual catalog](../README.md)

This reading companion explains the canonical Schedule model and where to change it. The [typed IR](../../src/open_cake_ir/compiler/ir/__init__.py), generated [JSON Schema](../../src/open_cake_ir/compiler/schema.py) and [Authoring Contract](../../compiler/AUTHORING_CONTRACT.md) own exact spellings and constraints. The [Glossary](GLOSSARY.md) owns terminology. This guide does not maintain a separate release or experimental results ledger.

## Scope of the reorganization

`open_cake_ir.compiler.ir` is one package with one definition of each type and rule. Moving definitions preserves existing imports, fields, enum values, parameter defaults, JSON input, rejection behavior and generated kernel source. The package exports the original objects rather than wrapping or duplicating them. Definition locations and Python `__module__` names now identify the owning submodule.

There is no new primitive, registry, layout algebra or alternative parser. The Compiler still has no dependency on Lab, Evaluation or Evidence. P1–P8 continue to constrain editing ergonomics, visible hardware decisions, canonical forms, typed construction, analysis consistency, the Corpus Gate and hardware-grounded interpretation.

Acceptance compares the public vocabulary, parsed corpus, structural Schema, findings, modeled analysis and lowering outputs, alongside relevant contract tests and the full Corpus Gate. A source move changes the frozen Compiler closure and requires a successor draft; release or merge still requires [independent review](../adr/0052-independent-agent-release-review.md). Static validation does not establish GPU correctness or performance.

## The Schedule model

A Schedule describes a concrete GPU computation: participants, storage, operations, coordinates, synchronization and repetition. The Workload and its oracle own the mathematical problem and correctness criterion.

Python authoring produces the same canonical document as JSON. `Schedule.from_dict` builds typed objects; the exact Target, Verifier and backend capability checks determine admissibility and lowering eligibility. Analysis reports modeled behavior. Eligible lowering produces generated source or a checked asset.

| Fields | Responsibility |
| --- | --- |
| `schema_version`, `schedule_id`, `target` | Document version, plan name and exact target |
| `lowering` | Backend and entry point |
| `grid` or `program_map` | Exactly one work-grid description |
| `roles`, `allocations`, `buffers` | Warp participants, storage regions and typed views |
| `pipelines`, `barriers` | Staging and producer/consumer handshakes |
| `operations` | Typed operations, read/write edges and explicit dependencies |
| Optional `tile_loops`, `access_maps` | Repetition and per-edge coordinates |
| Optional `residency` | Resident CTA commitment and register cap |
| `outputs` | Exported buffers in ABI order |
| `metadata` | Restricted contract and historical provenance fields |

The dataclasses are frozen, but nested metadata is not recursively immutable. Direct dataclass construction does not replace the parser or cross-object verification at an external input boundary.

## Module ownership

| Module under `compiler/ir/` | Owns |
| --- | --- |
| [__init__.py](../../src/open_cake_ir/compiler/ir/__init__.py) | Stable imports of the same canonical objects |
| [_parse.py](../../src/open_cake_ir/compiler/ir/_parse.py) | Structural checks and `ScheduleParseError` |
| [vocabulary.py](../../src/open_cake_ir/compiler/ir/vocabulary.py) | Closed enums and shared constants |
| [resources.py](../../src/open_cake_ir/compiler/ir/resources.py) | Roles, allocations, buffers, relations, synchronization and residency |
| [mapping.py](../../src/open_cake_ir/compiler/ir/mapping.py) | Program axes, tile loops and access maps |
| [operations.py](../../src/open_cake_ir/compiler/ir/operations.py) | Typed parameters, kind-specific parsing and `Operation` |
| [schedule.py](../../src/open_cake_ir/compiler/ir/schedule.py) | Top-level assembly, loading and derived structural queries |

Vocabulary and shared parsing depend only on the standard library. Resources, mapping and operations use those foundations without importing each other. Schedule assembles them. Internal modules import definitions directly rather than through the public package, avoiding initialization cycles.

Callers retain `from open_cake_ir.compiler.ir import Schedule, OperationKind, Buffer, DType`. There is no second Schedule type. The [vocabulary tool](../../tools/ir_vocabulary.py) continues to discover enums and dataclasses through that public module.

## Resources and coordinates

`DType` includes BF16, FP16, FP32, FP8 E4M3 and INT32. Parsing a dtype does not establish support for every operation or backend. `MemorySpace` describes global, shared, tensor or register storage; `BufferMode` separately describes input, output, caller-owned state or scratch usage.

`Role` identifies a contiguous warp interval and optional register budget. `Allocation` owns storage capacity and, for TMEM, its allocation role and column commitment. `Buffer` describes typed shape and a concrete storage view through allocation, offset, stages and swizzle. Its element count and byte extent are derived. `Residency` declares a register cap or requested resident CTA count; actual register allocation, spills and unmodeled shared memory require toolchain or device evidence.

`ScaleRelation` relates scales to a data buffer through granularity and axis order. `ValidExtentRelation` relates a fixed-capacity data axis to a runtime device prefix length. Both refer to existing buffers; AccessMap remains the coordinate authority.

`ProgramAxis` binds a grid axis to a buffer dimension and tile size. The logical tile count follows from the extent. A persistent `ProgramMap` launches a target/residency-derived number of CTAs to walk the logical work; optional traversal names the fastest-varying axes first.

`TileLoop` names its iterator, owning buffer dimension, tile size and ordered body of operations or nested loops. `RangeOptions` carries stages, unrolling and the existing backend loop choices. `LoopStop` provides a supported program-coordinate-derived exclusive bound, not arbitrary control flow. Schedule derives loop parents, depth and whether an MMA loop traverses the contraction axis.

`AccessMap` binds an operation/buffer pair to coordinates and a boundary policy. Index sources are `program`, `program_tile`, `loop_tile`, `dimension` and `buffer`. Dimension components may narrow a contiguous range. Multiple runtime buffer indices are zipped over a shared domain rather than multiplied into a Cartesian product.

For an `[8,128]` buffer, a scalar row program coordinate plus the complete second dimension loads a `[128]` register tile. A half-row access must produce the corresponding smaller result; an all-scalar access produces `[1]`. Masking an address does not establish unique write ownership. See the [load-domain rule](../adr/0051-load-values-follow-the-access-domain.md) and [indexed-store ownership rule](../adr/0036-atomic-reservation-proves-indexed-store-ownership.md).

## Operations and effects

`Operation` carries its id, kind, role, reads, writes, waits, signals, dependencies, pipeline and typed parameters. The JSON id key is `id`; the Python field is `op_id`. Kind selects the parameter parser. Dependencies do not substitute for required cross-role synchronization.

| Kind | Parameters | Meaning |
| --- | --- | --- |
| `load` | `LoadParameters` | Addressed data movement, global/TMA choice and reuse intent |
| `store` | `StoreParameters` | Writeback with a coalescing commitment and legal ownership |
| `elementwise` | `ElementwiseParameters` | Arithmetic, permitted scalar/broadcast use and instruction choice |
| `cast` | `CastParameters` | Explicit numeric conversion |
| `mma` | `MmaParameters` | Contraction of `[M,K]` and `[N,K]`, FP32 accumulation and atom/tile choices |
| `epilogue` | `EpilogueParameters` | Admitted formula, subtile and copy-atom commitments |
| `reduce` | `ReduceParameters` | Sum/max collapse of an axis, with scope and carried-state rules |
| `reduce_argmin` | `ReduceArgminParameters` | Minimum position with deterministic tie and NaN policy |
| `top_k` | `TopKParameters` | Greatest values and indices in descending order, optionally carried |
| `scan` | `ScanParameters` | Inclusive forward or reverse prefix sum |
| `index_expand` | `IndexExpandParameters` | Expand group positions using scale, extent and sentinel |
| `online_softmax` | `OnlineSoftmaxParameters` | Stable weighted softmax recurrence with explicit state |
| `atomic_rmw` | `AtomicRmwParameters` | Atomic update returning the previous value; add/relaxed/device vocabulary |

Elementwise arithmetic includes square, rsqrt, exp, relu, tanh, add, sub, mul, div and fma. The same enum owns arity. FMA requires three same-shaped FP32 register inputs and `ptx.fma.rn.f32`, with one RN-even rounding and no scalar/broadcast alternative. Tanh also requires an instruction contract. Operand placement is meaningful only for the MMA contracts that own it; a tile-level Triton dot does not accept arbitrary hardware placement fields.

Omitted `reduce.across_loop` preserves the historical true default; explicitly writing true is rejected. Its false spelling disables carried reduction. Argmin and top-k default to false. Resident signed INT32 top-k and carried FP32 top-k have distinct backend boundaries. `FenceProxyParameters` remains a Python type but has no corresponding authorable `OperationKind`.

Pipeline stages describe reuse. Barriers declare counts, producers, consumers and an mbarrier or named-barrier mechanism. The produced pipeline kind follows the operation: TMA load feeds UMMA, and MMA releases to threads. These are concrete hardware commitments, alongside swizzles, TMEM columns and copy/MMA atoms, not a separate layout language. See the [primitive examples](wiki/primitives.md).

## Read and inspect an existing plan

Run from the repository root with its `src` directory on the Python path:

```python
from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import read_schedule
from open_cake_ir.compiler.ir import Schedule

schedule = Schedule.load("corpus/schedules/fma-b8-smoke.json")
assert schedule.buffer("a").shape == (8, 128)
print([(op.op_id, op.kind.value) for op in schedule.operations])

compiler = Compiler.load(".", "compiler/revision.json")
authored = read_schedule("examples/python/fma.py")
assessment = compiler.assess(authored.document)
assert assessment.accepted and assessment.lowering_eligible, assessment.findings
lowering = compiler.lower(assessment)
assert "fma.rn.f32" in lowering.source
```

The example reuses the existing [Python plan](../../examples/python/fma.py) and [JSON plan](../../corpus/schedules/fma-b8-smoke.json). Development uses the draft descriptor; released execution uses a matching frozen lock. Parsing, assessment, source generation, GPU compilation, correctness and performance are separate evidence boundaries. Read findings and analysis coverage rather than treating a single accepted bit as proof of everything.

## Extending the model

First try composing existing primitives, buffer relations, loops and access maps. A new operator composed from existing capabilities needs its own Workload and Schedule, not a backend dispatch keyed by its name.

For a missing capability, specify typed inputs/outputs, effects, numerical order, address relationships, hardware behavior and rejection conditions. Add the smallest structure in its owning IR module, preserve the public entry, project the Schema and update authoring guidance. Change the Verifier, affected work/analysis and lowering together; never admit a field and silently ignore it during generation.

Add focused contract or reproduced-defect coverage after implementation, run the full Corpus Gate, and include new modules in the Compiler source set. Expectation changes require separate review. Use the [successor release process](../RUNBOOK.md); old Compiler, Executor and experimental closures replay from their original pinned Git sources.
