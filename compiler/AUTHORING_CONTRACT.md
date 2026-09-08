# Open Cake Schedule authoring contract

Author one complete Schedule as JSON or the restricted Python surface documented in
`docs/PYTHON_FRONTEND.md`. Python elaborates to the same JSON document conforming to
`schedule.schema.json`; it does not widen the semantic or backend contract. The Compiler—not the prompt—owns semantic
acceptance. Names are unique within each declaration list; operation dependencies refer only backward; every output
must be written; buffer allocation extents and role warps must fit the exact Target. `program_map` and `grid` are
mutually exclusive. Each role owns one ascending contiguous warp interval, and no warp belongs to two roles.
Findings carry a stable code and path. Each Finding independently declares whether it
blocks acceptance or lowering. A non-blocking Finding reports what the Schedule implies,
such as a declared residency bound; another Finding in the same Assessment may still
block it. Acceptance and lowering eligibility are the aggregate decisions. Unsupported
shapes, operations, instructions, memory spaces or uncalibrated analyses are explicit.
An otherwise-lowerable `mma` in a generated backend names the Target instruction contract that determines its
lowering; omitting it is a lowering-blocking candidate Finding rather than a late emitter failure.
The Triton backend admits multiple existing `mma` DAG nodes. Each operation independently
owns its instruction, operands, unique write and accumulation derivation; combining
partial contractions requires an explicit typed consumer such as FP32 `elementwise add`.
There is no multi-MMA mode, count parameter, implicit shared accumulator, or split-K
operation. A later MMA is checked and refused at its own path just like the first.
FP32 Triton MMA makes input precision explicit: `triton.dot.fp32_ieee` requests IEEE
input precision, while `triton.dot.fp32_tf32` requests TF32 tensor-core input precision.
Both keep FP32 operand Buffers and FP32 accumulation/results; the TF32 contract is not a
Buffer dtype or an implicit cast. A backend must emit the selected precision explicitly
and may not replace it with a default, fallback, or another precision mode. Numerical
acceptance belongs to the unchanged external Workload oracle and tolerance.
Emitter-only program-shape requirements have one owner in that backend's `preflight`.
Assessment projects them into lowering-blocking Findings before `lower`; a backend must
not silently reinterpret an unsupported epilogue formula or wait for emission to reject
its role, loop, pipeline, operation-count, load-movement or descriptor requirements.
Logical register-Buffer pressure is an uncalibrated structural feature, never a physical
register bound or legality gate. `residency.registers_per_thread` reaches the backend as
`maxnreg`; compiled-artifact evidence owns actual allocation and spills.
FP32 `elementwise fma` reads exactly three same-shaped register Buffers in a, b, c order
and writes one same-shaped FP32 register result. Its required instruction contract is
`ptx.fma.rn.f32`: one RN-even rounding of a*b+c, without FTZ, saturation, reassociation
or a rounded intermediate product. Scalar and broadcast fields are not admitted.
The Triton emitter uses an explicit instruction; unsupported backends refuse before
lowering. NaN payload identity is not promised. Preceding rounded producers and nested
FMA dependencies remain explicit. Work counts two arithmetic operations per result;
register pressure reuses the existing all-read dataflow model, not a physical-register
or latency estimate.
A block scale is an FP32 Buffer with one `scale_of` relation. `granularity` is written in
the FP8 data buffer's axis order; `axis_order` is the full permutation that gives the
scale buffer's physical grouped-axis order. The Compiler derives the scale shape and
requires loads to preserve the relation. The block-scaled MMA read order is data A, data
B, scale(A), scale(B); names and convenient shapes never substitute for those relations.
A padded global Buffer writes one runtime-valid prefix as `valid_extent`; `dimension`
names the padded axis, `buffer` names the global INT32 input that owns the lengths, and
`indexed_by` maps each length-buffer axis to one data axis. AccessMap remains the only
coordinate authority, so every access derives `coordinate < length` rather than
restating a predicate. The first Triton subset lowers one length axis indexed by one
scalar program axis and refuses wider mappings explicitly.
An AccessMap index with `source: buffer` names a rank-one register INT32 Buffer that the
operation also reads. Multiple such coordinates share one shape and are zipped into one
runtime-index domain; they are not a Cartesian product. `mask_tiled_axes` bounds both
sides of each runtime coordinate, and an invalid indexed load yields zero. The admitted
subset permits a direct global load and the atomic state transition below. Runtime-indexed
TMA is not admitted. An indexed store is admitted only when its coordinates are proved
unique by the reservation-owned store rule: the target index and returned old value of
a same-domain unit atomic increment jointly identify its destination. It requires the
same-role register ownership and bounded one-execution-per-program domain checked by
the verifier; a caller's uniqueness assertion is insufficient. Other indexed stores are
refused before lowering.
A direct global load into registers preserves the vector shape of its AccessMap.
Each program-tile, loop-tile and dimension component contributes its declared extent;
scalar program coordinates contribute no axis. An all-scalar address uses the canonical
one-value register shape [1]. A load cannot implicitly splat or reshape that result.
Dimension vectors must also fit the physical dimension they address, not merely the
dimension used to define their range. Invalid dimension references return Findings.
Triton preflight checks the arange intervals emitted for program tiles, loop tiles and
dimension walks. Their span must be a positive power of two within the backend's tensor
limit; a non-power-of-two global extent is still legal when power-of-two tiles and masks
cover it. This is a backend requirement, not a new arithmetic primitive or a guarantee
of toolchain compilation for every otherwise eligible program.
`state` is caller-owned global memory that an operation both reads and writes; read-only
and write-only Buffers remain `input` and `output`. The admitted `atomic_rmw` reads one
INT32 state target followed by its one runtime INT32 index, writes that same target and
one register result, adds its declared signed INT32 scalar with `order: relaxed` and
`scope: device`, and returns the old value. A masked coordinate has no memory effect and
returns zero. The Triton route requires the exact Target atomic contract; no backend may
silently strengthen, weaken or relocate the operation.
Resident `top_k` accepts rank-one FP32 scores or signed INT32 values. Both return values
in descending order and INT32 source positions, with the lowest source position winning
equal-value ties. INT32 ordering uses the ordinary signed order; its `nan_policy` field is
vacuous. Loop-carried INT32 top-k is explicitly unlowerable until a real carried-state use
case justifies its initialization and finalization contract.
Loop-carried FP32 `top_k` may declare `source_tiles_per_merge: 2` to batch exactly two
source tiles before updating its carried state; omission is the single canonical spelling
of the historical one-tile cadence. The final values and global source positions are
unchanged, including odd and partial tails, but the delayed results may be read only after
the loop. The current Triton slice admits this cadence only with an unflattened,
non-warp-specialized loop whose unroll factor is one. Pending keys are lowering-owned
derived state: one packed UINT64 composite key per source element, or
`8 * source_extent` logical bytes. The field makes that extent visible to analysis
without adding a destination-passing Buffer; physical register placement remains
compiled-artifact evidence.
The canonical Triton lowering may replace a two-source-tile carried merge by sorted-source
half-selection when the padded source pair and already ordered carried half each contain
`k` keys and the pinned Triton 3.7.1 frontend-network model removes at least one third of
comparison-lane work. The one-source cadence retains its frozen lowering identity. This
is a derived lowering choice, not author syntax: no
algorithm flag or second `top_k` spelling exists. Profile exposes that version-specific
structural model while abstaining from compiled registers, implicit shared memory,
instructions, cycles and latency until matched Executor evidence exists.
Declared work uses the same query-derived loop-stop domain as lowering: apply `add`, then
`floor_div`, clamp to the static loop extent, and tile the result for every scalar program
coordinate. One dynamic stop composes with static enclosing loops and other program-axis
multiplicity. More than one dynamic stop in an operation's loop chain is an explicit
analysis abstention, never a static-maximum fallback. Work and top-k cadence share this
domain owner. A roofline floor uses only `device_specification` ceilings; it may use
lower-bound arithmetic, but may use compulsory bytes only when their count is exact. A
microbenchmark reference or partial-addressing byte upper bound remains a comparative
ratio and cannot by itself refute a measurement.
An ordinary `store` may target state only when its AccessMap mechanically partitions the
state across every ProgramMap axis. Each Program axis must be owned by the same state
Buffer/dimension and consumed exactly once; remaining dimensions are covered in full.
Indirect coordinates, TileLoop placement, partial slices, unused launch axes and legacy
grid are refused. No `unique` assertion or Workload precondition substitutes for this
proof. The store keeps its existing dtype and ordering rules and lowers to the
caller-owned state pointer. A Schedule may return no output only when it updates
caller-owned state; the host wrapper returns an empty output tuple and the state effect
remains observable through the argument.
`reduce_argmin` compares FP32 values and returns INT32 source positions; admitting FP8
storage for another operation does not widen that semantic contract.

`lowering` owns only the materialization mechanism and executable symbol. Its `backend` is
one of `triton`, `cutlass_cute_dsl`, or `metal`; `entry_point` is an identifier.
The retired `checked_cuda_asset` spelling is a structural refusal. Historical Schedules
and results remain replay inputs for their pinned Git revisions, not a current source-selection path.
The executable argument signature is derived from global Buffers and is never restated as
an ABI label. Metadata may bind a Workload digest, but the Compiler does not infer a
Workload from a route or hard-code its tensor shapes.

For Flash-KMeans, the Workload Contract owns B/N/K/D, BF16/FP32/INT32 semantics, tie handling and oracle. A Study
narrows the public Compiler to one exact lowering route and supplies a complete `schedule-skeleton.json`; start from
that skeleton. A Schedule may change admitted block sizes, warps and stages, but must preserve its route, external
tensor shapes, `metadata.workload_contract_sha256`, operator semantics, and frozen Compiler Revision during a Run.

## Source, value-domain and hardware boundaries

Python-emitting backends require safe identifiers for emitted symbols and operation ids:
ASCII Python identifiers outside keywords, fixed imports and generated namespaces.
Input `out` and wrapper-validation locals cannot shadow wrapper state; an output buffer
named `out` remains valid. Backend source restrictions preserve IR acceptance and produce
localized capability Findings before lowering.

A register value stored to a direct global AccessMap has exactly the vector-domain shape
of that address (or canonical `[1]` for an all-scalar address). A store cannot implicitly
splat or reshape the value, just as a load cannot invent a different result shape.

The current Triton loop-carried argmin requires a proven static candidate domain with
complete tiles. Runtime candidate prefixes, candidate-dependent extent lookups, indirect
or unknown value domains and partial candidate tiles are refused. A runtime prefix on an
orthogonal row/feature dimension does not itself restrict candidate validity. Dynamic
`LoopStop` is currently supported only for transient computations feeding validity-masked
loop-carried top-k: no state/atomic/global-write effects, other carried state or escaping
private intermediates. Other stopped bodies receive a capability refusal; zero-fill
loads do not establish neutral values for arbitrary downstream operations.

Tensor-memory allocation columns are powers of two in `[32,512]` and must also fit the
Target capacity and declared bytes. MMA tile N contains whole declared instruction N
atoms; a wider instruction cannot write into a narrower accumulator tile. These checks
follow the [PTX tensor-memory allocation and MMA contracts](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
`Residency.registers_per_thread` is a compile-time cap bounded by the selected Target's
`maximum_registers_per_thread`. Its positive integer syntax is distinct from Role's
`setmaxnreg` immediate encoding: small or non-multiple-of-eight compile-time caps remain
valid. Missing Target cap coverage is reported, not treated as an unlimited resource.
The current CUDA Target capacity follows the [Blackwell hardware specification](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy),
while [.maxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#performance-tuning-directives-maxnreg)
and `setmaxnreg` retain their separate meanings.
