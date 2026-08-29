# Open Cake Schedule authoring contract

Submit one complete JSON document conforming to `schedule.schema.json`. The Compiler—not the prompt—owns semantic
acceptance. Names are unique within each declaration list; operation dependencies refer only backward; every output
must be written; buffer allocation extents and role warps must fit the exact Target. `program_map` and `grid` are
mutually exclusive. Each role owns one ascending contiguous warp interval, and no warp belongs to two roles.
Findings carry a stable code and path. A blocking Finding is a reason an Assessment is not
lowering-eligible and no candidate reaches the toolchain; a non-blocking Finding accompanies an Assessment that is,
and reports what the declared Schedule implies — such as which declared resource bounds its residency. Unsupported
shapes, operations, instructions, memory spaces or uncalibrated analyses are explicit.
An otherwise-lowerable `mma` in a generated backend names the Target instruction contract that determines its
lowering; omitting it is a lowering-blocking candidate Finding rather than a late emitter failure. Instruction-free
checked assets remain valid only when their asset-specific preflight proves the declared semantics match the asset.
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
subset is a direct global load plus the atomic state transition below: TMA and indexed
stores remain explicitly unlowerable.
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
`reduce_argmin` compares FP32 values and returns INT32 source positions; admitting FP8
storage for another operation does not widen that semantic contract.

`lowering` owns only the materialization mechanism and executable symbol. Its `backend` is
one of `triton`, `cutlass_cute_dsl`, or `checked_cuda_asset`; `entry_point` is an identifier.
The executable argument signature is derived from global Buffers and is never restated as
an ABI label. Metadata may bind a Workload digest, but the Compiler does not infer a
Workload from a route or hard-code its tensor shapes.

For Flash-KMeans, the Workload Contract owns B/N/K/D, BF16/FP32/INT32 semantics, tie handling and oracle. A Study
narrows the public Compiler to one exact lowering route and supplies a complete `schedule-skeleton.json`; start from
that skeleton. A Schedule may change admitted block sizes, warps and stages, but must preserve its route, external
tensor shapes, `metadata.workload_contract_sha256`, operator semantics, and frozen Compiler Revision during a Run.
