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
Emitter-only program-shape requirements have one owner in that backend's `preflight`.
Assessment projects them into lowering-blocking Findings before `lower`; a backend must
not silently reinterpret an unsupported epilogue formula or wait for emission to reject
its role, loop, pipeline, operation-count, load-movement or descriptor requirements.
Register findings are static bounds over declared logical storage, never a claim about ptxas's physical allocation.
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
subset is a direct global load: TMA and indexed stores remain explicitly unlowerable.
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
