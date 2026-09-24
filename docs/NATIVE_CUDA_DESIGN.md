# Native CUDA lowering (development successor)

The native backend owns operation traversal, warp dispatch, shared storage, tensor-memory
allocation and release, TMA tensor maps, instruction descriptors and circular stage/phase
protocols. NVIDIA nvcc/ptxas owns compilation and physical register allocation. The target
admits B200 / sm_100a and B300 / sm_103a separately, using their existing Target
contracts and the documented common tcgen05 f16 instruction subset. Neither target
inherits timing calibration or device evidence from the other. Existing releases replay
from their pinned Git. Successor preparation preserves released locks and approvals;
the canonical release cycle derives the new identity after the source change.

## Minimal contract and P1–P8

P1/P3: preserve Schedule and its operation DAG. Add `native_cuda` and one missing transfer:
`load(movement="tmem", source_atom=...)`. Transfer is independent of a formula. Existing
load, MMA, elementwise, argmin and store compose both GEMM+bias and KMeans. No operator
name dispatch, layout language, framework compiler or kernel template is introduced.

P2/P5/P8: support row-major BF16/FP16 operands in explicitly swizzled shared tiles,
K-major tcgen05 f16-family atoms, FP32 accumulators, CTA-group one, and 128-row TMEM
readout. One epilogue thread owns one row; stores therefore explicitly use
`coalesced=false`. Source maps identify operations and resource declarations. Circular
TMA stages use ready and consumed barriers with per-slot parity; consumers release
storage only after asynchronous MMA completes. The compiler derives addresses, never
algorithmic precision or placement. Register-role redistribution, persistent grids,
cluster operations and unsupported range controls are refused. Reported physical
registers and device correctness remain unknown until target compilation/evaluation.

P4/P7: the new transfer is typed tensor FP32 to identically shaped register FP32, with
an explicit tcgen05 load atom. The verifier checks these effects; other backends refuse
it. Backend admission also checks role ownership, access coordinates, instruction
shapes, synchronization scope and supported resource refinements before emission.

P6: implementation precedes focused public-interface and adversarial tests. The full
Corpus Gate keeps prior expectations. Independent ADR0052 review binds the exact new
commit and Gate; the author never writes release approval.

Only explicitly represented loops and accesses are supported. Examples are known-kernel
reproduction/structural cases, not clean-start or performance evidence. Remote testing
requires the active delivery's separate authorization and exact target contract.
Exact-target compilation, external-oracle checks, matched timing with L2 flush and profiler
remain separate evidence gates. Framework acceptance is not inferred from this compiler change.

PTX encoding reference: NVIDIA PTX ISA, sections 9.7.17.4 (matrix descriptors),
9.7.17.8 (TMEM allocation), 9.7.17.9 (TMEM transfer), and 9.7.17.10 (tcgen05 MMA):
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html .

## Transport and finite support domain

`load(movement="global")` may explicitly stage a masked half/BF16 global tile into
shared memory. Threads apply the declared swizzle, fence generic stores into the async
proxy, synchronize the producer warp and arrive at the ready mbarrier. This handles
contiguous K65/K66 inputs without hidden padding or host copies. `movement="tma"`
requires aligned global strides; its rank-2 box extends with a unit batch coordinate
when the AccessMap names a scalar batch ProgramAxis. Mixed global/TMA producers use
one ready barrier. Every stage has one MMA issuing thread; multiple MMA nodes queue
from that thread before the consumed commit, so the commit covers every stage reader.

An MMA owns a distinct completion barrier. Readout waits completion, fences thread
synchronization into tcgen05, issues complete-warp TMEM loads and waits for their
completion. Readout shares the contraction loop's parent scope. Four readout warps
must start at a multiple of four; their role-local rows then match physical TMEM lanes.
All output-tile readouts finish before the next output-tile iteration, and allocation
owners deallocate only after all readers finish. Pipeline stages overlap copy and MMA;
CTA barriers occur at allocation and output-tile/contraction boundaries, not between
individual stage operations.

Native code currently supports M128 atoms, N in [8,256] at multiples of eight, K16
atoms and K16/32/64 staged tiles selected by full-row 32/64/128-byte swizzles. Register
arithmetic supports FP32 add/sub/mul/div/relu/square/exp/rsqrt/reciprocal and explicit
floating-point casts. The last three are row-owned CUDA math operations in an
isolated KDA-preparation witness; no B300 numerical receipt yet establishes that
their error is acceptable for the complete recurrent Workload. A bounded FP32
sum now folds each 128-row register tile along its columns, and provenance from
that reduction permits one scalar per row to scale the original tile; a merely
replicated vector of the same shape is refused. The row-norm witness covers a
full 128-wide local reduction, matching KDA's head dimension. Prefix-product
scan and triangular chunk correction remain separate gaps.
The shared `scan(op="mul")` vocabulary can emit a Triton within-chunk prefix;
native CUDA still refuses that operation by name. A multi-stage preprocessor
would have to account for its extra launch and global traffic in the complete
KDA Workload rather than treating the scan alone as a speedup.
A bounded `sm_103a` H64/T8192 fixture now composes the gate arithmetic and
32-token prefix into one Triton preprocessor, writing a 256 MiB FP32 prefix
tensor. This is a typed component with CPU source-contract checks, not a
qualified prefix result or a complete two-stage KDA kernel. The extra launch,
write and later read must be measured against the 456 us adapted reference.
A separate BF16 output commitment halves the prefix tensor to 128 MiB. CPU
chunk algebra with that extra prefix rounding passed the complete T65 and T257
output/final-state oracles. A later combined full H64/T8192 CPU screen also
passes; B300 stage correctness remains unverified. Neither representation is
selected for a performance claim.
A typed midpoint-factor stage takes BF16 normalized Q/K and FP32 per-token
log-decay, scans its 32-token prefix, extracts the first/last log values, and
forms BF16 forward-key, backward-key and forward-query tiles. The bounded
H64/T8192 CPU recurrence screens pass, but the upstream normalization/gate
producer and generated-source B300 qualification remain open. Materializing
all three factor outputs costs 384 MiB, in addition to 512 MiB of factor-stage
inputs; fusion or another explicit traffic reduction may be needed to beat
the adapted single-kernel CAKE reference.
A separate typed coupling stage accepts midpoint-scaled BF16 key/query factors,
beta gates and explicit triangular masks. Three existing MMA nodes produce the
prediction matrix, its transposed view and the output correction matrix; row
versus column beta broadcasts are Schedule commitments. Full H64/T8192 and
tail guardrail CPU recurrence screens pass with BF16 factor operands. Its three
128 MiB factor inputs and three 32 MiB matrix outputs are still hypothetical
global traffic until the upstream factor producer and B300 timing are measured.
The numerical successor precomputes a 32x32 BF16 inverse factor per head/chunk,
independent of the current recurrent state. A typed Triton fixture uses five
unrolled doubling steps and 18 existing BF16 MMA operations; it maintains both
the matrix and its transpose because Cake MMA contracts the last axis of two
operands (`A @ B.T`). The coupling producer must preserve the two inverse
inputs' transpose relation. Full H64/T8192 and T65/T257 CPU recurrence screens pass,
but the generated inverse stage has no GPU compilation, register or timing
receipt, and its 32 MiB output plus two 32 MiB inputs are extra traffic.
Streaming argmin preserves global indices, lowest-index ties and centroid-tail masks.
Resident argmin uses local tile positions and is admitted only when a complete,
zero-origin candidate domain fits in one tile. `Schedule.argmin_domain` follows the
actual access and producer graph for both native CUDA and Triton. A 64-column physical
tile with four declared candidates therefore excludes columns 4 through 63. Numeric
broadcasts with conflicting candidate bounds are refused, not used to truncate the
candidate domain. Unsupported origins and nonzero subranges receive local findings.
The shared query does not model a runtime-valid candidate prefix. It refuses that
domain before emission; global storage capacity cannot replace a runtime extent.
The global addressing domain is bounded to signed-32-bit element counts. Unsupported
placement, precision, resource refinements, range controls and role arrangements fail
locally; there is no scalar-MMA or target fallback.

### B300 TMEM state prototype

The exact `sm_103a` prototype admits one root contraction in which four aligned
32-lane groups load a 128-row BF16 register tile and publish packed pairs through
`tmem_store` with `destination_atom` `tcgen05.St32x32b`, repetition 8. Each group
waits for its own
`tcgen05.st` completion; one leader per group arrives on the declared count-four
`mbarrier`. The MMA role waits for that barrier, then executes
`tcgen05.fence::after_thread_sync` before issuing `tcgen05.mma` with A in TMEM and B
in explicitly swizzled shared memory. FP32 TMEM readout and the final global store
retain their existing completion contract. A later bounded prototype accepts one
static outer `TileLoop.carried_buffers` tile: prologue store publishes phase zero,
each trip TMA-loads its B tile, waits for phase `trip & 1`, completes the MMA and
readout, then stores the next BF16 state and publishes the next phase. The per-trip
MMA completion barrier is drained and invalidated before reuse. Dynamic chunk bounds,
multiple carried tiles, KDA normalization, triangular solve and five-role overlap
remain outside this route. Core IR and native emission have CPU contract and Corpus
Gate coverage; both generated CUDA witnesses still need exact-target NVCC and B300
numerical validation. Neither witness is a KDA prefill performance candidate.

Native examples are complete hardware Schedules, not backend-renamed Triton inputs.
The KMeans core explicitly takes `centroid_sq`, matching the existing portfolio runtime
preparation contract. The original Workload still owns tokens/centroids and strict ids.
Norm preparation and its costs belong in qualification accounting; core-launch timing
is not the end-to-end timing of the original two-input Workload.

## Exact compile and launch boundary

Use explicit `--gpu-architecture=compute_100a --gpu-code=sm_100a` (or the matching
103a pair), never `-arch` shorthand that also emits non-specific virtual code. This was
confirmed by an independent CPU nvcc diagnostic: the shorthand failed PTXAS admission;
the same source with the exact pair compiled. `--fmad=false` keeps decomposed arithmetic
from acquiring an undeclared contraction. `CUtensorMap` retains its native alignment.

The emitted C ABI separates create/launch/destroy. Create encodes descriptors, validates
the exact device, records buffers and configures shared memory. It does not copy/pad,
compute norms or run kernels. Launch checks the device binding and launches on the
supplied stream. The caller owns all buffer storage and synchronization. The handle must
outlive queued launches. The emitted metadata is a projection of Schedule/Target, not a
second workload registry. GPU correctness is pending until the separate qualification
worker evaluates the unchanged oracle and case requirements.

## Completion phases and nested reduction scopes

A contraction loop carries its TMEM accumulator across K iterations. Its non-pipeline
completion barrier publishes that final value once in the contraction's parent scope:
initialize count one, issue all K iterations, commit once from the MMA issuing thread,
wait successfully on phase zero before TMEM readout, then invalidate after all readouts.
It never advances through unobserved intermediate phases. The stage ready/consumed
barriers continue their independent per-slot phase handshakes during the contraction;
every used consumed slot is explicitly drained before invalidation. TMA and MMA remain
concurrent roles. This follows [PTX primary-phase rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#primary-phase).

The immediate parent of a contraction owns its completion-barrier invalidation; a more
distant ancestor cannot invalidate it again. Likewise, carried argmin state initializes
on each dynamic entry to its own reduction loop, outside that loop but inside any
surrounding row/batch loops. State survives centroid tiles, never unrelated row tiles.
Unknown loop buffers or out-of-rank dimensions are common Schedule-semantics Findings.
Native shared staging outside a contraction pipeline and TMEM swizzles are explicit
backend refusals until their instruction and storage effects have an implementation.

The completion consumer contract is checked for every declared waiter: it must be a
TMEM LOAD of that MMA's result in the contraction parent scope. Other signal uses or
per-K consumers are refused, so final publication does not discard an observable
per-iteration notification. The original ready/free per-stage protocol is separate.

A single K tile is represented canonically by a contiguous root LOAD/MMA DAG and a
one-stage pipeline, with no authored TileLoop. The same emission path visits that
stage once and publishes once. More stages without a K loop are refused; declaring a
single-trip TileLoop remains a common canonical-form error. K2 and longer contractions
use their declared loop and stage count. The focused publication-scope checks cover
K1, K2, repeated ring wraps, multiple MMA results and nested row/centroid iterations.
