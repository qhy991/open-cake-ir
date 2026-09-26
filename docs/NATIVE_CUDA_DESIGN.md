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
A typed upstream stage now normalizes BF16 Q/K across the full 128-key dimension,
forms FP32 log-decay from G/A_log/dt_bias, and forms FP32 beta gates. Its four
outputs are preparation values, not the public KDA result. Separately
materializing them costs about 514 MiB of writes on the fixed H64/T8192 case,
before the factor, coupling and state stages read them. The upstream generated
source has CPU contract coverage only; fusion and B300 timing must decide which
intermediates deserve global storage.
A combined WorkBound screen for separately materialized upstream, factor,
coupling and inverse stages counts about 1.412 GB of logical reads and 1.076 GB
of logical writes before the state kernel. The three adjacent boundaries alone
account for about 2.013 GB of write-then-read values that fusion might avoid.
The sum of individual peak-bandwidth screens is 311 us, but this is neither a
measured latency nor a valid additive runtime lower bound: cache reuse,
instruction cost, launch order and on-chip resource pressure remain unknown.
A separate fused Cake fixture composes upstream preparation, midpoint factors
and three coupling MMAs into one Triton Schedule. It removes both global
upstream-to-factor and factor-to-coupling boundaries, about 1.879 GB of logical
write-then-read values in the four-stage screen; the fused Schedule declares
about 404 MB input and 101 MB output traffic before inverse/state work. This
only proves type legality and source generation. If register pressure or
serialization makes the fused kernel slow, the device result must choose a
different split rather than an automatic cost-model promotion.
A second fused fixture passes P and its transpose directly from coupling casts
into the five-step inverse, avoiding their global round trip. It emits 21 BF16
MMAs and returns only output coupling B and the BF16 inverse. Its declared
logical input/output are about 404/67 MB, excluding the recurrent state kernel.
This is a distinct mapping hypothesis: 21 dependent MMA operations may increase
register pressure or serial latency even though the traffic screen is smaller.
Target compilation and complete component numerics gate the choice; later
B300 measurements below reject this 21-MMA depth for the fixed H64 case.
A state-input successor keeps that fused 21-MMA body but also writes the three
BF16 tiles needed by the native state kernel (base key, base query, final key),
FP32 beta gate and FP32 chunk-end decay. It declares about 404 MB of input and
480 MB of output traffic on H64/T8192, rather than pretending the two-matrix
output is a complete preprocessor. A CPU screen with each planned BF16 operand
and storage rounding passes the complete T65/T257 and H64/T8192 token oracles.
That screen is not a GPU kernel receipt, and the high-retention held-out state
failure of chunk-level rounding still limits generalization.
The mixed-major seven-output successor subsequently passed exact-B300 AOT and
complete H64 component correctness. Its 21-MMA fused preparation alone took
637.606 us median under a five-round cold-L2 CUPTI diagnostic and used 227
registers. Replacing explicit inverse output with strict-lower P and a planned
serial forward substitution cut the typed preparation to two BF16 MMAs. That
preparation also passes complete B300 output checks, uses 185 registers, and
measured 389.507 us in a separate five-round CUPTI job. CPU serial substitution
passes the independent full output/final-state oracle. The earlier adapted
complete CAKE reference measured about 456 us in another job, leaving only
nonpaired component headroom; no full native state consumer or speedup exists.
A typed midpoint-factor stage takes BF16 normalized Q/K and FP32 per-token
log-decay, scans its 32-token prefix, extracts the first/last log values, and
forms BF16 forward-key, backward-key and forward-query tiles. The bounded
H64/T8192 CPU recurrence screens pass, but the generated upstream and factor
sources have no B300 qualification. Materializing all three factor outputs costs
384 MiB, in addition to 512 MiB of factor-stage
inputs; fusion or another explicit traffic reduction may be needed to beat
the adapted single-kernel CAKE reference.
A separate typed coupling stage accepts midpoint-scaled BF16 key/query factors,
beta gates and explicit triangular masks. Three existing MMA nodes produce the
prediction matrix, its transposed view and the output correction matrix; row
versus column beta broadcasts are Schedule commitments. Full H64/T8192 and
tail guardrail CPU recurrence screens pass with BF16 factor operands. Its three
128 MiB factor inputs and three 32 MiB matrix outputs are still hypothetical
global traffic until the generated factor producer and B300 timing are measured.
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
each trip TMA-loads its declared B tiles, waits for phase `trip & 1`, completes the MMAs and
readout, then stores the next BF16 state and publishes the next phase. The per-trip
MMA completion barrier is drained and invalidated before reuse. Dynamic chunk bounds,
multiple carried tiles, KDA normalization, triangular solve and five-role overlap
remain outside this route. Core IR and native emission have CPU contract and Corpus
Gate coverage. The root and two-chunk single-B generated CUDA witnesses compile for
exact `sm_103a` without spills and pass three-seed, complete-output broker-shared
B300 numerical checks. A successor matches each carried MMA to its own loop-indexed
staged B producer and requires the ready barrier's arrival count to cover every
load. Its typed two-B/two-MMA witness passes the full Corpus Gate (179 cases), exact
`sm_103a` AOT (168 registers, no spills), and three-seed B300 numerical checks of
both products (49,152 elements, zero failures); immutable inputs remain unchanged.
This qualifies only the two-B carried contraction, not the two-phase
projection/triangular-solve/correction sequence of a complete KDA prefill or a
performance result (F-2026-09-24-003, event 107).

An isolated successor pairs the shared-core `forward_substitute` meaning with a
bounded native CUDA root schedule. One copy warp stages BF16 P[32,32] into an
explicit 2 KiB swizzle-64B shared allocation and publishes one mbarrier; four
compute warps each own one of 128 FP32 RHS rows, wait for P, then issue the
ordered strict-lower C32 solve as 496 named `__fmaf_rn` updates per row. A later
explicit cast would own BF16 rounding. The route refuses wrong P placement,
shape or swizzle, missing completion wait and unrelated shared loads. Its
synthetic Target adds the operation kind only for this proof; the committed
`sm_103a` document and a complete KDA Schedule do not admit it yet. At clean
`56a145f3`, CPU contracts and Corpus Gate pass; exact-sm_103a AOT uses 119
registers with zero stack/spills. Device numerical proof and any latency
measurement remain separate gates (F-2026-09-24-003).

A further isolated two-chunk **K32/C32 synchronization witness** keeps one
BF16 `[128,32]` state tile in TMEM, not the full KDA K128 state. Its first
carried TMA/MMA phase projects that state, then a copy warp stages BF16
P[32,32] and the four row-owning compute warps solve FP32 U. An explicit cast
and TMEM store publish BF16 U before the second TMA/MMA phase forms a next
state and republishes the carried TMEM phase. Both pipeline groups must be
contiguous and ordered around the solve; `NATIVE_TWO_PHASE_ORDER` refuses a
correction before U publication. State/U/P handoffs have separate declared
barriers and parity, while the final global result is written by the same CTA
after each chunk. This checks the synchronization vocabulary and emitter
structure; it does not include KDA normalization, decay, beta or output formula.
The clean `2e365222` source compiles for exact B300 with 119 registers and no
stack/spills; three broker-shared device seeds match every BF16 final-state element
(4,096 per seed, maximum absolute error zero). A separate scalar unit-loop
coordinate and rank-3 native TMA route form the K128/C32 two-phase successor.
Its clean `a81a6948` source compiles with 230 registers and no stack/spills,
but device correctness remains unverified.

A further isolated BF16 TMEM read emits `tcgen05.ld.sync.aligned.32x32b`
into packed 32-bit registers, unpacks two BF16 values per word and waits on
the carried state's current mbarrier phase. Shared-core typing and the carried
phase proof require an identical register tile, whole copy-atom repetitions
and a read before the loop's state update. The native backend admits only that
proven carried-state subset; its emitted source has CPU contracts and no AOT
or device qualification yet. The read is a prerequisite for combining the
decayed prior state with correction MMA without a global state round trip.

The KDA base-key/query contraction needs K128 with a dynamic TMEM A tile. The
earlier K-major B stage is refused because one BF16 row occupies 256 bytes, beyond
the backend's modeled 128-byte swizzle row. A standalone edited PTX probe then
qualified one MN-major B mapping on B300: physical B `[K128,N32]`, 64-byte
swizzle, B-major instruction bit, and a 1024-byte shared-descriptor step per K16.
Its three-seed 128x32 FP32 outputs pass the independent product; the probe is
not a Cake lowering. The successor native route admits exactly BF16 TMEM-A
M128/N32/K128 with that swizzle on `sm_103a`, and refuses other geometries by
`NATIVE_MN_MAJOR_B_UNQUALIFIED`. The shared verifier reads the declared major
mode to require B `[K,N]` only when MN-major is named. The Cake-emitted
successor compiled for exact `sm_103a` with 40 registers and zero spills and
passed three full-output B300 checks against the independent product. The
complete Cake-admitted recurrent KDA state consumer and its paired timing remain
separate gates.

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
second workload registry. Every new route still needs a separate qualification
worker against its unchanged oracle and case requirements.

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
