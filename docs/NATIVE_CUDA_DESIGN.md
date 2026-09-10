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
arithmetic supports FP32 add/sub/mul/div/relu/square and explicit floating-point casts.
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
