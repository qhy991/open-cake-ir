# ADR 0021: Add Apple GPU family 9 as a successor Compiler Target

Status: accepted, 2026-08-25 by Compiler Revision `open-cake-ir-v14`.

## Context

The predecessor authority was `open-cake-ir-sm100a-v13`: one typed Schedule language,
eight admitted profiles, a nineteen-case Corpus Gate, structural Triton and CuTe DSL
emitters, and one retained CUDA asset. Schedule v1 already accepts a non-empty Target
identifier. The obsolete v3 assumptions in the first version of this proposal -- six
Corpus cases, three profiles, `target: const sm_100a`, and a second shadow compiler --
no longer describe the repository.

The real multi-Target gaps are narrower and more load-bearing. Target schema v1 requires
CUDA device names and compute capability, uses CTA/warp/shared-memory names, and gives
the Compiler a second handwritten Target parser beside the typed `Target` Module.
Lowering is selected by profile alone, so an understood profile can silently select an
`sm_100a` Adapter for another Target. Calibration coverage is also profile-only, and
ranking can accept a mixed-Target candidate set while selecting the first candidate's
Target for all of it.

The checked-in Metal probe establishes a capability seam on the local Apple M4: Xcode
compiles and links Metal 3.2 BF16 source, Apple GPU family 9 admits 32-wide SIMDgroups,
and an 8x8x8 BF16 SIMDgroup matrix multiply dispatches correctly. It does not establish
Flash-KMeans correctness, a Launchable Candidate, an Evaluation Receipt, or a
performance claim.

ApxInf supplies two relevant implementation lessons. First, an operator Adapter must
derive its SIMDgroup count, threadgroup size, bounded scratch, reduction order and
lowest-index tie rule from the frozen shape and tuning decisions; a source file that
only substitutes a digest is another capability probe. Second, full intermediate
materialization is the wrong locality: Flash-KMeans should stream 8x8 score fragments
into a running argmin rather than materialize a 256x64 FP32 distance tile, which alone
would require 64 KiB and exceed the observed 32 KiB threadgroup-memory limit.

ApxInf's larger wins -- persistent buffers, one command buffer/one wait, multi-kernel
fusion and atomic active/scratch state -- do not belong to this Compiler slice. ADR 0020
keeps one Schedule equal to one kernel; those transaction commitments belong to program
composition and Evaluation above Schedule.

## Decision

Add Metal through the normal successor Compiler Revision. Do not duplicate the typed
IR, verifier, Corpus machinery or live compiler into a shadow engine. The release cycle
archives the witnessed v13 content manifest before minting its successor; historical
source bytes remain recoverable through the bound Git history and retained Evidence,
while Study Contracts and observations remain immutable.

The first architecture contract is `apple_gpu_family9`, not a physical `MTLDevice`
name. A Target owns architecture semantics; exact device name, registry id, SDK path,
unified-memory observation and pipeline admission remain Evaluation/Executor facts.

Target schema v2 expresses a 32-wide execution group and workgroup/threadgroup resource
limits without invented CUDA fields. The typed `Target` Module is the sole parser and
normalizes v1 and v2 documents for the verifier, analysis and emitters. Schedule v1 and
its historical `roles[].warps` spelling remain unchanged; for this Target each index is
interpreted as one 32-wide SIMDgroup. A language rename is a separate migration and is
not required to add a second Target.

An admitted profile owns one or more exact Target implementations. Each implementation
selects its emitter or retained asset, toolchain contract and closed-semantics rule.
Assessment looks up the exact `(Target, profile)` pair. A missing pair produces a stable
blocking Finding and never falls back to another Target's Adapter. Calibration coverage
is likewise an exact Target/profile pair, and ranking refuses mixed-Target sets.

The first generated Metal Lowering is the fixed-shape Flash-KMeans b32 assignment
kernel. It preserves the four-buffer row-major ABI and derives these decisions from the
Schedule:

- four declared execution groups become 128 threads per threadgroup; an eight-group
  variant becomes 256 threads and must produce different source and launch requirements;
- the 256x64x128 macro tile is composed from Metal BF16 8x8x8 SIMDgroup MMA atoms;
- the loop-invariant token load retains sixteen distributed 8x8 BF16 fragments per token
  atom and reuses them across every centroid tile, matching the Schedule's loop placement;
- each SIMDgroup keeps only one 8x8 FP32 score fragment and a running per-token best
  score/index, with deterministic lowest-index ties;
- the declared token tile and centroid tile determine work division and grid;
- the first Adapter admits `num_stages=1`; unsupported range options produce Findings
  rather than being ignored or falling back to a scalar kernel.

The Lowering is `generated=true`. It contains operation source markers and Metal
toolchain requirements, but it is not a metallib and is not launchable. MSL-to-AIR and
AIR-to-metallib custody, runtime buffer ownership, correctness, timing and profiling
remain later Lab/Evaluation work.

The 1,024-byte four-group or 2,048-byte eight-group score-fragment array is
backend-introduced threadgroup scratch, not a logical 64 KiB Schedule distance-tile
allocation. Its exact bound is checked against the Target and exposed in the Lowering
toolchain receipt. Compilation fixes Metal 3.2, safe math and disabled FP contraction;
these flags constrain code generation but do not create a numerical Evaluation result.

## Smallest complete slice

1. Replace the duplicate Compiler-side Target parsing with the typed Target Module and
   admit Target schema v2 alongside the unchanged `sm_100a` v1 document.
2. Make profile implementation and Calibration selection exact by Target, with no
   fallback and mixed-Target ranking refused.
3. Add the `apple_gpu_family9` Target, one accepted Flash-KMeans Schedule, one sibling
   refused for an unsupported Metal scheduling commitment, and a deterministic MSL
   emitter that consumes the declared execution-group and tile decisions.
4. Preserve all nineteen v13 Corpus observations and extend the successor Corpus Gate
   to twenty-one cases.
5. Compile the emitted source to AIR and metallib on the local Apple M4. This is a
   toolchain check only; operator correctness and performance remain unclaimed.

## Acceptance gates

- All nineteen v13 cases retain their accepted/lowerable dispositions, Finding codes,
  Schedule digests and Lowering source digests.
- The Apple accepted case lowers deterministically with `generated=true`, exact Metal
  toolchain requirements and all Schedule operation markers.
- Changing four SIMDgroups to eight changes both source bytes and the derived
  threads-per-threadgroup requirement without changing the operator ABI.
- An unsupported Target/profile pair and an unsupported Metal range option each produce
  stable blocking Findings; neither reaches an emitter or another Target's Adapter.
- Apple Calibration is absent, and ranking refuses a mixed-Target set.
- Contract tests run without Metal. On a qualifying Mac, Xcode compiles and links the
  emitted MSL and the existing BF16 probe continues to dispatch exactly.
- No result from this slice is described as Workload correctness, latency, speedup,
  Calibration, Evaluation or Campaign evidence.

## Consequences

The new Seam is real because Triton/CuTe/CUDA and Metal are independent Target
implementations behind the same Lowering Interface. Target changes gain Locality in the
typed Target Module, and exact route selection prevents CUDA assumptions and Calibration
from leaking into Metal.

The first slice deliberately stops before runtime optimization. ApxInf's command-buffer
residency, persistent state and measurement protocol become useful only after Metal has
a sealed Launchable Artifact and a host-canonical correctness assay; adding them to this
Schedule would violate ADR 0020 and manufacture evidence the Compiler does not own.
