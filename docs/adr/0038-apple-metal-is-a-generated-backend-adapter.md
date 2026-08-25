# ADR 0038: Apple Metal is a generated backend Adapter

Status: proposed for the successor Compiler Revision.

## Context

Compiler v28 has one typed Schedule language, exact Target definitions, backend-owned
`preflight`, and a `LoweringRoute` that selects materialization mechanism plus entry
point. The earlier Apple prototype predated that architecture: it selected lowering by
workload profile and collided with the mainline v14 release history. Mainline v22 and
v26 now also express runtime-indexed gather and KDA weighted combine through existing
primitives, making them a smaller Mac correctness slice than the prototype's
Flash-KMeans matrix kernel.

The remaining portability gaps are architectural. Target schema v1 requires CUDA
identity and CTA-shaped resource names. The Compiler also parses that raw Target a
second time outside the typed Target Module. Neither is a sound Seam for an Apple
Implementation, and a CUDA fallback would violate exact Target selection.

## Decision

Add vendor-neutral Target schema v2. It names execution-group width and
workgroup/threadgroup resources, permits hardware budgets that the architecture does not
declare to remain absent, and normalizes to the compatibility properties used by the
current verifier. Schema v1 remains byte-compatible. The typed Target Module is the one
parser; Compiler TargetDefinition projects its normalized facts instead of interpreting
raw JSON again.

Add `metal` to `LoweringBackend` and register one generated backend Module at the existing
backend Seam. The Adapter identifies supported primitive graphs from Buffers, Operations,
AccessMaps, and dependencies; it does not infer a Workload from `entry_point` and does not
restore a profile registry.

The first two Implementations are:

- runtime-indexed BF16 gather: four 32-wide SIMDgroups map one thread to each of the
  8-by-16 route/feature outputs for a token;
- KDA weighted combine: one 32-wide SIMDgroup maps the first sixteen lanes to features,
  loads eight indexed rows and FP32 weights, reduces routes in ascending order, and
  rounds one BF16 output per feature.

Both preserve `mask_tiled_axes`: negative or upper-bound expert/row indices contribute
zero. Both use zero threadgroup scratch. MSL buffer order, grid, threads, language level,
and safe floating-point flags are exposed in the Lowering toolchain requirements.

Every Adapter-only constraint is returned by `preflight` as a BackendPrecondition and is
projected into an Assessment Finding. Direct emission consumes the same preflight. There
is no Metal-to-CUDA fallback.

The Compiler boundary ends at deterministic MSL. A local probe may compile, link, and
dispatch it as engineering verification, but that observation is not Calibration,
Evaluation Evidence, a performance measurement, or a Campaign result.

## Consequences

The Target Module gains Depth: vendor spelling changes are localized there while the
verifier and analysis retain one Interface. The backend map gains Leverage: adding Metal
does not add a parallel Compiler or duplicate Corpus machinery. Operator-specific
commitments stay local to the finite Metal Adapter and can later deepen into more generic
operation emitters as a third graph proves the repeated structure.

The successor Corpus deliberately adds two positive and two lowering-only negative Metal
cases. Existing v28 observations must remain unchanged. Release remains gated by an
external reviewer under ADR 0030; this change never creates or edits its own approval.
