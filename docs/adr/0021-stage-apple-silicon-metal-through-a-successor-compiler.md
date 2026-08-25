# ADR 0021: Stage Apple Silicon Metal through a successor Compiler Revision

Status: proposed, 2026-08-25.

## Context

The released `open-cake-ir-sm100a-v3` Compiler works on Apple Silicon for parsing, assessment, deterministic
Lowering and its six-case Corpus Gate because those operations do not execute GPU code. That does not make its
Lowerings portable: the three retained profiles emit Triton, CuTe DSL or CUDA C++, and Evaluation admits only
CUDA launch artifacts.

The current extension surface is also more NVIDIA-specific than the Target abstraction suggests. Schedule v1
fixes `target` to `sm_100a` in its JSON Schema and names execution groups `warps`. Target schema v1 requires CUDA
compute capability, CTA, warp and tensor-memory fields. Lowering assets, toolchain requirements and Calibration
coverage are keyed only by profile, so a second Target could accidentally select an `sm_100a` template or inherit
its Calibration. The b32 static-semantics digest and the closed-profile semantic digests include the Target as well.

Most importantly, the v3 release descriptor hashes the live `src/open_cake_ir/compiler/core.py` path. Editing that
file would make the released Revision fail its own source-closure verification and would invalidate frozen Lab
inputs. The Apple port therefore cannot begin as an in-place edit to v3 or as a Target JSON containing invented
CUDA values.

Apple's public Metal tables identify M4 as Apple GPU family 9, list 1,024 threads and 32 KiB threadgroup memory as
the family limits, and expose BF16 and SIMD-scoped matrix operations. Metal also requires the actual SIMD width and
per-pipeline thread limit to be checked from `MTLComputePipelineState`. The local Apple M4/Xcode installation can
compile Metal 3.2 `bfloat` source, link a metallib and dispatch a compute pipeline. These facts establish a viable
toolchain seam, not a released Compiler Target, Workload correctness result or Evaluation Receipt.

Primary references:

- [Metal GPU family definitions](https://developer.apple.com/documentation/metal/mtlgpufamily)
- [Metal feature-set and implementation-limit tables](https://developer.apple.com/metal/capabilities/)
- [`MTLComputePipelineState.threadExecutionWidth`](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth)
- [`MTLComputePipelineState.maxTotalThreadsPerThreadgroup`](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup)

## Proposed decision

Add Metal through a new shadow Compiler engine and a successor Compiler Revision. Do not modify any file bound by
`open-cake-ir-sm100a-v3`, and do not add a generic `apple_metal` Target. The first Target will be the narrow,
content-addressed `apple_m4` contract: Apple family 9, exact admitted device names, Metal language/toolchain
requirements and resource limits backed by Apple documentation plus a reproducible device observation.

The shadow remains Python for this migration so it can reuse the frozen v3 verifier behavior without coupling the
Metal port to a language rewrite. ADR 0003 remains proposed, but its Rust-v4 sequencing is deferred: any Rust
shadow starts only after the multi-Target Python semantics and Corpus are released, with its Revision numbering
reconsidered at that time.

The successor uses the lifecycle layout from ADR 0005:

```text
definitions/compiler/targets/       immutable Target documents
definitions/compiler/corpora/       successor Corpus manifests and Schedules
definitions/compiler/schemas/       successor Schedule and Target schemas
proposals/compiler/                  mutable draft Revision and source-set review inputs
releases/compiler/                  human-approved released descriptors only
```

The `proposals/` review lifecycle is an explicit extension to ADR 0005. Draft descriptors are neither immutable
Compiler definitions nor released identities; promotion regenerates the released descriptor under
`releases/compiler/` and does not retain a compatibility copy.

Target schema v2 uses architecture-neutral execution terms: workgroup, execution group, execution-group width,
per-memory-space limits and source-language memory bindings. Schedule memory spaces remain the Compiler vocabulary
(`global`, `shared`, `register`, and, where admitted, `tensor`); Metal Lowering maps them to `device`, `threadgroup`
and thread-local storage. Successor Schedule schema v2 replaces v1's `target: const sm_100a` with a non-empty Target
identifier that must be bound by the Compiler Revision. It deliberately retains `roles[].warps`; the shadow
interprets those values only through the Target's declared 32-wide execution-group contract. A later language
proposal may rename that field, but the Metal port will not silently change its semantics.

Lowering and toolchain selection become a closed `(target, profile)` registry. An understood profile without an
exact Target implementation produces a stable blocking Finding; it never falls back to another Target's template.
Closed-semantics digests and Calibration coverage are also keyed by `(target, profile)`. No latency or performance
Calibration transfers from B200.

The first Compiler-to-Metal slice is one complete Flash-KMeans smoke Schedule and one rejected sibling, a
deterministic MSL template, source map and Metal toolchain requirements. The existing six `sm_100a` Corpus cases
remain byte-identical. The successor Corpus Gate contains those six cases plus the accepted and rejected Apple
cases, and differential conformance must preserve every v3 Finding and Lowering observation. The Apple draft cannot
be promoted until that complete gate passes and a human approves it.

Common Evaluation remains a later boundary. A Metal artifact becomes a LaunchableCandidate only after its complete
metallib, entry point, buffer ABI and dispatch manifest are sealed. MTLDevice discovery, metallib loading, MTLBuffer
ownership, correctness oracles and timing do not enter Compiler source. MLX, MPS and Torch may be Evaluation or Lab
integration choices, never Target semantics.

## Delivery plan

1. **Toolchain and dispatch probe.** Compile a tiny Metal 3.2 BF16 SIMDgroup-matrix kernel to AIR/metallib, load it
   through `MTLDevice`, dispatch it and verify a fixed result. Keep this an engineering probe: no Compiler Revision,
   Campaign, Evaluation Receipt or performance claim.
2. **Successor Compiler shadow.** Add Schedule and Target schema v2, `apple_m4`, explicit engine dispatch,
   Target/profile keyed Lowering and Calibration, the six byte-identical v3 cases, one accepted and one rejected
   Apple Corpus case, and a deterministic MSL Lowering. Keep v3-bound sources untouched.
3. **Lab-owned Metal artifact builder.** Add compile-only custody for MSL to metallib and bind Xcode, SDK, language
   standard, source digest, metallib digest and entry-point reflection into a sealed LaunchableCandidate. This still
   authorizes zero kernel measurements and is not yet selected by an Authoring Environment.
4. **Common Evaluation slice.** Add a Metal LaunchableCandidate adapter, run the Workload-owned correctness assay,
   retain raw results and only then add timing. Torch MPS interop is admitted only with an explicit ownership and
   synchronization contract; the baseline may use a small Swift or Objective-C++ bridge with owned MTLBuffers.
5. **Lab integration.** Select the Metal artifact builder and machine admission in a future Authoring Environment.
   Freeze a new Study Contract and Executor Revision before any formal run. Provider/GPUQ campaign work remains
   last.

Each step must be independently removable and must not broaden the claim established by the previous step.

This proposal implements step 1 only. Run the fixed probe with:

```bash
python tools/probe_metal_toolchain.py
```

After command-line arguments are accepted, success writes one canonical JSON observation to stdout; an operational
failure writes one fail-closed JSON document to stderr and returns nonzero. Standard `argparse` usage errors retain
their normal text diagnostics and exit code 2. AIR and metallib files live in a temporary directory and are removed
after their digests and the dispatch observation have been collected. The probe sources live under
`tools/metal_probe/` and are not Compiler Lowering assets.

## Acceptance gates

The toolchain probe passes only when `xcrun metal` and `xcrun metallib` succeed, the metallib loads, the named
pipeline dispatches, the fixed BF16 matrix result is exact, and the probe reports toolchain/source/artifact digests.
It does not establish Flash-KMeans correctness or performance.

The successor Compiler slice additionally requires deterministic repeated Lowering, no cross-Target fallback, no
cross-Target Calibration, exact Target capability Findings, the six byte-identical v3 cases plus an accepted and
rejected Apple case, differential v3 conformance, the full declared Corpus Gate, contract tests on non-Metal hosts,
an optional macOS syntax/toolchain test, and human review.

Evaluation support requires an external oracle comparison before timing. Lab support requires the applicable
acceptance gates in `docs/ACCEPTANCE_GATES.md`, a released Compiler Revision, a released Executor Revision and a new
Campaign Lock outside the checkout.

## Consequences

- The front-end remains usable on Apple Silicon today, while GPU support advances without rewriting historical
  Compiler or Evidence identities.
- The first committed executable is deliberately smaller than a Flash-KMeans port. It proves the uncertain Metal
  compile/load/dispatch seam and cannot be cited as a migrated workload.
- Target v2 and Target/profile dispatch are real compiler work, not data-only registration. They are isolated in a
  successor engine because v3's live-path source closure is immutable.
- A scalar MSL implementation cannot satisfy a hardware-explicit `mma` Schedule merely because its numeric output
  matches. The Apple Corpus template must preserve the admitted operation and execution-group commitments; a
  correctness-only scalar reference belongs in Evaluation.
- SIMDgroup matrix support does not by itself imply useful speedup. Performance claims require Apple-specific
  Calibration and confirmatory Evaluation.
- M1, M2, M3, M4 Pro/Max and future chips are not aliases of `apple_m4`; they require an explicitly justified Target
  admission or a later family-level Target proposal.
