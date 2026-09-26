---
name: metax-c550-hardware-exploration
description: Research, extend lowering for, and optimize Cake kernels on MetaX C550 using version-matched MXMACA/Triton documentation, exact Target facts, and C550 measurements. Use for C550 hardware documents, backend gaps, kernel tuning, profiler interpretation, or hardware-limit analysis; not for generic GPU support or unqualified whole-model performance claims.
---

# MetaX C550 hardware exploration

Help an agent find C550-specific optimization opportunities and test them without turning adjacent-platform assumptions, vendor examples, or marketing peaks into compiler facts. Keep the exact Workload and target contract fixed; document what is known, what is hypothesized, and what evidence would decide the hypothesis.

## Start with the exact platform

Read the repository's [`docs/metax-c550.md`](../../docs/metax-c550.md), [`compiler/targets/xcore1002.json`](../../compiler/targets/xcore1002.json), and the captured [`runtime/hosts/xcore1002.json`](../../runtime/hosts/xcore1002.json). Check the current checkout and the actual campaign's `runtime.json`/report before using any version or capability from an older note. If one of these paths has moved, locate its current owner rather than creating a duplicate.

Keep these identities separate:

- Target/device identity `xcore1002` and physical device ISA;
- native code-generation family `xcore1000` and `mcfatbin` artifact;
- MACA Triton API target fields such as `GPUTarget("maca", 80, 64)`, where `80` is an API compatibility value, not NVIDIA SM80;
- exact C550 host, driver, MACA SDK, PyTorch, FlagTree/plugin, Triton, `mxcc`, MCPTI and profiler versions.

The active run's version wins over a portal's latest release. Never fill an undeclared or unmeasured Target field from another C-series card, another SDK release, CUDA analogy, marketing page, or an unverified kernel guide. A disagreement between official documentation, the captured host, and C550 device queries is a finding to investigate, not a reason to choose whichever value helps a candidate.

## Find and classify documentation

Use [references/document-map.md](references/document-map.md) for the curated MetaX and repository source map. For a live documentation request, search the official MetaX Developer Center first and open the exact document/release. Confirm the product applicability, title, version, date, section and whether the full document is accessible and redistributable. The portal's search result is an index, not proof that the linked PDF is the same version; verify the document itself when index and preview disagree.

Classify every useful statement before relying on it:

| Evidence class | What it can support | What it cannot establish by itself |
|---|---|---|
| Versioned vendor manual/release note | Documented API behavior, supported feature, limitation or known issue for that release and product family | That the live C550 image contains that release, or that a feature is efficient |
| Captured C550 Target/host query | A declared hardware/runtime fact under that captured host and software | A universal C550-family property or a performance peak |
| C550 correctness, timing or profiler receipt | The exact workload, shape, artifact, route and measurement boundary recorded in the receipt | Other shapes, kernels, serving behavior, or full-model speedup |
| Vendor/community sample, optimization guide or forum answer | A candidate hypothesis, a failure clue, or a search term for primary documentation | A legal Compiler capability, current version contract, or validated performance result |

Prefer the version-matched `mcTriton` guide and MXMACA release notes for the current Triton route. Use EID/error documentation and official support Q&A to interpret faults, while retaining their release and context. Do not copy proprietary manual text into the skill or repository; link to the authorized source and record only the concise facts needed for the task.

## Turn documentation into testable work

1. Freeze the Workload contract, exact target, source commit, host/runtime, precision, shapes, oracle, tolerance, baseline, timer, and allocation mode. Distinguish fixed-kernel optimization from whole-Program or serving work.
2. Retrieve only the manual sections relevant to the kernel mechanism. Record a short hypothesis with its source locator, exact applicability, predicted change, and a counterexample or falsification condition.
3. Express candidates only through the current Cake Schedule and admitted `triton-metax` path. A document or sample does not add Schedule vocabulary, an instruction contract, an emitter route, or a Compiler pass. If the needed mechanism is not expressible or admitted, route it to the proper IR/backend/verifier/Compiler owner and preserve the refusal evidence.
4. Treat static estimates and profiler attribution as guidance. Compile and admit the exact artifact; check full output correctness and input immutability; then use the C550-declared timer and reset protocol, paired baseline measurements, quality gates and independent confirmation. Keep profiling evidence separate from the timing score.
5. State measurement gaps plainly. In particular, distinguish MCPTI single-dispatch time from host/framework time; distinguish measured device-memory traffic from logical tensor bytes; do not call interconnect bandwidth HBM bandwidth. Build a Roofline only from a version-matched, target-applicable measured or vendor-documented ceiling, with assumptions and missing counters recorded.
6. Preserve negative, flat, slow, unsupported and provider-fault results. A provider fault is not a hardware limitation; a correct candidate is not necessarily faster; a task-level win is not an end-to-end win.

## Extend a MetaX lowering rule

Trace one Schedule from construction through `Compiler.assess`, `backends.triton.preflight`, emitted source, TTIR/TTGIR and device execution. Record the first divergence. A rejection by a Target contract, a MACA API keyword error, a compiler assertion, an incorrect result and a slow but correct kernel call for different changes. Keep the shared Triton emitter and put a MetaX-specific source spelling or admission rule with `compiler/backends/metax.py` when that is the actual difference; add a new `LoweringBackend` only if a distinct source-generation mechanism is required.

The captured 3.1 route accepts `num_stages` on `tl.range` but refuses other unsupported keywords. `tl.static_range` exists in that installation and is the narrow source spelling for an explicitly full-unrolled, fixed, single-stage loop; it is not a way to drop a partial-unroll or pipelining commitment. Check the emitted loop, actual TTIR/TTGIR and complete output on the exact captured runtime before extending the admitted scope. Upstream Triton API documentation explains the iterator, but does not qualify the installed MACA backend.

For a new rule, retain a positive Schedule and a counterexample naming the rule that refuses it. Validate unchanged source and findings for other vendors, then compile with the exact installed MetaX distribution. Run any authorized device check through the existing `maca` broker and compare against the Workload oracle; do not infer speedup from successful lowering or correctness.

For a rounding-sensitive instruction, include an input that distinguishes it from a composition of older operations. Compare complete output bits with an independent oracle; a `math.fma` node in TTIR/TTGIR establishes compiler intent, while the device result tests its realized numerics.

E4M3FN load/store or decoding evidence does not admit direct FP8 `tl.dot`. The captured 3.1 route and a later 3.6 C550-2 offline probe both failed before a native artifact. Keep the Target's direct FP8 dot contract absent, retain the exact compiler diagnostic, and treat any explicit FP8-to-BF16/FP32 matrix path as a separate numerical mechanism with its own oracle and precision boundary.
The admitted `maca.simt.fp8e4m3_compensated_fp32` mechanism is such a separate path: a fixed 64x64 case with 2x64x64 staged tiles, FP32 products and compensation. Read its current preflight and [platform evidence](../../docs/metax-c550.md) before using it; its name and receipt do not imply native FP8 matrix execution, other shapes, or a measured speedup.
For a proposed one-row variant, consult [F-2026-09-26-001](../../findings/2026-09-26-001-metax-fp8-one-row-capacity.json): a scalar program axis loses the singleton tile dimension before MMA, so merely widening the MetaX preflight cannot make the source legal. Treat a controlled latency direction as a reason to investigate the canonical IR/Verifier shape owner, not permission to hide a reshape in emission.
The existing `cast + broadcast + mul + reduce` composition expresses a fast one-row candidate for the frozen FP8 case, but its ordinary FP32 sum failed additional finite-input precision checks. Require held-out numerical evidence before replacing the compensated route; five-case timing and a close A/A control do not establish general precision eligibility.
An exploratory FP64 K-reduction route has bounded C550 correctness and a quality-passed directional timing successor, but the Cake DType vocabulary does not admit FP64. Read [F-2026-09-26-002](../../findings/2026-09-26-002-metax-fp64-reduction-capacity.json) before proposing an accumulator-precision or type change; standalone compilation and one paired diagnostic do not authorize a hidden cast, a new Target fact or a general speedup claim.

## Reference-access and action boundaries

Follow the repository's per-arm `reference_access` contract. Prose hardware manuals and high-level API specifications may be useful in clean-start work when the frozen Study permits them. Kernel source, complete low-level implementations, generated code and compiled artifacts are separate semantic reference roles; do not expose them to clean-start authors unless the arm explicitly allows that role. Known-kernel reproduction and direct-low-level arms follow their own frozen access rules.

This skill does not authorize downloading restricted manuals, changing a live SDK/container, launching or intervening in a GPU campaign, altering a frozen Workload, or widening a compiler capability set. Obtain the authorization required for those actions and preserve active-run source/evidence identities. Update hardware facts only in their canonical Target/host owner; put reusable optimization results in the platform's existing evidence/results path, and route cross-platform mechanisms through the repository's branch workflow.

## Report the outcome

Separate documented support, captured device facts, static/compiler admission, GPU correctness, timing quality, profiler attribution, independent confirmation and performance qualification. Cite document version/section and exact run/receipt paths. Give numerical speedups only for the exact paired workload case and baseline; include regressions and missing evidence. Do not claim a general C550 ceiling, compiler-wide benefit, full-model improvement or serving improvement from isolated kernel results.
