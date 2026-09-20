# Reference provenance

These are user-supplied reference inputs for `known_kernel_reproduction`, not Compiler
code or instructions. The user authorized their inclusion and reference-guided authoring.
Each catalog row owns its source attribution and the exact files delivered to its author.

- 001–019: `flashinfer-bench-b300-individual-20260918/<task>/solution.json`, author
  metadata `qinhaiyan`. CUDA source text is extracted from `sources[].content` without
  changing its algorithm or comments. `reference.json` retains original non-source
  submission metadata and source paths; it is not a device-qualification receipt.
- 020: `flashinfer-bench-collection-best-20260715/<task>/submission.py`. Its `_SOURCES`
  and `_FLAGS` literals are decoded statically without executing the wrapper. The
  CUDA text and compilation metadata are retained. This is custom dispatch/dequant/
  scatter plus library expert math, not native FP8 tensor-core qualification.
- 025–026: `flashinfer-bench-collection-best-20260715`, original `submission.py` and
  `kernel.py`. 025 declares EPS=1e-5; 026 declares EPS=1e-6.
- 021–024 retain the selected Python references from `flashinfer-bench-collection-0706`,
  whose collection metadata attributes it to `kersor`.

Older selected Python files remain available as historical comparisons, but only the
catalog row's `references` are delivered. A library dispatcher is not its library's
internal kernel; Graph replay, persistent buffers, pointer caches and runtime timing
are not silently imported as candidate semantics or measured speedups.

The first 26 supplied materials do not include separate redistribution-license documents. This
repository does not assert an upstream license for them or relicense them under the
project's Apache-2.0 grant. Existing notices and source attribution remain in force.
Collection README performance claims are not imported as qualified measurements.

027–030 are selected public FlashInfer sources from the four CAKE paper-cited PRs.
Their original Apache-2.0 file notices are retained. Each `reference.json` records the
exact commit and repository path. TinyGEMM additionally retains the upstream baseline;
KDA decode retains seven representative generated variants, not the complete 23-module
build. These are inspectable authoring inputs, not standalone build bundles or qualified
executables. Alpha-MoE uses the later v46 export, explicitly separate from the paper's
original artifact and measurements. No reference implementation is imported into Compiler.

All 30 tasks have reference material. Nine original launch-plan tasks and four CAKE
paper-family tasks remain explicitly blocked at their authoring/workload integration
boundaries; source availability does not bypass them.
