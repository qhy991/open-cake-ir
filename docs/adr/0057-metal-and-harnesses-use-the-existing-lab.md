# ADR 0057: Metal and CLI harnesses use the existing Lab

Status: implementation proposal from the owner's 2026-09-08 instructions; incomplete
Executor successor, not authorization for provider or GPU experiments.

The owner requires Python frontend operator tasks launched with explicit task,
backend, model, harness and effort, using a persistent workspace across iterations.
Codex and Claude Code must use the existing Lab. A separate tools-level optimization
loop is not an acceptable implementation of this request.

`TaskLab.execute` and `RalphController` remain the iteration authorities. Existing
candidate filtering, logical Evaluation attempts, fresh confirmation, attribution,
checkpoint projection and Evidence replay retain their owners. The launcher only
resolves the requested task/treatment/runtime bindings and persistent workspace.
No optimization loop belongs in the launcher. Task semantics and oracles remain under
`tasks/`; hardware build/launch/measurement remain behind existing common interfaces.

Python task references use the existing frontend parser without executing source.
Only Workload metadata may be inserted mechanically. Shape or arithmetic changes
must appear in the task-owned Python source. The existing `python_source` member is
submission transport; users and authors need not describe Schedule operations as JSON.

Claude is a native Provider adapter returning the existing ProviderTurn, sharing the
same qualified Run lifecycle and process supervisor. Its permission mode is not an
OS sandbox. Model, effort, executable, feature policy, observed session continuation
and usage require honest, harness-specific qualification; a CPU fixture is not that
qualification. The selected model must never be replaced silently.

Metal's compiled executable is a real `metal_binary_archive`, not a CUBIN or unbuilt
MSL source. A compile-only M1 Pro probe established source-free library loading from
such an archive and strict `failOnBinaryArchiveMiss` pipeline reconstruction. Changed
programs and empty archives were refused. Common Evaluation must load that exact
artifact without source compilation fallback. The host/OS/toolchain binding and new
Swift asset belong to the Executor successor.

Remaining integration is explicit:

- Express one Open Cake environment with a sealed fixed baseline inside existing
  `matched_search` / `artifact_optimization_only`; preserve scientific two-arm rules.
- Admit Metal timing/cache/profiler policies through common Evaluation and the broker;
  never label Metal observations CUPTI, NCU, CUDA occupancy or cold-L2 measurements.
- Bind a real Metal Executor host and include native assets in its release closure.
- Generalize provider qualification/configuration and replay for the admitted harness.
- Make workspace selection and continuation reach the existing Ralph path. Retain
  one workspace/session per Run; do not introduce a second checkpoint store or erase
  failed/ambiguous in-flight attempts. Cross-process recovery needs an explicit
  existing-engine continuation interface before it is claimed.
- Provide the thin launcher with required `--task`, `--backend`, `--model`, `--harness`,
  `--effort`, and persistent `--workspace`, then validate actual multi-turn execution.

Existing Compiler/Executor releases and historical evidence stay in pinned Git and
prior checkouts. Current CPU/compile-only seam tests establish only their tested
boundaries. They do not establish a working Metal Lab campaign, live Claude/Codex
qualification, framework acceptance, or a performance improvement.
