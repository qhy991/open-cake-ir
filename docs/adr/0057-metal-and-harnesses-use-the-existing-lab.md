# ADR 0057: Metal and CLI harnesses use the existing Lab

Status: implemented; validation baseline `555b403`, Metal Executor v64.
Provider-driven optimization remains unproven.

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
artifact without source compilation fallback. The released Metal Executor binds the
actual host/OS/toolchain and native Swift assets.

The [Metal task guide](../metal.md) documents the implemented path: one Open Cake
environment and sealed fixed baseline under `matched_search` /
`artifact_optimization_only`, common Metal timing and separate profiling, native
harness adapters with qualification gates, and a thin launcher requiring task,
backend, model, harness, effort and workspace. Scientific two-arm rules retain their existing scope. One
workspace/session is retained across Ralph turns; cross-process resume is unsupported.

## Verified scope and remaining limits

[Managed CI](https://github.com/qhy991/open-cake-ir/actions/runs/34180293661) passed on
exact commit `555b403` for Python 3.10 and 3.12. Native A/A checks then exercised
RMSNorm, LayerNorm and residual RMSNorm at `(128, 1024)` through the common
broker/Evaluation, using the same sealed program as candidate and baseline.
Correctness, input preservation and separate compute-stage timestamp profiling passed
for the examined cases. All timing-stability gates failed; no optimization result or
framework acceptance is established.

Real provider qualification attempts failed before a multi-turn task: the tested
Codex installation was incompatible with the requested model, and Claude Code hit
its quota. Successful two-turn qualification and actual persistent model-driven
optimization remain to be demonstrated. No model substitution or fixture qualification
can fill that gap. Stable measurement and fresh confirmation remain necessary for an
improvement claim.

Existing Compiler/Executor releases and historical evidence stay in pinned Git and
prior checkouts. Implementation, CI, native runtime checks and accepted optimization
results remain distinct evidence domains.
