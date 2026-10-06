# Metal mean30 / GLM successor experiment

Owner decision, 2026-10-06: subsequent experiments use `glm-5.3` through
`https://cloud.infini-ai.com/maas`. Credentials belong only to the private provider
environment outside source and evidence. Start with the existing Claude Code v4
artifact-optimization adapter; qualify initial/resume for the exact executable,
model and effort before the first search. Use `high` effort, as in the retained
GLM Metal engineering results; Codex qualifications do not qualify this treatment.

New Metal CLI launches default to `--metal-timing mean30`. Each participant has two
cohorts, in AB then BA order, with 3 untimed warmup calls and 15 timed command buffers
per cohort. Warmup and timed buffers each encode 64 dispatches; warmups are not samples. Divide each buffer's GPU interval
by 64, then take the arithmetic mean of all 30 samples. No trimming, outlier removal,
automatic repetition, or CV/IQR refusal. Raw samples and their dispersion remain
available. The 1.05 materiality threshold uses pooled means; pair wins are diagnostic.
Fresh confirmation applies exactly the same policy to the fixed nominated candidate.
These are amortized command-buffer measurements, not isolated kernel timestamps.

Use the already admitted observer's cohort mode. Mean30 does not require the proposed
interleaved observer, host-binding replacement, or a powermetrics session. Historical
v2/v3 investigations and failures remain unchanged. `--metal-timing legacy` explicitly
selects the previous assay; the Python helper's absent option also retains that policy.

Begin with a bounded two-turn RMSNorm run at R=128, C=1024, followed by GEMM+SiLU.
Declare these as engineering pilots for the evolution loop, not H1-H4 results or
independent search replications. Keep the Compiler fixed, retain distinct Cake
candidates, their own emitted MSL observations, negative findings and proposed compiler
changes. Automatic feedback currently supplies MSL and separate profiler evidence;
AIR guidance is included, but an AIR artifact is not automatically supplied. Record
unavailable AIR and physical register/ISA observations honestly. Each task keeps its own workspace, provider home and task instructions.
Claude safe mode disables native skill auto-discovery; deliver the complete Cake
skill and AIR guidance in the frozen AGENTS scaffold for this treatment, and report
that delivery mechanism honestly. A future native-skill treatment needs its own
qualification; it is not inferred from the Codex experiment.

The 30 timing samples are not 30 independent agent runs. Formal successor studies
still require fixed workload/model/material budgets, fresh searches and held-out
evaluation. Changing model and timing together creates a new engineering baseline;
it does not isolate a Compiler improvement against old Codex/median runs.
