# open-cake-ir local constraints

- The Compiler is the product core. It must not import Lab, provider, workload, campaign, evidence-store or claim code.
- The Lab may depend on a frozen Compiler Revision; a campaign may never mutate that revision.
- Workload Contract owns operator semantics and oracle. Study Contract owns treatment, estimand and analysis.
- `matched_search` and the evidence-justified exact-shape `portfolio` variant share one Lab path. Serving remains a
  future artifact handoff, not a runtime mode.
- KernelSeed and Workload-case specialization are owned by Lab; Compiler accepts complete Schedules and must remain
  unaware of held-out roles or Study policy.
- Compiler changes require a full Corpus Gate and human merge before producing a new Compiler Revision.
- Cake versus CUDA is an Authoring Environment assignment, not a syntax-only representation switch.
- Common Evaluation begins only after an arm produces a sealed launchable artifact.
- Every candidate, evaluation and terminal outcome is append-only; reports and status docs are derived views.
- Do not copy legacy `rXX`, `vN`, failure or archive runners. Historical implementation lives in pinned Git.
- No formal provider or GPU experiment is authorized before all applicable acceptance gates pass.
- Generated runs and secret bytes stay outside source. Cleanup of legacy data requires separate user authorization.
