# open-cake-ir local constraints

## Cake IR design principles (arXiv:2608.12629v1, Appendix B.1)

The paper states eight. They bind IR changes here; the global doctrine covers the rest.

- **P1 Ergonomic** — keep the editing model familiar to NumPy/PyTorch users, and avoid
  unnecessary destination-passing or grid bookkeeping.
- **P2 Performance-transparent** — keep performance-relevant hardware decisions visible
  and lowering behaviour inspectable.
- **P3 Canonical** — prefer one canonical form for each operation over equivalent
  alternative spellings.
- **P4 Statically type-checked** — use typing rules to constrain lowering and reject
  ill-typed programs during construction, not at emission.
- **P5 Analysis-friendly** — expose the information the supported static analyses need.
- **P6 Test-gated** — evaluate IR changes against the kernel-matrix tests.
- **P7 Analysis-consistent** — accompany changes to the IR data model with the
  corresponding analysis updates, in the same change.
- **P8 Hardware-grounded** — document the intended hardware behaviour of each operation.

## What the paper says not to do

- Layout is deliberately **not** a first-class abstraction. Do not introduce a layout
  algebra for an agent to manipulate. A Schedule records concrete storage and access
  commitments -- an SMEM view offset, an operand byte offset, a TMEM column range -- and
  the compiler verifies they stay mutually consistent. New primitives extend verification
  coverage without an agent learning a second language.
- Static analysis is a pre-compile gate **only within its modeled domain**. It does not
  prove global GPU correctness or capture all microarchitectural behaviour, and both
  false positives and false negatives occur. On-device measurement is the ground truth;
  a cost estimate never replaces it.
- Feedback to an author is localized correctness and performance diagnostics. A pass/fail
  bit, or one latency number, is the failure mode this harness exists to avoid.
- Timing-model coverage is evidence-gated per target. A target without its own calibration
  reports a coverage limitation; it never inherits another target's estimates.
- Human judgement gates compiler evolution. An automatic estimate does not authorize a
  Revision.

- The Compiler is the product core. It must not import Lab, provider, workload, campaign, evidence-store or claim code.
- The Lab may depend on a frozen Compiler Revision; a campaign may never mutate that revision.
- Workload Contract owns operator semantics and oracle. Study Contract owns treatment, estimand and analysis.
- `matched_search` and the evidence-justified exact-shape `portfolio` variant share one Lab path. Serving remains a
  future artifact handoff, not a runtime mode.
- `artifact_optimization_only` is a non-scientific Claim Scope on `matched_search`, not a mode. It may expose
  provider-default features, but Candidate promotion still requires common confirmatory Evaluation and never forms
  an arm comparison.
- KernelSeed and Workload-case specialization are owned by Lab; Compiler accepts complete Schedules and must remain
  unaware of held-out roles or Study policy.
- Compiler changes require a full Corpus Gate and human merge before producing a new Compiler Revision.
- A Revision id is derived by its cycle script, never chosen by hand. An id some frozen artifact names is history and
  its bytes are immutable; an id nothing names is a working artifact and is re-released in place. Editing a
  Revision-bound source is therefore free of version churn until a sealed run witnesses the id.
- Never regenerate Corpus Gate expectations to make the gate pass — that reports a match it just manufactured. Adopt
  new expectations as a separate, reviewed act (`tools/refresh_corpus_expectations.py --write`) and name the reason in
  the release approval basis.
- Cake versus CUDA is an Authoring Environment assignment, not a syntax-only representation switch.
- Common Evaluation begins only after an arm produces a sealed launchable artifact.
- Every candidate, evaluation and terminal outcome is append-only; reports and status docs are derived views.
- Do not copy legacy `rXX`, `vN`, failure or archive runners. Historical implementation lives in pinned Git.
- No formal provider or GPU experiment is authorized before all applicable acceptance gates pass.
- Generated runs and secret bytes stay outside source. Cleanup of legacy data requires separate user authorization.
- Every new Campaign Lock, Evidence root and report stays outside the checkout; historical in-checkout Campaigns are
  read-only replay inputs. New lifecycle layout follows ADR 0005 only at successor Revision boundaries.
