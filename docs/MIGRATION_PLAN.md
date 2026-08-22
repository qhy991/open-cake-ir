# Compiler-first one-way migration

## 1. Rule

The migration is a one-way cutover from a campaign-shaped legacy repository to a compiler-first product with a
dependent Research Lab. The legacy repository remains a read-only evidence source. No formal run dual-writes and
no legacy runner is copied for convenience.

Implementation status: P0-P7 code/evidence paths are implemented in the new architecture, including the real r43-r45
Portfolio and a non-scientific two-Run system Study. G7 r4 and G8 r6 pass on the remote B200 host; sole-owner Git
cutover remains pending because this migration does not authorize an automatic commit or deletion of legacy history.

## 2. What migrates

| Legacy fact or behavior | Target owner | Migration method |
| --- | --- | --- |
| Accepted/rejected Schedule semantics from r16/r25/r31 | Compiler Corpus | Distill canonical cases and expected Findings |
| Target capabilities used by accepted B200 paths | Compiler Target | Re-derive exact sm_100a facts with source citations |
| Canonical verifier and lowering behavior | Compiler Revision | Reimplement one path and prove corpus parity |
| Workload inputs, oracle and tolerances | Workload Contract | Hash-reference source facts and add contract tests |
| Cake/CUDA authoring surfaces | Lab Authoring Environment revisions | Freeze complete prompt/tool/compiler/feedback bundles |
| Common correctness and timing behavior | Evaluation Module | Extract after arm-specific build and prove raw parity |
| CAS, manifests and ledger behavior | Evidence Module | Deepen into one writer/replayer |
| r35/r36/r38/r40 failures | Regression Corpus | Replay earliest divergence through public Interfaces |
| Historical source implementations | Git revision | Reference only; never copy `rXX`/`vN` topology |
| README/Roadmap/claim map | Claim View | Regenerate from accepted Study Reports |
| Workspaces, pycache, probes and temporary clones | Nothing | Do not migrate; cleanup requires separate authorization |
| Credential and signing-key bytes | External custody | Migrate only non-secret identity/custody references |

## 3. Migration phases

### P0 — Freeze the legacy source set

Bind the clean final Stage 6 revision rather than a moving branch. Record repository/tree IDs, origin divergence,
evidence roots and every admitted object digest in `legacy_manifest.jsonl`. The final revision is `2fa79092...`,
tree `b02d730b...`; all r41-r45 non-raw files and five raw Evidence trees are covered. The revision was initially
observed 68 commits ahead, is now present at origin, and also has a verified complete-history Git bundle.

No legacy object is deleted or rewritten. The manifest classifies each source as compiler corpus, workload fact,
evaluation fixture, evidence fixture, historical implementation or ephemeral state.

### P1 — Distill the Compiler Corpus

Start from real accepted schedules, not a new universal IR design:

- r16 minimal Schedule path;
- r25 complete Flash-KMeans assignment path;
- r31 TinyGEMM2 path as the real second family;
- the corresponding rejected schedules and capability failures.

Resolve synonyms across `ir.py`, Schedule-v2, explicit-resource and production-resource models into one canonical
Schedule vocabulary. Each proposed construct must either express a retained corpus case or enforce a stable Target
invariant. Record source digests and expected Findings/Lowering observations before implementing the new Compiler.

### P2 — Compiler vertical slice

Implement the smallest independent Compiler path through its public Interface:

```text
Schedule -> assess -> localized Findings -> eligible -> lower -> inspectable target source
```

Begin with one complete r25-derived fixed-shape Schedule plus one rejected sibling. The slice must be independent of
providers, Lab, Evidence storage and GPU execution. Exact Target mismatch and unsupported analysis/calibration fail
explicitly. Red-green cycles add one corpus behavior at a time.

### P3 — Compiler Revision and Corpus Gate

Add the r31 TinyGEMM2 family to test whether abstractions earn their depth. Freeze a content-addressed Compiler
Revision only when the full accepted/rejected Corpus reproduces its declared assessments and deterministic lowerings.
A human-reviewed Corpus Gate report is the release evidence. Compiler gaps discovered later create proposals for
the next revision; they do not patch this one.

### P4 — Common Evaluation vertical slice

Migrate Workload Contracts for Flash-KMeans and TinyGEMM2, then implement common Evaluation from a sealed
LaunchableCandidate. Prove that Open Cake and direct CUDA launchable artifacts cross the same materialization,
oracle, correctness, timing and route-accounting Interface.

Use r37 correctness and r39 raw timing as parity fixtures. Separate search measurements from confirmatory
measurements, and permit multiple Evaluation Receipts for one Candidate. Arm-specific source checks and toolchain
build remain in their Authoring Environments, not Evaluation.

### P5 — Evidence vertical slice

Migrate the content-addressed store, append-only ledger and terminal archive as one Evidence Module. Import r39
success and r40 failure bytes by reference, without changing their historical claim level. The same replay path
must accept intact success and failure archives, report protocol deviations independently of integrity, and rebuild
all projections after deleting them.

### P6 — Research Lab vertical slice

Implement `matched_search` only. Separate Workload Contract from Study Contract; preflight resolves both plus exact
Authoring Environment, provider, Compiler, toolchain and machine identities into one Campaign Lock.

First execute a zero-GPU fake-provider Run covering initial turn, resume, rejection, protocol fault, terminal archive
and the three checkpoint states. Then execute one two-turn live-provider qualification without GPU to settle actual
resume behavior. No scientific Campaign is authorized in P6.

### P7 — Bounded system pilot

Freeze one infrastructure-only Study Contract with one small Run per Authoring Environment. Use a clean pinned
checkout and the same public Interfaces from preflight through fresh-process audit. The pilot validates the system,
not the paper effect, and never reuses r40 outcomes.

### P8 — Cutover

After the pilot and completion audit pass:

- anchor the new repository and Compiler Revision remotely;
- make the legacy repository read-only for historical replay;
- prohibit old/new dual-writing and old runner invocation;
- publish a rollback procedure to the pinned legacy revision;
- authorize the first full matched-search Study Contract.

A successor study adds contract data, not `r41_runner.py` or another runtime path.

### P9 — Claim-stage boundary

A fixed-shape Study may emit a sealed Kernel Seed. The final remote run supplied the real inputs and endpoints for a
closed `portfolio` Study, now implemented without adding a mode. Serving remains future work because no serving
framework or endpoint was implemented remotely; portfolio correctness cannot satisfy that gate.

## 4. Deletion and compatibility rules

- No artifact is deleted before its legacy manifest entry and digest verify.
- Historical execution uses its pinned legacy checkout; the new runtime does not preserve old implementation seams.
- A compatibility reader survives only at ingest and only for a current evidence consumer.
- There is exactly one writer for new Evidence and one runnable repository after cutover.
- Compiler source never imports a legacy reader, campaign, provider or evidence writer.
- Old workspaces may be retired only after necessary bytes are in canonical storage and the user authorizes cleanup.

## 5. Stop conditions

Migration stops and returns to design if a proposed abstraction lacks a second corpus use, if one fact gains two
owners, if a test observes source layout instead of an Interface, or if parity can be achieved only by introducing a
new versioned runtime path.
