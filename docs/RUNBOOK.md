# Canonical execution runbook

[中文阅读](zh-CN/RUNBOOK.md) · [Bilingual catalog](README.md)

This runbook describes the supported operating path, not authorization or historical
experiment results. A live run still requires the exact Workload and Study gates, a
qualified provider, released Compiler and Executor authorities, admitted GPU
infrastructure, new external output roots, and explicit permission for external effects.

New users should first follow [`GETTING_STARTED.md`](GETTING_STARTED.md), which runs no
provider Campaign and makes no performance claim. Terms in this runbook are defined in
[`GLOSSARY.md`](GLOSSARY.md).

## 1. Resolve the source identity and the host

Do not copy identities from prose. The identity is this checkout's commit:

```bash
git rev-parse HEAD
git status --porcelain --untracked-files=all   # must print nothing
ls runtime/hosts
.venv/bin/python tools/render_current_status.py --check
```

A Compiler is `open-cake-ir@<commit>` and an Executor is `<target>@<commit>` over the
committed capture in `runtime/hosts/<target>.json`
([ADR 0065](adr/0065-source-identity-is-the-commit.md)). A checkout carrying modified or
untracked files has no identity, and every Lab boundary refuses it with the paths that
differ.

A Study template may use `{"binding":"current_release"}`. Lab preflight resolves that
moving reference once into exact Compiler and Executor references in the CampaignLock. A
frozen Study always carries exact references and never follows a later commit.

## 2. Verify the Compiler and capture a host

Run the full Corpus Gate before using the Compiler:

```bash
open-cake-ir compiler check-corpus --revision compiler/revision.json
```

CI runs that same check on every commit. No release document is minted per change: a
change reaches campaigns when it merges to `main`, and the merge carries one review by
someone other than its author. Changing a Target definition is not an ordinary source
edit; review the hardware contract and its citations with it. Never regenerate Corpus
expectations merely to make a proposed change pass. Adopting new expectations is a
separate reviewed action (`tools/refresh_corpus_expectations.py --write`).

Capture a host on that host, using its intended Python invocation and the installed
distribution names, then commit the file it writes:

```bash
PYTHONPATH=src /absolute/environment/bin/python tools/capture_executor_host.py \
  --target sm_103a \
  --package torch --package triton --package numpy \
  --package cuda-bindings --package flashinfer-python \
  --cupti-distribution cupti-python --flashinfer-distribution flashinfer-python \
  --ncu /absolute/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu
```

The command writes `runtime/hosts/<target>.json`. It records the current interpreter and
installed versions, binds CUPTI's non-bytecode package files plus its distribution
metadata and RECORD, and binds FlashInfer's distribution-owned
`flashinfer/testing/utils.py`. It reads `ncu --version` from the supplied executable. Host
admission, including the real CUPTI and helper imports, and profiler admission must
succeed before the file is written. Run it as a fresh process and retain stdout/stderr,
including any cold-import failure; it installs nothing and repairs nothing. A Metal host
passes `--kind metal` with its Swift, archive and observer executables instead. Recapture
only when the host itself changes, with `--replace`, and commit that change. This
establishes the host boundary, not GPU correctness or performance: no kernel is
dispatched.

### The external AKA corpus (closed)

AKA review was a read-only challenge-corpus workflow, not a Compiler Corpus Gate, an IR
relevance judgment, or GPU/performance evidence ([ADR 0068](adr/0068-aka-is-a-challenge-corpus-not-a-compiler-corpus.md)).
The campaign is closed. Its harness -- `tools/audit_aka_corpus.py`,
`tools/review_aka_expressibility.py`, the `tools/run_aka_*.py` runners and their
plan, admission and summary tools, with their contract tests -- lives on the `history`
branch at its original paths (`git show history:tools/<name>.py`), and its published
dataset stays under `docs/data/` with `tools/verify_aka_qualified_review_export.py` and
`tools/verify_aka_fma_v41_reaudit.py` as the retained verifiers. Every retained result
remains provisional with `semantic_binding=reviewer_claimed` and `gpu_test=not_run`; do
not infer a new primitive from a model Turn alone.

## 3. Select and freeze a Study

Choose by question, not by historical sequence number:

| Question | Template or Study kind |
| --- | --- |
| Does the zero-GPU matched control plane compose? | `matched-search-infrastructure-template.json` |
| Does the two-file Ralph path compose with time/token/Evaluation budgets? | `matched-search-system-qualification-ralph-template.json` |
| What is the best confirmed artifact through the two-file Ralph loop? | `artifact-optimization-ralph-template.json` |
| Does an implementation-free matched reference boundary hold? | `matched-search-clean-start-reference-template.json` |
| Can Cake and native Triton optimize the same B300 baseline through Ralph? | `matched-search-triton-b300-optimization-template.json` |
| Does a frozen exact-shape specialist set generalize to its declared cases? | Historical replay at its pinned commit (ADR 0071) |

Template names are discovery aids; the content-bound Study Contract is the authority.
Preflight writes a new CampaignLock outside the checkout:

```bash
open-cake-ir lab preflight contracts/studies/<study>.json \
  --output /new/external/path/campaign.lock.json
```

`tools/create_study_successor.py` creates an explicit zero-GPU successor and validates it
through Lab preflight. It does not refresh a live worker command. Use
`tools/freeze_live_matched_study.py` when provider qualification, Executor, runtime
configuration, broker command, model, reasoning effort, or custody changes.

Templates whose execution fields use `{"binding":"campaign_lock"}` also require an
external execution binding. This includes the B300 Triton templates. Follow
[the B300 binding example](B300.md) and [ADR 0056](adr/0056-fixed-baseline-paired-execution.md):

```bash
open-cake-ir lab preflight contracts/studies/matched-search-triton-b300-optimization-template.json \
  --execution-bindings /new/external/path/execution-bindings.json \
  --output /new/external/path/campaign.lock.json
```

When the Study declares external advisory cost selection, also pass
`--empirical-cost-model /external/path/model.json`. The model must cover the exact
declared target; it does not replace measured acceptance.

All matched-search Studies use `task_agents_ralph_v1`; Preflight renders no Prompt template. Live composition creates
one read-only `TASK.md` and `AGENTS.md` in each Run workspace and retains their exact bytes
with every StateCard. The workspace may contain only those files plus
`candidate-set.json` after a Turn.

The candidate envelope accepts valid UTF-8 JSON with arbitrary whitespace and object-key
order. Keys must be unique at every nesting level, numbers finite, and `schema_version`
the integer `1`. Each Turn retains the exact submitted file bytes. Audit projects those
bytes into the ordered canonical members again and checks their archived identities;
direct CUDA source strings retain their decoded UTF-8 bytes.

Clean-start references are replaced as one paired operation:

```bash
python tools/create_study_successor.py \
  --source contracts/studies/matched-search-infrastructure-template.json \
  --output contracts/studies/<new-clean-reference-study>.json \
  --study-id <new-clean-reference-study-id> \
  --open-cake-schedule-skeleton contracts/scaffolds/open-cake-clean-start-v1.json \
  --direct-cuda-candidate-skeleton contracts/scaffolds/direct-cuda-clean-start-v1.cu
```

The Cake IR starter remains structurally incomplete and the CUDA body remains empty. This
checks reference access only; it does not create a paper-aligned Study.

## 4. Qualify the provider without GPU

Provider qualification uses a fresh workspace, absolute external output paths, Evidence root, and Run id.
All outputs stay outside every enclosing Git checkout. Harness, exact model and reasoning effort are required.
For single-arm Metal tasks and Claude Code, use the [TaskLab task launcher](metal.md); executable test doubles
use `--fixture-only` and cannot issue live authority.
Cached login status alone is insufficient; the qualification must execute the exact
provider binary and frozen policy. For Codex, the native CLI's local Code Mode host is required
and explicitly enabled. Its selected path and bytes belong to the existing provider
configuration identity. Package resources take precedence over the native executable's
sibling helper; a changed selection or changed bytes rejects both initial and resumed
calls. Runtime configuration still supplies only the provider executable and workspace
root, with no helper override. Closed research disables shell, browser and web search.
The qualifier retains real file-change events and rejects startup or in-turn errors.

Closed matched authoring example:

```bash
python tools/qualify_codex_provider.py \
  --harness codex --model "<exact-model-id>" \
  --executable /absolute/path/to/codex \
  --provider-revision codex-cli-<version>-sha<digest-prefix> \
  --output-schema contracts/providers/codex-turn-output-schema-v1.json \
  --workspace /new/external/path/closed-provider-workspace \
  --receipt-output /new/external/path/closed-receipt.json \
  --anchor-output /new/external/path/closed-anchor.json \
  --evidence-root /new/external/path/closed-provider-evidence \
  --run-id <new-closed-provider-run> \
  --reasoning-effort xhigh \
  --maximum-candidates-per-turn 3 \
  --feature-policy closed_research
```

Provider-default artifact-optimization example:

```bash
python tools/qualify_codex_provider.py \
  --harness codex --model "<exact-model-id>" \
  --executable /absolute/path/to/codex \
  --provider-revision codex-cli-<version>-sha<digest-prefix> \
  --output-schema contracts/providers/codex-turn-output-schema-v2.json \
  --workspace /new/external/path/provider-default-workspace \
  --receipt-output /new/external/path/provider-default-receipt.json \
  --anchor-output /new/external/path/provider-default-anchor.json \
  --evidence-root /new/external/path/provider-default-evidence \
  --run-id <new-provider-default-run> \
  --reasoning-effort max \
  --maximum-candidates-per-turn 3 \
  --feature-policy provider_defaults_optimization
```

For the B300 Cake/native-Triton template, pass
`contracts/providers/codex-triton-optimization-output-schema-v1.json` as `--output-schema`.
The qualifier derives the two arms from this schema and validates both add/update paths.

Both policies use Ralph: two immutable task files and one candidate-set envelope.
There is no agent-interface selector or legacy prompt qualification.

Failure remains a sealed observation and issues no passing receipt. Reauthenticate before
using another Run id; never delete or rewrite a failed archive.

Freeze a live system-qualification successor from its matching receipt, seal anchor,
committed host capture, and exact runtime configuration:

```bash
python tools/freeze_live_matched_study.py \
  --project-root . \
  --template contracts/studies/matched-search-system-qualification-ralph-template.json \
  --qualification contracts/providers/<new-closed-receipt>.json \
  --qualification-anchor evidence/qualifications/<new-closed-anchor>.json \
  --executor runtime/hosts/<target>.json \
  --runtime-config /new/external/path/runtime.json \
  --reasoning-effort xhigh \
  --study-id <new-system-qualification-study-id> \
  --output contracts/studies/<new-system-qualification-study>.json \
  --enable-attribution
```

Reasoning effort has no implicit default. Changing it requires a matching provider
qualification and successor Study. A provider transport qualification neither authorizes
GPU work nor supplies a scientific result.

Every matched-search Study freezes one budget vector in its `budget` object:

- provider-token limit and checkpoints;
- maximum Turns and Candidates per Turn;
- wall-time safety limit;
- active provider-authoring time limit;
- independent search, confirmatory and attribution Evaluation limits.

The controller starts a Turn only when the remaining Evaluation budget can admit that
Turn's declared worst-case assay set. Queue or evaluator time is not charged as active
authoring time, but it remains inside the wall-time safety bound.

## 5. Preflight and execute

Use the `--execution-bindings` and, when required by the Study,
`--empirical-cost-model` arguments from section 3 for campaign-bound templates.

Every new CampaignLock, runtime configuration, provider workspace, and Evidence root must
be outside the checkout and create-only. Historical in-checkout Campaign material is
read-only replay input.

Before execution, verify that the broker service user can traverse every checkout parent
and read the sealed candidate handoff without weakening ownership modes. Broker admission,
GPU allocation, and production-job preservation remain external infrastructure gates.

```bash
open-cake-ir lab preflight contracts/studies/<frozen-live-study>.json \
  --output /new/external/path/campaign.lock.json

sg gpuq-users -c 'open-cake-ir lab execute \
  --lock /new/external/path/campaign.lock.json \
  --runtime-config /new/external/path/runtime.json \
  --evidence-root /new/external/path/evidence'
```

Lab—not the provider or worker—owns Turns, cumulative budget, bounded zero-work admission
resubmission, checkpoints, search selection, confirmatory promotion, and terminal sealing.

For `system_qualification_only`, exactly one Run per Authoring Environment exercises the
real canonical path. The report may state that the system path qualified; Estimand,
estimate, uncertainty, pooling, and comparative statistics remain absent.

For `artifact_optimization_only`, one multi-Turn Run per Authoring Environment may receive
measured feedback and promote one confirmatory-qualified Candidate. Auxiliary activity is
retained, the primary provider thread is the sole candidate-set writer, and the external
Lab is the sole evaluator. Per-Run observations must not be compared as an arm effect.

For profiler attribution, every correctness-qualified searched survivor receives a
separate no-timing profiler Evaluation. Raw profiler output is retained and replayed; its
duration is never Candidate latency.

## 6. Historical Portfolio replay

The Portfolio Study CLI is retired (ADR 0071). Replay old campaigns at their pinned
source commit. The template and runtime example are preserved on `history`.

## 7. Audit offline

```bash
open-cake-ir lab audit \
  --lock /new/external/path/campaign.lock.json \
  --evidence-root /new/external/path/evidence
```

Audit is read-only. It rebuilds Run Audits, checkpoints, endpoints, route counts,
Measurement Quality, and the Study's permitted report from retained authorities. It does
not normalize permissions or rerun a failed environment.

Archive Integrity may pass while Filesystem Custody is unverified. Such bytes remain
inspectable but cannot support promotion, system qualification, or a scientific estimate.
Missing or invalid evidence yields an unavailable conclusion, not a repaired environment
followed by a second attempt under the same identity.

## 8. Historical instruments and observations

Files under `inventory/`, `evidence/`, and dated analysis documents retain bound
observations. Do not infer that an old calibration driver, Workload adapter, or run command
is current merely because it remains in the repository. Check its pinned source closure
and owning contract before reuse.

Observation tools use create-only outputs. A renewed measurement gets a new path, date,
and bound authority; the earlier record remains history. Current calibration coverage is
read only from the released Compiler Revision.

## 9. Cutover and rollback

Source-authority cutover is an explicitly approved transaction, never an automatic
consequence of system qualification. Recheck the clean source revision, released
authorities, complete-history bundle, storage threshold, contract suite, offline audit,
remote target, and rollback plan immediately before cutover. Never run two active writers.

Rollback likewise requires explicit authorization. Stop active writers and retain their
terminal state, then create a new empty checkout from the verified legacy bundle. Do not
modify either existing tree or copy new Evidence into legacy state.

Task-specific Python/CLI entrypoints now live under `src/open_cake_ir/tasks/`; see [task ownership](en/TASKS.md).
