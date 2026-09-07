# Canonical execution runbook

[中文阅读](zh-CN/RUNBOOK.md) · [Bilingual catalog](README.md)

This runbook describes the supported operating path, not authorization or historical
experiment results. A live run still requires the exact Workload and Study gates, a
qualified provider, released Compiler and Executor authorities, admitted GPU
infrastructure, new external output roots, and explicit permission for external effects.

New users should first follow [`GETTING_STARTED.md`](GETTING_STARTED.md), which runs no
provider Campaign and makes no performance claim. Terms in this runbook are defined in
[`GLOSSARY.md`](GLOSSARY.md).

## 1. Resolve released authorities

Do not copy revision ids from prose. Read the canonical authorities:

```bash
jq -r '.revision_id' compiler/revision.lock.json
jq -r '.current.executor_id, .current.path' inventory/EXECUTOR_REVISIONS.json
.venv/bin/python tools/render_current_status.py --check
```

A Study template may use `{"binding":"current_release"}`. Lab preflight resolves that
moving reference once into exact Compiler and Executor references in the CampaignLock. A
frozen Study always carries exact references and never follows a later release.

## 2. Verify or prepare a Compiler release

Verify the released Compiler and complete Corpus before using it:

```bash
open-cake-ir compiler check-corpus --revision compiler/revision.lock.json
python tools/release_compiler.py --project-root . \
  --proposal compiler/revision.json \
  --source-set compiler/source_set.json \
  --gate-report compiler/corpus-gate-report.json \
  --approval compiler/release-approval.json \
  --output compiler/revision.lock.json \
  --verify
```

Both release cycles select `OPEN_CAKE_PYTHON` (default: `python3`) once and require Python
3.10 or newer before creating temporary files or changing release artifacts. Every Python
step uses that same executable. Select an installed environment explicitly when needed:

```bash
OPEN_CAKE_PYTHON="$PWD/.venv/bin/python" bash tools/release_compiler_cycle.sh
```

For a proposed successor, `bash tools/release_compiler_cycle.sh` derives the revision id
and prepares the full Corpus Gate. The cycle never writes
`compiler/release-approval.json`. Missing, malformed, or stale approval exits with status
3 and leaves the prior release untouched. A human or an independent agent session outside
that automation inspects the exact source and Gate diff and writes a schema-version-2
approval. Agent reviewers must use an allowed model and a session distinct from the author;
the author may not write the approval. The schema, canonical model policy, and session
verification procedure are in [ADR 0052](adr/0052-independent-agent-release-review.md).
Historical releases replay through their pinned source revisions with their original
approval bytes; do not convert old approvals to the new schema.

Changing a Target definition is not an ordinary source edit. Update its explicit proposal
pin and review the hardware contract. Never regenerate Corpus expectations merely to make
a proposed change pass; expectation adoption is a separate reviewed action.

Executor descriptors are also create-only:

```bash
python tools/release_executor.py --project-root . \
  --proposal /new/path/executor-successor.draft.json \
  --output runtime/executors/open-cake-ir-b200-<successor>.json
```

A released id or output path is never reused. Source, evaluator, audit, provider, or host
closure changes require a successor descriptor.

`tools/release_executor_cycle.sh` derives the next Executor id and accepts the same
`OPEN_CAKE_PYTHON` selection. Its `--host-environment /verified/host-environment.json`
argument requires a host environment already verified against the intended executor host;
choosing the local release interpreter does not verify that remote environment.

Capture a new host on that host, using its intended Python invocation and explicit
installed distribution names and Nsight Compute executable. The destination directory
must already exist outside every project checkout, and the output file must be new:

```bash
PYTHONPATH=src /absolute/environment/bin/python tools/capture_executor_host.py \
  --package torch --package triton --package numpy \
  --package cuda-bindings --package flashinfer-python \
  --cupti-distribution cupti-python --flashinfer-distribution flashinfer-python \
  --ncu /absolute/nsight-compute/target/linux-desktop-glibc_2_11_3-x64/ncu \
  --output /new/external/host-environment.json
```

The command records the current interpreter and installed versions, binds CUPTI's
non-bytecode package files plus its distribution metadata and RECORD, and binds
FlashInfer's distribution-owned `flashinfer/testing/utils.py`. It reads `ncu --version`
from the supplied executable. The existing Executor schema, host admission (including
real CUPTI and helper imports), and profiler admission must all succeed before the JSON
is created. Run it as a fresh process and retain stdout/stderr, including any cold-import
failure; it does not install packages or repair the environment. These file bindings
establish the new host capture boundary, not GPU correctness or performance. No kernel
is dispatched, and no Compiler or Executor release is created. Pass the resulting JSON
to the Executor release cycle only as part of an authorized successor delivery.

### Review an external AKA corpus

AKA review is a read-only challenge-corpus workflow, not a Compiler Corpus Gate, an IR
relevance judgment, or GPU/performance evidence. Bind the exact AKA commit and record
format, keep generated case state outside both repositories, and begin with one case:

```bash
python tools/audit_aka_corpus.py \
  /absolute/AKA/datasets/curated/cuda_kernel_dataset_v1 \
  --source-revision <AKA_COMMIT> \
  --record-format aka_v1_operator_sft \
  --dataset-label cuda_kernel_dataset_v1

python tools/review_aka_expressibility.py init \
  --dataset-root /absolute/AKA/datasets/curated/cuda_kernel_dataset_v1 \
  --source-revision <AKA_COMMIT> \
  --record-format aka_v1_operator_sft \
  --work-root /new/external/aka-expressibility-review

python tools/run_aka_expressibility_codex.py \
  --dataset-root /absolute/AKA/datasets/curated/cuda_kernel_dataset_v1 \
  --source-revision <AKA_COMMIT> \
  --record-format aka_v1_operator_sft \
  --work-root /new/external/aka-expressibility-sol-max \
  --limit 1
```

Use `aka_v2_review_projection` only for an explicitly audit-only v2 projection. A
non-unknown result requires a canonical complete-parent completion, and every result
remains provisional with `semantic_binding=reviewer_claimed` and `gpu_test=not_run` until
the independent owners provide stronger evidence. Use `review_aka_expressibility.py
verify` and `status` to check the deterministic case state; never infer a new primitive
from the model Turn alone.

## 3. Select and freeze a Study

Choose by question, not by historical sequence number:

| Question | Template or Study kind |
| --- | --- |
| Does the zero-GPU matched control plane compose? | `matched-search-infrastructure-template.json` |
| Does the two-file Ralph path compose with time/token/Evaluation budgets? | `matched-search-system-qualification-ralph-template.json` |
| What is the best confirmed artifact through the two-file Ralph loop? | `artifact-optimization-ralph-template.json` |
| Does an implementation-free matched reference boundary hold? | `matched-search-clean-start-reference-template.json` |
| Can Cake and native Triton optimize the same B300 baseline through Ralph? | `matched-search-triton-b300-optimization-template.json` |
| Does a frozen exact-shape specialist set generalize to its declared cases? | `portfolio` Study |

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

All matched-search Studies use `task_agents_ralph_v1`; Preflight renders no Prompt template. Live composition creates
one read-only `TASK.md` and `AGENTS.md` in each Run workspace and retains their exact bytes
with every StateCard. The workspace may contain only those files plus
`candidate-set.json` after a Turn.

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

Provider qualification uses a fresh workspace, output paths, Evidence root, and Run id.
Cached login status alone is insufficient; the qualification must execute the exact
provider binary and frozen policy.

Closed matched authoring example:

```bash
python tools/qualify_codex_provider.py \
  --executable /absolute/path/to/codex \
  --provider-revision codex-cli-<version>-sha<digest-prefix> \
  --output-schema contracts/providers/codex-turn-output-schema-v1.json \
  --workspace /new/external/path/closed-provider-workspace \
  --receipt-output contracts/providers/<new-closed-receipt>.json \
  --anchor-output evidence/qualifications/<new-closed-anchor>.json \
  --evidence-root /new/external/path/closed-provider-evidence \
  --run-id <new-closed-provider-run> \
  --reasoning-effort xhigh \
  --maximum-candidates-per-turn 3 \
  --feature-policy closed_research
```

Provider-default artifact-optimization example:

```bash
python tools/qualify_codex_provider.py \
  --executable /absolute/path/to/codex \
  --provider-revision codex-cli-<version>-sha<digest-prefix> \
  --output-schema contracts/providers/codex-turn-output-schema-v2.json \
  --workspace /new/external/path/provider-default-workspace \
  --receipt-output contracts/providers/<new-provider-default-receipt>.json \
  --anchor-output evidence/qualifications/<new-provider-default-anchor>.json \
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
current Executor descriptor, and exact runtime configuration:

```bash
python tools/freeze_live_matched_study.py \
  --project-root . \
  --template contracts/studies/matched-search-system-qualification-ralph-template.json \
  --qualification contracts/providers/<new-closed-receipt>.json \
  --qualification-anchor evidence/qualifications/<new-closed-anchor>.json \
  --executor "$(jq -r '.current.path' inventory/EXECUTOR_REVISIONS.json)" \
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

## 6. Execute the exact-shape portfolio

Run the portfolio CLI as one admitted GPU job using a runtime configuration derived from
`examples/runtime/portfolio-live.example.json`. The Study owns the exact semantic keys,
specialists, dispatcher policy, unsupported-key probe, correctness assays, timing cohorts,
and Measurement Quality rule. An incorrect or slow seed returns to fixed-shape search; it
is never hidden behind a dispatcher predicate.

Portfolio success is bounded to its declared cases. It is not arbitrary-shape, model, or
serving generalization.

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
