# Canonical execution runbook

This runbook describes capability, not authorization. A live run still requires a current provider qualification,
new output roots and the applicable Acceptance Gates.

This is an operator reference. New users should first run the single-kernel B200 tutorial in
[`GETTING_STARTED.md`](GETTING_STARTED.md); it does not call a provider or start a Campaign.

## 1. Verify immutable inputs

```bash
open-cake-ir compiler check-corpus --revision compiler/revision.lock.json
python tools/release_compiler.py --project-root . --proposal compiler/revision.json \
  --source-set compiler/source_set.json --gate-report compiler/corpus-gate-report.json \
  --approval compiler/release-approval.json --output compiler/revision.lock.json --verify
python tools/release_executor.py --project-root . --proposal /new/path/executor-vN.draft.json \
  --output runtime/executors/open-cake-ir-b200-vN.json
python tools/build_legacy_manifest.py --source-set migration/source_set.json \
  --checkout /tmp/checkout-from-final-bundle --output migration/legacy_manifest.jsonl --verify
python tools/archive_compiler_release.py --project-root . --revision compiler/revision.lock.json \
  --evidence-root /new/path/compiler-release-evidence --run-id compiler-release \
  --index-output /separate/new/path/compiler-release-index.json
```

Executor output paths are create-only, and a released ID cannot be reused from another path. A source or audit change
requires a new Executor ID and output file; never turn a released descriptor back into a draft.

Every new Campaign custody root must be outside the checkout. `lab preflight --output` and `lab execute
--evidence-root` reject in-checkout paths before writing files or invoking execution inputs. Historical in-checkout
Locks and Evidence remain valid only for read-only `lab audit`.

Run `bash tools/release_compiler_cycle.sh` to derive the id and prepare the Corpus Gate. The command never writes
`compiler/release-approval.json`; it exits with status 3 when the existing approval is missing or does not bind the
new Gate. A reviewer outside that automation must inspect the exact Gate diff, then write the sole approval artifact
with `schema_version`, `decision: "approved"`, the Gate path and canonical SHA-256, a reviewer identity and an
approval basis. Rerun the same no-argument command to validate that artifact and install the verified lock. The
v24 approval predates this protocol and remains historical same-actor evidence. The current
v25 approval was written by the distinct tmux 882 window1 reviewer after it independently
bound the final 48-source, 32-case Gate.

The cycle does not re-pin a **Target definition**: `compiler/revision.json` holds each Target's `canonical_sha256`, and editing
`compiler/targets/*.json` means updating that pin by hand first, or the cycle stops at `target definition ... bytes
differ`. That is deliberate — a source edit is routine and a hardware description changing is not — but it is a step
the script will not take for you.

Expectations are never regenerated inside a release. `tools/refresh_corpus_expectations.py` prints the diff and exits
non-zero; `--write` adopts it. Read the diff first: adopting before reading is how a gate becomes a tautology.

`attribution_evaluation=correctness_then_profile_each_search_survivor` opts a matched
Study into a separate assay for every search survivor whose common launch passes
correctness. `tools/evaluate_flash_candidate.py` admits the exact NCU binary pinned by the
Executor, launches each sealed CUBIN once under NCU, checks that launch against the
external oracle, and retains the raw CSV plus its recomputed summary. Only the selected
survivor's projection enters next-Turn feedback. The receipt structurally has
`timing=null` and zero timing samples: profiler duration is never candidate latency. A
missing target-kernel row, missing metric, incorrect output, changed tool byte or
projection/raw mismatch fails the attempt. The older `correctness_then_profile` spelling
is frozen selected-only replay compatibility, not the current authoring operation.

## Instruments

These produce evidence rather than artifacts. Each one is checked in so a claim it supports can be repeated instead
of trusted; `docs/ANALYSIS_CALIBRATION.md` reads their output and `evidence/calibration/` holds it.

```bash
python tools/calibrate_wave_term.py --first 60 --last 400 --step 2      # exclusive: benchmark
python tools/calibrate_ranking_at_scale.py --size 512 --observed-at <iso8601> \
  --out /new/path/ranking.json                  # historical pre-v24 closure only
python tools/check_ranking_calibration.py                               # no GPU; exit 1 is a retained negative decision
python tools/observe_lowered_kernel.py --out inventory/<NEW>.json
python tools/profile_lowered_kernel.py --schedule <path> --out <NEW>.json
python tools/ir_vocabulary.py                                           # no GPU
python tools/audit_aka_corpus.py /absolute/AKA/datasets/curated/cuda_kernel_dataset_v1 \
  --source-revision <AKA_COMMIT>                                        # no GPU; read-only
python tools/report_schedule_work.py --target compiler/targets/sm_100a.json   # no GPU
python tools/observe_target_peak.py --out evidence/calibration/<NEW>.json    # exclusive: timing
```

The AKA audit is an external challenge-corpus projection, not a Compiler Corpus Gate. Its
summary reports storage, syntactic scope and lexical incidence; `--emit cases` produces a
reference-only human review queue. Both leave complete-parent and delta expressibility
unknown, ignore `excluded/`, copy no model-visible fields or evidence, and never authorize a
Schedule or vocabulary change. Use the exact Git revision that owns the selected shards.

Read-only Evidence audit does not normalize clone-time modes. It reports archive content
integrity and `filesystem_custody_verified` separately; weak modes leave intact bytes
replayable but block claim-bearing projections. Writer admission remains strict. Changing
modes immediately before audit is not evidence of continuous historical custody.

The retained ranking drivers and v6-v8 plans bind pre-v24 Schedule syntax and exact source
bytes. They are historical evidence, not current instruments: in the current checkout the
checker fails explicitly with `calibration Schedule differs`. A v24 successor driver and
plan must be created before another measurement; old thresholds, records and drivers are
never rewritten. Released `calibration_coverage` remains empty.

Both `observe_lowered_kernel.py` and `profile_lowered_kernel.py` build their inputs from
`tools/kernel_inputs.py`, so a kernel one of them profiles is a kernel the other checked.
Shapes, dtypes and argument order are derived from the Schedule's global buffers, because
that is where they are already declared and where the emitted host function validates
them. The adapter projects the retained builder and owns only post-freeze inputs such as
FP8 E4M3 tensors and the fixed empty/partial/full runtime extents of the admitted ragged
slice. Current correctness answers come from `tools/kernel_oracles.py`; it likewise
projects the retained base registry because `kernel_cases.py` bytes are part of frozen
ranking calibration authority and cannot be rewritten to add a new operator.

`observe_lowered_kernel.py` refuses to overwrite. A record is what happened once, so renewing one means a new file
and a new date; the earlier record stays as history for the Revision it was taken under.

## 2. Resolve a Study

The checked-in zero-GPU fixtures are stable `template` Studies. Their sole moving
spelling is `{"binding":"current_release"}`; preflight resolves that once into exact
Compiler and Executor references in the CampaignLock. A live or historical `frozen`
Study always carries exact references and never follows the inventory.

```bash
open-cake-ir lab preflight contracts/studies/matched-search-infrastructure-template.json \
  --output /new/path/campaign.lock.json
```

Which Study Contract is current is not a fact this document owns. Release scripts never rewrite a frozen contract;
`tools/create_study_successor.py` binds a new explicit fixture successor to the current Compiler and Executor and
validates it through Lab preflight. It deliberately refuses a live Study because changing a worker also changes the
broker command digest; `freeze_live_matched_study.py` is the sole path that refreshes that complete authority. The
names below were current when written.

The current checked-in scientific matched contract, `matched-search-infrastructure-template.json`, uses a zero-GPU fixture
provider and intentionally cannot start a live provider. Its Analysis Plan is the ADR 0013 successor: the terminal
budget comes only from `budget.limit`, candidate failure is observed, external failure is missing, and a complete
estimate requires conditional latency in both arms. Its Evidence policy is the ADR 0014 successor: every event kind
is closed, Run boundaries are checked, and search/diagnosis projections are derived during replay. The current
non-scientific G8 template is `matched-search-system-qualification-template.json`; earlier versions remain frozen
historical records. Freeze
a live successor only after a real two-Turn qualification emits a receipt with
the matching closed or tool-rich scope and binds the exact executable, model, reasoning effort, service tier, output
schema, prompt/scaffold bytes, removed environment, reference visibility, feature overrides and event contract.
`artifact-optimization-template.json` is the current zero-GPU contract fixture. The frozen
`artifact-optimization-verda-v7.json` remains a historical live authority for Executor v8; it is not executable
from the current source closure. Re-freeze a successor with the exact accessible checkout and broker command before
launching a live Campaign. Earlier revisions remain historical.

`matched-search-clean-start-reference-template.json` is the reference-access fixture. It inherits the scientific
template's local 150k/`max` factors, so preflight validates only that both arms receive implementation-free starters;
it is not a runnable paper result. Create later reference successors through the paired operation below so one arm
cannot silently retain a task implementation:

```bash
python tools/create_study_successor.py \
  --source contracts/studies/matched-search-infrastructure-template.json \
  --output contracts/studies/<new-clean-reference-study>.json \
  --study-id <new-clean-reference-study-id> \
  --open-cake-schedule-skeleton contracts/scaffolds/open-cake-clean-start-v1.json \
  --direct-cuda-candidate-skeleton contracts/scaffolds/direct-cuda-clean-start-v1.cu
```

The Open Cake starter must remain structurally incomplete and the direct CUDA function body empty. A future live,
paper-aligned successor separately needs an `xhigh` qualification, the declared 80M endpoint and the complete
isolation/provenance audit.

## 3. Qualify the live provider without GPU

First ensure a fresh process can authenticate; cached `codex login status` alone is insufficient. Use new paths and
one Run id exactly once:

```bash
python tools/qualify_codex_provider.py \
  --executable /absolute/path/to/codex \
  --provider-revision codex-cli-<version>-sha<digest-prefix> \
  --output-schema contracts/providers/codex-turn-output-schema-v1.json \
  --workspace /new/external/path/provider-qualification-workspace \
  --receipt-output contracts/providers/<new-live-receipt>.json \
  --anchor-output evidence/qualifications/<new-live-anchor>.json \
  --evidence-root evidence/qualifications/<new-live-run> \
  --run-id <new-live-run> \
  --reasoning-effort xhigh \
  --maximum-candidates-per-turn 3 \
  --feature-policy closed_research
```

Failure remains a sealed, externally anchored observation and issues no receipt. Reauthenticate before using a new
Run id; never delete or rewrite the failed archive.

For artifact optimization, use output schema v2 and the provider-default feature policy. This injects no
`--disable` flags and requires a real auxiliary shell lifecycle on both Turns:

```bash
python tools/qualify_codex_provider.py \
  --executable /absolute/path/to/codex \
  --provider-revision codex-cli-<version>-sha<digest-prefix> \
  --output-schema contracts/providers/codex-turn-output-schema-v2.json \
  --workspace /new/external/path/tool-rich-qualification-workspace \
  --receipt-output contracts/providers/<new-tool-rich-receipt>.json \
  --anchor-output evidence/qualifications/<new-tool-rich-anchor>.json \
  --evidence-root evidence/qualifications/<new-tool-rich-run> \
  --run-id <new-tool-rich-run> \
  --reasoning-effort max \
  --maximum-candidates-per-turn 3 \
  --feature-policy provider_defaults_optimization
```

Then freeze either non-scientific Study from its matching receipt, seal anchor and exact runtime configuration:

```bash
python tools/freeze_live_matched_study.py \
  --project-root . \
  --template contracts/studies/matched-search-system-qualification-template.json \
  --qualification contracts/providers/<new-live-receipt>.json \
  --qualification-anchor evidence/qualifications/<new-live-anchor>.json \
  --executor "$(jq -r '.current.path' inventory/EXECUTOR_REVISIONS.json)" \
  --runtime-config /new/path/runtime.json \
  --reasoning-effort xhigh \
  --study-id <new-g8-study-id> \
  --output contracts/studies/<new-g8-study>.json \
  --enable-attribution
open-cake-ir lab preflight contracts/studies/<new-g8-study>.json \
  --output /new/path/g8-campaign.lock.json
```

For an artifact-only successor that is intended to observe real Evaluation feedback, the freeze command may also
declare `--provider-token-limit <N> --maximum-turns <M>`. They are one operation: both are required, the limit becomes
the sole terminal checkpoint, and other Claim Scopes reject the override. Choose `N` from retained prior provider
usage and keep `M` as the independent hard call bound; neither value creates a scientific budget claim.

Reasoning effort has no implicit operator default. Qualification and freezing both name the exact value, and the
qualification digest must match it. Use `xhigh` for a future paper-aligned scientific treatment; `max` remains a
distinct engineering choice for the existing task-informed artifact lane. Changing the value requires a new
qualification and successor Study, not a field edit.

## 4. Execute matched search, system qualification or artifact optimization

Copy `examples/runtime/matched-live.example.json`, replacing every absolute path. Run the CLI with effective
`gpuq-users` access; the broker configuration must declare `service_user=gpuq` and
`service_group=gpuq-users`. Its command must be `gpu-run ... tools/evaluate_flash_candidate.py` so GPUQ owns
`CUDA_VISIBLE_DEVICES`; the setgid handoff directory allows the service user to read sealed candidates and return
raw results without weakening ownership checks. Until the broker exports a real worker job ID, include
`--env GPUQ_JOB_ID=gpuq-000000000000`; the parent normalizes the real ID from `gpu-run` stderr.

```bash
sg gpuq-users -c 'open-cake-ir lab execute --lock /new/path/campaign.lock.json \
  --runtime-config /new/path/runtime.json --evidence-root /new/path/evidence'
```

Lab—not the provider or worker—owns Turns, cumulative tokens, bounded admission resubmission, checkpoints,
search/confirmatory promotion and terminal sealing.

For `system_qualification_only`, exactly one Run per Authoring Environment executes. Its report can set only
`system_qualification_passed`; `estimand`, `estimate`, and `uncertainty` remain null and comparative statistics are
forbidden.

The accepted candidate-set qualifications establish one host precondition: every path needed by the
broker service user must be traversable, including all checkout parents. The immutable v1 attempt failed before a
GPU launch because a temporary checkout had private parent permissions; v2 succeeded from a durable accessible
worktree. Keep the runtime config, Campaign Lock and Evidence root outside the checkout, and test service-user path
admission before execution. The retained v2 Evidence root is
`/home/qinhaiyan/open-cake-ir-evidence/campaigns/candidate-set-campaign-live-v2`. The current Executor-v18 successor
at `/home/qinhaiyan/open-cake-ir-evidence/campaigns/candidate-set-campaign-live-v3` additionally retains one profile
for every correct searched survivor. Both are system qualification only.

For `artifact_optimization_only`, one multi-Turn Run per Authoring Environment uses the same command. Auxiliary
Apps/MCP/shell/browser/plugin/subagent activity is retained in raw Evidence, so the operator must approve its source
data for archival before launch. In candidate-set successors, auxiliary agents are read-only and the primary thread
is the sole writer; only `candidate-set.json` may remain at the Turn boundary. Audit promotes the
lowest-latency confirmatory-qualified Candidate per Run, with earliest Turn as
tie-break; it never reports qualification rates, arm medians, ratios, uncertainty or scientific inclusion. External
mutation and direct GPU measurement remain unauthorized even when those tools are visible.

The current retained Executor-v16 feedback Campaign is
`/home/qinhaiyan/open-cake-ir-evidence/campaigns/artifact-optimization-live-v4`. It declares one 8M terminal
provider-token checkpoint and a hard two-Turn bound. Both Runs adhere, resume the same thread once, and fresh audit
reports archive integrity, semantic replay and `artifact_optimization_complete=true`. Open Cake uses 2,921,505 total
provider tokens and promotes its Turn-2 artifact at 1.417239 ms after a 1.851297 ms Turn-1 confirmation. Direct CUDA
uses 16,934,227 total provider tokens and retains its 4.156109 ms Turn-1 artifact after the Turn-2 selection confirms
at 8.814379 ms. The Open Cake checkpoint is `unreached`; Direct CUDA's crossing Turn cannot backfill the checkpoint,
so its Turn-1 artifact remains the endpoint at 8M. Artifact promotion remains a separate per-Run view over every
confirmatory receipt.

All four raw provider Turns contain command/file activity and no auxiliary-agent lifecycle. Feature exposure is not
evidence that an agent was used. v4 validates a KDA-style authority boundary—one resumed writer and an external
judge—not a multi-agent result. The retained per-Run latencies and provider usage are artifact-ranking observations
only and must not be compared as an arm effect or with the paper's 80M-token Study. v3 remains the immutable earlier
Campaign that exposed the ineffective 150k feedback horizon.

## 5. Execute the exact-shape Portfolio

Run the entire Portfolio CLI as one exclusive B200 job, using a config derived from
`examples/runtime/portfolio-live.example.json`. It compiles three seed-bound lowerings, persistently loads three
CUBINs, runs direct/dispatcher/postflight correctness, 15 CUPTI and 15 host cohorts, rejects one unsupported key,
unloads all modules and stores the raw receipt.

## 6. Audit offline

```bash
open-cake-ir lab audit --lock /new/path/campaign.lock.json --evidence-root /new/path/evidence
```

Audit opens Evidence read-only. Portfolio audit rebuilds CV, medians, route counts and Claim View from raw retained
cohorts; matched audit rebuilds turn-discrete checkpoints and endpoints from event/receipt objects. A structurally
intact archive can still be semantically unavailable. `archive_integrity_passed=true` with
`filesystem_custody_verified=false` means the bytes replayed at inspection time but cannot support promotion,
system qualification or a scientific estimate without a separate custody/anchor fact.

## 7. Sole-owner cutover and rollback

Cutover is one explicitly approved transaction, never an automatic consequence of G8. Immediately before it,
recheck the legacy HEAD/tree and clean status, the complete-history bundle, free-space threshold, full contract
suite, Compiler release, Executor host admission, G8 offline audit and the proposed Git index. The frozen migration
Evidence currently beside the source is reviewed acceptance input; every future `--evidence-root`, qualification
workspace, Campaign Lock and report path must be outside the checkout.

The transaction must create one new Git revision and remote anchor, declare `open-cake-ir` the sole runnable writer,
and leave `cake-repro@2fa79092c143fd8c2d9caa93fd84ad79a7504836` read-only. Never run both writers. Do not start a
scientific Study as part of cutover.

If the new sole-owner path fails after cutover, first stop its writers and record any terminal state. With explicit
rollback authorization, create a new empty checkout from the verified bundle rather than modifying either existing
tree:

```bash
git bundle verify migration/bundles/cake-repro-final-2fa79092.bundle
git clone migration/bundles/cake-repro-final-2fa79092.bundle /new/empty/path/cake-repro-rollback
git -C /new/empty/path/cake-repro-rollback switch --detach \
  2fa79092c143fd8c2d9caa93fd84ad79a7504836
git -C /new/empty/path/cake-repro-rollback rev-parse HEAD
git -C /new/empty/path/cake-repro-rollback rev-parse HEAD^{tree}
python tools/build_legacy_manifest.py --source-set migration/source_set.json \
  --checkout /new/empty/path/cake-repro-rollback --output migration/legacy_manifest.jsonl --verify
```

The two expected identities are commit `2fa79092c143fd8c2d9caa93fd84ad79a7504836` and tree
`b02d730bb892629f250a20b8c5bd5869262e5c03`. Only after they and the manifest verify may the detached legacy
checkout become the sole writer. Keep new-repository Evidence read-only, never copy it into legacy state, and use a
new explicitly approved transaction to return ownership to `open-cake-ir`.
