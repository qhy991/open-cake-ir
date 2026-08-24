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

`tools/release_compiler_cycle.sh "<basis>"` drives that pipeline end to end and derives the id, but it does not
re-pin a **Target definition**: `compiler/revision.json` holds each Target's `canonical_sha256`, and editing
`compiler/targets/*.json` means updating that pin by hand first, or the cycle stops at `target definition ... bytes
differ`. That is deliberate — a source edit is routine and a hardware description changing is not — but it is a step
the script will not take for you.

Expectations are never regenerated inside a release. `tools/refresh_corpus_expectations.py` prints the diff and exits
non-zero; `--write` adopts it. Read the diff first: adopting before reading is how a gate becomes a tautology.

`attribution_evaluation=correctness_then_profile` opts a matched Study into a separate
post-confirmation assay. `tools/evaluate_flash_candidate.py` admits the exact NCU binary
pinned by the Executor, launches the same sealed CUBIN once under NCU, checks that launch
against the external oracle, and retains the raw CSV plus its recomputed summary. The
receipt structurally has `timing=null` and zero timing samples: profiler duration is never
candidate latency. A missing target-kernel row, missing metric, incorrect output, changed
tool byte or projection/raw mismatch fails the attempt.

## Instruments

These produce evidence rather than artifacts. Each one is checked in so a claim it supports can be repeated instead
of trusted; `docs/ANALYSIS_CALIBRATION.md` reads their output and `evidence/calibration/` holds it.

```bash
python tools/calibrate_wave_term.py --first 60 --last 400 --step 2      # exclusive: benchmark
python tools/calibrate_ranking_at_scale.py --size 512 --observed-at <iso8601> \
  --out /new/path/ranking.json                                          # exclusive: benchmark
python tools/observe_lowered_kernel.py --observed-at <iso8601> --out inventory/<NEW>.json
python tools/profile_lowered_kernel.py --schedule <path> --observed-at <iso8601> --out <NEW>.json
python tools/ir_vocabulary.py                                           # no GPU
```

The ranking calibration measures the dormant structural hypothesis even when the released
Compiler has no profile coverage. Its output informs a later reviewed Revision; running
the instrument never changes `calibration_coverage`.

Both `observe_lowered_kernel.py` and `profile_lowered_kernel.py` build their inputs from
`tools/kernel_cases.py`, so a kernel one of them profiles is a kernel the other checked.
Adding an operator adds an oracle there and nothing else: shapes, dtypes and argument
order are derived from the Schedule's global buffers, because that is where they are
already declared and where the emitted host function already validates them.

`observe_lowered_kernel.py` refuses to overwrite. A record is what happened once, so renewing one means a new file
and a new date; the earlier record stays as history for the Revision it was taken under.

## 2. Resolve a Study

```bash
open-cake-ir lab preflight contracts/studies/matched-search-infrastructure-v14.json \
  --output /new/path/campaign.lock.json
```

Which Study Contract is current is not a fact this document owns. Release scripts never rewrite a frozen contract;
`tools/create_study_successor.py` binds a new explicit fixture successor to the current Compiler and Executor and
validates it through Lab preflight. It deliberately refuses a live Study because changing a worker also changes the
broker command digest; `freeze_live_matched_study.py` is the sole path that refreshes that complete authority. The
names below were current when written.

The current checked-in scientific matched contract, `matched-search-infrastructure-v14.json`, uses a zero-GPU fixture
provider and intentionally cannot start a live provider. The current non-scientific G8 template is
`matched-search-system-qualification-v14.json`; earlier versions remain frozen historical records. Freeze
a live successor only after a real two-Turn qualification emits a receipt with
the matching closed or tool-rich scope and binds the exact executable, model, reasoning effort, service tier, output
schema, prompt/scaffold bytes, removed environment, reference visibility, feature overrides and event contract.
`artifact-optimization-v14.json` is the current zero-GPU contract fixture. The frozen
`artifact-optimization-verda-v7.json` remains a historical live authority for Executor v8; it is not executable
from the current source closure. Re-freeze a successor with the exact accessible checkout and broker command before
launching a live Campaign. Earlier revisions remain historical.

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
  --maximum-candidates-per-turn 3 \
  --feature-policy provider_defaults_optimization
```

Then freeze either non-scientific Study from its matching receipt, seal anchor and exact runtime configuration:

```bash
python tools/freeze_live_matched_study.py \
  --project-root . \
  --template contracts/studies/matched-search-system-qualification-v14.json \
  --qualification contracts/providers/<new-live-receipt>.json \
  --qualification-anchor evidence/qualifications/<new-live-anchor>.json \
  --executor runtime/executors/open-cake-ir-b200-v16.json \
  --runtime-config /new/path/runtime.json \
  --study-id <new-g8-study-id> \
  --output contracts/studies/<new-g8-study>.json \
  --enable-attribution
open-cake-ir lab preflight contracts/studies/<new-g8-study>.json \
  --output /new/path/g8-campaign.lock.json
```

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

The accepted candidate-set v2 qualification also establishes one host precondition: every path needed by the
broker service user must be traversable, including all checkout parents. The immutable v1 attempt failed before a
GPU launch because a temporary checkout had private parent permissions; v2 succeeded from a durable accessible
worktree. Keep the runtime config, Campaign Lock and Evidence root outside the checkout, and test service-user path
admission before execution. The retained v2 Evidence root is
`/home/qinhaiyan/open-cake-ir-evidence/campaigns/candidate-set-campaign-live-v2`; it is system qualification only.

For `artifact_optimization_only`, one multi-Turn Run per Authoring Environment uses the same command. Auxiliary
Apps/MCP/shell/browser/plugin/subagent activity is retained in raw Evidence, so the operator must approve its source
data for archival before launch. In candidate-set successors, auxiliary agents are read-only and the primary thread
is the sole writer; only `candidate-set.json` may remain at the Turn boundary. Audit promotes the
lowest-latency confirmatory-qualified Candidate per Run, with earliest Turn as
tie-break; it never reports qualification rates, arm medians, ratios, uncertainty or scientific inclusion. External
mutation and direct GPU measurement remain unauthorized even when those tools are visible.

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
intact archive can still be semantically unavailable.

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
