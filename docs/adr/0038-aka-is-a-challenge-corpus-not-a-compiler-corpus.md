# ADR 0038: AKA is a challenge corpus, not a Compiler Corpus

Status: accepted for external-corpus assessment, 2026-08-28.

## Outcome

AKA is an external, out-of-sample **challenge corpus** for Cake expressibility. It may
falsify a vocabulary or ownership boundary by exposing a real CUDA mechanism that Cake
cannot place. Its four-field rows are not Compiler Corpus cases, correctness oracles,
performance verdicts, or independent kernels.

Two questions remain separate for every reviewed row:

1. can the current Cake owners name the baseline-to-candidate mechanism; and
2. can Cake describe the complete parent kernel or program endpoint?

The first may pass while the second is unknown or fails. A mechanism match must never be
reported as whole-artifact coverage. This is the same distinction ADR 0020 applies to KDA,
now required for a task-oriented rather than chronological source collection.

This decision adds no Schedule metadata, operator mode, mechanism enum, performance label,
or automatic CUDA-to-Schedule translator. AKA remains the source-record authority. Cake
retains only reference-based assessments, distilled design decisions, and its own minimal
Schedule positives and falsifiers.

## Reviewed source snapshots

The admitted discovery snapshot is AKA Git revision
`becde652253d0ce287555097b632149f621cad4e`, directory
`datasets/curated/cuda_kernel_dataset_v1/categories/`:

- 1,362 English records in 142 operator shards;
- nine coarse categories and 54 reported operator folders;
- 902 analysis, 120 generation, 36 debug, 150 historical optimization-positive and
  154 historical optimization-neutral rows;
- exactly four strings per row: `instruction`, `input`, `reasoning`, and `output`.

The 51 records under `excluded/` are outside this snapshot. Historical positive/neutral
filenames are source partitions, not accepted GPU outcomes.

The fetched AKA `origin/main` revision `8dbbf7a` also contains
`cuda_kernel_dataset_v2`, but it is not admitted here. Its README calls v2 a preservation
of every v1 row plus 48 additions, while exact row comparison against v1 finds only 230
common rows: 1,132 v1 rows are absent byte-for-byte and 748 v2 rows are new byte-for-byte.
In particular, no analysis row is identical across the versions. Some source inputs have
lineage-like matches, but the model-visible reasoning/output was rewritten and no stable
sample/parent identifiers prove the claimed preservation. A successor review may admit v2
after it owns that transformation and lineage explicitly; silently widening this snapshot
would make the denominator unknowable.

The adapter may inspect that tree under the required `aka_v2_review_projection` record
format without admitting it here. That format treats v2 `optimization_neutral` and
`optimization_negative` rows as `review_context=input` and `review=output`, and deliberately
derives no CUDA scope or lexical signal from either field. The admitted v1 snapshot instead
uses `aka_v1_operator_sft`; a format is an explicit review contract, not an inference from a
directory name.

## What the v1 rows can and cannot test

The source breadth is useful. A bounded lexical audit reports these primary-artifact scope
signals; these are the analysis source, generated implementation, broken implementation and
optimization baseline respectively:

| Syntactic scope signal | Rows | Consequence |
| --- | ---: | --- |
| one `__global__` definition and at most one launch marker | 951 | primary artifact may enter single-Schedule review; inspect its paired artifact independently |
| multiple kernel definitions or launch markers | 398 | route the primary artifact first to program/composition review |
| no visible `__global__` definition | 13 | primary fragment, wrapper or opaque-library review |

Signals are retained separately for every code-artifact role. In particular, the 304
optimization candidates split into 223 single-kernel, 80 multi-kernel-or-launch and one
fragment-or-library artifact, rather than inheriting the 238/65/1 baseline split. The 36
repaired artifacts retain their own 25/11/0 split. The rows stress thread mapping, shared
memory, barriers, warp collectives, atomics, vectorized access, inline PTX, library wrappers
and collectives. That is enough to discover missing concepts. It is not enough to calculate
an IR success rate because the records do not structurally own:

- source repository, revision, path, licence, stable sample id or split group;
- concrete API, dtype/index specializations, launch geometry, stream and valid domain;
- frozen shapes/layouts, reference, tolerance or workload;
- compilation, full-output correctness, sanitizer or profiler receipts;
- paired absolute timing, noise state, node/run locator or accepted verdict.

Exact records are not duplicated, but two primary-source strings are repeated across four
rows, including one cross-category pair. Folder names are therefore reported sampling axes,
not semantic operation truth or split ownership.

The active mechanism-augmentation workspace reinforces the boundary rather than repairing
it automatically. Its verifier-owned terminal records overwhelmingly route rows to
`parent_invalid` because launch, specialization, layout/ABI, stream, oracle or workload
facts are missing. Those are contract failures before they are IR vocabulary failures.
The workspace is active, unversioned external state; its counts and self-reported result
views are not copied into this decision.

## Ownership routing

| Observed fact | Cake owner |
| --- | --- |
| one kernel's arithmetic, buffers, dependencies, access and hardware commitments | `Schedule` under one Compiler Revision |
| operator semantics, valid inputs, reference and tolerance | Workload Contract |
| exact-shape specialist choice, dtype/shape dispatch and fallback | portfolio above `Schedule` |
| multiple launches, fusion delta, streams, graph or collective edges | program composition above `Schedule` |
| parent/candidate relation, experiment treatment and estimand | Study/Lab contract |
| compile, correctness, sanitizer, profiler, timing and verdict | external evaluation evidence |
| source lineage, task label, dataset version and split | AKA/corpus authority |

One fused endpoint may be a Schedule. The statement that several baseline kernels became
one endpoint is a relation between program artifacts, not a `fused: true` Schedule field.
Likewise, a framework wrapper or CUB call is not a missing Cake primitive until its concrete
kernel semantics and supported boundary are independently materialized.

## Read-only projection

`tools/audit_aka_corpus.py` is the only adapter introduced by this decision. The
commit-addressed, role-scoped summary and case projections are schema v2; the singular
primary-field signals and caller-asserted source identity from v1 are not retained. It:

1. requires one declared record format, `aka_v1_operator_sft` or
   `aka_v2_review_projection`, and emits it in every summary and case;
2. enumerates only regular Git blobs at
   `<dataset>/categories/<category>/<operator>/<task>.jsonl` in the exact selected commit,
   rejecting symlinks, non-blobs and malformed or nested shard paths;
3. reads those commit blobs directly with Git replacement objects disabled, never mutable
   worktree shard bytes, then fails closed
   unless every row has the exact four-string schema and the task's parent field is non-empty;
4. preserves `(credential-stripped observed origin hint, Git revision, full
   repository-relative dataset/path, line, primary field)` as a locator and references the task's source/generated,
   broken/repaired, baseline/candidate or review-context/review fields according to the
   declared format; the user-supplied dataset label is reported metadata, not source authority;
5. refuses a non-commit revision and makes replacement refs plus ignored, untracked,
   filtered or modified worktree
   shard bytes irrelevant to the projection;
6. reports syntactic scope and overlapping lexical incidence separately for every declared
   code-artifact role, never as gold labels or for v2 neutral/negative review fields;
7. leaves owner scope, whole-parent expressibility and delta expressibility `unknown`;
8. emits no model-visible content, timing, verdict or copied evidence;
9. never writes a Compiler manifest or mutates AKA.

The projection is a review queue. A reviewer must resolve at least:

```text
source_ref: sanitized observed remote hint, revision, repository-relative dataset/path, line and field
reported dataset label
record format
reported category / operator / task
relation: analyze | generate | repair | optimize
artifact field roles
scope and lexical signals per code-artifact role
owner_scope
assessment_scope: contract | fixed_instance
complete_parent_expressibility
delta_expressibility
provenance/evidence/split status
review_state
```

An actual Cake bridge record may additionally reference a source-complete augmentation
contract and verifier-owned terminal record, then retain Cake's Assessment, Schedule case
ids and missing primitive paths. It must reference external evidence rather than copying
its facts.

## Iterative review layer

`tools/review_aka_expressibility.py` consumes the immutable projection without changing its
authority. It creates one external case at a time, keeps parent readiness independent from
whole-parent and delta review, and derives classifications instead of accepting a reviewer
verdict. Any non-unknown proposal requires a canonical complete-kernel-parent completion;
case-local Schedule candidates are checked with the exact frozen Compiler `assess/lower`
path. Compiler acceptance and lowering are not semantic equivalence: the result remains
`semantic_binding=reviewer_claimed`, `review_state=checked`, and `gpu_test=not_run`.

The Schedule check has one explicit scope owner. `schedule_candidate_lowerable` and
`schedule_candidate_backend_blocked` are reserved for a reviewer claim about the whole
narrowed parent contract. A static case-local Schedule from a wider runtime parent is
instead `fixed_instance_candidate_lowerable` or
`fixed_instance_candidate_backend_blocked`. A fixed binding may also be
`fixed_instance_gap_candidate`, `fixed_instance_unknown`, or a fixed-instance
Program/Portfolio redirect; none of those states classifies the rest of the runtime domain.
Contract-scoped outputs remain provisional `schedule_gap_candidate` or program/portfolio
redirect candidates. They must not be aggregated as released IR coverage. Lexical signal
counts are retrieval aids only, never gap counts or priority. A future independent semantic
adjudicator and workload-owned external evaluation are required before any row may become
described/tested evidence or motivate a Compiler Corpus change.

`tools/run_aka_expressibility_codex.py` is a transport-only sequential consumer of that
review layer. It fixes GPT-5.6 Sol at max reasoning, disables unrelated tools and agent
orchestration, retains raw Codex receipts, and stops at the first invalid or unchecked case.
It cannot write a derived class directly: the deterministic reviewer still recomputes every
parent, owner, Schedule, Finding and lowering result. A Codex completion therefore changes no
claim boundary and never authorizes GPU work.

`tools/plan_aka_expressibility_queue.py` reconciles the admitted 1,362-row source snapshot
with one explicit mechanism-augmentation ledger without importing that ledger as Cake
authority. In the observed B200 campaign, 1,182 rows have invalid parents and 180 have
historically qualified parents. Of those 180, only 15 also have terminal valid augmentation
results; 14 are syntactic single-kernel cases and one is multi-kernel. These are priority
facts, not IR acceptance. The strongest direct-Schedule queue is therefore the 14 terminal
valid single-kernel rows; the multi-kernel row enters program ownership review instead.

`tools/run_aka_qualified_ir_codex.py` is the create-only executable front door for that
strongest queue. For each row, in order, it copies the preserved parent/evaluator artifacts
into an external case, asks the same fixed Sol/max treatment only to normalize them into the
canonical complete-kernel-parent record, runs the independent parent validator, and invokes
the ordinary expressibility runner only if that gate passes. It is CPU-only, performs no GPU
rerun, never retries or skips a failure, and leaves the remaining 1,348 rows untouched. A
historical `parent_status=qualified` is thus not silently upgraded into a current IR claim.

The intended invocation is:

```bash
python3 tools/run_aka_qualified_ir_codex.py \
  --dataset-root /absolute/AKA/datasets/curated/cuda_kernel_dataset_v1 \
  --source-revision becde652253d0ce287555097b632149f621cad4e \
  --campaign-root /absolute/terra-high-b200-10-20260827 \
  --completion-root /new/external/aka-parent-bridges \
  --review-root /new/external/aka-ir-reviews \
  --limit 1
```

Both output roots are create-only. Continue with a larger limit only in a fresh pair of roots
after the one-case bridge and review pass deterministically; an existing failed root is
evidence to inspect, not a queue to overwrite.

## Portable qualified-parent review

AKA commit `4d041e5e88c3da157ce19fd016eed1ce44c4816b` publishes 50 portable
`aka.portable-kernel-parent.v1` projections under
`datasets/curated/cuda_kernel_parent_completions_v1`. Each projection has an explicit
narrowed contract, baseline/reference/harness source bundle, unique fixed B200 locator and
passed/valid compile, complete-output correctness and sanitizer projection. The projection
is not the node-owned route/result closure and cannot be replayed through the qualified
parent validator as GPU authority.

`tools/run_aka_portable_parents_codex.py` therefore reads every source and artifact directly
from that exact Git commit, checks the portable row against its v1 source field, and creates
an external canonical `runnable_unqualified` completion. The fixed complete-kernel-parent
validator must accept that runnable bundle before the ordinary Sol/max reviewer sees it.
The resulting parent status is `runnable_by_parent_validator`, while node custody stays
`not_assessed`, semantic binding stays `reviewer_claimed`, and `gpu_test` stays `not_run`.
Portable qualification is retained as evidence but never promoted into a current GPU claim.

Forty-nine of the 50 records use the v1 task's canonical primary field. One debug record
qualifies its repaired `output` while the v1 review queue owns the broken `input` as primary;
the adapter records `source_field_override_required` for that row rather than relabeling the
completion. This is a deterministic admission outcome, not a model failure or a skipped
review.

The create-only invocation is:

```bash
python3 tools/run_aka_portable_parents_codex.py \
  --source-dataset-root /absolute/AKA/datasets/curated/cuda_kernel_dataset_v1 \
  --portable-dataset-root /absolute/AKA/datasets/curated/cuda_kernel_parent_completions_v1 \
  --source-revision 4d041e5e88c3da157ce19fd016eed1ce44c4816b \
  --completion-root /new/external/aka-portable-runnable-parents \
  --review-root /new/external/aka-portable-ir-reviews \
  --limit 49
```

These reviews assess whether the frozen Compiler can describe a source-complete derived
operator. They do not import the portable set into the Compiler Corpus, establish the
portable stage projection as node authority, test generated Cake code on GPU, or evaluate
the unimplemented optimization handoff.

After a successful one-case canary, a fresh create-only batch may continue without
repeating it by passing that canary's exact portable case id to `--start-after`. The value
selects only within the commit-bound review-ready order; an absent, blocked or repeated id
is refused before creating either output root.

AKA commit `b6607f78148b152bfa48133af98a8f58148c3684` adds the immutable
`cuda_kernel_parent_completions_v2` snapshot: the 50-record v1 set plus 100 new source
identities. `tools/run_aka_portable_parent_pool.py` computes that difference from the two
commit-bound source identities rather than trusting a filename, count label or row position.
The pool requires exactly 150 current, 50 prior and 100 new records before launch.

Each selected parent receives an independent completion root, review root, Codex receipt
and deterministic checked result. At most 30 workers may run concurrently. A failed case is
preserved without retry; it neither cancels nor rolls back already running siblings, and no
worker uses a GPU. Controller completion is successful only when all 100 workers finish with
ordinary `completed` status; partial success and controller failures remain explicit.

The create-only pool invocation is:

```bash
python3 tools/run_aka_portable_parent_pool.py \
  --source-dataset-root /absolute/AKA/datasets/curated/cuda_kernel_dataset_v1 \
  --portable-dataset-root /absolute/AKA/datasets/curated/cuda_kernel_parent_completions_v2 \
  --prior-portable-dataset-root /absolute/AKA/datasets/curated/cuda_kernel_parent_completions_v1 \
  --source-revision b6607f78148b152bfa48133af98a8f58148c3684 \
  --portable-record-count 150 \
  --prior-record-count 50 \
  --expected-new-count 100 \
  --batch-root /new/external/aka-portable-100-pool \
  --max-workers 30
```

## Initial IR assessment

The current vocabulary is plausible but not broad enough to call complete. The useful
initial mapping is:

| AKA pressure | Current Schedule | Decision |
| --- | --- | --- |
| reduction and scan | sum/max CTA reduction; sum prefix in two directions | partial; do not add operators/scopes without concrete cases |
| vectorized/global access | load movement and reuse intent; store coalescing claim | missing concrete width, alignment and lane/address commitment |
| shared-memory staging | allocation, buffer offsets/stages/swizzle, pipelines and barriers | partial; dynamic lifetime/aliasing requires a source-backed resource decision |
| general layout/index arithmetic | concrete AccessMaps and a narrow runtime-buffer index | partial; div/mod/stride/predicate forms are absent, but no layout algebra is admitted |
| atomics/fences | one INT32 relaxed device-scope add returning the old value | deliberately narrow; fence and other operations are absent |
| control flow | tiled-axis mask only | comparison/select/general predicate and branch are absent |
| fusion/multiple launches/collectives | outside one Schedule | program-composition evidence, not a kernel vocabulary defect |
| runtime/dtype/shape dispatch | outside one Schedule | portfolio decision, not an operation |

This review also exposed structural defects that precede vocabulary growth: the canonical
parser accepted declaration names that the authoring Schema rejected and that produced
invalid lowered source; `Compiler.assess` raised exceptions for some structural documents
instead of returning `SCHEDULE_STRUCTURE`; and the Schema admitted empty residency that the
parser refused. The accompanying successor proposal closes those three executable defects
and documents parser ownership of arbitrary-length warp adjacency. Because v29 already has
a retained observation, the fixes require a successor Compiler Revision and external release
approval before an AKA coverage run can treat them as the current Compiler boundary.

No new primitive is justified by row frequency alone. As a governance policy rather than a
finding of the AKA corpus, a vocabulary candidate must have at least two independent,
source-complete cases that require the same irreducible commitment. Current candidates for
that review are concrete vector width/alignment/lane mapping and a runtime scalar value;
neither is accepted by this ADR.

## Promotion into the Compiler Corpus

One reviewed AKA seed may produce Compiler cases only through this sequence:

1. freeze the exact source, API, dtype/index matrix, shape/layout, target and oracle;
2. classify the owner before attempting a Schedule;
3. author one complete Schedule, or record the exact missing primitive/effect;
4. keep whole-parent and delta expressibility as independent outcomes;
5. for a vocabulary change, update IR, Schema, verifier, analysis and backend together;
6. add one minimal positive and one near-miss falsifier with expected Findings;
7. run the full Corpus Gate and obtain external human release approval;
8. obtain independent target correctness before making a correctness claim;
9. keep AKA/augmentation speed evidence external and make no Cake performance claim from it.

Parent invalidity, missing evidence, infrastructure failure and measurement noise stay
`unknown` for IR expressibility unless an independently materialized Schedule demonstrates
otherwise.

## Acceptance evidence

Against the admitted v1 revision, the adapter reads 1,362/1,362 active records directly from
the commit and reports the 951/398/13 primary-artifact scope split, the separate 223/80/1
optimization-candidate split, zero duplicate full-record groups, two repeated primary-source
groups and one cross-category source group. Focused contract tests cover reference-only
projection, role-scoped source/candidate signals, duplicate reporting, exclusion of
`excluded/`, ignored and modified worktree isolation, replacement-ref isolation, symlink and
nested-path refusal, v2 review-role suppression, exact Git snapshot admission and fail-closed
malformed input. Separate focused Compiler tests cover
identifier rejection, root structural Assessments and residency Schema/parser agreement;
the draft Compiler replays all 39 retained Corpus cases without a disposition, Finding or
lowering mismatch.

No CUDA compilation, GPU run, profiler measurement, external campaign intervention or
Compiler release approval is part of this decision.
