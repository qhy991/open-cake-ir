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

The admitted outputs are therefore provisional `schedule_candidate_lowerable`,
`schedule_candidate_backend_blocked`, `schedule_gap_candidate`, or program/portfolio
redirect candidates. They must not be aggregated as released IR coverage. A future
independent semantic adjudicator and workload-owned external evaluation are required before
any row may become described/tested evidence or motivate a Compiler Corpus change.

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

No new primitive is justified by row frequency alone. The first vocabulary candidate must
have at least two independent, source-complete cases that require the same irreducible
commitment. Current candidates for that review are concrete vector width/alignment/lane
mapping and a runtime scalar value; neither is accepted by this ADR.

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
