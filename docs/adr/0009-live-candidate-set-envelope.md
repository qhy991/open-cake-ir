# ADR 0009: live candidate sets use one sealed envelope

Status: accepted, 2026-08-24.

## Outcome and non-goals

A successor live Study lets one provider Turn author several Candidates through one
canonical `candidate-set.json` file. The Lab already consumes an ordered candidate tuple;
this closes the missing live transport without adding a second Lab path or making
`searches_per_turn` mean two things.

This decision does not change frozen Studies, promote the dormant cost model, increase GPU
work implicitly, admit runtime-shaped Schedules, or require every Turn to fill its allowed
set. The transport, zero-GPU fixture slice and bounded live system-qualification slice are
implemented and exercised.

## Owners and canonical form

- The successor Study budget owns `maximum_candidates_per_turn`, an integer greater than
  zero. Its presence selects the envelope transport; absence retains the frozen singleton
  transport. It is the authoring bound and is granted identically to both arms in a
  scientific Study. There is deliberately no second Study flag saying the same thing.
- `evaluation_protocol.searches_per_turn` remains the independent GPU-measurement bound.
  It may not authorize more candidates than the Turn wrote, and it does not change the
  authoring bound.
- One canonical envelope owns candidate order. It contains `schema_version`, `arm` and a
  non-empty `candidates` array. Open Cake entries are Schedule JSON objects; direct CUDA
  entries are UTF-8 source strings. The envelope itself must be canonical JSON, so its
  exact bytes are reconstructible rather than becoming a second handwritten truth. The
  file spelling is sorted-key compact UTF-8 JSON followed by exactly one LF.
- Provider qualification records the derived `candidate_set_envelope_v1` submission
  capability in its configuration digest. That is a qualification projection of the
  Study budget, not another writable choice.
- `ProviderTurn.candidates` is the sole semantic projection consumed by the Lab. Schedule
  objects are projected to canonical Schedule bytes and CUDA strings to their exact UTF-8
  bytes. Existing candidate hashes, Evidence roles and replay keys remain unchanged.

One envelope deliberately reuses the qualified one-file add/update lifecycle. A directory
of numbered files would add workspace enumeration and multi-file event semantics without
adding candidate capability; an envelope of escaped Schedule strings would discard the
Schedule's existing canonical JSON form.

## Dataflow, failure and compatibility

The provider adds or updates the fixed envelope, the adapter seals it without following
links, validates the arm and bound, and projects its ordered entries. The existing path
then builds and gates every entry, deduplicates equivalent programs, applies only released
ranking coverage, and measures at most `searches_per_turn` survivors.

An absent, non-canonical, empty, over-bound or wrong-arm envelope is a provider protocol
fault before compilation or GPU work. A malformed member remains attributable to that
member through the existing Environment rejection path. Missing ranking coverage retains
provider order. No automatic retry changes either bound.

Frozen single-candidate Studies keep their exact prompt, filename and adapter semantics as
a bounded compatibility edge. Only a successor Authoring Environment may select the
envelope contract; old and new forms are never writable in the same Run.

## Agent ownership

The tool-rich optimization scope follows KDA-internal's useful ownership boundary without
copying its orchestration framework: auxiliary agents may perform distinct read-only
investigations, while the primary provider thread is the only submission writer. At the
Turn boundary the workspace contains only `candidate-set.json`; raw auxiliary lifecycles
remain in provider Evidence, and the external Lab remains the sole evaluator. Agent
personas or managers may influence exploration style but cannot change the Study,
candidate budget, Evaluation protocol or acceptance decision.

## Smallest complete vertical slice

Acceptance required all three pieces, in order:

1. a zero-GPU fixture Turn with three entries proves ordering, the authoring bound,
   per-candidate rejection, semantic deduplication, selection and fresh-process replay;
2. a two-Turn live provider qualification observes the same envelope add/update lifecycle
   for both arms;
3. one non-scientific live Campaign writes at least two structurally distinct launchable
   Candidates in a Turn and evaluates two under an explicitly frozen
   `searches_per_turn=2` protocol.

As of 2026-08-24, item 1 passes in the 319-test/223-subtest local suite and item 2 passes
for both the closed and tool-rich Codex 0.144.4 policies. The first item-3 attempt retained
three launchable Open Cake members but reached no Evaluation because the broker worker
could not traverse the temporary checkout; direct CUDA then hit a JSON-whitespace
normalization fault. That immutable v1 Campaign remains bounded failure evidence.

The successor Study `open-cake-ir-candidate-set-system-v2` (canonical SHA
`6b7c916c61bf01f393383c5c9f7f0929ad834511cbd30237575c8cf7e0302538`) passed item 3 under
Executor v14. Its external Campaign authority SHA is
`88605aa1b1557b58aec9dacb4e043290d9a03205b0847d507fa72fc68fad633b`, and its Evidence root
is `/home/qinhaiyan/open-cake-ir-evidence/campaigns/candidate-set-campaign-live-v2`.
Each arm submitted three non-deduplicated launchable Candidates and searched two. The Open
Cake set is structurally distinct at the Schedule level (`m256-c32-w8-s3`,
`m128-c64-w8-s3`, and `m128-c32-w4-s2`). Each selected Candidate then received fresh
confirmatory and attribution Evaluation, yielding four receipts per arm. All eight receipts
pass the frozen tie-aware correctness oracle, observe one target-kernel call and zero
fallback calls. Fresh-process audit reconstructs archive integrity, semantic replay,
protocol adherence and `system_qualification_passed=true`; the two terminal seals are
`9d3fb8b890f38c10655e4327f07c320de30333d16c2d1cee9ebd0c3cd692cb33` and
`e4ac6d746eb5b1bd01df786aeae4642a2d09cae82efc4fd79d8c57875e92157c`.

This accepts the candidate-set composition boundary only. The Study forbids comparative
statistics, and its estimand, estimate and uncertainty are all null. The observed arm
latencies are operational evidence for the path, not a treatment comparison or a paper
result.
