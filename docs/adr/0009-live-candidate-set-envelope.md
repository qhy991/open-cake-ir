# ADR 0009: live candidate sets use one sealed envelope

Status: proposed, 2026-08-24.

## Outcome and non-goals

A successor live Study will let one provider Turn author several Candidates through one
canonical `candidate-set.json` file. The Lab already consumes an ordered candidate tuple;
this closes the missing live transport without adding a second Lab path or making
`searches_per_turn` mean two things.

This proposal does not change frozen Studies, promote the dormant cost model, increase GPU
work implicitly, admit runtime-shaped Schedules, or require every Turn to fill its allowed
set. It is not implemented or live-qualified yet.

## Owners and canonical form

- The successor Study budget owns `maximum_candidates_per_turn`, an integer greater than
  zero. It is the authoring bound and is granted identically to both arms in a scientific
  Study.
- `evaluation_protocol.searches_per_turn` remains the independent GPU-measurement bound.
  It may not authorize more candidates than the Turn wrote, and it does not change the
  authoring bound.
- One canonical envelope owns candidate order. It contains `schema_version`, `arm` and a
  non-empty `candidates` array. Open Cake entries are Schedule JSON objects; direct CUDA
  entries are UTF-8 source strings. The envelope itself must be canonical JSON, so its
  exact bytes are reconstructible rather than becoming a second handwritten truth.
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

## Smallest complete vertical slice

Acceptance requires all three pieces, in order:

1. a zero-GPU fixture Turn with three entries proves ordering, the authoring bound,
   per-candidate rejection, semantic deduplication, selection and fresh-process replay;
2. a two-Turn live provider qualification observes the same envelope add/update lifecycle
   for both arms;
3. one non-scientific live Campaign writes at least two structurally distinct launchable
   Candidates in a Turn and evaluates two under an explicitly frozen
   `searches_per_turn=2` protocol.

Until the third item passes, candidate-set composition is an internally tested capability,
not an exercised live paper stage.
