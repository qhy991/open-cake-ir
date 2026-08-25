# ADR 0028: Study templates defer revision binding to CampaignLock

Status: accepted, 2026-08-25.

## Outcome and non-goals

Stop minting five near-identical frozen Study Contracts after every Compiler or Executor
revision.  A `template` Study carries the stable experimental design and writes the one
canonical revision spelling `{"binding":"current_release"}` for Compiler and Executor.
`Lab.preflight` resolves both once and puts their exact id, path and digest into the
CampaignLock.  A `frozen` Study continues to carry exact references and never resolves a
moving binding.

This does not weaken completed Studies, edit historical contracts, make CampaignLocks
dynamic, add a registry, or let an unresolved Study become execution authority.  It also
does not solve Compiler profile/workload ownership; that remains a separate semantic
migration.

## Minimal primitives and authority

- `StudyContract.state` is the closed choice `template | frozen`.
- A template has exactly one revision form: `current_release` for both authorities.
- A frozen Study has exactly one execution form: exact content-bound references.
- `Lab.preflight` is the sole resolver and CampaignLock is the execution SSOT.
- `create_study_successor.py` and `freeze_live_matched_study.py` turn a template into a
  frozen Study by replacing both bindings atomically before validation.

The narrower template type is not a compatibility alias.  It makes two illegal states
unrepresentable: a template cannot pretend to pin one revision, and a frozen Study cannot
silently follow current HEAD.  Existing frozen reference shapes remain bounded read
compatibility and are never rewritten.

## Dataflow and failure semantics

```text
template Study --preflight--> exact CampaignLock --execute--> Evidence
       |                 exact Compiler + Executor refs
       `--freeze tool--> frozen Study --preflight--> exact CampaignLock
```

Resolution fails when the current Compiler is not released/corpus-gated, the current
Executor inventory reference does not verify, either binding uses another spelling, or
a frozen Study contains a moving binding.  The template's canonical digest and the two
resolved revision digests all survive in the CampaignLock, so later audit can distinguish
design bytes from execution authority.

## Smallest complete slice and evidence

1. The five current zero-GPU fixtures become stable templates with no exact Compiler or
   Executor identity.
2. Preflight of each template resolves current v23/v27 while preserving the non-revision
   design inputs of its v37 frozen predecessor.
3. A template with one inline authority and a frozen Study with one current binding both
   fail.
4. Both freeze tools emit a frozen Study with exact references that preflights.
5. Existing v37 frozen Studies remain byte-identical and keep their exact historical
   references. Like other unexecuted superseded Studies, they require their original
   source checkout to preflight; only terminal evidence earns a copied Executor archive.
