# ADR 0052: Independent agent sessions may review Compiler releases

Status: accepted owner policy, 2026-09-06; implementation requires independent review.
Supersedes the human-only interpretation of ADR 0030. Its external-writer boundary stays.

## Contract

The owner permits a new reviewer session using Opus 5, Fable 5, Fable 5.1, or GPT-6 Astra
to approve a Compiler change. A human may still review it. The goal is to remove the
mandatory human wait while retaining an independent examination of the actual change and
full Corpus Gate. No IR primitive, analysis, kernel, GPU acceptance rule, or cost-model
calibration coverage changes in this decision; P1-P8 remain applicable.

`compiler/release-approval.json` remains the only approval authority. The release cycle
and implementation session cannot write it. The exact machine identifiers have one
canonical owner: `ALLOWED_REVIEW_MODELS` in `src/open_cake_ir/compiler/release.py`. A model
outside that set, an unavailable chosen model, or an unverified effective model does not
authorize a fallback. The owner's “5.1” is interpreted as Fable 5.1.

## Review procedure

1. The author prepares an isolated successor and the full Corpus Gate, preserving prior
   releases and existing Corpus expectations. Freeze the review commit and record the
   author session id in the review request.
2. Start a distinct reviewer session with an explicitly selected allowed model. Check the
   actual launch/session metadata; a requested model alone does not establish execution.
   The reviewer reads the change, acceptance contract, focused regression evidence and
   complete Gate, and independently exercises the relevant public boundary. Review
   reports and scratch outputs stay outside the checkout.
3. The reviewer returns `approved`, `request-changes`, or `rejected`. Only `approved`
   authorizes that reviewer to write the approval for the exact Gate. Its basis describes
   the evidence examined and any limitations. The reviewer does not modify implementation,
   expectations, or release locks. A source or Gate change requires review of the new Gate.
4. The author reruns the existing cycle. It validates the approval and creates the
   successor lock, or exits 3 while preserving the prior lock and approval bytes.

The caller can inspect model/session metadata without reading credentials. No new model
provider, scheduler, account system, cryptographic identity service, or second release
command is introduced. The Compiler does not import Lab or provider code.

## Approval format and assurance

New Gate reports use schema version 2 and `decision: awaiting_review`. New approvals use
schema version 2 and the same five top-level fields as before:

```json
{
  "schema_version": 2,
  "decision": "approved",
  "gate_report": {
    "path": "compiler/corpus-gate-report.json",
    "canonical_sha256": "<identity of the exact reviewed Gate>"
  },
  "reviewer": {
    "kind": "agent_session",
    "model": "gpt-6-astra",
    "session_id": "<actual reviewer session id>",
    "author_session_id": "<author session id from the review request>"
  },
  "approval_basis": "<reviewer-authored findings and validation basis>"
}
```

For a human, `reviewer` is `{"kind": "human", "name": "<reviewer identity>"}`.
These are format examples, not approvals. The validator requires an allowed exact model,
nonblank unpadded session identities, different author/reviewer ids (case insensitive),
a nonblank basis, an affirmative decision, and the existing exact Gate binding.

As in ADR 0030, identity assurance remains a workflow fact: JSON fields cannot prove who
ran a session or which model served it. The launcher/session record is the provenance for
that check; the validator enforces the declared model and separation and does not claim
to authenticate a self-declared identity. An author cannot relabel an agent as a human to
bypass this policy.

Historical schema-version-1 approvals, Gate reports, locks, and source revisions are
immutable replay inputs. Replay them using their original pinned implementation, as in
the existing history runbook. New release construction rejects the legacy unstructured
reviewer instead of using it to skip model and session checks.

## Acceptance evidence

Public release-builder tests cover all allowed models, disallowed or missing models,
same-session approval, malformed reviewer records, human approval, legacy approval,
nonaffirmative decisions and stale Gate binding. The cycle contract confirms that failed
approval leaves both authorities untouched, independent approval permits release, and
an unchanged released successor remains a no-op. The complete Corpus Gate must still pass.
