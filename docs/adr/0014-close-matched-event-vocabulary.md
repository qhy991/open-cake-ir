# ADR 0014: Close the matched Run semantic event vocabulary

Status: accepted, 2026-08-24.

## Outcome and non-goals

A successor matched Study names one closed semantic event vocabulary. Semantic replay
must account for every retained event, validate the Run start, and derive every routed
diagnosis from facts retained elsewhere in the same Run. An unknown or unjustified event
therefore makes semantic replay fail instead of being silently ignored.

This does not change Evidence-v2 storage, add timestamps or an `active_evolve_time`
surrogate, introduce another execution path, rerun a Compiler during audit, or rewrite a
frozen Study, Campaign, event or report.

## Authorities and canonical form

`study.evidence.event_vocabulary = matched_run_v1` selects the current contract. It is
copied into the Campaign Lock and is the sole authority for strict semantic replay. The
vocabulary contains only the event kinds already emitted by the matched execution path;
it is not a registry that callers may extend.

The first event is the sole `run_started`, whose sequence and assigned arm are derived
from `allocation.order`. The sole `checkpoints_projected` immediately precedes the sole
terminal event. All other events occur between those boundaries and belong to one
provider Turn or its terminal fault.

`candidate_set_filtered.order` owns the disposition, optional released cost and optional
Compiler semantic digest for every submitted Candidate. A same-program collapse is
derived from that ordered projection and the Study's search bound. A cost-model routing
diagnosis is derived from that same order, qualified search Receipts and the frozen
materiality ratio. `diagnosis_routed` is consequently a checked projection, not a second
writable truth.

## Failure and compatibility semantics

For `matched_run_v1`, an unknown kind, duplicate boundary, changed start, extra payload
field, unaccounted object role, impossible collapse or unsupported cost-model diagnosis
fails semantic replay. Archive content integrity and filesystem custody remain separate
facts under the later ADR 0031.

Earlier matched Studies did not declare a semantic vocabulary. Their frozen bytes remain
readable through one bounded legacy adapter with the earlier replay behavior. Every new
matched successor adopts `matched_run_v1`; neither successor tool may regenerate the
legacy open vocabulary.

## Acceptance evidence

One complete current matched Run must replay in a fresh process. Contract probes then
inject an unknown event, alter `run_started`, and alter a routed diagnosis while retaining
the otherwise valid replay objects; all three must fail semantic replay. Existing frozen
legacy Campaign audits must keep their historical result.
