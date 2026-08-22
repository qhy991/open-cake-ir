# ADR 0004: Restore provider-default features only for artifact optimization

Status: accepted, 2026-08-23.

## Context

G7 r4 and G8 r6 deliberately disabled Apps/MCP, browser, shell, plugins, subagents and related Codex features. That
closed Authoring Environment was necessary to qualify a controlled provider seam and preserve the interpretation of
scientific comparisons. It is too restrictive for the final engineering optimization loop, where the objective is
the strongest confirmed artifact rather than a treatment-effect estimate.

Removing the denylist alone is insufficient: tool-rich Codex Turns emit auxiliary JSONL items, shell can write the
Candidate without a `file_change` item, and the v1 terminal schema's model-reported `tool_calls=1` is not an activity
authority. Reusing `scientific_matched_search` would also let engineering output leak into an Estimand.

## Decision

Keep the same `matched_search` Lab, Evaluation and Evidence path and add the
`artifact_optimization_only` Claim Scope; do not add a runtime mode. It has one Run per existing Authoring
Environment, multiple bounded Turns and no Estimand, comparison, pooling or scientific inclusion. Each Run may
promote only its lowest-latency confirmatory-qualified Candidate, with earliest Turn as the tie-break.

The successor Authoring Environment sets `disabled_features=[]`. This means only that `open-cake-ir` injects no
`--disable` overrides: the pinned Codex binary's defaults, account capabilities and administrator requirements still
determine the effective catalog. A distinct live qualification must bind that configuration. G7 r4/G8 r6 retain the
historical closed denylist and strict event contract.

The provider seam uses `tool_rich_candidate_v1`: known auxiliary item lifecycles are retained as typed activity and
raw JSONL, while the final no-follow Candidate postcondition remains the submission Interface. Static output schema
v2 removes the model-reported tool count. Auxiliary tools may help authoring, but they do not authorize external
mutation or direct GPU measurement; correctness and performance promotion still come only from common Evaluation
Receipts.

## Consequences

- Full feature exposure and scientific data use are independent policies.
- Apps/MCP, shell, browser, plugins and subagents may appear in optimization traces without becoming measurements.
- A tool-rich Run may create scratch files; only the fixed Candidate path is sealed as the Candidate.
- Raw auxiliary output is retained in Evidence, so operators must authorize the source data for archival before a
  live optimization Campaign.
- Multi-agent token usage remains the provider's reported Turn usage; maximum Turns provides an independent hard
  bound, and no scientific token-budget claim is permitted for this scope.
- A promoted Candidate becomes a KernelSeed only through a later explicit freeze; Portfolio remains unchanged.
