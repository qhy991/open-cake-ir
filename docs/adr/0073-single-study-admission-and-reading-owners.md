# ADR 0073: One Study parse and one reading index per responsibility

Status: accepted

## Context

TaskLab parsed a Study before common preflight parsed it again. Replay helpers occupied
unrelated-looking top-level modules. Four document indexes repeated the full ADR list,
while current guides still described the retired Portfolio Study lifecycle.

## Decision

Common preflight parses Study once and passes that object through the task admission hook
before resolving execution bindings. Timing-coverage and execution-mode refusals retain
their ordering. The facade acquires no new state.

Independent replay readers live under `lab.replay`, with the public replay entry unchanged.
Internal callers use their owning modules; no compatibility forwarding modules are added.
Provider modules remain separate because they own distinct execution and protocol boundaries.

`docs/README.md` is the short reading entry, `docs/catalog.md` retains the detailed reading
and historical catalog, and `docs/adr/README.md` alone indexes all ADRs. Language gateways
link to it. The Lab implementation map owns module navigation.

## Consequences and validation

Tests exercise one real Study parse, task refusals before dependency resolution, and replay
imports that exclude live writers. Documentation checks cover the canonical index and links.
No IR syntax, analysis, emission or hardware contract changes; P1–P8 remain unchanged.

Historical evidence, inventory, migration bundles and dated data remain in place: tests and
fixed workload provenance still reference them. Consolidating navigation is not authorization
to rewrite evidence or break historical locators.
