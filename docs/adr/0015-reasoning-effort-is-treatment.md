# ADR 0015: Reasoning effort is an explicit treatment factor

Status: accepted, 2026-08-24.

## Outcome and non-goals

The exact provider reasoning effort belongs to each matched Study's shared provider
configuration. Both arms must use the same value, and a live provider qualification must
prove that exact configuration before a Study can freeze. Qualification and live freezing
therefore require the operator to name the value; there is no implicit default.

This does not reinterpret `max` as the paper's `xhigh`, rewrite a frozen Study or provider
receipt, introduce a paper mode, add an agent-manager state machine, or claim that the
current task-informed skeletons are clean-start inputs.

## Authorities and dataflow

`study.arms.*.provider.reasoning_effort` is the canonical treatment fact. Matched
preflight already requires the complete provider documents of both arms to be identical.
The qualification receipt's configuration digest proves the chosen value together with
model, service tier, output schema, removed environment, sandbox, reference visibility,
feature policy and submission contract. The CampaignLock copies that frozen provider
document; the invocation builder emits the same value on initial and resumed Turns.

The Lab validates a non-empty value and the qualification digest. It does not maintain a
second list of effort levels: support is a property of the exact qualified provider
executable, whose bytes are also bound by the receipt.

## Compatibility and agent boundary

Existing `max` Studies and receipts retain their bytes and meaning. A future paper-aligned
Study must first qualify `xhigh`, then freeze a successor with `xhigh` explicitly. Merely
editing a Study value leaves the qualification digest inconsistent and fails preflight.

KDA-internal confirms the useful agent boundary already present here: an isolated worker
hands one whitelisted artifact to an external authoritative Judge. Auxiliary investigation
does not own submissions or measurements. No additional orchestration primitive is needed
to make reasoning effort explicit, and KDA-internal does not become a runtime dependency or
authority of this repository.

## Acceptance evidence

A zero-GPU qualification using `xhigh` must archive an invocation containing that exact
configuration. Live-freeze must project `xhigh` into both arms and pass preflight against
the matching receipt. Changing a current `max` Study to `xhigh` without replacing its
qualification must fail preflight.
