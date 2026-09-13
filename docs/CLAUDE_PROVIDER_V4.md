# Claude artifact-only event contract v4

`claude_stream_candidate_v4` is a prospective Lab/Executor contract for the existing
single-arm `artifact_optimization_only` Claude treatment. It does not expand the
Compiler or admit Claude to scientific matched comparisons. Its evidence is
F-2026-09-13-003: eight real compactions and four malformed terminal outcomes from
the B300 E109 Kimi-K3 sweep.

## Compaction

The parser validates the observed CLI 2.1.263 records in order:

1. One or more `system/status: compacting` progress records.
2. A `system/status` record with null status and `compact_result: success`.
3. A `system/compact_boundary` with the observed metadata and retained-message
   identities. It closes the lifecycle before another compaction or the terminal.

Unknown fields, malformed types, failed or incomplete lifecycles, and a different
session remain refusals. Historical retained-message UUIDs need not appear in this
invocation. Active tool invocations and writes are not cleared by compaction.

Every admitted record becomes a `ProviderAuxiliaryActivity` with
`item_type: context_compaction`, its raw event UUID, and the phase as status.
The complete native JSONL owns the metadata. This projection records a changed
author context; it does not claim to reconstruct the provider's internal state or
to make compacted and uncompacted arms scientifically comparable.

Usage still comes from native `modelUsage` once. Context sizes, dropped tokens,
durations and progress notifications do not add to provider token accounting.
`--autocompact 1M` is only a CLI request: the observed model context can be smaller.

## Exact terminal request

The stable provider configuration and qualification bind the terminal-schema
template. For a v4 invocation, the Adapter deterministically binds its `arm` and
`turn` constants from the trusted Run request before invoking the CLI. The template
is not rewritten in a Study or qualification per turn. Recorded invocation plans
retain that template; the actual request schema is its deterministic projection
using the retained expected terminal. The caller's invocation object is unchanged.

The returned native `structured_output` and any successful StructuredOutput tool
still have to match that exact terminal. Missing output, wrong arm/turn, changed
session/model, invalid candidate writes and unsupported tools remain refusals.
Natural language or a candidate file cannot manufacture a terminal. There is one
supervised CLI call per Lab Turn; no repair invocation or uncharged retry is added.
This prevents an unconstrained requested turn number, but cannot guarantee that a
provider returns a terminal. A live canary must measure that behavior separately.

## Compatibility and publication

`claude_stream_candidate_v3` keeps its original compaction refusal and unbound
schema request. Execution, qualification parsing, fault usage and campaign replay
dispatch on the declared contract. Current templates choose v4; old frozen Studies
remain v3 and are never relabeled. v3/v4 configurations differ, so v4 requires a new
matching live qualification before a formal campaign. CPU fixtures are not one.

These changes belong to a successor Executor source closure. Compiler v80 and old
Executor descriptors remain intact. Replay admission of retained v3 fault streams
under v4 is defect verification, not retroactive promotion of the old Runs. The
four malformed terminal streams stay faults; live v4 qualification and a bounded
task canary precede another full sweep.
