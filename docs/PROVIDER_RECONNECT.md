# Bounded recovered-stream notices

The native Codex 0.153.4 JSONL stream can contain a top-level `error` saying
`Reconnecting... n/5 (stream disconnected before completion: ...)` and later
complete the same turn. F-2026-09-20-002 records the observed first qualification
turn. The original failed qualification stays failed.

For `tool_rich_candidate_v1`, the parser now recognizes only that bounded
notification shape. It still requires one thread start, one turn start and one
final completion; complete functional item lifecycles; a matching terminal
message; valid usage; and, at normalization, a valid candidate envelope. A
`turn.failed`, unknown error, malformed notification, missing completion or
invalid candidate is still refused. The legacy closed-file contract is unchanged.

The full notification remains in `raw_events`. Its auxiliary projection is
`transport_reconnect` / `recovered`; `item_id=jsonl:N` is its zero-based location
in those original bytes, not a native item identifier. It cannot count as tool
activity or replace a write. Replay derives the same projection from the same
raw stream, and all reported input/output tokens remain charged.

This does not retry a provider invocation, resume a failed campaign, or assert
that a proxy is reliable. CPU counterexamples cover fatal errors, failed and
truncated turns, missing file events, invalid candidates and ambiguous JSON.
At commit `1b51db28`, precise replay of the remote failure preserved all original
events and token accounting and matched the frozen candidate plan. A fresh
two-turn live qualification also passed on B300-M3 through the jump proxy; the
Finding records both external evidence paths. This verifies the bounded adapter
repair, without changing the old failure or establishing network reliability.

Upstream references: [JSONL event documentation](https://developers.openai.com/codex/noninteractive),
and `openai/codex` tag `rust-v0.153.4`,
`codex-rs/exec/src/event_processor_with_jsonl_output.rs`. The native implementation
emits an Error notification while staying Running; TurnStatus owns final failure
versus completion. An enum comment alone calling every top-level error fatal
does not describe the recovered stream observed here.
