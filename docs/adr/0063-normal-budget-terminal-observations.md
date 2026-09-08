# ADR 0063: Normal budget terminal observations

Status: accepted design, 2026-09-09; implementation and release acceptance pending.
Related: ADR 0048 and issue #69.

## Decision

A prospective matched-search Analysis Plan may declare
`"endpoint_policy": "normal_budget_terminal_v1"`. This closed value defines the
terminal-budget endpoint at the first normal stop derived by the existing Ralph
controller over its budget vector: provider tokens, wall time, active authoring
time, maximum Turns, or Evaluation capacity. Existing ordering resolves simultaneous
limits. The controller and its retained StateCard remain the authority for that stop;
this policy does not change scheduling, limits, timing, retries or replacement rules.

The endpoint selects the lowest confirmed latency among fully completed
TurnObservations by that stop, breaking ties by the earliest Turn. A normal stop with
no qualified completed Turn observes `no_qualified_candidate`, including a normal
zero-work stop. A provider, harness, broker or custody fault remains missing data.
A partially completed Turn cannot supply a terminal candidate. Known-subtotal usage
from a failed call is not a complete budget total and cannot rescue a fault endpoint.

The new endpoint records `observation_basis: normal_budget_terminal_v1`, the derived
`terminal_reason`, and actual terminal provider tokens in `budget`. Its existing
qualified-by-budget and best-confirmed-candidate fields retain their meanings under
this explicit vector-terminal policy. Exact Lab semantic replay validates the closed
projection. Evidence remains domain-generic; it does not import Lab endpoint policy.

Token checkpoint rows are independent observations and remain unchanged. In
particular, a token checkpoint above actual terminal usage stays `unreached`; the
terminal endpoint does not backfill it. Reports show token-limit checkpoint state
separately from terminal reason and distinguish normal stopping before a checkpoint
from a protocol fault. The scientific estimand's term “terminal-budget” means this
vector-terminal observation only when the Analysis Plan explicitly declares it.

Absent `endpoint_policy`, existing token-limit checkpoint endpoint shape and semantics
are preserved exactly. Frozen Studies and historical Runs are never rewritten. Active
matched-search templates and the normalization Study factory adopt this policy as a
design successor, without minting per-Compiler or per-Executor Study ids. Successor and
freeze helpers preserve the source Study's explicit policy. Distinct plans remain
separate Campaign authorities; no automatic pooling is authorized.

## Independent Run audit

Every available Run is considered independently for semantic replay after its archive
and Campaign authority checks. A missing or invalid Run still makes the aggregate
Study unavailable. It does not erase another Run's valid receipt count or eligible
per-Run artifact. Per-Run artifact promotion still requires that Run's archive,
filesystem custody, adherence, semantic replay and qualifying confirmation evidence.
Scientific inclusion and comparative availability remain separate decisions.

Reports derive `reference_access_by_arm` directly from the resolved arm declarations;
this readability projection creates no new reference authority.

## Acceptance

Exercise each normal budget axis and simultaneous limits; confirmed, unqualified,
zero-work and partially completed/faulted Turns; actual terminal usage below a token
checkpoint; legacy absent policy; unknown/malformed policy and endpoint extras; and
live-to-semantic replay with a tampered stop or endpoint. Exercise failed-first and
failed-last Campaign orderings, preserving valid per-Run artifacts and receipt counts
while aggregate availability stays false. Fault usage retains observed-zero versus
unavailable-subtotal distinctions and faults never become candidate failures.

All acceptance here is CPU semantic testing with explicit dependencies unless separately
reported. It does not establish device correctness, performance, host qualification or
new release authority.

## Logical evaluator invocation witness

A new `evaluation_attempt_started` event records only Turn, purpose and sealed
candidate identity immediately before the Lab records its Ralph Evaluation budget
consumption and invokes the evaluator. This is a logical-call witness. It is not
broker admission, physical GPU dispatch, a receipt, correctness or successful timing.
If the evaluator raises without returning an attempt, logical invocation count is
known while physical work is unknown. The fault remains missing data.

Replay derives Ralph Evaluation counts from validated ordered start witnesses.
A start requires the already sealed candidate and the corresponding filtered search,
selected qualified candidate or profile dependency. Each start has at most one
ordered attempt completion and one receipt. Only the final matching evaluation fault
can leave a transaction unfinished; no further evaluator transaction follows a fault.
Counts cannot exceed the declared Evaluation limits. No completion or receipt can
invent a start. Old absent-witness histories retain their pinned Executor interpreter;
new code does not append witnesses to old Runs or subtract attempted calls to obtain
a passing replay.
