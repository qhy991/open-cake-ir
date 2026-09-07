# ADR 0055: Task implementations live outside the common Lab

Status: accepted by the repository owner, 2026-09-07.

## Decision

The owner requested complete task ownership isolation after observing QSA-specific code
exported as a common Lab capability. Task code belongs under `src/open_cake_ir/tasks/`.
The common Lab and Evaluation modules must not import concrete tasks or dispatch by operator.

Task modules retain the exact semantic validators, oracles, fixed launch ABIs and preparation
rules. TaskLab is the built-in composition of those functions with the common Ralph engine.
WorkloadContract is the shared data/ABI type; the task-owned loader selects exact validators
statically. Shared CUPTI timing remains common because both QSA and Flash consume it.

The Flash portfolio, including its fixed shape key, seed and execution/audit operations,
is task-owned. One ExactShape represents B/N/K/D for both specialization and dispatch.
QSA's obsolete TurnRequest constructor and CLI turn command are deleted. The Ralph controller
remains the sole iteration authority.

## Frozen boundaries

Compiler sources and Corpus expectations are unchanged. Existing Workload and Program
contracts, release descriptors and historical evidence are not rewritten. The QSA oracle
is relocated with its pinned bytes intact. Historical callable names inside frozen contracts
remain identity labels, not runtime imports. The Executor successor recursively includes
all task Python modules and explicitly includes task assets.

## Acceptance

Exercise strict task loading, QSA compiler/evaluation projection and native/Flash candidate
boundaries; task-package generation must preserve Workload, target and exact source rules.
Run portfolio execution/replay, matched Ralph contracts and CLI composition with a coherent
Executor successor. Enforce the import direction and refusal of retired interfaces in
contract tests. Moving an oracle does not establish new GPU or performance evidence.
