# ADR 0067: Retired release outputs live on the `history` branch

Status: accepted by the repository owner, 2026-09-18 (owner decision D1 of the
2026-09-18 architecture audit). Supersedes one sentence of
[ADR 0065](0065-source-identity-is-the-commit.md), "History stays where it is"; every
other clause of ADR 0065 stands. The identity reservations of
[ADR 0049](0049-released-executor-descriptors-reserve-their-identities.md) and
[ADR 0050](0050-released-compiler-locks-reserve-their-identities.md) are kept.

## Later branch cleanup, 2026-09-20

The owner requested that GitHub retain only the five maintained branches and active task
branches. The former `history` branch tip, `387ff34c`, is retained as the immutable
`history` tag before the remote branch is deleted. `git fetch origin tag history` retrieves
it, and `git show history:<path>` continues to resolve the same tree. No historical object,
path, identity or evidence is rewritten. The legacy `history_branch` key in
`docs/history/identities.json` remains compatible: its value is a Git ref, now a tag.
Future archival additions use new versioned tags; published tags are not moved.

This changes the original decision's branch lifecycle only. The original rationale and
move set below remain the record of the 2026-09-18 decision. Current task and archive
routing is owned by [Development branches](../DEVELOPMENT_BRANCHES.md).

## Context

ADR 0065 made the clean commit the source identity and removed the per-change release
cycle. Its last clause kept the cycle's outputs in the working tree: released locks under
`compiler/releases/`, descriptors under `runtime/executors/` and the frozen Executor
inventory, "replaying at their own commits with the tools of those commits".

The producer of those outputs was deleted in `7c951715` (`compiler/release.py`, the
release-cycle tools and their tests). Measured at `c75ee4b3`, the outputs it left behind
were 15.0 MB, 31.1% of the tracked bytes of the repository. `compiler/releases/` had zero
working readers: no source module, tool or test on `main` opened a released lock. The only
readers of `runtime/executors/` were two test fixtures. A retained result replays at the
commit that produced it, where those files are present, so nothing on `main` needed them
in the working tree; they were being carried by every clone and every worktree for no
consumer.

## Decision

- **An orphan branch `history` holds the move set, byte-identical at the original
  paths.** Its first commit is `d6d764e7`. `git show history:<path>` resolves every moved
  file; a historical Evidence root or Campaign Lock that cites one still names the same
  path.
- **`docs/history/identities.json` maps every retired identity to its producing commit
  and its path on the `history` branch.** A finding or Evidence record that cites
  `open-cake-ir-sm100a-v75` or `open-cake-ir-b200-v42` stays resolvable without a copy
  of the document on `main`.
- **Identities stay reserved.** ADR 0049 and ADR 0050 continue to hold: no retired
  Compiler Revision or Executor id is reused. The reservation now lives in the identities
  table rather than in the presence of a directory.
- **Nothing on `main` reads the moved files.** No tool, test or fixture on `main` opens a
  released lock, a descriptor, a calibration plan or the retired study successors. A
  reader that needs one checks out the commit that produced it, exactly as ADR 0065
  already required for replay.
- **Historical Evidence replays as before**, at its own commit, with the tools of that
  commit. This decision changes no Evidence record and no replay path.

## What moved

The move set is the list the audit produced; each entry is on `history` at the same path.

- `compiler/releases/` -- every released Compiler lock, Gate report, source set and
  approval record, including the final `v75+<authority>` and `v84+<authority>` forms.
- `runtime/executors/` -- every released Executor descriptor.
- `evidence/executors/` -- the retained per-Executor source closures.
- `evidence/calibration/` and `contracts/calibrations/` -- the three preregistered
  ranking-calibration plans and their evidence; owner decision D4 records that no
  ranking calibration is coming.
- The thirteen `runtime/g8-system-r1..r6.{campaign.lock,preflight}.json` and
  `runtime/v28-rope-r1.campaign.lock.json` documents, which pin Compiler v3/v28 and
  Executor v1/v29.
- The 36 `flash-kmeans-r45-portfolio-reconstruction-*` Study successors under
  `contracts/studies/`; each was minted per Revision, the practice ADR 0028 and the
  frozen-template rule already forbid.
- The closed AKA campaign's ten `tools/*aka*` runners and nine `tests/contracts/test_aka_*`
  tests (owner decision D7), and the three ranking-calibration tools with
  `tests/contracts/test_ranking_calibration_tool.py`.

## What stayed, and why

- `inventory/` -- the frozen historical snapshot that `docs/adr/0047` and
  `tests/contracts/test_documentation.py` name as a stable entry document.
- `docs/data/` -- a live published dataset with external readers, so the AKA
  verification tools that read it (`verify_aka_qualified_review_export.py`,
  `verify_aka_fma_v41_reaudit.py`) and their tests stayed with it.
- `migration/` -- the migration bundle and capability matrix, still read by
  `tests/contracts/test_migration.py`.
- `evidence/campaigns/` -- retained Evidence; append-only and never relocated.
- The eight live `runtime/*.json` campaign configurations and `runtime/hosts/`, which
  ADR 0065 made the Executor's own document.
- `contracts/studies/flash-kmeans-r45-portfolio-reconstruction-template.json` -- the one
  template the 36 successors were stamped from.

## Consequences

- A clone of `main` is about a third smaller, and no session pays for release
  bookkeeping it cannot read.
- A reference to a retired id resolves through `docs/history/identities.json` and a
  `git show history:<path>`; a reference that resolves nowhere is a defect in the table,
  not a reason to copy the document back.
- The two test fixtures that read `runtime/executors/` inline the descriptor they need
  (`tests/contracts/test_executor_host_capture.py`); no test reads the `history` branch.
- `.github/workflows/ci.yml` keeps `fetch-depth: 0`, because replay contracts still
  execute pinned Git sources at their own commits.
- The `history` branch is append-only in the same sense as Evidence: a later move adds
  a commit; nothing there is rewritten.
