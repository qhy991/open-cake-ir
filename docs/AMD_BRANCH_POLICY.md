# AMD optimization branch policy

`codex/amd-gfx1151-consolidated` is the sole active integration branch for AMD work. It
retains the gfx1151, Q4/Q8, AITER RMSNorm and rocprofv3 lineage and is rebased onto
GitHub `main@415a1f0f4d090e7f97292adcf3a539be67269550`. The earlier
`codex/amd-gfx1151-aiter-rmsnorm-baseline`, optimization, candidate, sync and Q4
conformance refs remain read-only historical lineage inputs, not parallel publication
branches.

This integration branch is exploratory until the exact gfx1151 Executor and required
correctness runs exist. Rebasing it onto a fixed `main` keeps development current; it
does not advance a protected/formal AMD candidate, release a Compiler or qualify GPU
evidence. The formal advancement gate below remains unchanged.

The GitHub refs are the branch-state source of truth. Every alignment fixes both ref SHAs
before acting and uses an isolated clean worktree. The active AMD commits are rebased
onto the fixed main SHA, while historical evidence refs remain unchanged. Conflict,
authentication failure, unavailable hardware, failed tests or ref drift leaves the
remote active branch unchanged. No process modifies main, uses an unleased force push,
bulk-merges historical refs, creates a PR or treats a collector failure as evidence of
no update.

A rebased active branch may be published only after:

1. proving the pre-rebase branch already contains every non-obsolete AMD line and that
   superseded v25/v26 or patch-equivalent refs are not replayed;
2. recording the old active ref and fixed main SHA, then proving the rebased tree is
   byte-identical except for an explicit alignment-policy update;
3. passing the Compiler Corpus Gate verification and focused AMD/Executor contract tests;
4. re-reading main and the destination ref and proving neither moved during validation;
5. updating only the active feature branch, with `--force-with-lease` bound to the
   observed destination SHA when that branch already exists.

This synchronization gate is not GPU qualification. A new optimization campaign or
promotion still requires the exact released gfx1151 Executor and fresh applicable
correctness evidence; a B200 Executor, source-only load or successful rebase cannot
substitute for that boundary.

Raw GPU artifacts remain outside the checkout. A candidate commit status owns the
pass/fail result, and the AMD sync ledger Issue owns append-only conflict and attempt
history. The normal cadence is every Monday at 10:00 Asia/Shanghai and once immediately
before each new AMD optimization campaign. Synchronization never reruns a performance
search or mints a Compiler/Executor revision by itself.
