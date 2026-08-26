# AMD optimization branch policy

`codex/amd-gfx1151-aiter-rmsnorm-baseline` is the active integration branch for the
source-pinned AMD library comparison. It retains the earlier gfx1151/Q4 lineage and has
merged GitHub `main@415a1f0f4d090e7f97292adcf3a539be67269550`. The earlier
`codex/amd-gfx1151-optimization`, tuning and Q4 conformance refs remain historical
lineage inputs, not parallel places to publish this slice.

This integration branch is exploratory until the exact gfx1151 Executor and required
correctness runs exist. Merging `main` into it keeps development current; it does not
advance a protected/formal AMD candidate, release a Compiler or qualify GPU evidence.
The formal advancement gate below remains unchanged.

The GitHub refs are the branch-state source of truth. Every alignment fixes both ref SHAs
before acting and uses a unique temporary branch. GitHub performs the merge of the fixed
main SHA into that temporary branch. Conflict, authentication failure, unavailable
hardware, failed tests or ref drift leaves the formal AMD branch unchanged. No process
modifies main, force-pushes, automatically resolves conflicts, creates a PR or treats a
collector failure as evidence of no update.

A candidate alignment may fast-forward the AMD branch with `force=false` only after:

1. downloading a clean archive for the candidate SHA;
2. releasing and admitting the exact gfx1151 Executor on the target host; a B200
   Executor or source-only load is not an AMD execution qualification;
3. passing the complete contract suite except the explicitly B200-only Nsight Compute
   custody check;
4. passing both gfx1151 SwiGLU and llama RMSNorm correctness quickstarts with new external
   evidence directories;
5. re-reading main and AMD refs and proving neither moved during validation.

Raw GPU artifacts remain outside the checkout. A candidate commit status owns the
pass/fail result, and the AMD sync ledger Issue owns append-only conflict and attempt
history. The normal cadence is every Monday at 10:00 Asia/Shanghai and once immediately
before each new AMD optimization campaign. Synchronization never reruns a performance
search or mints a Compiler/Executor revision by itself.
