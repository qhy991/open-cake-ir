# AMD optimization branch policy

`codex/amd-gfx1151-optimization` is the sole active AMD optimization branch. Its initial
base was advanced without feature changes to GitHub
`main@f02320b17ee1f6ed12d1d858e3439bc16b9da4ed`; the earlier
`codex/amd-gfx1151-tuning` branch and its `STOP_CLOSE_NULL` evidence are immutable
historical inputs, not a parallel current implementation.

The GitHub refs are the branch-state source of truth. Every alignment fixes both ref SHAs
before acting and uses a unique temporary branch. GitHub performs the merge of the fixed
main SHA into that temporary branch. Conflict, authentication failure, unavailable
hardware, failed tests or ref drift leaves the formal AMD branch unchanged. No process
modifies main, force-pushes, automatically resolves conflicts, creates a PR or treats a
collector failure as evidence of no update.

A candidate alignment may fast-forward the AMD branch with `force=false` only after:

1. downloading a clean archive for the candidate SHA;
2. passing the complete contract suite except the explicitly B200-only Nsight Compute
   custody check;
3. passing both gfx1151 SwiGLU and llama RMSNorm correctness quickstarts with new external
   evidence directories;
4. re-reading main and AMD refs and proving neither moved during validation.

Raw GPU artifacts remain outside the checkout. A candidate commit status owns the
pass/fail result, and the AMD sync ledger Issue owns append-only conflict and attempt
history. The normal cadence is every Monday at 10:00 Asia/Shanghai and once immediately
before each new AMD optimization campaign. Synchronization never reruns a performance
search or mints a Compiler/Executor revision by itself.
