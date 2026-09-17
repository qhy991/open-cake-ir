# Historical AMD Executor v57 reservation

This unchanged descriptor reserves the already released `open-cake-ir-b200-v57` identity through the existing release scan. Its alternate filename follows ADR 0049. It is not the current main Executor and does not modify `inventory/EXECUTOR_REVISIONS.json`. Current-release consumers continue to resolve the inventory's explicit current path.

Replay the original source-bound runtime at Git commit `71e53664e4563485120912d5b49517427e5d2106`, available on `codex/archive-amd-executor-v57-20260907`, using the original `runtime/executors/open-cake-ir-b200-v57.json` path and complete pinned Git tree. The 98-source document inherits the host declaration of main `49fa65b` Executor v56. This archive is not a new live host capture or GPU/performance qualification.

Do not point current main execution at this archive alias: its source paths belong to the original AMD tree. The existing cycle observes the reserved executor_id and derives a later identity for future releases.
