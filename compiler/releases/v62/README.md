# Historical native Compiler v62 reservation

These four authority files preserve the already released native Compiler v62 unchanged. They reserve its identity for the canonical release planner; they do not activate native lowering in this checkout or change the current main Compiler.

Replay the original release at Git commit `6752307cfccf79e1f9616a40c2e5f136aaf36ba8` with its original `compiler/revision.lock.json`, `compiler/corpus-gate-report.json`, `compiler/release-approval.json` and `compiler/source_set.json` paths. Source files named by the lock belong to that pinned Git tree; the current main tree is not a substitute replay environment. The four metadata files are not a complete source archive.

The files were copied from the independently approved native release, not regenerated or newly approved. This reservation does not establish GPU correctness or performance qualification.
