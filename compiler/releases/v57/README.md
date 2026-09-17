# Historical AMD Compiler v57 reservation

These four authority files preserve the already released AMD Compiler v57 unchanged. They reserve its identity for the existing canonical release planner; they do not activate AMD support in this checkout or change the current main Compiler.

Replay the original release at Git commit `b2d0a42409afcdef05e66da4723df44a6a09dfef` with its original `compiler/revision.lock.json`, `compiler/corpus-gate-report.json`, `compiler/release-approval.json` and `compiler/source_set.json` paths. Source files named by the archived lock belong to that pinned Git tree; the current main tree is not a substitute replay environment.

The archive was copied from the accepted AMD history, not regenerated or newly approved. GPU correctness, live host admission and performance claims do not follow from this reservation.
