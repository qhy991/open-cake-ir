# Evidence custody

The evidence already present in this migration tree is a frozen acceptance fixture: historical indexes, external
seal anchors, released Compiler archives, G7 provider archives and the G8 r1-r6 qualification archives. It is
immutable input to replay and cutover review, not an active output root.

After cutover, every new provider qualification, Compiler release archive and Campaign must use a new absolute
evidence root outside the Git checkout. The checkout retains only explicitly reviewed evidence promoted into a
later revision. Reports remain deletable projections; terminal archives and their external anchors remain the
authority. No process may dual-write a Run into the legacy repository and this repository.

`executors/open-cake-ir-b200-v1-5bdc9106/` is the complete 30-source Executor closure used by G8 r6. Verify it through
the current `ExecutorRevision` loader with that directory as `project_root`;
`runtime/executors/open-cake-ir-b200-v2.json` is the distinct current Revision and must never be substituted into
the historical Campaign Lock.

The unversioned `open_cake_turn.md` and `direct_cuda_turn.md` prompt bytes are likewise retained for historical G8
references. Current Study templates use the v2 prompt files, which state the actual embedded-bundle boundary.
