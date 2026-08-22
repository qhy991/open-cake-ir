# Evidence custody

The evidence already present in this migration tree is a frozen acceptance fixture: historical indexes, external
seal anchors, released Compiler archives, G7 provider archives and the G8 r1-r6 qualification archives. It is
immutable input to replay and cutover review, not an active output root.

After cutover, every new provider qualification, Compiler release archive and Campaign must use a new absolute
evidence root outside the Git checkout. The checkout retains only explicitly reviewed evidence promoted into a
later revision. Reports remain deletable projections; terminal archives and their external anchors remain the
authority. No process may dual-write a Run into the legacy repository and this repository.

`executors/open-cake-ir-b200-v1-5bdc9106/` is the complete 30-source Executor closure used by G8 r6, while
`executors/open-cake-ir-b200-v2-4be390bf/` preserves the first Git-anchored migration baseline. Verify either through
the current `ExecutorRevision` loader with its directory as `project_root`.
`runtime/executors/open-cake-ir-b200-v3.json` is the distinct current Revision and must never be substituted into a
historical Campaign Lock.

The unversioned `open_cake_turn.md` and `direct_cuda_turn.md` prompt bytes are likewise retained for historical G8
references. Controlled current Study templates use the v2 prompt files; artifact optimization uses its separate
tool-rich prompts.

`qualifications/codex-cli-0.144.3-tool-rich-v1/` is the zero-GPU live observation for provider-default optimization.
It contains shell activity and may not be reused as scientific evidence. Future tool-rich traces can contain private
auxiliary results; launching them requires explicit operator approval for raw Evidence retention.
