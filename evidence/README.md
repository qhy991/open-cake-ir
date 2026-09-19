# Evidence custody

The evidence already present in this migration tree is a frozen acceptance fixture: historical indexes, external
seal anchors, released Compiler archives, G7 provider archives and the G8 r1-r6 qualification archives. It is
immutable input to replay and cutover review, not an active output root.

After cutover, every new provider qualification, Compiler release archive and Campaign must use a new absolute
evidence root outside the Git checkout. The checkout retains only explicitly reviewed evidence promoted into a
later revision. Reports remain deletable projections; terminal archives and their external anchors remain the
authority. No process may dual-write a Run into the legacy repository and this repository.

The retired Executor descriptors (`runtime/executors/*.json`) and the vendored source closures that preceded them
(`evidence/executors/open-cake-ir-b200-v1-5bdc9106/` for G8 r6, `-v2-4be390bf/` for the first Git-anchored
migration baseline, `-v3-1c18cbfb/` for the tool-rich cutover baseline, `-v4-203b2d8f/` for the first GPU
teaching-smoke runner and `-v5-7f437598/` for the last pre-custody runtime closure) no longer live in the working
tree. They are on the `history` branch, byte-identical at their original paths, and `docs/history/identities.json`
maps every retired Compiler Revision and Executor id to the commit that produced it and the path that
`git show history:<path>` resolves. They replay at their own commits with the tools of that commit, never against
today's tree or through today's `ExecutorRevision` loader (ADR 0065); a retired identity is never substituted into a
historical Campaign Lock.

The unversioned `open_cake_turn.md` and `direct_cuda_turn.md` prompt bytes are likewise retained for historical G8
references. Controlled current Study templates use the v2 prompt files; artifact optimization uses its separate
tool-rich prompts.

`qualifications/codex-cli-0.144.3-tool-rich-v1/` is the zero-GPU live observation for provider-default optimization.
It contains shell activity and may not be reused as scientific evidence. Future tool-rich traces can contain private
auxiliary results; launching them requires explicit operator approval for raw Evidence retention.

`inventory/GPU_QUICKSTART_QUALIFICATION_V3_20260823.json` binds the current beginner teaching smoke's create-only result,
stdout and parent `gpu-run` stderr. It proves one current B200 correctness path, not Campaign-level semantic replay,
performance stability or a paper claim. `gpu-quickstart-v3-attempt1/` preserves the preceding zero-result handoff
permission fault. Its worker result was not retained, so kernel calls are unknown and no Candidate conclusion is
authorized. Earlier inventories remain historical.
