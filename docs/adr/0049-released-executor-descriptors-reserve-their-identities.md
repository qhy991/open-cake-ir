# ADR 0049: released Executor descriptors reserve their identities

Status: accepted, 2026-09-06. Supersedes witness-based reclamation for Executor
Revisions only; Compiler release and approval rules are unchanged.

The branch review found different released bytes under the same Executor v41 and v42
identities: main at `7fdac036` bound 45 sources, while the Kimi-K3 clone at `0a0528d0`
bound 43. The cycle scanned only local Studies, Evidence, and inventory observations,
then deleted descriptors without a local witness before releasing their ids again.
External runs cannot be enumerated by that scan. This also contradicted the underlying
create-only release command, which refuses every previously released identity it finds.

Released descriptors are the sole owner of reserved Executor identities. The cycle
uses the release command's descriptor set in `runtime/executors` and archived
`evidence/executors`, retains every released file, and derives the next B200 ordinal
after their maximum. Inventory remains a projection; missing witnesses cannot authorize
reclamation. Divergent existing histories remain bound to their original commits and
must not replace each other's descriptors during integration.

An explicit `--host-environment` JSON may supply a host already verified by the release
operator when the previous host no longer exists. The cycle validates its document
structure and preserves its contents; it does not turn that structural check into a
live-host claim. Without this input, the existing host/profiler discovery path remains.

A focused regression runs two actual releases with no local witness references,
including an archived higher ordinal, and requires increasing ids, preserved old
descriptor bytes, and retained inventory history.

On 2026-09-07, concurrent branches independently produced v48. The cost-loop release
at `4a9d2df0f994a230ceb4b589bf405022a2887abd` keeps its original
`runtime/executors/open-cake-ir-b200-v48.json`. The R2 B200 release at
`966eba4bcfa773b302501df45c9f640e95c7268b` is retained unchanged as
`runtime/executors/open-cake-ir-b200-v48-r2.json` for identity reservation. Its replay
uses the complete original Git tree at `966eba4`, including the descriptor's original
path, Compiler v52 and all required source files; it must not load against current
sources. Both commits remain in the integration ancestry. The R2 GPU records keep
their original binding, and the combined runtime requires a separately derived successor.
