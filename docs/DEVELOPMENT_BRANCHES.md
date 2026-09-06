# Development branches

`main` is the maintained source. Its published Compiler and Executor are described by the
[generated release view](../reports/current/STATUS.md).

The following branches retain work that needs a separate integration and validation step:

| Branch | Purpose | Integration boundary |
| --- | --- | --- |
| `codex/amd-gfx1151-consolidated` | Consolidated gfx1151, Q4/Q8, AITER RMSNorm, and ROCm analysis work | Historical Compiler draft; port to current main, resolve frozen revision boundaries, and independently review before integrating. Earlier AMD candidate/sync branches are superseded by this development line. |
| `metal` | Apple Metal lowering, source analysis, and hardware-informed experiments | Historical target/backend draft; requires a current Compiler successor and independent review. |
| `codex/kda-decode-cake-vs-internal-r1` | Caller-indexed recurrent state and a KDA comparison proposal | Contains unintegrated Compiler changes; the earlier fused-decode branch is its ancestor. Requires a current successor, complete corpus validation, and independent review. |

These are development lines, not additional targets promised by the current main release.
Historical branch names may also appear in retained experiment records; deleting a stale
branch does not turn its earlier results into evidence for the current Compiler.
