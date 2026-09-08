# Development branches

`main` is the maintained source. Its published Compiler and Executor are described by the
[generated release view](../reports/current/STATUS.md).

This table tracks three historical development lines; it is not a live branch inventory.
Use Git and the linked pull requests for integration status:

| Branch | Purpose | Integration boundary |
| --- | --- | --- |
| `codex/amd-gfx1151-consolidated` | Consolidated gfx1151, Q4/Q8, AITER RMSNorm, and ROCm analysis work | Historical Compiler draft; port to current main, resolve frozen revision boundaries, and independently review before integrating. Earlier AMD candidate/sync branches are superseded by this development line. |
| `metal` and `codex/metal-*` | Historical Apple Metal development and M1 Pro integration | The M1 Pro backend and existing-Lab task launcher were merged through [PR #76](https://github.com/qhy991/open-cake-ir/pull/76), superseding PR #46/#47. Historical branches remain replay inputs; any further change uses the current release boundary. |
| `codex/kda-decode-cake-vs-internal-r1` | Caller-indexed recurrent state and a KDA comparison proposal | Contains unintegrated Compiler changes; the earlier fused-decode branch is its ancestor. Requires a current successor, complete corpus validation, and independent review. |

Target availability and release identity come from the generated release view above.
A retained development branch does not establish a measured performance claim.
Historical branch names may also appear in retained experiment records; deleting a stale
branch does not turn its earlier results into evidence for the current Compiler.
