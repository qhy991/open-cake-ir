# ADR 0033: Workload materialization owns TinyGEMM2 input bytes

Status: accepted, 2026-08-25.

## Outcome and non-goals

Supersede the under-specified TinyGEMM2 v1 Workload with a v2 Contract whose sole case
pins the raw bytes of input, weight, bias, the independent FP32-linear/BF16 oracle and
the retained upstream parent output.
The adapter reproduces the upstream seed-0 CUDA distribution and refuses any generated
tensor whose raw digest or size differs. This restores the `/8` scaling of input and
weight that the v1 provenance already named but its case did not own.

This change does not edit the frozen v1 Contract, add a second distribution vocabulary,
copy the legacy 527-line runner, claim current checked-asset correctness, or change the
Compiler. It also does not treat a static digest as kernel correctness evidence.

## Invariants and authorities

- The current v2 case's `materialized` receipts are the sole authority for exact tensor
  bytes. Seed, mode and adapter code describe how to reproduce them; they cannot override
  a receipt mismatch.
- v1 remains immutable historical input. It may be loaded for custody inspection, but
  the evaluation adapter fails closed because v1 has no materialized authority.
- Four expected receipts come from the retained r31 B200 result and are adopted only if
  one preregistered brokered regeneration of the digest-bound historical generator
  reproduces all four. The fifth receipt is the retained r31 parent output; it is not
  presented as a newly launched observation. Missing or different bytes fail closed.
- Changing the Workload consumer or adapter requires a newly content-bound Executor
  release. The release cycle may reclaim an unwitnessed working id; it never rewrites a
  witnessed revision. Compiler v25 and its independent approval are outside this change's
  source closure.
- A future checked-asset launch must bind the v2 Workload and still prove the canonical
  parent-output and FP32-oracle rules. Materialization equality is necessary, not
  sufficient, for that claim.

## Dataflow, failure and compatibility

```text
historical generator + seed --brokered regeneration--> four observed receipts
retained r31 result ----------------> four expected receipts + parent output
                     exact match --> frozen v2 Workload materialized authority
v2 Workload + adapter --------generate, hash, compare--> tensors or fail closed
```

The v2 path is the only current evaluation path. The v1 file and validator branch remain
only so old references can be inspected; no new Study may use them. Digest mismatch,
non-CUDA materialization or missing receipts aborts before candidate launch. Existing
TinyGEMM2 evidence is unaffected because the current checked-asset partition is missing.

## Smallest complete slice and acceptance evidence

The slice is one case, one generator and five receipts. The predeclared observation must
match input, weight, bias and oracle bytes with zero retries; the retained parent output
must remain bound to the exact r31 result bytes. Contract tests then prove
that v2 is current, v1 is not silently rewritten, CPU/unit-normal generation fails, and
the adapter verifies materialization before launch. Metrics separately require bitwise
parent equality and FP32-oracle tolerance. A newly content-bound Executor release binds
the changed consumer bytes. Only a later brokered launch of the current checked asset
can close the separate TinyGEMM2 correctness partition.
