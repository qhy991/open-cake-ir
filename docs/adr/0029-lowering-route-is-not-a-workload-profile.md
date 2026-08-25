# ADR 0029: Lowering route is not a workload profile

Status: accepted in Compiler v24, 2026-08-25.

## Outcome and non-goals

Replace `metadata.profile`, a lookup whose thirteen use-case names simultaneously choose
a backend, entry symbol, unused ABI label, and workload-specific conformance function.
A Schedule instead carries one typed top-level `lowering` route with only `backend` and
`entry_point`. Generated backends decide whether they can emit the declared operations;
the emitter derives its actual argument signature from global Buffers. A Workload
Contract and its consuming Environment own external tensor meaning, cases, oracle, and
correctness.

This does not add Workload Contracts for synthetic Compiler corpus slices, claim that a
lowerable Schedule implements an unstated Workload, generate the remaining TinyGEMM2
asset, redesign empty ranking coverage, or preserve `profile` as a second writable
spelling.

## Minimal primitives and authorities

- `LoweringRoute {backend, entry_point}` is Schedule syntax. The backend vocabulary is
  `triton | cutlass_cute_dsl | checked_cuda_asset`; the entry point is a symbol, not an
  operator identity.
- Schedule operations, Buffers, AccessMaps, and Target contracts own program and hardware
  legality. Generated backend `preflight` owns emitter-only limitations.
- `metadata.workload_contract_sha256` is an optional opaque content binding. Compiler
  checks its spelling but does not resolve or interpret it. The Workload consumer checks
  tensor/case/oracle agreement.
- The generated-backend table has two rows and carries only module plus toolchain
  mechanism. New operators using those primitives do not add a row.
- A checked source asset is the bounded exception: its entry symbol selects immutable
  source custody, exact Schedule-semantics pin, and asset-specific preflight. TinyGEMM2's
  four-part CTA sum and bias-add/BF16-round epilogue live there because the retained body
  implements exactly those facts.

The old numeric ABI labels are removed. They had no consumer and one was already false
(`softmax_b8_smoke` declared four tensors while its Schedule has two). Generated source
and compile metadata already derive the real signature from ordered global Buffers.

## Dataflow and failure semantics

```text
opaque Workload binding ----> Workload consumer ----> oracle/case conformance
Schedule + Target ----------> generic verifier -----> accepted program
Schedule.lowering ----------> backend preflight ----> lowering eligible
                         `---> emitter/asset --------> exact source + symbol
```

An unknown Workload digest does not affect Compiler acceptance or lowering. An unsupported
backend spelling is structural failure. A generated backend missing an operation, dtype,
or emitter precondition yields a localized lowering-only Finding. A checked asset with an
unknown symbol or mismatched semantics is accepted as an IR program but not lowerable by
that route. Ranking remains unavailable while the released coverage domain is empty; a
backend name must never become profile-wide calibration coverage.

Historical Compiler revisions retain the old syntax in their archived closures. The new
Compiler has no `profile` alias: current Corpus Schedules, schema, Study templates, and
authoring checks migrate together.

## Smallest complete slice and evidence

1. All twelve generated Schedules route through two backend records; the thirteenth uses
   one checked-asset record. No operator-named lowering registry remains.
2. Changing the GEMM bias extent no longer blocks lowering merely because it is outside
   the named workload; generic access masking still determines program legality.
3. Operation/dtype/backend-preflight negatives remain failure-capable with localized
   paths.
4. TinyGEMM2 remains `generated=false`; three partials or the wrong epilogue formula are
   lowering-only asset-precondition failures.
5. The public schema rejects `metadata.profile`, and the Lab freezes/checks the exact
   lowering route instead.
6. Corpus expectation adoption is reviewed separately from the release gate; the full
   gate and contract suite pass under the successor Compiler/Executor.

The reviewed adoption changed all 32 Schedule digests, made the GEMM bias-extent variant
lowerable, and made the TinyGEMM2 reduction drift accepted but asset-unlowerable. The v24
release gate matched 32/32 cases. Frozen pre-v24 GPU observations and ranking plans remain
historical; no current evidence is inferred from their passed results.
