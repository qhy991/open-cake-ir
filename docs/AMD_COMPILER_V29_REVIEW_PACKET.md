# AMD Compiler v29 independent-review packet

This document is a review map, not an approval. The canonical review surfaces remain
`compiler/revision.json`, `compiler/source_set.json`,
`compiler/corpus-gate-report.json` and `compiler/release-approval.json`. An independent
reviewer must inspect the exact checked-out bytes and write the approval artifact in
their own process; the release automation never writes it.

## Preconditions and current boundary

The candidate is `open-cake-ir-v29-draft`. Its retained Corpus Gate is
`awaiting_human_review` and matches 47 of 47 cases over 64 bound sources. The existing
approval remains the released mainline v28 approval for 37 cases and 53 sources, so it
does not authorize this Gate.

Before reviewing v29, verify the v28 identity incident is closed as recorded in
`inventory/COMPILER_V28_IDENTITY_INCIDENT_20260826.json`. The reservation-owned 35-case
release and the later RoPE 37-case release independently used
`open-cake-ir-sm100a-v28`. Both variants are frozen and neither may be rewritten. The
release cycle may advance past the occupied v28 archive only while that incident still
matches both exact variants; an unknown mismatch remains a hard failure. The generic
v29 identity is the first successor after the ambiguous v28 name.

## Gate delta from the reviewed mainline

There are no removed cases. Each of the prior 37 Gate case records is unchanged. The
ten added cases are:

| Case | Expected disposition | Required finding boundary |
| --- | --- | --- |
| `swiglu-gfx1151-accepted` | accepted and lowerable | `RESIDENCY_TARGET_UNMODELED` |
| `swiglu-gfx1151-instruction-unsupported` | rejected | `TARGET_INSTRUCTION_UNSUPPORTED`, plus unmodeled residency |
| `llama-rmsnorm-mul-gfx1151-r64-w4-accepted` | accepted and lowerable | `RESIDENCY_TARGET_UNMODELED` |
| `llama-rmsnorm-mul-gfx1151-r1-w8-accepted` | accepted and lowerable | `RESIDENCY_TARGET_UNMODELED` |
| `packed-q4_0-record-copy-gfx1151-accepted` | accepted and lowerable | `RESIDENCY_TARGET_UNMODELED` |
| `packed-q8_1-record-copy-gfx1151-accepted` | accepted and lowerable | `RESIDENCY_TARGET_UNMODELED` |
| `int8-byte-copy-gfx1151-accepted` | accepted and lowerable | `RESIDENCY_TARGET_UNMODELED` |
| `packed-q8_1-direct-dimension-gfx1151-unsupported` | accepted, lowering blocked | two distinct `TRITON_ARANGE_EXTENT_UNSUPPORTED` findings, plus unmodeled residency |
| `packed-q8_1-producer-gfx1151-accepted` | accepted and lowerable | `RESIDENCY_TARGET_UNMODELED` |
| `packed-q8_1-producer-gfx1151-record-count-drift` | rejected | `PACKED_BLOCK_STORE_RECORD_COUNT`, plus unmodeled residency |

The source set adds one exact Target definition and the ten Schedules above. No source
path is removed. Nine previously bound paths also change and require semantic review:

- `compiler/AUTHORING_CONTRACT.md` and `corpus/manifest.json`;
- `src/open_cake_ir/compiler/core.py`, `ir.py`, `schema.py`, `target.py` and
  `verifier.py`;
- `src/open_cake_ir/compiler/emit_triton.py` and `emit_cutedsl.py`.

## Reviewer checklist

1. Confirm gfx1151 is an exact Target with wave32 execution groups. Missing occupancy
   calibration must remain explicit as `RESIDENCY_TARGET_UNMODELED`; no CUDA or other
   target estimate may be inherited.
2. Confirm `program_tile` is an AccessMap-owned vectorization fact and the one-row
   RMSNorm Schedule remains rank-correct while lowering to a one-element row vector.
3. Confirm UINT8/INT8 storage, reshape, arithmetic/cast rounding and `xor_tree_32`
   rules are statically typed and have matching verifier and lowering coverage.
4. Confirm Q4_0/Q8_1 packed records describe raw storage and typed field encoding, not
   an opaque quantized dot operation. Record-count drift must block acceptance.
5. Confirm unsupported direct Q8 dimensions and unsupported gfx1151 instructions fail
   before lowering without target fallback.
6. Confirm the prior 37 case records have no disposition, finding, Schedule or lowering
   drift.
7. Confirm the v28 collision registration matches both frozen variants and that removing
   or weakening the registration makes the release cycle fail closed.

The existing Gate can be checked without submitting GPU work:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /path/to/python-3.10-or-newer \
  tools/release_compiler.py \
  --project-root . \
  --proposal compiler/revision.json \
  --source-set compiler/source_set.json \
  --output compiler/corpus-gate-report.json \
  --prepare-gate --verify
```

The reviewer should also run the repository's applicable CPU contract suite in a
declared environment. A missing test dependency or wrong Python version is an
environment failure, not a passing or failing Compiler result.

## Deliberate non-claims

This Gate proves deterministic Compiler construction, verification and lowering for
the listed corpus only. It does not prove gfx1151 GPU correctness, performance,
occupancy, a complete Q4_0/Q8_1 MMVQ consumer, llama.cpp integration or serving
performance. Formal RMSNorm timing remains separately gated by a released Compiler, an
exact gfx1151 Executor, a frozen Search Contract, external-oracle correctness and the
no-profiler noise and paired-timing protocol.

After completing this review, the independent reviewer either rejects the proposal or
writes `compiler/release-approval.json` so that it binds the exact Gate they reviewed.
This packet intentionally supplies no reviewer identity and no approval wording.
