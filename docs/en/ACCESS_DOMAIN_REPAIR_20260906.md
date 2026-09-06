# Access-domain repair following the FMA re-audit

[中文原文](../ACCESS_DOMAIN_REPAIR_20260906.md) · [English home](README.md)

This historical repair first reproduced defects at e4cf49e from the v41 parent re-audit, then repaired the current Compiler. The twelve historical rows remain unchanged.

| Reproduced input | Before | After |
| --- | --- | --- |
| Scalar access declared as a 128-value register result | Accepted/lowered, possible hidden replication | LOAD_ACCESS_SHAPE_MISMATCH blocks acceptance |
| Nonexistent dimension 99 | IndexError | Localized ACCESS_DIMENSION_MISMATCH |
| Eight-element coordinate used on a four-element dimension | Accepted/lowered, out-of-range read possible | ACCESS_DIMENSION_COORDINATE_RANGE |
| Three-element Triton vector span | Accepted/lowered, compile failed later | TRITON_ARANGE_RANGE_UNSUPPORTED blocks lowering only |

An all-scalar load retains canonical [1]. A global dimension of nine may still use masked four-value tiles. Nonzero starts are checked by end-start, matching the inspected Triton 3.7.1 implementation. The original 72 Corpus expectations remain; one scalar positive and four negatives were added, with public-interface regressions for rank, references, masks, and nonzero starts. No splat, reshape, or runtime-scalar syntax was added.

The four fixed affine candidates were compared against their original Phase A mathematics and flattened addresses and remain lowerable. The rejected GroupNorm plan now gets a shape finding before generation. This is not GPU correctness or complete dynamic-ABI coverage. Old compilation metadata is historical; the projection cannot verify PTX/CUBIN bytes it does not carry.

## Replay the frozen Compiler

The projection verifier requires an explicit v41 source checkout and invokes that checkout's real CLI in a new process:

```bash
CAKE_V41_REPLAY=$(mktemp -d)
git worktree add --detach "$CAKE_V41_REPLAY/source" \
  d9d56e835cf96eecf70e0259b65bc1b20c4f6f0d
.venv/bin/python tools/verify_aka_fma_v41_reaudit.py \
  --data docs/data/aka-fma-v41-reaudit-20260906 \
  --compiler-root "$CAKE_V41_REPLAY/source" \
  --output "$CAKE_V41_REPLAY/verification.json"
```

The new v2 projection reports projection consistency and frozen-Compiler replay. Original v1 verification remains. Output is create-only, and summary or compilation metadata cannot upgrade GPU/performance flags.

## Observed verification

Preparation commit 1718383669ecace011bb51c935d04be531e1ab41 underwent actual offline target compilation on Verda/Triton 3.7.1. Four affine candidates and three controls—scalar load, nonzero start, and nine columns tiled by four—produced sm_100a PTX and ELF CUBIN. Existing ordinary/nested FMA rounding compile checks passed. These are new CPU compilation observations, not replacements for v41 artifacts or GPU launches.

The full Corpus matched 77/77. CPU regressions covered access, indexed writes, atomics, IR, and emitters. Two groups needing Torch used an existing server environment; no historical environment was repaired. Runtime FP32 scalar and splat remain separate design work.
