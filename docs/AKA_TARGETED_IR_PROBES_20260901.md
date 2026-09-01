# AKA targeted IR probes — 2026-09-01

Status: FP32 FMA numerical probe executable and passing; cosine contract still pending;
no FMA/cosine Compiler change or GPU run.

## Frozen review evidence

- AKA portable source: `d64639253fa101e2af5052ace5bbbb25deec57c8`.
- Original v4-minus-v2 pool: 257 selected, 233 completed, 24 failed.
- Reviewer-v4 recovery implementation: `df34c97e79080f5a51da668c4025a14ce4a5d1f4`.
- Recovery: 18/18 completed, bound to the original terminal `pool.final.json`.
- Recovery result: nine fixed-instance gaps, one fixed-instance lowerable and eight
  contract-scoped gaps; all remain `reviewer_claimed` and `gpu_test=not_run`.

The review records have no upstream repository/revision authority and retain material
unknowns from direct-derived parent completion. They identify pressure and define probes;
they are not source-independent semantic or correctness evidence.

## Decision table

| Candidate | Current decision | Evidence | Missing before IR admission |
| --- | --- | --- | --- |
| single-writer state store | successor proposal prepared | fixed `case-000524`; independent pressure in `000367`, `000517`, `000539`, `000540`, `000568` | external approval and independent target correctness |
| ternary FP32 FMA | numerical probe passed; IR admission pending | fixed gap `case-000557`; related pressure in `000375`, `000376`, `000556`; fused/separate result differs by 1 ULP | independent source provenance and target correctness |
| unary FP32 cosine | targeted probe admitted | `case-000172`, `case-000453` from distinct source paths | fixed-instance review, one accuracy/instruction contract, fixed oracle |
| generic runtime shape/index/control | rejected as one probe | bundled across many heterogeneous gaps | successor system ADR and separately isolated primitives |

## Probe F1 — ternary FP32 FMA

### Goal

Decide whether one closed ternary register operation is irreducible relative to the
current `mul` plus `add` composition.

### Workload contract

- three contiguous FP32 inputs `a`, `b`, `c`, each shape `[1024]`;
- one contiguous FP32 output `y`, shape `[1024]`;
- `y[i]` is one IEEE binary32 fused multiply-add of `a[i]`, `b[i]`, `c[i]`;
- fixed shape and launch; no runtime scalar, broadcast, state, alias, implicit cast or
  dynamic extent;
- NaN, infinity, signed zero and subnormal policy must be stated before admission;
- the oracle contains at least one finite bit pattern where fused evaluation differs from
  separately rounded multiplication and addition.

### Required evidence

1. source-complete examples from two independent repositories require the same FP32
   single-round operation;
2. a current-IR `mul` + `add` near miss fails the distinguishing oracle input;
3. the proposed lowering names an exact backend instruction/accuracy contract;
4. positive and instruction-drift Corpus cases are reviewed independently;
5. complete-output target correctness precedes any coverage claim.

The FP16-result case `000199`, state-update case `000489`, runtime/broadcast cases and
ordered GEMM accumulation are excluded from F1; they require separate cast/effect/domain
owners.

### Executable result

`tools/probe_fma_fp32.py` performs exact rational arithmetic and IEEE binary32
round-to-nearest ties-to-even, without relying on host `fma` behavior. The frozen finite
input is:

| Operand | FP32 bits | Value |
| --- | --- | ---: |
| `a` | `0xc0ce69db` | -6.4504218101501465 |
| `b` | `0xc022ec9e` | -2.545691967010498 |
| `c` | `0xc07e0807` | -3.9692399501800537 |

It produces:

```text
single-round FMA: 0x41473989
FP32 mul + add:   0x4147398a
```

The 1-ULP distinction proves that FMA is not an equivalent spelling of the admitted
`mul`/`add` composition. Triton's public API has a dedicated
[`triton.language.fma`](https://triton-lang.org/main/python-api/generated/triton.language.fma.html)
operation, so a backend body exists in principle; target code generation and numerical
correctness remain unverified.

## Probe C1 — unary FP32 cosine

### Goal

Determine whether `cosine` can be one closed unary elementwise member without importing a
general math-expression language.

### Workload contract

- one contiguous FP32 input and output, each shape `[1024]`;
- fixed shape and launch; no derivative multiply, runtime extent or scalar parameter;
- the operation is unary cosine only;
- the accuracy contract must choose one backend realization explicitly rather than leave
  accurate-versus-approximate behavior to lowering;
- the oracle covers representative small/large magnitudes, multiples near pi/2, signed
  zero, infinities and NaNs with a preregistered tolerance/bit policy.

### Required evidence

1. re-review `case-000172` and `case-000453` as fixed instances and separate cosine from
   runtime shape and derivative arithmetic;
2. establish that their required numerical contracts match;
3. add one positive and one instruction/accuracy near miss only after that match;
4. verify backend/toolchain support and complete-output target correctness.

If the two sources require different numerical contracts, they remain two proposals or no
proposal; an unqualified `cos` token is rejected.

Triton's public API exposes
[`triton.language.cos`](https://triton-lang.org/main/python-api/generated/triton.language.cos.html),
but that API page states only elementwise cosine and does not define an accuracy or target
instruction contract. Backend surface availability therefore does not close C1.

The current CUDA Programming Guide maps regular FP32 `cos` to `cosf` with a reported
maximum error of 2 ULP, while `--use_fast_math` translates it to the distinct `__cosf`
intrinsic with a range-dependent approximation contract. See the
[CUDA mathematical-functions appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).
Neither behavior can be attributed to `tl.cos` from its public API alone. C1 therefore
stops before IR admission until generated target code and a preregistered accuracy policy
identify which contract the backend actually implements.

## Stop conditions

- Do not add FMA or cosine to `ElementwiseOp` from lexical frequency or model prose.
- Do not combine either probe with runtime shape, scalar-backed valid extents, state-store
  effects, broadcasting, dtype expansion or Program composition.
- Do not modify `compiler/release-approval.json`.
- A static Compiler Gate is not GPU correctness or performance evidence.
