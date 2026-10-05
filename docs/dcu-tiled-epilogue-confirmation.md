# Rounded tiled epilogue fusion on gfx938

The shared `fuse_tiled_epilogue` Program rewrite derives one Schedule from a
private tiled producer and matching pointwise consumer. The qualification below
uses a clean DCU successor, `a1ef4d20b3781db39005cc4f3010a17b5c38785a`; it does
not update Target facts or alter the frozen optimization runs. The API and
refusal domain are owned by [the fusion documentation](EPILOGUE_FUSION_PASS.md).

## Original Task confirmation

`L1/048_fused_gate_up_projection_with_swiglu` actually uses GELU-tanh. Its
retained `dualgemm-v1-aa` incumbent already has manual dual-GEMM/epilogue fusion.
The successor reconstructs four small-M two-stage Programs, then applies the
Compiler rewrite. The producer keeps both MMA operations and the original K
loop; the two explicit BF16 projection casts remain before the FP32 GELU and
multiply. Matching private stores/reloads disappear. M128, M131, M256 and M512
use the derived emissions. The ten other emission receipts retain their
original logical Schedules.

The original wrapper, SOL source/data lock, all 16 workloads, oracle,
tolerances, precision, strong FlagGems GELU baseline and fixed DTK image stay
unchanged. All original workloads passed ten correctness rounds: **160/160**.
Separate complete-call timing uses 30 samples in each order and a baseline A/A
control. Across all 16 cells, the geometric mean of the lesser forward/reverse
speedup is **2.15384×** against the community baseline; the minimum is
**1.37665×** and maximum relative A/A median drift is **0.22815%**. All measured
inputs remain unchanged.

These results establish reusable Compiler derivation and original Task
compatibility. They do not establish an extra performance gain over the earlier
manually fused incumbent. No model or serving claim follows. The per-user HCU
gateway provides serialized local admission; its receipts explicitly do not
prove physical exclusivity against external users.

## Actual kernel profiling

The independent M128 collection invokes the same unchanged emitted callable
with the original Task inputs. It bypasses the graph wrapper for attribution;
this collection supplies no complete-call or graph performance score. The
installed `dcu-rocprof-report-skill` CSV parser validates five target rows:

| observed field | value per target dispatch |
|---|---:|
| kernel | `_l1048_dualgemm_m128_bm128bn128bk64s1g4_kernel.kd` |
| grid / workgroup | 49152 / 256 work-items (192 CTAs) |
| wave width / dispatched waves | 64 / 768 |
| LDS | 16384 bytes |
| architectural VGPR / SGPR | 256 / 32 |
| scratch (`scr`) | 32 |
| VALUInsts / SALUInsts | 11447 / 838 |
| FETCH_SIZE | 296778.5 KB |

The first two filtered wrapper collections produced zero contexts and remain
retained. The valid third collection changed both filtering and dispatch scope;
it does not isolate why the earlier collections were empty. No WRITE_SIZE was
collected. No total-traffic, bandwidth-floor, exact-occupancy, dominant-bottleneck
or additional speedup claim is supported. VGPR and scratch are a resource lead,
not proof that fusion caused a regression or that the IR needs a new primitive.

## Evidence and custody

The owning evidence root is on `bw1100-1`:

`/data3/testuser01/experiments/bw1100-cake-tiled-fusion-gateup-confirmation-20261005`

- `REPLAY-SOURCE.json` binds the four pass-derived emissions and ten unchanged
  epilogues to the measured Compiler successor.
- `campaign/results/correctness.json` owns full original correctness.
- `campaign/results/community-paired-wall.json` owns raw timing;
  `TIMING-SUMMARY.json` is its derived interpretation.
- `profile/tiled-gateup-003/reports/metrics.csv` retains all ten collected rows;
  `analysis/validated.json` identifies the five target rows and actual resources.
- The terminal receipts are `fusion-correctness-001-admission-terminal.json`,
  `fusion-timing-001-admission-terminal.json` and
  `fusion-profile-00{1,2,3}-admission-terminal.json` under `.local/`.

All five admissions ended normally with exit zero, HCU2 VRAM at 0%, no task KFD
context and no remaining container. HCU0's unrelated load and the frozen
overnight queue were preserved. No new rolling optimization batch was started.

Software gates retain 193 DCU Corpus cases and 106 source snapshots unchanged.
The shared clean full suite passed 2659 contracts, with 26 skips and 9512
subtests. Independent Compiler review includes valid transposed-consumer,
role-control and public-intermediate refusal counterexamples. Integration uses
shared PR #310 to main followed by the DCU carrier; the measured pin remains
immutable across that branch integration.
