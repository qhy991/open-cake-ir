# DCU multi-region MMA: bounded device qualification

Shared change: [PR385](https://github.com/qhy991/open-cake-ir/pull/385), measured at
`bdcdf319274d0a6e232edfb5288b7ade96c29592`, merged into main as `1b5cca1b`.
The previous Compiler is `6ff0051249f4a399284774a35c2c23f5a3e0f90e`.
The platform integration preserves both ancestors and earlier DCU observations.

The retained `attention_decode/flash16` Schedule was accepted by the IR but refused
by the old multi-region Triton operand rule. It now lowers and executes with
loop-invariant computed weights and loaded/transposed V tiles. QK accumulates along
K; AV creates a fresh result per output-column tile. This is a capability
qualification. **The new candidate is slower than the existing incumbent and is not
promoted as a performance improvement.**

The same shared change exposes existing ProgramMap options through Python. Its
Python/JSON equivalence and combined persistent/MMA behavior have CPU coverage;
this replay uses the retained ordinary program map and does not establish a
persistent-map device speedup.

## Fixed contract and evidence

- Hardware: BW1100/gfx938, HCU1, fixed DTK image
  `sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260`.
- FP32 Q `[1024,256]`, K/V `[64,256]`, scale `0.0625`, noncausal attention and no dropout.
- All five original Task cases, original oracle, `atol=0.000298`, `rtol=2e-5`,
  input invariance and fresh contiguous nonaliasing output are retained.
- Evidence root:
  `/data3/testuser01/experiments/bw1100-compiler-multi-region-replay-20261008/run-bdcdf319`.
  `replay-manifest.json` binds generated source, workload and input bytes;
  `correctness-comparison.json`, `timing.json`, `timing-analysis.json`,
  `admissions/` and `profile/` retain the observations.
- Three Cake arms and two Torch SDPA references pass all 33 numerical/caller checks:
  25 original-case observations, five storage-rebinding/retained-output checks and
  three explicit-output checks. Output ABI is checked before flattening.

The old incumbent re-emitted by the new Compiler has byte-identical source and is
an A/A-style control, not a distinct performance implementation. The new flash16
source remains the original retained candidate; the owner did not retune it to
select a favorable result.

## Paired complete-call timing

Each comparison retains 30 AB samples, 30 BA samples and 30 A/A pairs in one
admission. Timing includes the public callable and device synchronization. The
minimum of the two median ratios is compared against the larger of 1% and observed
A/A drift. Inputs remain unchanged. No profiler duration is used as a score.

| Left / right | AB ratio | BA ratio | Conservative ratio | Decision |
|---|---:|---:|---:|---|
| Old-final / new-same | 1.00113 | 0.99536 | 0.99536 | Identical-source control, within noise |
| Old-final / new-flash16 | 0.94073 | 0.93749 | 0.93749 | New candidate regresses; retain old-final |
| Torch default SDPA / new-flash16 | 2.01260 | 2.25355 | 2.01260 | This image's default FP32 SDPA comparison only |
| Torch forced MATH / new-flash16 | 2.41926 | 2.70233 | 2.41926 | Includes per-call backend-context overhead; not a strongest-reference claim |

Old-final measures 79.36/79.34 us in AB/BA; new-flash16 measures 84.36/84.63 us.
Its latency is 6.30–6.67% higher. The paired old-final A/A drift is 0.353%, below the
1% floor. The Torch image reports memory-efficient attention is not compiled;
these references do not stand for every available attention library. The default
and forced-MATH A/A drifts are 4.153% and 3.248%, respectively. Do not compare a
candidate time across these different pairs as if they were one simultaneous sample.

## Executed-kernel counters and limits

Both rocprof captures validate all 50 target rows. Each row has 16,384 work items,
256 threads per CTA, wave size 64 and 256 dispatched waves.

| Reported per-dispatch value | Old-final | New-flash16 |
|---|---:|---:|
| LDS bytes / CTA | 65,536 | 20,480 |
| Architectural VGPR count | 104 | 68 |
| SGPR count | 32 | 32 |
| VALUInsts metric | 366 | 459 |
| SALUInsts metric | 36 | 58 |
| FETCH_SIZE, KiB | 1,156.875 | 1,156.3125 |

Lower LDS/register allocation did not lower whole-call latency. More reported
instruction work with similar fetch volume is a diagnostic lead, not proof that
one particular loop or resource caused the regression. The candidate also differs
in K tiling and pipeline depth, so this is not a single-parameter ablation.

The skill helper sums 50 wave observations then compares that sum with one grid;
its resulting wave mismatch warning is false. WRITE_SIZE was not requested and
is unknown, despite a displayed default zero. No peak bandwidth is assumed.
Earlier CPU-only compilation at `8af09545` reports different allocations from the
executed JIT dispatches. Those offline artifacts remain separate evidence; source
identity alone must not be used to substitute them for the executed specialization.

## Release and acceptance scope

Correctness `bw-66ec281b7f52`, timing `bw-4074046ac80f`, and profiling
`bw-98f11f721ce3` / `bw-73da6ea196f2` all end with exit0, stopped containers,
0% observed VRAM and no visible KFD use. The gateway serializes this user;
external GPU activity is not excluded and physical exclusivity is not claimed.

At the measured source, all 2,952 contract tests and 10,497 subtests pass, with
26 environment-dependent skips; Corpus204/204 and115 source snapshots pass.
Independent source and result reviews passed, and all five shared PR CI checks
passed. The prior failed software receipts and slower candidate remain retained.

Promote the bounded lowering capability and Python access to existing mapping
semantics. Keep the old operator incumbent. This is one-shape Compiler-development
replay, not an independent bw1100-bench score, same-budget author-search result,
general attention ranking or evidence of a new hardware instruction.
