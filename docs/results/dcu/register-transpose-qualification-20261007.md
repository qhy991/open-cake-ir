# gfx938 register-transpose qualification

The new value operation maps a rank-two register tile from [M,N] to [N,M] without
arithmetic or dtype conversion. It makes the existing A[M,K] × B[N,K] MMA contract
usable with a correctly loaded global B[K,N] tile. It is not a new hardware instruction,
a general layout algebra, or an automatic decision to use MMA.

Implementation `0e62a1fdf119d4f5cb428dc18c8174f684a137e6` is qualified below. The successor
`c79de020` adds only the canonical primitive reference. Shared PR377 owns software
integration status; F-2026-10-07-005 owns the originating capability observation.

Evidence root: `/data3/testuser01/experiments/bw1100-register-transpose-qualification-20261007/run`.
The original input/oracle/comparator owner remains Compiler `d0d0acab` for the two Tasks.
The existing DTK image and gateway are retained in each admission receipt.

## Component and Task coverage

`results/qualification-r1/summary.json` contains **26/26 passed checks**:

- Sixteen bit-exact transpose copies: FP32, FP16, BF16 and INT32 at shapes1×1,
  17×33,35×19 and64×128. Floating patterns include signed zero, infinity, quiet NaN,
  subnormal and ordinary values. Outputs are checked by their integer bit views;
  this is a pure copy contract, not extension of a finite-input numerical Task.
- The original GEMM and GEMM+SiLU Workloads, each R1024/K256/N64, pass all five
  original cases. Inputs remain unchanged; outputs retain shape/dtype/contiguity
  and fresh storage. Each Task's maximum absolute error is6.103515625e-5 under
  its original tolerance. Operands and accumulation stay FP32 with the existing
  `triton.dot.fp32_ieee` contract; no TF32 or FP16 narrowing was introduced.

`results/tail-regression/summary.json` records a separate exact-integer FP32 component:
M35/N19/K65 with16×32×64 tiles and a partial final K tile. All665 outputs equal the
independent integer-product/sum reference exactly, and inputs remain unchanged. This
component does not add another original Task or justify arbitrary-shape performance.

The initial CPU preparation failed while Torch tried to serialize differently typed
views sharing one storage. The successor gives the saved input an independent storage;
original `results/qualification` remains retained. No GPU request was made for that
failed preparation, and no oracle, expected bits or tolerance was changed.

## Actual execution and source evidence

- `results/qualification-r1/manifest.json` binds18 emissions:16 copy sources and
  two Task sources. Candidate/typed Schedule/generated source files are retained
  under the same directory. `kit-r1/prepare.py` invokes the original oracle in a
  separate process before importing the successor Compiler.
- `kit-r1/device.py` executes only retained emissions and saves observations;
  `kit-r1/compare.py` checks bit copies and the original Task comparator after release.
- Job `bw-37233ef178b2` on HCU5 completed with exit0, after_vram0%, no visible KFD
  context and no running container. The tail job `bw-436b28759d8b` has the same
  release result. Receipts preserve the actual image and runtime.
- `results/qualification-r1/isa-observation.json` links actual JIT artifacts to
  their TTIR source locations. Both Task artifacts contain `v_mmac_16x16x8_f32`.
  GEMM+SiLU also has scalar epilogue instructions. This proves instruction presence
  for these executed artifacts, not throughput or a speedup.

## Software and promotion boundaries

Typing requires one rank-two register input/result, reversed shape, identical supported
dtype and empty parameters. Contraction and argmin domain analyses trace the same axis
swap through unique valid cast/transpose chains. Malformed/cyclic/ambiguous chains provide
no proof. The existing nested-MMA direct-load restriction remains a refusal. No other
Target receives a transpose declaration in this change.

Software verification includes non-square/tail copies, double transpose, exact K-loop
MMA reference execution, argmin provenance and negative type/effect cases. The Corpus
contains199 cases and111 lowering snapshots; full CI results belong to PR377 and this
platform integration. Promote this bounded capability only. No timing, community-baseline
or equal-budget search claim is made; the next search must use a new frozen Compiler.
