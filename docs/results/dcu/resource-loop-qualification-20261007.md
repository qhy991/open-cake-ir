# DCU compiled-resource feedback and explicit store-loop depth

This successor records two bounded improvements from the completed development batch:
resource observations reach existing author feedback, and an existing loop-depth choice
can be generated through a guarded Compiler action. Neither is a performance or
same-budget search result. Frozen experiments remain pinned to their original source.

## Resource feedback

Implementation: `982755aa` plus `ccaec6c6` (scope the explanation to bound observations),
shared PR [374](https://github.com/qhy991/open-cake-ir/pull/374).
Evidence root: `/data3/testuser01/experiments/bw1100-hsaco-resource-feedback-20261007/run`.
The actual DTK CPU-only compile uses the original RMSNorm `starter-base` and
`rows128-g16` Schedule documents. No device kernel is launched.

| Candidate | VGPR per lane | Private bytes per lane | Fixed LDS bytes | Dynamic launch LDS bytes |
|---|---:|---:|---:|---:|
| starter | 51 | 0 | 0 | 0 |
| large row tile | 192 | 420 | 0 | 8192 |

The toolchain is Triton `3.6.0+dtk2604.torch2110.2608271608.g5a3e96`, target `gfx938`.
The new compilation is not the old profiler's binary; its private allocation must not be
relabeled to match a historical counter. Separate stack bytes are unknown (`null`).
`results/compiled-r1/report.json` and its `0000`/`0001` source, AMDGCN and HSACO artifacts
own the observations. `results/profile-r3.json` replays those artifacts, and
`results/author-feedback-r3.json` contains the same bound allocation in the existing
`project_feedback.py compiler` output. The last code change only restores the original
static explanation for targets without a compiled observation; it changes no compilation,
record schema, parser, identity check or resource number.

The first equality check between static LDS and launch shared allocation was disproved
by the actual `0/8192` result. Its failed `results/compiled` and `results-compile.log`
remain retained. Static and dynamic shared values are independent and are summed.
The CUDA register-capacity formula also produced zero CTAs for the retained HCU kernel;
HSACO bank/wave allocation is unmodeled, so VGPR counts remain observations without a
register-based CTA bound. Thread/shared capacity upper bounds remain, never achieved
occupancy or calibrated latency. Private bytes are not measured dynamic spill traffic.

## Store-loop action

Implementation: `166f82a1`; final `3589b069` adds only a negative contract test,
shared PR [375](https://github.com/qhy991/open-cake-ir/pull/375).
Evidence root: `/data3/testuser01/experiments/bw1100-store-loop-qualification-20261007/run`.
The original RMSNorm `colloop256-g1` and LayerNorm `colloop-t512` Workloads and Schedules
are retained in `fixtures/`. The action changes only the selected output-loop depth
and result identity: RMSNorm depth 4, LayerNorm depth 2. Earlier sum loops, arithmetic,
access maps, dtype, precision and tolerance remain unchanged.

`results/qualification/manifest.json` binds four emissions: each original Schedule and
its action-generated counterpart. All four pass their five original cases, for **20/20
candidate-case checks**. Maximum absolute error is `9.5367431640625e-7` in each group;
all inputs remain unchanged. `kit/compare.py` loads the original `d0d0acab` comparator.
No timing samples were collected. `summary.json` owns the result; the admission and
terminal records own job `bw-22acbddf9500` on HCU5, exit0, VRAM0%, no visible KFD context
and no running container after release. The existing gateway remains the per-user
serialization owner and does not exclude unrelated users' physical activity.

## Review and integration scope

Independent source review covered both actions and the corrected resource boundaries.
The shared Corpus has197 cases and109 retained lowering snapshots. The whole-Program
consumer and source-selection negative cases are exercised in the contract suites.
The source and actual toolchain/device evidence above are qualification of the named
mechanisms; PR checks own full integration status.

Promote only these capabilities. Do not set a default stage count, a register threshold,
a calibrated cost model, an automatic transpose policy, or a new native instruction from
these observations. Independent Bench comparisons and equal-budget new searches require
new frozen runs with their own confirmation; the old Bench confirmation failure remains
an independent protocol issue.
