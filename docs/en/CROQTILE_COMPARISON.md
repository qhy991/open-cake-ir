# Croqtile and Open-Cake: engineering scope and research position

[Technical report](../README.md) · [Detailed review and pinned sources](../CROQTILE_COMPARISON.md)

This 2026-09-21 review inspected public documentation, source and test definitions, without
building Croqtile or reproducing performance/agent-accuracy claims. Snapshots: compiler
`8ff94800498b450ad14f27dadd798ec19c96a0dd`, tutorial
`81d42df73dd80929c7e363015dbcffdfeada21ac`, tuner
`f3d590602ea64d373d5fae6420266783beaa1198`; Open-Cake `dbc67382`.

Croqtile includes a GPU language/compiler and a separate agent tuning system. Both projects
combine structured programs, compiler feedback and device measurement. Agent-driven DSL
authoring or static diagnostics alone are not unique Open-Cake contributions.

| Dimension | Croqtile / Tuner, as inspected | Open-Cake, with current limits |
| --- | --- | --- |
| Representation | C++-embedded `.co`, shaped data and scheduling constructs; also a Python builder and MLIR CoIR | Restricted Python/JSON Schedule, composed Program, separate mathematical Workload and oracle |
| Shapes | Symbolic dimensions and inference, with residual runtime assertions | Concrete case ABIs and sealed launches; loop/index expressibility is not arbitrary dynamic-shape binary support |
| Scheduling | DMA/TMA, parallel roles, events and multi-buffering; vectorization, memory reuse, layout and fence passes | Explicit storage/access/synchronization commitments, guarded rewrites and backend optimizations; no first-class layout algebra |
| Targets | CUDA/CuTe, CoIR-to-PTX, HIP, CPU and heterogeneous registrations/implementations | NVIDIA, Apple, AMD, Hygon and MetaX paths with distinct device/measurement coverage |
| Agent control | Skill workflow and TypeScript/Pi SDK v2, controller-side build/measurement, snapshots and trajectories | Frozen Run permissions and evaluation, Study assignments, independent confirmation and audit |
| Author tools | v2 exposes read/write/bash to the author session | Restricted candidate Runs freeze tool/write permissions and delegate build/evaluation to the controller; comparisons must account for this difference |
| Knowledge | DSL-skill injection, examples, round memory and mock harness comparisons | Proposed separately controlled explanation E and callable-transform P study; transfer benefits remain unverified |

The source is broader than the tutorial's CUDA-only description. Calling Croqtile “C++ only”,
“without an IR”, or “without a harness” would also be incorrect. Registered backends and test
files do not establish device qualification in this review. See the fixed
[compiler target code](https://github.com/LancerLab/croqtile/tree/8ff94800498b450ad14f27dadd798ec19c96a0dd/lib/Target),
[Python interface](https://github.com/LancerLab/croqtile/blob/8ff94800498b450ad14f27dadd798ec19c96a0dd/tools/co-py/src/croqtile/__init__.py),
[CoIR](https://github.com/LancerLab/croqtile/blob/8ff94800498b450ad14f27dadd798ec19c96a0dd/tools/coir/README.md)
and [Tuner v2](https://github.com/LancerLab/croqtile-tuner/blob/f3d590602ea64d373d5fae6420266783beaa1198/v2/README.md).

In the inspected v2, KEEP/REJECT uses parsed TFLOPS and a fixed 0.995 relative tolerance;
its source notes that noise calibration is pending. Correctness and timing guarantees depend
on the configured commands. The loop has no separate terminal confirmation phase equivalent
to Open-Cake's, but this does not imply that every Croqtile benchmark lacks validation.
A specific static risk is that a failed baseline build still proceeds to measurement: a command
that can execute an old binary could supply a stale baseline. This path has not been reproduced
here and does not invalidate all existing results. Relevant implementation:
[tuner.ts](https://github.com/LancerLab/croqtile-tuner/blob/f3d590602ea64d373d5fae6420266783beaa1198/v2/src/tuner.ts),
[measure.ts](https://github.com/LancerLab/croqtile-tuner/blob/f3d590602ea64d373d5fae6420266783beaa1198/v2/src/measure.ts),
[decide.ts](https://github.com/LancerLab/croqtile-tuner/blob/f3d590602ea64d373d5fae6420266783beaa1198/v2/src/decide.ts).

Croqtile is a useful reference for compact scheduling syntax and symbolic-shape diagnostics.
Open-Cake should evaluate improvements through real author failures and repair/search cost,
while preserving its existing frontend, analysis and acceptance owners. A fair future comparison
must separate language representation under a common harness from the native tuning stacks;
freeze supported hardware, workload, oracle, precision, baseline, timing, model, permissions,
materials and budgets before measuring. Neither public H800 tutorial numbers nor local
B300/C550 observations establish comparative superiority. No comparison was launched or
material injected into active clean-start/E0 runs during this review.
