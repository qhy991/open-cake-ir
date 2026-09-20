# Cross-hardware transfer of executable optimization knowledge

[Technical report](../README.md) · [中文](../OPTIMIZATION_TRANSFER.md) · [Ablation protocol](../OPTIMIZATION_TRANSFER_ABLATION.md)

We propose **optimization mechanisms with explicit applicability conditions** as the unit
of cross-hardware knowledge transfer. An agent explores operator fusion, tiling and memory
hierarchy optimization on a source platform. Deterministic semantic rewrites become explicit
Compiler passes; when to apply them and how to choose parameters remain Lab decisions.
Destination Targets and backends determine realizability, and destination measurements
determine correctness and benefit.

The research contribution is the loop from **experience discovery to executable rewrites,
destination validation and feedback**. Experience can inform agent reasoning or provide a
callable optimization tool. Rejections and negative transfer constrain subsequent reuse.
The question is whether this accumulation reduces search cost on new hardware, and how much
of the effect comes from extra explanations versus callable transformations.

![Source experience reaches the destination agent through separately controlled explanations E and callable rewrites P; destination validation remains mandatory](../figures/optimization-knowledge-transfer.svg)

*E/P are separately allocated experimental factors over a shared Run execution path.
The diagram contains no measured transfer results.*

A shared IR rewrite is reused when destination lowering supports it. Hardware-specific
instructions, storage and synchronization remain backend responsibilities. For example,
fusing Add and SiLU may eliminate a private BF16 intermediate's global store/reload while
preserving its rounding boundary. Tile sizes and execution groups are selected again on
the destination; a source speedup is not a destination result.

The proposed 2×2 study controls **E**, extra mechanism explanations and cases, and **P**,
permission to invoke a frozen transformation. E0P0 supplies neither; E1P0 supplies explanations
for manual rewriting; E0P1 supplies only the callable interface and its necessary contract;
E1P1 supplies both. P1 inherently carries some knowledge, so E estimates the incremental
value of explanatory material, not the presence of all knowledge. All groups share the same
base IR/lowering capabilities, oracle, numerical constraints and measurement gates.

The primary outcome is qualified optimization success within a matched budget. Secondary
outcomes include search cost and destination-baseline-relative performance. Discovery and
adaptation cost are reported separately. The [method appendix](../OPTIMIZATION_TRANSFER_ABLATION.md)
owns assignment, isolation, estimands and failure handling; these are proposed experiments,
not results of existing artifact-only campaigns.

Existing [bounded passes](../../src/open_cake_ir/compiler/passes.py) and
[experience-material preparation](../OMOE_TRANSFER.md) provide implementation foundations.
The prototype now connects complete Programs, independent Runs, controlled material/pass
access, preassigned E/P Studies and audited analysis. Software tests exercise these protocol
paths; production orchestration, target adaptations and measured transfer benefits remain
to be qualified. Automatic mechanism extraction is outside the current implementation.
The design builds on explicit transformations such as
[MLIR Transform](https://mlir.llvm.org/docs/Tutorials/transform/) and prior schedule reuse such
as [Transfer-Tuning](https://arxiv.org/abs/2201.05587v2), focusing on agent-produced experience,
destination qualification and controlled attribution of its effects.
