# ADR 0053: B300 is an exact target on the existing Triton path

Status: proposed; independent design review completed, release review pending.

The first B300 slice runs the existing Python Schedule/Triton path for RMSNorm,
GEMM+bias and indexed gather. B300 is compute capability 10.3 (`sm_103a`);
B200 is 10.0 (`sm_100a`). Architecture-specific code is not interchangeable.
The canonical hardware facts live in separate Target documents. Their consumers
share those documents rather than maintain separate device/resource tables.

No IR primitive, layout language, operator route table or Lab mode is added.
Python buffer indexing for loads now elaborates to the existing indexed-load
IR: rank-one INT32 register indices share one zipped shape, appear once in the
ordered read set, and participate in normal dependency/lifetime analysis.
Explicit destinations cannot bypass index validation; other indexed operations
remain refused by this frontend slice.
Independent review exposed a repeated tiled-coordinate mismatch in indexed
loads. Python and equivalent JSON/Triton preflight now refuse this pattern;
scalar program coordinates may still repeat. The impossible-reduction-axis
reproducer is retained as a separate negative Corpus case.
Workload v2 successors retain the three operators' mathematics, ABI, input domains,
15 cases and tolerances, and explicitly select B300 corpus seeds. Historical
Workloads, B200 Target bytes, releases, calibrations and evidence stay immutable.
Provenance binds the new B300 seed to its actual source commit and retains a
separate reference to the B200 seed from which it was adapted.

The target must agree across Workload, Schedule, lowering requirements, actual
Triton output, sealed Candidate, Tensor launch manifest, device admission and
loaded CUDA function. A mismatch is refused before module load or kernel launch.
The shared structural CUDA launch specification no longer requires a Tensor ABI
to impersonate the historical Flash ABI. That Flash/direct-CUDA path and the
portfolio study remain scoped to B200. CuTe and checked CUDA assets receive a
localized B300 refusal until separately supported and qualified.

B300 has no inherited occupancy, peak or calibration evidence. Missing facts
are reported; a persistent grid without an SM count is refused before emission.
Target-independent role-budget legality still runs when occupancy is unavailable.
The offline Triton jail decodes the exact architecture name without reading the
checkout. Runtime consumers read Target documents pinned by their Executor.

P1-P8 review: familiar authoring is unchanged (P1); target and lowering remain
visible, with absent model coverage explicit (P2); hardware facts and structural
launch checks have one owner (P3); mixed-target and unsupported-backend refusals
are static (P4); existing analyses receive the target facts they model (P5/P7);
focused crossed-target tests and the complete Corpus Gate gate release (P6);
NVIDIA's CC tables, PTX specification and separate B300 observations ground the
supported hardware domain (P8).

The historical Compiler/Executor id prefixes remain opaque release namespaces.
In particular, an `open-cake-ir-b200-vN` Executor id does not qualify B200 hardware:
its descriptor owns software and host files, while Target/Workload/admission own
the actual GPU. A B300 run needs its own verified host environment and exact
toolchain/device checks. CPU/source-only results are not GPU or performance proof.

Acceptance requires independent review of the exact successor and complete Gate,
then exact B300 compilation and external-oracle correctness for both paired
baseline paths across all 15 cases. Timing, profiler, provider qualification and
agent optimization comparisons remain separately labeled evidence.

Sources: [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus),
[CUDA feature targets and resource tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html),
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
