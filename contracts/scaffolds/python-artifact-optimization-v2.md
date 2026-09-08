# Python artifact optimization — scaffold v2

Use the supplied Workload Contract and Python starter to propose structurally distinct
Schedules for the declared exact target and shape. Preserve Workload metadata, argument
order, tensor shapes, FP32 numerical semantics and output ownership. Author Python only;
submit each source as a `python_source` member of the candidate-set envelope named by the
StateCard. Keep the starter's ABI and target binding intact.

The current Compiler Metal route enforces these authoring limits:

- Keep one role with `warps=[0]` (`METAL_ROLE_UNSUPPORTED`).
- Use scalar program indices with `tile=1` (`METAL_PROGRAM_TILE_UNSUPPORTED`).
- Access tensors through scalar program indices and finite contiguous dimension slices
  (`METAL_ACCESS_UNSUPPORTED`). Preserve every varying program index in output stores.
- Set stores to `coalesced=False` (`METAL_COALESCING_UNSUPPORTED`).
- Keep the Workload's FP32 buffers and finite FP32 scalar literals. Leave CUDA/PTX
  instruction contracts unset (`METAL_INSTRUCTION_UNSUPPORTED`).
- Put scalar literals on the right of frontend binary expressions, as in `x * 2.0`;
  preserve operand order and the Workload's numerical contract.

These are limits of this Compiler route, not physical Apple GPU maxima. The
[Metal guide](../../docs/metal.md), [Python frontend](../../docs/PYTHON_FRONTEND.md), and
localized Compiler Findings explain the supported representation. The bound Compiler
Revision and its diagnostics remain authoritative.

Explore distinct dataflow, equivalent arithmetic formulations, load reuse and lifetimes
of private intermediates within this route. Read localized Compiler and Evaluation
feedback before the next turn. The common Lab owns compilation, checks across every
input case, timing, profiling and stopping. Do not invoke those tools, a GPU, the network
or another task. Change only the named candidate envelope; keep TASK.md and AGENTS.md
unchanged. Do not author serialized Schedule JSON or insert a low-level baseline.

This v2 scaffold is a separate authoring treatment from v1. Keep its Runs separate;
do not pool, relabel or extend earlier evidence. A locally confirmed artifact establishes
no arm-comparison or cross-shape claim.
