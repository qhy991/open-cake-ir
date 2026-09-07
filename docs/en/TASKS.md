# Task implementations and the common Lab

[中文](../TASKS.md)

`lab/` owns Ralph iteration, budgets, candidate submission and common evidence replay.
`evaluation/` owns shared workload data types, tensor ABI, receipts, CUDA lifecycle and
measurement primitives. Neither imports a concrete task.

Operator-specific contracts, input materialization, oracles, preparation and diagnostics
live under `src/open_cake_ir/tasks/`: `qsa`, `flash_kmeans`, `tinygemm`, `tiles`, `dsa`, `kda`.
The three tensor tasks share the existing `tiles` implementation instead of duplicating it.
QSA Program execution and feedback belong to QSA. Fixed B/N/K/D portfolio specialization,
dispatch, assay and historical result interpretation belong to Flash-KMeans.

`tasks.runtime.TaskLab` binds the built-in task functions to the common Lab constructor.
`tasks.compose` is the live composition boundary used by the CLI. The engine receives
explicit workload loading, schedule preparation, task admission and launch-manifest parsing
functions; there is no plugin discovery or dynamic import configuration.

`tasks.workloads.load_workload` selects the unchanged exact task validators from one static
table. The common WorkloadContract checks shared structure; it does not select operators.
The Flash workload view resolves its fixed ABI. The common authoring environment derives
tensors and target from that same validated workload and constructs advisory selection
from the Study binding and its actual runtime context.

Frozen contracts remain their original semantic authority. Historical oracle identifiers
are not dynamically imported. QSA's source-bound oracle bytes are unchanged. Historical
release descriptors and evidence remain unmodified and replay with their original Git tree.

Use `python -m open_cake_ir.tasks.qsa.project_feedback compiler|evaluation` for QSA reports.
The old turn builder and `turn` subcommand are removed: only the Ralph controller derives
the next StateCard and TurnRequest. No old import path or CLI forwarding shim is retained.
