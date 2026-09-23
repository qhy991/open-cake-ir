# Python clean-start authoring

The supplied `schedule-starter.py` declares only the Workload's public tensor ABI,
exact target and lowering route. Its `...` body is intentionally incomplete. Write a
complete Cake Python Schedule by replacing that placeholder with your own roles,
coordinates, operations, storage and synchronization. Do not inspect or request an
existing low-level implementation of this target.

Submit authored implementations as `python_source` members in the frozen candidate
envelope. The envelope is transport; do not author a Schedule or Program as JSON.
The Lab owns compilation, GPU allocation, external correctness and performance
measurement. Use feedback from this Run only within its stated reference access,
tool permissions and budget. Do not infer correctness or a speedup from a static
assessment or one timing result.
