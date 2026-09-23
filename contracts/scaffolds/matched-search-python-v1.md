# Clean-start source authoring

The Cake arm receives `schedule-starter.py` with only the Workload's public tensor
ABI, exact target and lowering route. Its `...` body is intentionally incomplete.
Write a complete Cake Python Schedule by replacing that placeholder with your own
roles, coordinates, operations, storage and synchronization. The direct low-level
arm receives its own CUDA skeleton and follows that treatment's syntax. Neither arm
may inspect or request an existing low-level implementation of this target.

Cake authors submit Python source in `python_source` members of the frozen candidate
envelope; direct low-level authors follow their assigned CUDA payload contract. The
envelope is transport, not a Schedule/Program authoring format.
The Lab owns compilation, GPU allocation, external correctness and performance
measurement. Use feedback from this Run only within its stated reference access,
tool permissions and budget. Do not infer correctness or a speedup from a static
assessment or one timing result.
