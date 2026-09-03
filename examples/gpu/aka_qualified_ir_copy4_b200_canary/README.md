# AKA qualified IR B200 canary

This create-only v6 package validates one fixed `n=1024` Cake lowering selected by the
AKA qualified-parent Phase A ledger. `prepare.py` copies the unedited generated Triton
source from the external admission root and binds a correctness plus memcheck/racecheck
task. The candidate-owned judge uses four deterministic input distributions and checks
all 1,024 output elements bitwise, the read-only input, supplied-output ABI, disjoint
pointers, B200 identity, and complete-output artifacts.

This is an execution-path canary. It makes no arbitrary-shape, complete-parent,
performance, release, training, operator, model, or serving claim. Five-way controller
concurrency is authorized only after this exact task completes with valid correctness,
memcheck, racecheck, and collectable artifacts.

The preserved v1 and v2 tasks stopped before GPU custody because their candidate and
judge paths remained below a non-traversable home/transfer parent. V3 moved node state
and the immutable candidate under `/tmp`, reached broker custody, and then exposed the
same issue for GPU Infra's own `exec_guard.py` source path. V4 places both the exact
GPU Infra deployment and node state under a fresh `/tmp` root owned by
`qhy-sol:gpuq-users`, then stops at an unavailable NumPy import in the shared runtime.
V5 uses PyTorch alone for deterministic inputs and bitwise comparison. It does not
modify or retry an earlier run identity, but its task generator resolves the virtualenv
entry symlink to the system Python and therefore loses Torch. V6 preserves the explicit
virtualenv entry path while checking that it is executable.
