# Released state-store B200 correctness successor

This create-only bundle validates exactly one fixed `[8,128]` instance of the
released `state-store-b8-smoke.json` lowering. `prepare_candidate.py` loads
`compiler/revision.lock.json`, assesses the positive and two drift Schedules through
the public Compiler interface, and writes the positive generated Triton source without
editing it. The candidate-owned judge compares all 1,024 state elements in four
deterministic exact-FP32 domains, preserves complete before/after artifacts, checks the
empty-tuple wrapper ABI, verifies in-place pointer identity, and proves the update input
is bitwise unchanged.

The checked task is the create-only `v4` broker-result-readable successor to the
preserved `v1`, `v2`, and `v3` infrastructure-unknown runs. It uses the same four
workloads and acceptance boundary; only the task identity and isolated Kernel Infra
permission/deployment boundary differ.

The AKA parent is `contiguous_apply2_add_fp32_u32_block256_v1`, source record
`data_movement_and_layout__memory_addressing__analysis__l000001_b200_v1__directderived_sol_ultra_v2`.
It is a source-complete direct-derived contract with no upstream repository/revision
authority. This task claims only the fixed `[8,128]` instance, not arbitrary `n`, full
parent coverage, performance, or operator/model/serving/training qualification.
