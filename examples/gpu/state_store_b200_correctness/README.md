# Released state-store B200 correctness successor

This create-only v5 bundle prepares exactly one fixed `[8,128]` instance of the
state-store lowering from released Compiler v41. GPU validation is pending. The four
workloads, numerical judge and acceptance boundary are unchanged from historical v4.

`prepare_candidate.py` requires a Git checkout containing Compiler source commit
`ee32b76870c4166c8143e8774c37c1212a0b6e1c`. It compares the current released lock with
that commit's lock, checks that the executing Compiler comes from this checkout, and
uses `Compiler.load` to verify the complete released source closure. It then assesses
the positive and two drift Schedules through the public Compiler interface and writes
the generated Triton source without editing it. A different release or source closure
is refused before creating the output directory; preparing another Compiler requires
an explicit task successor. `candidate/provenance.json` records the verified
`compiler_source_commit`.

From this checkout, prepare an unused output directory outside the repository:

```bash
python3 examples/gpu/state_store_b200_correctness/prepare_candidate.py \
  --output-root /tmp/open-cake-state-store-v5-preflight
```

Preparation uses no provider, GPU or daemon. The task and catalog name isolated v5
deployment paths; their presence does not establish that a runtime or broker has been
deployed there. The bundle's static preflight validates the generated wrapper; actual
B200 correctness remains pending.

The candidate-owned judge compares all 1,024 state elements in four deterministic
exact-FP32 domains, preserves complete before/after artifacts, checks the empty-tuple
wrapper ABI, verifies in-place pointer identity, and proves the update input is bitwise
unchanged. The AKA parent is `contiguous_apply2_add_fp32_u32_block256_v1`, source record
`data_movement_and_layout__memory_addressing__analysis__l000001_b200_v1__directderived_sol_ultra_v2`.
It is a source-complete direct-derived contract with no upstream repository/revision
authority. This task claims only the fixed `[8,128]` instance, not arbitrary `n`, full
parent coverage, performance, or operator/model/serving/training qualification.

Historical v1-v4 tasks and their outcomes remain recorded in
[the correctness report](../../../docs/STATE_STORE_B200_CORRECTNESS_20260902.md).
Replay v4 using its complete Git checkout at
`82491bb7658a4da92be444fa029b26b02635a154`, including that checkout's Compiler code and
released v40 lock. The current v5 task does not inherit v4's GPU result.
