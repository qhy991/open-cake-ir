# Inspect candidate MSL as AIR

Use only when the task grants the compiler, source and output paths. Read the
candidate-bound MSL first; select the operation/region whose lowering would test
your hypothesis. A compilation failure is retained evidence, not permission to
change the host toolchain or the frozen task environment.

With the approved Xcode 16 macOS toolchain, this invocation emitted readable
AIR/LLVM IR from an already generated RMSNorm source:

```sh
"$TASK_METAL_COMPILER" -S -emit-llvm -std=macos-metal2.3 -fno-fast-math \
  "$TASK_CANDIDATE_MSL" -o "$TASK_AIR_OUTPUT"
```

These variables denote exact task-granted paths, and the output must be a new
per-attempt file. The task also fixes child-process DEVELOPER_DIR/SDKROOT; do not
change global xcode-select. `metal2.3` alone was refused by this toolchain: the
tested platform spelling is `macos-metal2.3`. Other toolchains need their own
declared invocation; this recipe does not select a fallback architecture.

Retain the source identity, command, toolchain facts, stderr and IR. Trace memory
accesses, reduction intrinsics and synchronization back to the Cake region. An
AIR `alloca` describes intermediate private storage, not a physical register
allocation or proof of a spill. An intrinsic name is evidence of this offline
lowering stage, not proof of the machine instruction in the runtime archive.

Report offline AIR separately from MTLDevice runtime compilation, the sealed
archive and instrumented device observations. Different compilation routes may
produce different code. Qualified timing and oracle checks decide performance;
readable IR helps select the next falsifiable hypothesis.
