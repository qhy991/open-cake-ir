# Paired Studies bind provider transport per arm

Status: accepted for new Python Cake versus native Studies. No live provider or GPU
qualification is inferred from the CPU fixtures.

A matched Study keeps one model, reasoning effort, scaffold, tool policy, budget,
Workload, target, baseline and Evaluation contract. Its two authoring environments
need not write the same file format. Cake writes complete Schedules and Programs in
`candidate-set.py`; native Triton and CuTeDSL write the existing JSON envelope.
Treating the Provider's submission contract as a shared control would either force
Cake back to JSON or misidentify one arm's actual invocation.

Each arm therefore binds its own `ProviderQualificationReceipt` and evidence anchor.
The receipt's configuration identity includes that arm's submission contract, and
preflight and per-Run admission check the exact receipt against the exact provider
configuration. The existing two-turn qualification tool is run separately for each
arm under the same model and tool policy; a native JSON receipt cannot authorize the
Cake Python arm. External execution bindings version 3 name both receipt and anchor
paths by arm while keeping one runtime configuration and sealed fixed baseline.
Versions 1 and 2 retain their original meaning and refuse mixed transports.

The five NVIDIA Python Study successors are new treatments: B200 RMSNorm and B300
RMSNorm, GEMM+bias, gather and CuTe GEMM. They declare every Workload validation case.
The older static templates and Campaigns keep their original JSON transport for
pinned replay; they are not silently translated. A new Campaign still requires real
two-turn qualifications, target-specific Executor and toolchain admission, external
oracle correctness, paired on-device timing and profiler evidence.
