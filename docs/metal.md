# Apple Metal tasks through TaskLab

The Compiler supports exact `Apple M1 Pro` / `apple_gpu_family7` and `Apple M2` /
`apple_gpu_family8` targets. The task launcher admits both, as `--backend metal-m1-pro`
and `--backend metal-m2`. One backend selects exactly one target and one admitted device
name; a Workload frozen for one device never validates against the other. The launcher
checks the exact device, OS, toolchain and released Executor, and refuses a released
Executor bound to the other Apple GPU; there is no device fallback or borrowed Apple
performance calibration. Running on M2 therefore requires a released Metal Executor
captured on that M2 host, not the M1 Pro one.

The built-in tasks are `rmsnorm`, `layernorm` (affine, centered population variance),
and `residual_rmsnorm` (FP32-rounded residual addition before normalization). Their
[task-owned Workloads and oracles](../src/open_cake_ir/tasks/normalization/workload.py)
fix the rank-2 FP32 ABI, epsilon, input bounds, seeds and tolerances. Each Workload
covers one selected shape and five required input cases: primary, zeros, near-zero,
alternating signs and mixed magnitudes. Every case must pass before timing primary.
Different shapes are separate Workloads; a per-shape result establishes no portfolio
or framework claim.

## Launch one task

Use an existing Apple Silicon/macOS 15+ environment and a reviewed Compiler with a
passing full Corpus Gate. The current released Executor must be a Metal Executor whose
Python, Swift, SDK, device/OS and native helpers match the host. The launcher reports a
missing or mismatched prerequisite; it does not install, repair or release a runtime.

From the checkout, using the Executor's Python executable:

```sh
python3 tools/launch_task.py \
  --task rmsnorm --backend metal-m2 \
  --harness codex --model "<exact-model-id>" --effort high \
  --workspace "$HOME/.local/share/open-cake-ir/runs/metal-rmsnorm-example" \
  --rows 128 --columns 1024 --turns 4 --token-budget 150000
```

Replace the model placeholder with the exact configured model. For Claude Code, use
`--harness claude-code` and its exact model identifier and supported effort. The required
harness, model and effort are retained as treatment choices. No alias or fallback is
silently selected; Claude's native events must report the requested model.

`--workspace` must be a new absolute path outside **every** Git checkout, including a
parent repository. Its actor workspace is created once and retained through all Ralph
turns; the provider resumes the same session. Reusing an existing task root refuses
instead of resetting its state. `--provider-executable` selects an explicit CLI binary.
The launcher discovers `codex` or `claude` on PATH when that option is omitted. A known
Codex npm wrapper resolves to its own native executable; another installation is never substituted.

The broker and its worker use the Executor's Python with `-I` and an absolute
source bootstrap from the same checkout. Their imports do not require ambient
`PYTHONPATH`; inherited `PYTHONPATH` and `PYTHONHOME` cannot select another checkout.
The broker still execs its worker, retaining the admitted job and lock descriptor.

The [thin launcher](../tools/launch_task.py) writes the Workload, readable `starter.py`,
Study template and runtime bindings under that root. It then prepares a sealed baseline
through the common Open Cake environment, obtains or verifies live provider qualification,
runs `TaskLab.preflight`, saves `campaign-lock.json`, and calls the existing
[TaskLab composer](../src/open_cake_ir/tasks/compose.py). Ralph owns candidate filtering,
feedback, confirmation, token/time accounting and stopping.

To reuse existing authorities, supply `--fixed-baseline-bundle`, `--qualification` and
`--qualification-anchor` with their external paths. Preflight verifies their bindings.
`--preflight-only` stops after saving the Campaign Lock; it can still compile the baseline
and invoke provider qualification, so it is not an offline test option.

## Qualification and evaluation boundaries

The shared [provider qualifier](../tools/qualify_codex_provider.py) observes two actual
turns through the same immutable TASK.md/AGENTS.md package: add a Python-source candidate
envelope, then update it in the same workspace and session. It retains native events,
actual token usage and protected-file checks. Executable test doubles use
`--fixture-only`; those receipts cannot authorize a live task.

Common Evaluation receives a sealed Metal binary archive and explicit Workload launch
ABI. It reloads with a strict archive hit and never compiles candidate source. The
[local broker](../src/open_cake_ir/evaluation/local_broker.py) serializes this project's
jobs; it does not claim that other applications are absent from the GPU.

The [task Study policy](../src/open_cake_ir/tasks/normalization/study.py) declares ten
alternating candidate/baseline pairs, 25 samples per cohort after three warmups,
materiality ratio 1.05 and six required pair wins. These are engineering assay choices,
not target calibration or assumed speedups. The timer is the completed Metal
command-buffer interval; it is not pure kernel latency and does not inherit CUPTI/L2-flush
semantics.

`fixed_baseline_paired_metal_v2` declares two further values. `dispatches_per_sample`
is how many dispatches each timed command buffer encodes back to back, and a sample is
that buffer divided by them. Encode, submit and completion cost is paid once per buffer,
so a kernel shorter than that cost is otherwise measured mostly through it; the kernel
rewrites its whole output from unchanged inputs, so repeating it leaves the checked
buffers identical. `maximum_relative_iqr` gates cohort dispersion on the relative
interquartile range of the raw samples rather than their coefficient of variation, at the
same bound. Both describe spread; this assay states that other GPU clients are not
excluded, and an isolated disturbed sample should not veto a cohort whose bulk is stable.
The original `fixed_baseline_paired_metal_v1` keeps its exact single-dispatch, CV-gated
meaning, so Studies frozen against it replay unchanged. Profiling is a separate compute-stage timestamp observation. Missing native
profiling capability remains a refusal, and physical registers, spills, occupancy,
bandwidth and instruction counts are not inferred.

Static lowering uses 32-lane striped ownership, safe MSL 2.3 math and explicit launch
metadata. CPU semantics, native compilation, device correctness, stable timing and
framework acceptance are distinct evidence domains. No calibrated Apple cost ranker
exists; the pre-GPU filter retains that abstention.

Portable checks invoke neither a live provider nor a GPU:

```sh
PYTHONPATH=src python3 -m unittest \
  tests.contracts.test_normalization_tasks tests.contracts.test_task_launch \
  tests.contracts.test_metal_task_composition tests.contracts.test_harness_qualification
```

Generated-body checks use a C++ CPU adapter when available; qualification tests use
explicit executable fixtures. Actual Campaign qualification and performance still
require reviewed releases and device execution.
