# Main tasks on Metal: development inputs, not the independent Bench

`main@0061770e` is already an ancestor of `metal@0a9ebb4c`. Its registered task
implementations are present in Metal without a second copy. The preparation command
projects that existing catalog onto exact M4 starter shapes; it owns neither task
semantics nor a new launch path.

```sh
python tools/metal/prepare_tasks.py --output /absolute/external/development-defaults
python tools/metal/prepare_tasks.py --output /absolute/external/development-contractions \
  --task gemm --task gemm_silu --task pairwise_sqdist --task attention_decode \
  --task gemm_bias --task aka_gemm_nt_bias --rows 128 --columns 32 --depth 256
```

Each task has its own `workload.json`, `starter.py`, available `starter.metal`,
`assessment.json` and task notes. Refused tasks remain in the inventory. Output must
be outside Git, new, and prepared from a clean committed checkout. No provider,
native compilation or GPU call is made. The canonical `tools/launch_task.py` still
owns admission and execution. Use fresh external run directories, exact provider
qualification, `--backend metal-m4`, `--wall-seconds 10800`, and no `--token-budget`.
Private author environments and actual read boundaries are separate requirements;
these folders do not claim OS read isolation.

At the audited `00381d7e` source, all 55 registered tasks were retained: 25 default
starters emitted MSL, six were refused by the Metal private-storage bound, and 24
were refused by their original dtype/operation/backend constraints. The six explicitly
smaller contraction development cases all emitted MSL. These are 31 task names with
CPU lowering evidence, **not 31 device-qualified tasks**. Reports and full per-task
artifacts are outside source at `/private/tmp/metal-main-tasks-{default,bounded}-20261006/`.

The remaining gaps include BF16/FP16, integer and compare/select/coordinate/cast
operations, and NVIDIA/MetaX-specific reproduction tasks. A lowerable starter is not
a proof of full-shape coverage. Keep the original task identity and dtype; do not
silently turn a BF16 benchmark into an FP32 problem or widen target declarations.
Proposed Compiler capabilities need their own semantics, analyses, counterexamples,
Corpus Gate and device evidence. This change modifies no Compiler primitive or gate.

For development, authors should explore structurally different Cake hypotheses,
compare their own emitted MSL, preserve actual diagnostic codes and negative results,
and propose a responsible owner and minimal reproducer. A new independent Metal Bench
will own its task semantics, oracle, scoring and reference implementations without
using this development inventory as a held-out claim. The two completed GLM pilots
are development evidence. Token usage is accounting only for new Runs.

Local validation at `00381d7e` retained 34 passed, one skipped and one environment
failure in the selected existing suites: the CPU C++ oracle test selected incompatible
system CLT/SDK27 libraries. That run is not an acceptance pass and was not normalized
and rerun. Independent CI on the final commit owns full software acceptance.
