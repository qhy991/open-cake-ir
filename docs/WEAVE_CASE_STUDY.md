# B300-M4 MoE event pipeline: a bounded Cake case study

**Date:** 2026-09-28. **Status:** documentation-only addition to `main`.
The experimental Compiler/CUDA implementation remains on
`task/nvidia-weave-persistent-ep-pipeline`; its device runs pin clean source
commit `2c638aad856909de36efaa234aefb48799d42bcc`. This page does not
claim that the experimental lowering is available from `main`.

## Question and implementation

The question was whether Cake could express and lower spatial producer/consumer
allocation, temporal event overlap and bounded work stealing for a four-rank
BF16 MoE layer on `sm_103a`, using at most four B300-M4 GPUs. The synthetic
model geometry is four ranks, 512 tokens per rank, 128 experts, eight routes
per token, hidden width 2048 and FFN intermediate width 768. The temporal
control called `K` in the retained experiments means **one, two or four
source chunks**, not the eight routes per token.

The experimental Program contains up/gate, BF16 SwiGLU activation and down
Schedules, followed by a weighted combine. `RankedTileEffects` declares the
source-event publication, per-tile predecessor conditions and steal budget.
The native CUDA lowering composes the checked stage bodies with a resident
cooperative grid. The first `c` CTAs produce routing, bins and ready tiles;
other CTAs consume stage tasks. Producer CTAs may claim remaining work after
their event work, subject to the rank-local budget. GPU-scope PTX barriers
order same-device producers, while cross-GPU handoffs retain system scope.
The consumer fences the global-to-async TMA handoff after acquiring a ready
tile. A later Schedule variant maps six activation rows to six warps and
increases tensor-core N from 64 to 128. The N128 variant derives TMA boxes,
MMA shape, TMEM columns, shared-memory offsets and store coordinates from the
Schedule, reducing per-tile stage work units from 78 to 50.

This is a **system implementation claim**. The repository has not established
that these mechanisms are novel relative to all prior MoE runtimes.

## Retained observations

Each GPU job used the broker's four-GPU exclusive lease and released it.
Correctness was checked against the same external FP64 oracle, a CPU tile-event
planner and the earlier v3 output. These measurements have development scope.

| Observation | Retained result and job | Scope |
| --- | --- | --- |
| N128 route/chunk matrix | `gpuq-1a73f0c69804`: 12/12 controls, 0/50,331,648 oracle failures, 960/960 tile-event slots, zero v3 output bit differences, 12/12 positive task-progress overlaps | Four frozen routes × one/two/four source chunks, `c=64`, full analyzed steal capacity 12,750 |
| N128 spatial/steal sweep | `gpuq-c4eac1dae550`: 6/6 controls, 0/25,165,824 oracle failures, 480/480 tile-event slots, zero v3 bit differences, 6/6 positive overlaps | `c=1/74/95`; zero budget stole nothing, positive budgets produced bounded steals |
| N64 versus N128 | Reversed-order `gpuq-f1c87673109b`: two-arm medians 6.344 versus 6.091 ms (1.042× N64/N128) | Same-source, same-input, same-lease CUPTI development comparison on synthetic fanin, four chunks, `c=64`; not qualified external latency |
| Timing instability | `gpuq-4111913e8f16`: the last N64 arm rose to 27.082 ms while its first arm was 6.447 ms | Its automatic 2.696× aggregate ratio is rejected; `interpretation-v2.json` selects no performance ratio |
| Open baseline | SGLang `v0.5.12.post1` + DeepEP `1.2.1`: one Nsight development joint GPU span 2.149 ms; Cake N128 one span 7.335 ms | Exact synthetic input and FP64 oracle relation, but separate leases and Nsight versions; **no qualified Cake-to-baseline speedup** |

The full-array input audit found zero differences across all ranks in hidden
BF16 values, expert IDs, FP32 route weights and gate/up/down BF16 weights;
the FP32 oracle outputs were bitwise identical. The two implementations both
passed that oracle, but their BF16 outputs are not bitwise interchangeable.
The original first N128 device attempt (`gpuq-dd1cdbaba524`) was refused by
an Evaluation capacity check before the model kernel ran. Successor
`2c638aad` derived that check from ranked-tile analysis; a CPU replay of the
failed candidate and plan passed before the successful GPU trials.

## Evidence and limits

The primary research log is
`docs/WEAVE_PERSISTENT_EVENT_PIPELINE.md` on the experimental task branch at
`73065ad672c2e7f7ef1c8bb9b7cb76b488d392b9`. Its external evidence roots
remain outside the source checkout. Key B300-M4 roots are:

- `/home/qinhaiyan/cake-weave-n128-matrix12-2c638aad/` — oracle and overlap reports for 12 controls.
- `/home/qinhaiyan/cake-weave-n128-sweep6-2c638aad/` — six spatial/steal controls.
- `/home/qinhaiyan/cake-weave-n64-vs-n128-baab-2c638aad-fanin/` — reversed-order CUPTI comparison.
- `/home/qinhaiyan/cake-weave-n64-vs-n128-abba-2c638aad-fanin/` — anomalous first comparison and its appended interpretation.
- `/home/qinhaiyan/cake-weave-n128-nsys-fanin-2c638aad/` and `/home/qinhaiyan/cake-weave-sglang-nsys-warm-20260928/` — single instrumented layer traces.

The six-control c=95 pass does not close the intermittent route-location
Finding `F-2026-09-27-001`. Target clock/reset qualification, repeated common
timing against the open baseline and broader workloads remain open. The
observed 1.042× improvement is a bounded internal N-tile comparison; it
cannot be multiplied by gains from earlier experiments or presented as a
general MoE speedup.
