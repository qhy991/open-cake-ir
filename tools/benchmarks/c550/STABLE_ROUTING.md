# Stable expert-routing starter

Tracking issue: [#404](https://github.com/qhy991/open-cake-ir/issues/404).
Draft PR: [#405](https://github.com/qhy991/open-cake-ir/pull/405).

`stable_routing.py` implements a pure CAKE starting Program for the original
`L1/058_moe_expert_token_radix_sort_with_prefix_sum` task. It preserves the
ordered public ABI:

| Argument | Type and shape |
|---|---|
| `topk_idx` | INT32 `[B,S,8]`, input |
| `sorted_token_indices` | INT32 `[B*S*8]`, output |
| `expert_offsets` | INT32 `[257]`, output |

The permutation contains flattened **assignment positions**. A token with eight
expert assignments contributes eight distinct positions. Ties retain the
original batch/sequence/slot order.

## Construction

1. Each original assignment `i` owns `ranks[i]`. Its rank is the number of keys
   smaller than its key, plus equal keys whose original position precedes `i`.
2. Each output position `p` scans those ranks and writes the unique assignment
   index with rank `p`. The candidate uses no scatter or atomic operation.
3. Each expert boundary `e` independently counts keys below `e`. This produces
   offsets for `e=0..256`, including initial zero and final assignment count.

All arithmetic is INT32. The original sixteen shapes contain at most 65,536
assignments. Input storage keeps its original rank-three shape. Each stage has
one direct store with program-owned output coordinates. Existing comparisons,
integer arithmetic, reductions, nested tile loops and `Program` composition
express the implementation; no sort or scan primitive was added.

The generator removes single-trip loops at fixed shapes. An outer-loop
coordinate is constructed in its own loop before an inner loop reads it. These
are task-side choices that satisfy existing Compiler rules.

## API and verification

- `source_for(B,S)` returns static CAKE Python source.
- `program_for(B,S)` parses the complete Program.
- `source_for_workload(workload, case_id)` checks the exact original ordered ABI.

At clean C1 `5bb474c6`, all sixteen original shapes construct successfully and
all 48 stages pass assessment and Triton source lowering. Five fixed-commit CPU
contracts pass without skips. They execute actual emitted stage arithmetic
against an independent stable-sort/count oracle, including ties across batches,
rows and slots; masked tails; empty experts; boundary offsets; exact permutation;
unchanged inputs; and one write per output. A wrong tie rule still produces a
permutation but fails the independent stability check.

Small tiles exercise repeated loop iterations in CPU controls; an additional
control uses the baseline's actual 128-row tile. These checks do not establish
native compiler behavior or device correctness. Original definitions, reference
code, workload UUIDs and generated records remain outside Git.

## Cost and remaining gates

This is an untuned quadratic baseline: counted rank and inverse permutation each
scan all assignments for every output. It uses 525,316 bytes of output and
intermediate storage at the largest original shape, excluding inputs and
repeated evaluation argument sets. Its runtime has not been measured.

Native compilation and full original-device correctness remain required before
using it as an optimization baseline. Complete Program timing/attribution must
also be qualified by the runtime owner. No Compiler change or optimization pass
is promoted from this task; current primitives were sufficient for this source
construction. No GPU/provider execution, speedup, or new hardware capability is
claimed here.
