# Original variable-length vision attention starter

Related issue: #411. `varlen_attention.py` builds an untuned nine-stage CAKE
Program for `L2/018_cu_seqlens_variable_length_vision_attention`. It uses the
software integration base `dd20bb56` and changes no Compiler implementation.

## Semantics and implementation

The public binder preserves the original eight inputs and one output. Hidden
states and output have shape `[T, 1152]`; cosine and sine have shape
`[T, 16, 72]`. Cumulative sequence endpoints remain INT64 without a leading
zero. Other tensors retain the original BF16 shapes and order.

The stages project Q, K and V; apply rotary embeddings; assign sequence IDs;
compute masked scores; compute row maximum and denominator; apply attention
values; and project the concatenated heads. Attention is bidirectional within
each sequence. Repeated endpoints represent empty sequences. An all-empty
sequence list returns zero, including when the output projection bias is nonzero.

SIMT multiply/reduce implements the projections and attention products. The
candidate materializes dense masked scores and has quadratic attention work and
storage in the token count. It uses existing operations and concrete tensor
accesses, with no transpose operation or Torch computation in the candidate.

The candidate preserves BF16 projection outputs, separate BF16 rotary products
before their sum, BF16 score products and scaled scores, FP32 softmax followed by
a BF16 probability cast, and BF16 attention values. Floating-point reduction order
still requires validation against the unchanged original device comparator.

## Software evidence

At `bee713cc`, seven CPU contracts pass with no skips. They execute the emitted
Triton source in a small CPU model with FP32 and BF16 rounding, and compare with
an independent staged scalar reference. Controls cover repeated and trailing
empty segments, the all-empty branch, bidirectional attention, projection and
key-loop tails, original tensor ABI refusal, and rotary product rounding. Removing
the cross-sequence mask fails the numerical control.

The same commit constructs, assesses and lowers all 16 original workload shapes,
giving 144 stage emissions. Private source material and generated records stay
outside Git. The local evidence is
`/tmp/cake-varlen-full-software-20261009/result.json` and
`/tmp/cake-varlen-controls-bee713cc.log`.

The largest declared Program scratch allocation is 1,744,203,776 bytes. With this
starter on both arms, the current paired tensor allocation has a maximum derived
lower bound of 31,813,282,240 bytes using `2I + 18O + 18S`. This includes both arms'
resident inputs, outputs and scratch plus sixteen fresh active-arm output/scratch
sets. It excludes runtime state, private spill and oracle allocation, so it does
not establish that device evaluation fits.

Native compilation, device correctness, memory admission and timing remain
unqualified. No optimization Run or GPU work was started by this software task.
