# C550-Bench L1/048 starter

`gate_up.py` constructs a complete CAKE Program for
`L1/048_fused_gate_up_projection_with_swiglu`. The original reference applies
**GELU-tanh**, despite the task identifier's SwiGLU name.

The public ABI is unchanged and contiguous:

- `x`: BF16 `[batch_size, seq_len, 3072]`
- `gate_proj`, `up_proj`: BF16 `[24576, 3072]`
- `output`: BF16 `[batch_size, seq_len, 24576]`

`source_for(batch_size, seq_len)` returns a static Python CAKE Program.
`program_for(batch_size, seq_len)` parses that source into a `Program`.
`source_for_workload(workload, case_id)` checks the exact ordered external ABI
before returning the same source. The module does not load private benchmark
data, execute a reference, or provide host-side arithmetic.

The fixed starter has three stages:

1. BF16 gate projection with FP32 dot accumulation, then BF16 output rounding.
2. BF16 up projection with FP32 dot accumulation, then BF16 output rounding.
3. FP32 GELU-tanh on the rounded gate, BF16 activation rounding, multiplication
   with the rounded up projection, then BF16 output rounding.

Each projection uses a `[16,32,64]` M/N/K tile and four execution groups. Batch
uses its own grid axis. The final stage uses 256-element column tiles. Sequence
tails remain their original sizes and use the backend's masked accesses. No
flattening or dispatcher replaces the original shape.

The software tests execute emitted arithmetic with the existing bounds-checked
CPU memory double. They expose a projection midpoint that must round to BF16
and an activation/product example that changes when the activation rounding is
removed. They also check original rank, stage count, tail grids, ABI ordering and
all required rounding operations. This does not model native dot reduction
order, native tanh approximation, or FP32 contraction by the vendor compiler.

This is a starting implementation. Native compilation, all original 16 workload
correctness checks, full Program measurement qualification and performance
optimization remain separate gates. There is no measured speedup claim.
