# Original expert execution starter

Related issue: #412. This module generates an untuned Program for the original
L2/024 tensor ABI. It does not run a Torch or native fallback.

The three stages compute the gate/up intermediate, the weighted down projection,
and the final token output. Projection operands are widened to FP32. The
intermediate and per-slot contributions remain FP32. Only the final output is
stored as BF16. Each repeated expert assignment retains its own routing weight.
The final stage sums in expert-index order and uses original slot order for ties.

The baseline uses explicit multiply/reduce projections with 16 output columns
and 128 reduction elements per tile. This preserves the FP32 computation path
without assuming that a BF16 dot has the same rounding. It is deliberately
untuned and can be expensive. There are no atomics or indirect output stores.

At `3a7a5dc3`, all 16 original shapes constructed, passed assessment and lowered
their 48 stages. Three CPU contracts executed emitted arithmetic and checked
duplicate routes, routing weights, masked tails, early BF16 rounding and a
counterexample to summing in arbitrary slot order. These checks do not model
native reduction order or native exponential error.

Native compilation, full original-device correctness, peak memory, event timing
and separate profiling remain unqualified. Scratch storage per invocation is
`4 * tokens * 8 * (2048 + 4096)` bytes. The largest original case uses 1.5 GiB
of scratch per invocation. A measurement adapter that retains 18 sets would add
27 GiB of scratch, before its other tensors and runtime allocations. Software
lowering does not establish memory admission.

Generated Workloads, original reference material, native binaries and Run
artifacts stay outside the source checkout. This starter must pass the original
comparison before it becomes a fixed optimization baseline.
