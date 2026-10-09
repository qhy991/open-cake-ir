# Original ConvNextV2 starter

Related issue: #415. The implementation plan uses seven existing CAKE stages:

1. Depthwise 7x7 convolution with zero padding, storing `[B, H*W, C]`.
2. Channel LayerNorm with a separate centered variance calculation.
3. FP32 pointwise expansion and an explicit approximation to erf GELU.
4. Spatial L2 norm for each batch and expanded channel.
5. Channel mean normalization of the spatial norms.
6. GRN arithmetic inside the FP32 pointwise projection.
7. Bias, original residual and NCHW output.

Fusing the GRN arithmetic into projection avoids another expanded intermediate.
Each stage retains a concrete tensor ABI and inspectable generated source. The
candidate has no Torch computation or opaque native implementation.

The GELU approximation is a candidate implementation choice. It must pass the
original unchanged comparator. It is not a claim of exact erf semantics. Tests
will compare the approximation with an independent scalar erf, and exercise
padding, normalization axes and the residual. The original scalar epsilons and
all 16 shapes remain fixed.

The original public tensor ABI remains NCHW. Intermediate tensors use explicit
`[B, H*W, C]` or `[B, H*W, 4C]` storage, and the final stage writes NCHW. The
candidate uses no transpose operation. Both projections request the Target's
existing `triton.dot.fp32_ieee` contract. The binder checks all eleven FP32 input
tensors, the output, and the original `eps` and `layer_norm_eps` scalar bindings.

## Software evidence

At `3d8fc2ac`, six CPU contracts pass with zero skips. They execute emitted Triton
arithmetic in a CPU model and cover:

- A complete small Program against an independent scalar reference using `math.erf`.
- Depthwise zero padding across row, channel and batch boundaries.
- Centered LayerNorm variance, channel tails and a nondefault epsilon.
- Spatial norms across separate batches, a 129-element tail and zero inputs.
- GELU approximation on 4,097 points in `[-16, 16]`, with maximum absolute error
  below `2e-6` against `math.erf` in this model.
- Exact ordered tensor ABI and scalar provenance refusal.

The GELU implementation evaluates an explicit polynomial approximation to erfc.
Its negative branch uses the small erfc result directly to avoid cancellation.
The bounded CPU error check is not an error bound for all inputs or for native
device arithmetic. The original Bench comparator remains authoritative.

All 16 original workloads construct, assess and lower at the same commit, giving
112 stage emissions. Shape, dtype and scalar epsilons are unchanged. Evidence is
outside Git at `/tmp/cake-convnext-full-software-20261009/result.json` and
`/tmp/cake-convnext-controls-003.log`. This task adds a starter and changes no
Compiler implementation.

The largest declared scratch allocation is 2,158,166,016 bytes. With the same
starter on both arms, the current paired allocation model gives a maximum tensor
lower bound of 45,013,254,912 bytes using `2I + 18O + 18S`. This includes both arms'
resident inputs, outputs and scratch plus sixteen fresh active-arm output/scratch
sets. Runtime state, private spill and oracle allocations are excluded.

Native compilation, device correctness, memory admission and timing remain
unqualified. No GPU/provider work or optimization Run was started by this task.
