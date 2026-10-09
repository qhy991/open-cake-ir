# Original ConvNextV2 starter

Related issue: #415. The implementation plan uses seven existing CAKE stages:

1. Depthwise 7x7 convolution with zero padding, retaining NCHW storage.
2. Channel LayerNorm with a separate centered variance pass, producing NHWC.
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

This document records the design before implementation. There is no native,
device, memory or timing qualification yet.
