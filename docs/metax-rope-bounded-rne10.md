# Bounded RoPE frequency conversion proposal

This is a CPU design review. It does not change the failed TF32 probe, the original Workload or the production Target.

## Exact domain

For nonnegative frequency f in [2^-19, 2047], 32f is within the normal FP16 interval [2^-14, 65504]. Both power-of-two scalings are exact in FP32. FP16 round-to-nearest ties-to-even retains 10 fraction bits; widening that result and dividing by 32 preserves those bits. Zero is exact separately.

Therefore `fp32(fp16_rne(32*f))/32` equals a rounding of f to 10 fraction bits in this domain. The original sequential INT64 positions are in [0,2047], so they are exact in FP32 and already have at most 11 significant bits. The product of the two reduced operands has at most 22 significant bits and fits FP32 exactly.

The three retained frequency vectors lie in [2.4551407022954663e-06,0.125]. Their scaled values lie inside the proved domain. This range is derived for the other original shapes from the identical factory expression and constant head_dim=128, not from 13 new device observations.

## CPU evidence

- All 409,984 retained original HIGH angle words match the proposed conversion and product exactly.
- The integer FP16 conversion model and the CPU binary16 cast agree on all retained input words.
- 153,596 control inputs cover every retained significand across the admitted exponent range at exact, just-before-halfway, halfway, just-after-halfway, and last-in-bin values. The integer model, CPU binary16 cast and direct RNE10 rounding agree.
- Independent mathematical review confirms the exponent translation and ties-to-even argument.

This is word coverage within three shapes. Repeated batch values are not independent experiments.

## Refusing use outside this domain

- f = 2^-20 + 2^-30: scaling enters FP16 subnormals. The proposed cast returns 2^-20 after reverse scaling, whereas RNE10 keeps the original value.
- f = 2047.5: scaled FP16 conversion overflows, whereas RNE10(f) is the finite value 2048.
- position = 2049: an unquantized position differs from RNE10(position)=2048. The current proposal relies on the original position bound.

## Existing IR expression

```python
scaled_frequency = frequencies * 32.0
frequency_half = lm.cast(scaled_frequency, to="fp16")
frequency_wide = lm.cast(frequency_half, to="fp32")
frequency_rne10 = frequency_wide * 0.03125
angles = positions * frequency_rne10
```

The external B2/S131 candidate parsed, passed real Compiler assessment and lowered at fixed public source 06be782e. The emitted source retains both explicit casts. The production Target is unchanged; no MMA or TF32 instruction contract is used.

## Implementation and remaining gates

Source `02423b3f` implements the starter and its focused contracts. `source_for_workload` requires the fixed original task, Bench commit, HIGH policy, scalar and position bound, then delegates the complete factory/reference binding to the existing original Bench document validator. It uses `source_for_rne10`; `source_for` retains the former IEEE expression as a negative control. Original frequency scaling, shape, scalar, output ordering and final BF16 conversion are preserved. A loose FP32 ABI alone does not establish these value bounds.

At that fixed source, eight CPU contracts passed with no skips, including actual emitted arithmetic and complete BF16 outputs, 153,596 rounding-domain controls, domain counterexamples, and original-owner refusals. Independent source review passed. This is software verification; the native/device successor is tracked in #445.

No IR, Compiler primitive, general TF32 conversion rule or Target admission is needed for this specialized expression. A new fixed native build and device comparison must establish that MACA performs the intended FP16 cast and does not remove or alter it. Then the complete original RoPE oracle must pass all 16 shapes and 10 inputs per shape. The CPU proof does not replace these gates.

Tracking issue: #444. Related investigations: #401 and #440. This branch inherits an unintegrated experiment stack; it does not request a maintained-branch merge.
