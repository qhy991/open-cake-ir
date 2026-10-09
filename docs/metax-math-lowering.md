# MACA FP32 sine and cosine

Tracking issue: [#401](https://github.com/qhy991/open-cake-ir/issues/401).
Draft implementation: [#403](https://github.com/qhy991/open-cake-ir/pull/403).

## Observed implementation

The C550-2 container uses MetaX Triton 3.6.0. Its installed
`triton/backends/metax/compiler.py` redirects
`triton.language.extra.libdevice` to the compatible CUDA-named Python module
(lines 234–235 in the observed installation). That backend links
`ext_maca_mathlib.bc`, `maca_kernellib.bc`, and `maca_mathlib.bc`
(lines 161–176 and 392–396).

The installed `triton/language/extra/cuda/libdevice.py` declares these FP32
entry points:

| Python call | Compatible symbol | Library selected by the observed backend |
|---|---|---|
| `libdevice.sin(x)` | `__nv_sinf` | MACA math libraries |
| `libdevice.cos(x)` | `__nv_cosf` | MACA math libraries |
| `libdevice.erf(x)` | `__nv_erff` | MACA math libraries; not implemented in Cake here |

The compatible symbol names do not identify NVIDIA machine code. The backend
and native output format establish which toolchain compiled the calls.

## Implemented software boundary

`maca.sin.f32` and `maca.cos.f32` have explicit instruction records for the
existing unary FP32 SIN/COS operations. Their Triton emission requires the
MCFATBIN code object and the matching named contract. OCML retains its own
HSACO contract. A contract from another code object, implicit dtype promotion,
function aliases, keywords, extra operands, and unrelated library calls are
refused by their owning checks.

The production `xcore1002` Target still declares neither new contract. It returns
`TARGET_INSTRUCTION_UNSUPPORTED`. No Target document, hardware fact, or Corpus
expectation changed in this work.

## Qualification evidence and limits

At `7cdd24a9`, both FP32 probes compiled through the existing isolated CPU
compiler to native MCFATBIN, with 64 threads per CTA and zero dynamic shared
memory. GPU visibility was empty; no kernel or provider ran. Source, compiler
requirements, build artifacts and results are retained on C550-2 under:

```text
/root/open-cake-runs-reviewed/c550-evolution-parallel-author-20261006/
  math-trig-native-7cdd24a9-20261009/
```

The first attempt, `math-trig-native-137a28f9-20261009`, remains preserved. Its
probe preparation omitted the target/compiler/source-language fields normally
bound by `Compiler.lower`; the compile-contract check rejected it before vendor
math compilation. The successor repaired this code boundary and added a
regression contract. The environment was not repaired or replaced.

`tools/qualify_metax_trig.py` is a pre-admission CPU probe. It uses a clearly
scoped in-memory Target declaration to exercise emission while confirming that
the committed Target still refuses the operation. It cannot grant production
admission or device numerical correctness.

Before adding real Target declarations, retain original-domain device checks
for signed zero, representative angle magnitudes and trigonometric boundaries,
then validate the complete original RoPE task. Record actual error statistics,
exceptional-value coverage, exact toolchain and input domain. CPU compilation
does not establish those facts. Update the Target declaration and its pinned
admitted-contract test only in a reviewed successor supported by that evidence.

## Original RoPE starter

`tools/benchmarks/c550/rope.py` retains INT64 positions `[B,S]`, FP32 `inv_freq[64]`,
and BF16 `cos_sin[B,S,128,2]`. It requires the integer-storage successor. One CTA
handles a batch/sequence/frequency coordinate and writes its two cosine/sine
components. This is a conservative baseline, not an optimized mapping.

The original custom input factory already computes the frequency scaling and
returns `attention_scaling` as Python float `1.0`. That value comes from the
factory, not a `scalar.value` entry in the raw workload. The task adapter records
it in `fixed_scalar_inputs` and verifies the factory result. The final wrapper
must also reject an original scalar argument that differs from the frozen value.
The candidate receives no extra scalar tensor and performs no second frequency
scaling.

All sixteen original shapes pass construction, verification and source emission
in the integration-only software probe that combines the integer-storage and
math changes. The same real Target continues to refuse unqualified trig. CPU
contracts exercise interleaved output order, repeated frequencies, FP32 casting
of large INT64 positions, BF16 output rounding, input preservation and scalar
binding. They do not model the native library's numerical error.

## ERF follow-up

The installed library exposes an FP32 ERF API, but Cake has no ERF primitive.
That requires a separate proposal covering syntax, canonical type/effect rules,
legality, lowering, diagnostic/cost coverage and counterexamples under P1–P8.
This change adds no ERF registry entry, source whitelist or Target capability.

## Complete original RoPE check and numerical diagnosis

The pre-admission bridge at `d6985ddd` passed five CPU contracts and built all
16 original shapes through the isolated native compiler. The original Bench
check retained its inputs, reference, comparator and ten-round sequence. It
passed ten invocations of the first shape, then stopped at the first invocation
of B=1, S=2048. The maximum output absolute error was 0.0418701171875.
This failed result remains sealed as `rope-original-20261009`; the production
math declarations remain closed.

A separate one-case diagnosis compared three computations using the same original
inputs. The CAKE candidate and an FP32 elementwise-angle computation followed by
Torch cosine/sine produced identical BF16 values at all 524,288 output positions.
Both failed the original comparison. The original reference uses a matrix
multiplication for its angles; its angles differed from the FP32 products by up
to 0.0419921875. Inputs were unchanged and the native module closed.

The captured environment sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` and reports
`float32_matmul_precision=high`, `allow_tf32=true`. A fresh CPU process importing
only Torch reports the same policy. The original `set_seed` function changes
random seeds, not that policy. The reference's effective math policy is therefore
an execution input that needs explicit binding. Removing the environment override
would change this observation's reference semantics.

**Promotion disposition: no promotion.** This diagnosis locates the discrepancy
in angle computation. It does not qualify the C550 TF32 route or establish its
rounding rule. The existing global IR already has `triton.dot.fp32_tf32`; the
C550 Target does not admit it, and its prior RNE10 hypothesis has retained
failures. A new bounded probe will compare the actual K=1 matrix path against
the original reference. Production admission and the complete original RoPE
check remain separate pending decisions. No tolerance, oracle or failed result
has been changed.
