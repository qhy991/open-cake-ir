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

The initial software tick left both contracts unadmitted. The subsequent
[bounded device qualification and original RoPE result](metax-sincos-admission.md)
support declaring them on exact `xcore1002`. The current Target admits those
named FP32 library calls; TF32 remains unadmitted. Other code objects and dtypes
still require their own contracts. No Corpus expectation was refreshed.

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

The later device checks cover signed zero, sampled angles and trigonometric
boundaries, followed by the complete original RoPE task. Their retained numerical
contract, actual errors, toolchain and coverage limits are documented in the
[admission record](metax-sincos-admission.md). CPU compilation alone did not
establish those facts. The historical pre-admission tools still refuse an already
admitted Target; replay their old evidence at its producing commit.

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

The initial integration-only software probe emitted all sixteen original shapes
under a synthetic declaration. That evidence remains software-only. The original
IEEE angle starter then failed a device case, and explicit TF32 failed its own
angle comparison. The [bounded RNE10 starter](metax-rope-bounded-rne10.md) preserves
the original HIGH policy and subsequently passed all original 16×10 checks.
Current CPU contracts exercise the admitted route, interleaved output order,
producer casts, BF16 rounding, input preservation and scalar binding. They do
not model the native library's numerical error or turn the earlier failures
into passing results.

## ERF follow-up

The installed library exposes an FP32 ERF API, but Cake has no ERF primitive.
That requires a separate proposal covering syntax, canonical type/effect rules,
legality, lowering, diagnostic/cost coverage and counterexamples under P1–P8.
This change adds no ERF registry entry, source whitelist or Target capability.
