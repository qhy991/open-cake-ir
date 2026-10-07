# DCU explicit OCML FP32 FMA qualification

Shared implementation: [PR382](https://github.com/qhy991/open-cake-ir/pull/382),
source `eb816734`; final `119665a0` updates evidence and coverage projections only.
The existing same-shape three-register FP32 FMA now accepts `ocml.fma.f32` on gfx938.
The emitter uses the installed DTK OCML route. This exposes an existing hardware and
Triton operation; it adds no new ISA instruction and performs no automatic mul/add rewrite.

Evidence root: `/data3/testuser01/experiments/bw1100-fma-contract-probe-20261007/run`.
Native investigation is under `results/native`, and clean generated-source qualification
is under `results/generated-eb816734-r1`. Protocols and producer scripts remain beside them.

The native tl.fma, OCML and direct ISA probes each match12288 result words across ordinary
and nested cases. The unfused negative control differs on1921 words. The independent
integer oracle covers cancellation, signed zero, subnormals, overflow and nonfinite values.
Only quiet-NaN classification is required for NaNs; other output words must match exactly.
Native controls do not count as Compiler-generated or Task performance evidence.

The unchanged generated public callables pass12 component cases/12288 words and15 original
cases of SiLU, PReLU and softplus-gradient. The retained SwiGLU expressibility probe says
explicitly that its x*x+up formula is intentionally wrong; four of its five original Task
cases still fail. It is a negative control, not a successful SwiGLU implementation.

Actual generated-call metadata records `enable_fp_fusion=true`, `allow_flush_denorm=false`,
and gfx938. Executed descriptors retain round-mode0 and denorm-mode3. Ordinary FMA reaches
`v_fmac_f32_e32`; the nested component retains independent `v_mul_f32_e32`, `v_fma_f32`
and `v_fmac_f32_e32`, preserving the separately rounded producer. No native-probe options
were injected into those public calls.

The first generated worker saved observations but failed serializing GPUTarget metadata.
Its failed admission and numerical verdicts remain unchanged. The declared r1 successor
repairs serialization, preserves the same sources/input/oracle, and records successful job
`bw-1a2611e48fa7`: exit0, VRAM0%, no visible KFD context, no live container. The gateway
serializes this user only; physical activity from other users is not excluded.

Independent review and the exact shared PR CI passed. The two new Corpus cases preserve
all199 previous expectations: total201 cases and113 source snapshots. This platform
integration preserves the earlier register-transpose observation beside the FMA citation.

Promote the explicit operation contract. Do not claim an operator speedup, relax Task
precision, accept the wrong SwiGLU formula, or count these components as independent
bw1100-bench results. Performance and fixed-budget author-search effects require new runs.
