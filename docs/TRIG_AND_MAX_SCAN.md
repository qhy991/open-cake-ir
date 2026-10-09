# OCML trig and resident integer maximum scan

RoPE materializes a FP32 position-times-frequency tensor before computing cosine
and sine. A schedule can now express those two functions as elementwise `sin` and
`cos`, each with an explicit `instruction={"contract": "ocml.sin.f32"}` or
`ocml.cos.f32`. Each reads and produces FP32 register values of the same shape.
An explicit cast is required before using a narrower input or rounding an output.
The existing arithmetic, effect, access and consumer rules apply; this change does
not commute, reassociate or fuse any producer across a trig operation.

The contract names the OCML FP32 software function. Triton emits direct unary
`libdevice.sin` / `libdevice.cos` calls. It does not select an approximate hardware
trig instruction or promise correctly rounded results beyond the named library's
semantics. Only HSACO source admission accepts these direct calls. A Target must
separately admit the named contracts after qualification; registry membership is
not hardware evidence. Unsupported backends refuse them. Performance analysis
reports their work as uncounted, rather than inventing a FLOP count.

For expert sorting, a segment-start vector such as `[0,0,2,0,4,0,0,7]` needs
`[0,0,2,2,4,4,4,7]`. Sum scan gives a different answer. `lm.scan(values, op="max",
axis=..., direction="forward")` returns the inclusive maximum along that resident
axis. Reverse direction returns an inclusive suffix maximum. Inputs and outputs
must be INT32 registers with identical shapes; integers beyond FP32's exact range
are never converted to floating point. Scalar scan is identity. Floating max
scan is refused until its NaN and signed-zero policy is defined.

Triton emits `associative_scan` with one exact pure JIT maximum-combine helper.
Source projection retains that helper; source admission rejects a changed helper,
callback alias, arbitrary callback, duplicate helper, or use outside the direct
`combine_fn` argument. This does not add a sort primitive, an indirect global
scatter, cross-CTA prefix carry, or a full sorting algorithm.

`tests/contracts/test_trig_max_scan.py` checks parsing/schema, explicit contracts,
types, refused target/backend routes, cost coverage, forward/reverse signed
integer semantics and public source admission. Device and complete-task
qualification remain separate evidence. The changes alone do not establish a
whole-call speedup or better results within a fixed agent budget.
