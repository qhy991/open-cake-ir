# MACA explicit-rounding lowering probes

Use the [platform record](../../../docs/metax-c550.md#显式舍入-addfma-的工具链探针)
as the result owner. Evidence is outside source under `metax-rounding-offline-20260927`
and `metax-rounding-device-v2-20260927` in the external evidence root. The frozen
Compiler source was66ca6c83, installed Triton3.6/MACA3.8.0.4.c600u. This is a
standalone sealed-native diagnostic, not Cake contract admission or performance.

## What to inspect

Follow the imported module map, wrapper, linked library and complete linked LLVM
function body. Generic libdevice placeholders alone do not determine support, and
CUDA wrapper names do not determine the physical vendor. Do not reduce a function
to just its arithmetic lines: this implementation saves, changes and restores a
hardware rounding state around ordinary add/FMA. The observed get/sethwreg selector
is2049; its physical register/bitfield decoding remains unknown. Bind any such state observation
to the installed release, and retain the full dependencies in the evidence.

## Numerical acceptance and remaining questions

Eight variants each pass1024 complete words against an exact Fraction oracle;
six directed variants differ from nearest-rounding reference. The local verifier
also checks adjacent IEEE-word inequalities for directed results. Inputs are finite,
exact results nonzero/finite. Full outputs and oracle inputs are retained; do not
extend the result to NaN/inf, exact signed zero, exception flags or every input.

Before exposing a Cake rule, test mixed arithmetic in one kernel after the directed
call: the linked function contains restoration, but separate kernel launches do not
prove observable state restoration within a composed Schedule. Then trace the exact
InstructionContract, instruction admission, emitter and offline source validator.
FMA has an existing explicit-instruction primitive; plain ADD currently refuses an
instruction. Reuse the proper owner rather than smuggling a register mutation into
ordinary arithmetic or introducing a global rounding mode. Shared changes follow
the development-branch workflow. No promotion is valid until those gaps are closed.

The first CPU oracle assertion found an accidental exact-zero add on a FMA test row;
no broker allocation occurred. Preserve that failure and its successor rather than
restarting or relabelling it as hardware failure. This diagnostic has no timing score,
profiler or generic precision guarantee; resource queries are not performance proof.

## Mixed-state and edge successor

Read the [new platform record](../../../docs/metax-c550.md#定向-fma-后的混合算术与边界结果)
and [F-2026-09-27-004](../../../findings/2026-09-27-004-metax-directed-fma-lowering.json).
Job maca-25874d0cb439 uses four sealed FMA mode kernels; each directed result feeds
ordinary ADD and a tl.fma with a loaded multiplier. All12288positions pass:12228exact
words and60NaN-class checks. Directed modes distinguish leaked state from restored
RN at508/510/506rows, and subsequent ordinary outputs match RN. Raw outputs retained.

The successor also covers exact signed zero, cancellation, subnormal underflow,
overflow/fused cancellation, infinity and NaN. This closes the previously untested
compositions for these inputs, not every hidden state bit or exception/NaN payload.
The original finite-only scope remains unchanged in its old receipts. Current Cake
still refuses the proposed names and native source. Shared contract/source work
must precede platform admission; require real Cake-generated successor evidence,
not this standalone native result, before closing the capacity finding.

## Actual Cake prototype and output-write control

The [generated prototype record](../../../docs/metax-c550.md#实际-cake-定向-fma-原型)
uses sourceb5da553f, existing lm.fma, explicit RZ/RD/RU names and same-shaped FP32
register operands. Real TaskOpenCakeEnvironment output seals and loads native code;
two jobs match12228exact words and60NaN classes each. The second uses an initializer
that differs at every reference position and is finite for NaN references, so an
unwritten output cannot pass the numerical oracle. Retain complete raw words.

The tested ordinary ADD/FMA depend on the directed result, with508/510/506rows
that distinguish restored RN from leaked mode. Do not turn this observation into
hidden-state, NaN-payload, exception-flag or arbitrary-runtime guarantees. Normal
FMA reports14registers/thread versus58for directed calls; no latency or causal
performance inference follows. Shared registration/source review and platform
integration are still pending; the capacity finding remains proposed.
