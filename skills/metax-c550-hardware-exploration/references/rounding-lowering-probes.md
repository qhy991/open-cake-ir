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
hardware rounding register around ordinary add/FMA. Bind any such state observation
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
