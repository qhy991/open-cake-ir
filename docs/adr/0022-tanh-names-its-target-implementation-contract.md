# ADR 0022: tanh names its target implementation contract

Status: accepted, 2026-08-25.

## Outcome and non-goals

An `elementwise` `tanh` operation names the target implementation contract that realizes
it. The admitted SM100 contract is `libdevice.tanh.f32`; the Triton backend lowers that
contract explicitly instead of choosing it behind the Schedule.

The KDA mechanism `tanh.approx.f32` is structurally expressible but is not admitted by
the current Target. This decision does not add a global `fast_math` mode, a generic
accuracy flag, an error-budget language, approximate `exp`, approximate reciprocal, or
a KDA-version profile. Those choices have different numerical effects and require their
own workload contracts and evidence.

## Evidence and authority

The external KDA MoE history uses `cute.math.tanh(..., approx=True)` for the SwiGLU
change in v11->v12, the routing experiment in v12->v13, and the retained routing change
in v20->v21. The arithmetic identity and the physical implementation are different
facts: v13 is later rolled back while its SwiGLU use remains, and v21 is accepted only
after numerical and NCU review. KDA owns those historical outcomes; Cake owns only
whether a Schedule can state and verify the resulting machine decision.

Before this decision, `ElementwiseOp.TANH` was emitted as `libdevice.tanh` with no
Schedule field naming that choice. Two implementations with materially different cost
and numerical behaviour therefore had one representation. That contradicts the paper's
explicit-machine-schedule boundary: the Schedule says which implementation consumes an
operand, lowering derives its mechanics, and the Target admits the contract.

An initial successor lowered the KDA choice directly to PTX `tanh.approx.f32`. The
generated kernel compiled and launched on B200, but failed the existing independent
standalone FP32 SwiGLU gate: maximum absolute deviation was
`3.24249267578125e-05`, with 2,304 of 524,288 elements above `1e-5`. This was a temporary
probe, not retained evidence and not a performance measurement.

The failure cannot be repaired by importing KDA's end-to-end FP8 MoE tolerance. That
contract admits `atol=0.05`, `rtol=0.15`, a 99% matched ratio and relative L2 below 0.25;
it answers a different workload. NVIDIA's PTX contract gives `tanh.approx.f32` a maximum
relative error of `2^-11`, which likewise does not prove the composed standalone output
meets an absolute `1e-5` gate. Widening the gate after observing the result would make the
evidence unable to fail.

## Minimal primitive and owners

No new operation or math mode is needed. `ElementwiseParameters` gains one target
implementation contract object, required exactly when `op` is `tanh` and forbidden
otherwise in the current vocabulary:

```json
{"op": "tanh", "instruction": {"contract": "libdevice.tanh.f32"}}
```

- the typed IR owns the one canonical spelling and the tanh-only structural rule;
- the Target owns whether the contract is available;
- the verifier localizes an unavailable contract or a contract admitted for the wrong
  operation kind at the operation parameter;
- the Triton emitter owns the exact lowering;
- the external oracle and B200 observation own numerical correctness.

The contract is not an optional hint. Omitting it would restore the hidden backend
choice, while attaching it to `add` or `mul` would create syntax with no defined effect.
The string remains Target-defined for the same reason MMA instruction contracts do:
architecture admission is not a global IR enum.

## Smallest complete slice and failure semantics

The existing standalone SwiGLU profile is the positive slice. Its tanh operation names
`libdevice.tanh.f32`, generated Triton contains the named call, and the complete kernel
must remain within the pre-existing `1e-5` tolerance of the independent definition-level
oracle on B200.

A retained sibling changes only that contract to KDA's `tanh.approx.f32`. It must parse
but fail hardware conformance before lowering with `TARGET_INSTRUCTION_UNSUPPORTED`.
The existing shape-drift sibling remains a separate failure for the workload boundary.
This separation keeps evidence able to distinguish an unavailable numerical contract
from wrong tensor geometry.

This slice closes the hidden implementation choice; it does not close the KDA approximate
tanh delta. Complete KDA-version coverage remains 0/57 because v1 still requires grouped
routing composition, quantization-scale relations, grouped/ragged GEMM, scatter, and a
multi-kernel program DAG. Historical Compiler releases and observations remain immutable;
v15 retains the first explicit implementation observation. Source review then exposed a
separate flat-Target hazard: an MMA contract admitted by the Target could be attached to
`tanh` and reach an emitter that could not honor it. Compiler v16 makes that mismatch a
blocking `ELEMENTWISE_INSTRUCTION_KIND_DIFFERS` Finding before lowering and receives its
own successor correctness observation; the emitted SwiGLU source itself is unchanged.

## References

- [CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/abs/2608.12629v1)
- [NVIDIA PTX ISA 12.9.1, `tanh.approx`](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html)
