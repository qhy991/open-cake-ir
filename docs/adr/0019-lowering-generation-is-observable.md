# ADR 0019: Lowering generation is observable

Status: accepted, 2026-08-24.

## Outcome and non-goals

`Lowering.generated` states whether the returned source was emitted from Schedule
operations. Backend emission sets it to `true`; substituting the Schedule digest into a
closed source asset sets it to `false`. The public CLI and new observations project this
fact directly.

This does not rename asset materialization as emission, remove the retained TinyGEMM2
capability, add a CUDA-emitter framework, or claim that its current Schedule contains the
lane mapping, access maps and loop commitments needed to generate that kernel.

## Authority and dataflow

The profile's existing exclusive `backend`/`asset` choice owns the fact. `Compiler.lower`
copies it into the immutable result; readers must not infer it from profile, language or
filename. Existing frozen observations retain their bytes. Successor instruments read the
field instead of writing an unconditional `generated=true` projection.

## Acceptance evidence

An emitted profile reports `true`, TinyGEMM2 reports `false`, and the CLI exposes the same
value. Corpus source digests remain unchanged because this change labels the existing two
paths; it does not alter either source.
