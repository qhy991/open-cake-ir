# Compiler context

The Compiler context owns the open, independently usable Cake-like language and its deterministic interpretation.

## Language

**Schedule**:
A typed, hardware-explicit declaration of roles, resources, operations, dependencies and pipeline decisions.
_Avoid_: Program, config, IR candidate

**Target**:
An exact architecture definition containing the supported instruction, resource and synchronization contracts.
_Avoid_: GPU name, backend flag

**Finding**:
A localized statement that a Schedule satisfies, violates or lacks coverage for a compiler rule.
_Avoid_: Error string, diagnostic blob

**Assessment**:
The immutable findings, analysis coverage and lowering eligibility for one Schedule under one Compiler Revision and Target.
_Avoid_: Validation result, preflight

**Lowering**:
The deterministic derivation of inspectable target source and source mapping from an eligible Assessment.
_Avoid_: Compilation, execution

**Compiler Revision**:
A content-addressed combination of Schedule semantics, Target definitions, verifier, analysis, lowering and calibration coverage.
_Avoid_: Current compiler, Git HEAD

**Corpus**:
A versioned set of accepted and rejected schedules with their expected compiler observations.
_Avoid_: Tests, examples

**Corpus Gate**:
The release decision that a proposed compiler change preserves or deliberately updates the entire Corpus.
_Avoid_: Unit test pass

**Calibration**:
Measured target-specific evidence authorizing a bounded performance estimate.
_Avoid_: Heuristic, default cost

## Relationships

- A **Schedule** is assessed by exactly one **Compiler Revision** against exactly one **Target**.
- An **Assessment** contains zero or more **Findings** and either permits or blocks **Lowering**.
- A **Compiler Revision** is released only after one **Corpus Gate** covers its full **Corpus**.
- **Calibration** belongs to one Compiler Revision and Target pair and never transfers implicitly.

## Example dialogue

> **Agent:** “The Schedule is valid on B200; may I reuse the latency estimate on H100?”
>
> **Compiler maintainer:** “No. The Target may accept the Schedule, but H100 needs its own Calibration.”

## Flagged ambiguities

- “compile” previously meant IR lowering, CUDA toolchain build and CUBIN load; this context uses **Lowering** only.
- “target” previously meant both architecture and physical GPU; **Target** is architecture semantics, not machine admission.

