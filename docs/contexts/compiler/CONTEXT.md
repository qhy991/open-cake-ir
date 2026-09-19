# Compiler context

[中文阅读](../../zh-CN/contexts/compiler/CONTEXT.md) · [Bilingual catalog](../../README.md)

The Compiler context owns the independently usable Cake-like language and its deterministic
interpretation. Canonical definitions are in the
[`Glossary`](../../GLOSSARY.md#compiler-terms); this document owns responsibilities and
relationships only.

## Owned terms

- [`Schedule`](../../GLOSSARY.md#schedule)
- [`Target`](../../GLOSSARY.md#target)
- [`Finding`](../../GLOSSARY.md#finding)
- [`Assessment`](../../GLOSSARY.md#assessment)
- [`Lowering`](../../GLOSSARY.md#lowering)
- [`Compiler Revision`](../../GLOSSARY.md#compiler-revision)
- [`Corpus`](../../GLOSSARY.md#corpus) and
  [`Corpus Gate`](../../GLOSSARY.md#corpus-gate)
- [`Calibration`](../../GLOSSARY.md#calibration)

## Responsibilities

- Parse and type-check complete Schedules under one exact Target.
- Emit localized Findings for schedule semantics, hardware conformance, data consistency,
  program safety, and backend lowering preconditions.
- Report modeled analysis coverage without presenting estimates as GPU truth.
- Deterministically lower eligible Schedules to inspectable source for a current backend.
- Change semantics only through the full Corpus Gate and one review at the merge to `main`.

The Compiler does not own Workload semantics, input materialization, correctness oracles,
providers, Campaigns, GPU allocation, Evaluation, Evidence, or claims.

## Relationships

- A Schedule is assessed by exactly one Compiler Revision against exactly one Target.
- An Assessment contains zero or more Findings and separately records acceptance and
  lowering eligibility.
- A Compiler Revision at one clean commit binds one complete Corpus expectation set.
- Calibration belongs to one Compiler Revision and Target domain and never transfers
  implicitly.
- The Lab may consume a Compiler Revision at a clean commit; the Compiler never imports the Lab.

## Boundary examples

“The Schedule is accepted” does not mean “the backend can lower it”, “the generated source
compiles”, or “the kernel is correct”. Each statement crosses a separate boundary named in
the Glossary.

“Target `sm_100a` accepts this instruction contract” does not mean that a physical B200 is
available or that another architecture inherits the same calibration.

## Backend input policies

A backend registration may bind a raw-input contract checked after typed construction
and before canonicalization. Each Python emitter declares its PythonNamespace; shared
code applies the lexical and collision checks without owning another emitter's symbols.
See [ADR 0072](../../adr/0072-backend-owned-input-and-native-adapters.md).
