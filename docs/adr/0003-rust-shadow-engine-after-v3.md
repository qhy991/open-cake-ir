# ADR 0003: Consider Rust only as a Corpus-equivalent v4 shadow engine

Status: proposed, 2026-08-22.

## Context

The current Compiler parses content-bound JSON Schedules and Targets, produces ordered Findings and deterministic
Lowerings, and releases a Compiler Revision through a Corpus Gate. It does not compile or execute GPU code. Python
keeps the migration closure small because the legacy implementation, Research Lab, Triton and CuTe DSL environment
are already Python-based, while the current Corpus is too small for parser throughput to be material.

Rust could eventually deepen the Compiler Module: typed enums and immutable records can replace dynamic mappings,
and a standalone binary can provide a stable multi-language interface for untrusted Schedule input. Those benefits
depend on stable semantics. Today, several profile rules remain more precise in the verifier than in the public JSON
Schema; rewriting now would either freeze exploratory semantics or reproduce them with `serde_json::Value`.

## Proposed decision

Finish and freeze Python Compiler v3 before starting a Rust implementation. A Rust successor must begin as a shadow
engine behind the same language-neutral interface: `load`, `assess`, `lower`, `check-corpus` and release verification.
It must reproduce Finding order/code/path, Assessment fields, Lowering source bytes and digest, one-based inclusive
Source Maps, toolchain requirements, canonical JSON and Corpus Gate documents exactly.

The Rust crate, `Cargo.toml`, `Cargo.lock`, pinned toolchain, Schedule Schema, assets, Targets and Corpus become the
source set of a new `open-cake-ir-sm100a-v4-draft`. It becomes authoritative only after differential conformance,
the full Corpus Gate and human approval. A temporary Python Adapter may preserve the Lab seam; after v4 adoption,
the Python semantic implementation is deleted rather than maintained as a second owner.

If the Compiler later becomes primarily an MLIR/LLVM pass pipeline, this proposal must be revisited because the
C++/MLIR ecosystem may offer a deeper implementation than Rust.

## Consequences

- No language rewrite is part of the current migration or v3 identity.
- Rust is evaluated for type safety, distribution and embeddability, not assumed GPU-performance gains.
- Historical Campaigns remain bound to Python v3; a Rust rewrite cannot silently reuse that Compiler Revision.
- Compiler/kernel co-evolution is unchanged: kernels evolve inside a frozen Campaign revision, while Compiler
  evolution occurs between Campaigns through evidence, a Corpus Gate and human approval.
