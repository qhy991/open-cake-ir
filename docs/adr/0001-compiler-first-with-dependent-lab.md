---
status: accepted
---

# Make the compiler the product core and the research lab a dependent application

`open-cake-ir` will expose a standalone Cake-like Compiler, while a separate Research Lab freezes a Compiler
Revision and evaluates complete Open Cake versus direct CUDA Authoring Environments. We rejected a compiler-only
repository because it could not reproduce the compiler-agent research loop, and rejected a reproduction-harness
core because experiment-specific control flow had already caused the legacy architecture to grow by `rXX`/`vN`.
The dependency is permanently one-way: Lab may use Compiler; Compiler may not know about campaigns, providers,
workloads, evidence storage or claims.

Consequently, kernel evolution runs inside a fixed Compiler Revision, while compiler evolution occurs only between
Campaigns through change proposals, a full Corpus Gate and human merge. Cake versus CUDA is modeled as assignment
to a complete Authoring Environment, not as a syntax-only representation Adapter.

