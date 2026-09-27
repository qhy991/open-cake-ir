# Ordered Python candidate bundles

Status: accepted for new known-kernel Cake Runs; no live Provider or GPU result is
inferred from software fixtures.

The default new Cake author file is `candidate-set.py`. It has one Cake frontend
import and an ordered sequence of complete `@cake.schedule(...)` functions and,
only when granted, static `cake.transform(parent=..., transformation=...,
parameters=...)` declarations. The Lab parses this file as AST without executing
it. It projects each function into the existing single-Schedule Python frontend
and each transform into the existing action owner. Unsupported top-level code,
dynamic transform arguments, duplicate candidates and a proposal count above the
Run budget are refused before compilation.
Static `cake.program(...)` declarations can compose complete Schedule functions in
the same source into one ordered Program candidate. Referenced stage functions are
components, not extra proposals; the Compiler's existing typed Program checks tensor
bindings, single producers and read-before-write without an author-written JSON graph.

This retains the existing single-file Provider add/update lifecycle while preserving
the order of multiple proposals and transform actions. The original UTF-8 file is
sealed separately from every projected candidate. Replay reprojects that file and
compares the ordered Candidate bytes; a changed source or substituted evidence role
cannot support the old result. Codex and Claude qualification binds this versioned
submission contract. The per-Turn candidate/search budget is not reduced to make
Python source authoring work.

The older `candidate_set_envelope_v1` remains readable for frozen Runs, and the
single-candidate `python_source_file_v1` remains a separate treatment. No implicit
conversion changes their submitted bytes or retrospective analysis. The Compiler's
canonical Schedule serialization and the Lab's private action/Candidate documents
remain internal; authors of new known-kernel Runs do not fill Schedule JSON.
