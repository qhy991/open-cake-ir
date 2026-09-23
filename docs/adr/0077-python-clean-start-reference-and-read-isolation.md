# Python clean-start reference and read isolation

Status: accepted for reference construction and preflight; live execution refused until read isolation is qualified.

A new Cake clean-start reference is a deterministic, intentionally incomplete Python
source file. The Lab derives its decorator, exact target, lowering route and tensor
signature from the frozen Workload; the body is only `...`. Admission compares the
entire file to that rendering. Parsing an arbitrary Python file and checking that its
result looks empty would miss comments, imports or ignored source carrying a complete
implementation. The normal Compiler frontend accepts only complete Schedules and is
not the parser for this reference.

New independent Runs and paired Cake-versus-direct-CUDA Study successors use
`python_starter` and `python_source_v1`. Frozen JSON stubs and their Study templates
remain replay inputs. The candidate-set envelope remains a separate provider
transport contract.

The current task launcher cannot qualify clean-start reference access. It stores a
complete baseline source and compiled artifacts near the actor workspace, while the
provider can read outside its delivered TaskPackage; provider qualification also
uses the complete source. A safe prompt or starter does not isolate those bytes.
The launcher and common execution paths therefore refuse clean-start before provider
work or Evidence creation. Enabling execution requires a qualified filesystem read
boundary for the author process and a qualification task that contains no target
implementation. That successor must test the actual provider's read permissions,
not infer them from a declared sandbox mode.
