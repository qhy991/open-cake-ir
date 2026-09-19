# Backend-owned input policies and native adapters

Status: accepted

The owner requested refactoring three measured cross-backend couplings: a Triton input
named `cute` was refused by CuTe's reserved namespace; the Compiler facade inspected the
Triton route to enforce a list-only loop body; native pairing treated every non-Triton
implementation as CuTe.

Python-emitting backends now provide an immutable PythonNamespace to the shared name
checker. The checker owns lexical validity and collision detection; the emitter owns
reserved imports, generated prefixes and implementation-specific restrictions. CuTe's
three implementations share its own namespace declarations. Foreign backend names may be
used when they do not collide with this emitter's actual generated code. Python keywords,
unsafe identifiers and real collisions remain refused.

The backend registration has an optional validate_input callable. Compiler first parses
the typed IR and applies common input/name checks, then invokes the selected backend's
raw-input contract before canonicalization. Triton owns its Mapping API's list-only loop
body check. This preserves existing error ordering and leaves other backends unaffected.

NativeBackend explicitly binds a NativeAdapter implementing build factories, source
admission, launch-block projection and native baseline construction. Triton and CuTe own
separate adapters. Factories import their dependencies when called, preserving the existing
module import cycles and test seams. The registry still refuses unknown backends and arms;
there is no default adapter. Existing public pairing helpers and native payload shapes stay
stable. A synthetic third adapter tests all six operations without either incumbent path.

The change adds no IR operation, layout abstraction, implicit target conversion or new
hardware claim (P1-P8). The Corpus and generated source snapshots stay unchanged. The
intentional admission change is that an otherwise legal name is no longer refused because
another backend reserves it. Validation exercises that behavior and real namespace
collisions, backend-specific input dispatch, existing error precedence, and native pairing.
