# Reference provenance

The files under `references/` are the original selected Python entry points and
implementation dependencies from the user-supplied `flashinfer-bench-collection-0706`
collection. Its `COLLECTION_META.json` attributes the collection to **kersor** and
identifies 15 Python tasks and 11 tasks whose optimized CUDA `solution.json` is absent.
The user requested that the usable task inputs be included in this GitHub repository.

These files are reference data for `known_kernel_reproduction`, not project-authored
Compiler code or instructions. They are preserved without rewriting their algorithms,
comments or entry-point dependencies. Task 010 includes the file actually loaded by
`submission.py`, `n1a1-num_stages4.py`; its unused alternative implementations are not
included. Tasks 008 and 009 are library wrappers, not hand-authored GPU kernels.

The supplied collection contains no separate redistribution-license document. This
repository does not assert an upstream license for these files or relicense them under
the project's Apache-2.0 grant. Existing file notices and source attribution remain in
force. No collection timing scores are imported or treated as measured evidence.

`catalog.json` accounts for all 26 collection tasks. Missing optimized references, the
h4096 epsilon conflict, and unavailable authoring routes are excluded from launch.
