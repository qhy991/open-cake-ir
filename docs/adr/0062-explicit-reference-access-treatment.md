# ADR 0062: Explicit per-arm reference access treatment

Status: proposed prospective Authoring Environment boundary.

## Decision

Each matched-search arm must declare exactly one `reference_access` category:
`clean_start`, `known_kernel_reproduction`, or `direct_low_level`. This is an Authoring
Environment treatment, independent of scientific versus artifact-only Claim Scope.
The declaration passes unchanged into the resolved arm in CampaignLock, its existing
arm identity, and the task's run authority. Missing or unknown declarations are refused
by the current Study and Lock readers.

The infrastructure template keeps its scientific matched-search scope and complete
Flash-KMeans references, explicitly declaring known-kernel reproduction for both arms.
Other complete-baseline templates, including native Triton/CuTe and the Metal task factory,
also declare known-kernel reproduction. The clean-start template declares clean synthesis
for Cake and direct low-level synthesis for CUDA. No historical frozen Study is relabeled.

## Controlled reference handoff

Reference access is determined by semantic artifact roles, not file extension. Workload
mathematics and its high-level oracle specification, hardware contracts and public
programming-interface documentation are allowed in restricted arms. Complete target-specific
Cake or Python Schedules are target references; Python is not a blanket exemption.

The initial restricted-reference domain is deliberately bounded. It admits the existing
reviewed operation-free Cake stub and empty CUDA stub, with the reviewed matched-search
instruction scaffold. These three assets are owned by the successor Executor closure.
JSON stubs are compared as parsed documents, then rendered from that document. CUDA and
instruction scaffolds are compared as complete bytes. External copies of those same vetted
assets are admitted through the same rule. A different or unreviewed target reference is
refused, even if it claims to be incomplete. An arbitrary Python source is not accepted as
a vetted stub merely because its parsed Schedule looks empty; raw comments or ignored
source could otherwise carry additional reference material.

Inherited native baselines generated from the Cake reference are target implementations
and require known-kernel reproduction. The inherited route cannot hide reference access
behind an indirect path. Restricted task packages omit complete Python examples and paired
implementation material while retaining mathematical/oracle and interface documentation.
Package sections expose their controlled semantic roles and declared access.

Preflight validates the policy. Package construction independently reapplies it and refuses
an arm argument different from the frozen assigned arm. Thus a manually constructed Lock
or direct package call cannot bypass the content rule. Existing reference identity and path
checks remain in force, including external paths; no new digest catalogue is introduced.

## Scope and limits

This is admission of explicitly controlled reference channels, not a universal text scan,
empty-function correctness proof, arbitrary-program noninterference theorem or filesystem
confinement claim. Workload mathematics/oracle correctness, Compiler/API documentation,
provider visibility and the original reviewed stub classification retain their own trusted
owners and acceptance gates. Additional safe stubs or reference channels require reviewed
source changes; an author-supplied role label cannot grant access to a target implementation.

Different access declarations change the existing arm and Campaign identities. Reports
project the resolved per-arm treatment, and the scientific Analysis Plan's existing
no-pooling-without-successor rule continues to apply. Declaring the same generic Claim Scope
does not make clean synthesis and known-kernel reproduction interchangeable populations.

## Revision and validation

The current contract requires these declarations at an Executor successor boundary.
Historical frozen Studies, Campaigns, provider records and results remain untouched and
are replayed with their pinned implementations. The earlier missing category establishes
an under-specified treatment; it does not prove a historical experiment was misused.

CPU tests cover missing/unknown declarations, known-kernel scientific infrastructure,
restricted math/API/stub packages, complete CUDA and Python refusal, external identical
stubs versus external full implementations, inherited native baselines, scaffold smuggling,
and locked-package/direct-call bypasses. No provider call, native compilation, GPU dispatch,
new qualification or scientific performance claim is involved.
