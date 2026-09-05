# ADR 0051: load values follow the access domain

Status: accepted under the current task's delegated repository authority, 2026-09-06.

The v41 FMA parent re-audit exposed a scalar load declared as a vector register result.
At main `e4cf49e` the public Compiler still accepted/lowered a scalar `a[batch,batch]`
as a [128] FMA operand. Three adjacent probes also reproduced accepted/lowered
`tl.arange(0,3)`, an out-of-rank dimension raising IndexError, and a vector defined
from an eight-element dimension addressing a four-element dimension without refusal.

The existing AccessMap verifier now checks the rank of direct global-to-register
loads, including the canonical [1] representation of an all-scalar address. Existing
same-rank extent and zipped runtime-index checks remain their owners. Unknown
references remain localized Findings rather than exceptions. Dimension coordinates
must fit the dimension actually addressed. No splat, reshape, layout or new scalar
primitive is introduced; subsequent analysis can continue using the validated Buffer
shapes.

The Triton preflight owns its generated arange limits for program tiles, loop tiles
and dimension walks. The installed Triton 3.7.1 semantic implementation checks the
span, not whether each endpoint itself is a power of two. The check preserves legal
nonzero-start spans and masked power-of-two tiles over non-power-of-two global sizes.
The [Triton arange reference](https://triton-lang.org/main/python-api/generated/triton.language.arange.html)
also documents the maximum tensor extent. Existing top-k/index-expand constraints
remain with their operation checks; this is not a universal compilation guarantee.

The unchanged 72 Corpus entries plus one scalar-load control and four reproduced
negative cases gate the successor. No old expectation is regenerated to hide a change.
The 12-parent report remains historical v41 evidence: current repair checks cannot be
used to rewrite its earlier dispositions or claim GPU correctness.
