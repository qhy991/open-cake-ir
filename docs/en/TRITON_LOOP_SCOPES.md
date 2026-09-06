# Triton TileLoop scopes

The Compiler owns this source-level lowering contract. Existing TileLoop bodies remain
the only control-flow representation. R1 admits zero or one loop and a static, two-deep
loop nest: an outer output tile may contain an inner sum/max reduction or K contraction,
then consume its finalized result and store the outer tile. Root and outer invariant
operations execute once at their declared scope. No primitive or authoring mode is added.

Each operation belongs to one lexical scope. Expanding child bodies must preserve the
contiguous operation declaration order. Loop coordinates are visible only inside that
loop and its descendants. A carried result survives its immediate loop; it does not
automatically survive an enclosing output loop. Reduction identities and MMA zero state
are allocated immediately before their owning loop, with the declared result shape.

The bounded nested slice excludes sibling loop trees, deeper nests, dynamic stops,
flattening, warp specialization, nested argmin/top-k/online-softmax, and MMA that would
carry over an ancestor of its direct loop. Nested MMA requires two directly loaded
operands, because the existing access-map query proves K accumulation only for those
producers. Other producers receive a localized refusal. Each nested `tl.range` retains
its own scheduling options; there is no shared launch-wide stage override. These
restrictions receive localized preflight Findings.
In-loop scans remain explicitly refused: the existing primitive has no cross-tile
prefix carry. Ordinary in-loop stores are admitted only for output buffers with affine
coordinates that uniquely cover every active loop and program axis. Indexed and state
stores retain their existing restrictions. No role, TMA or TMEM-transfer capability is
added.

Acceptance is CPU contract verification and execution of generated source with a small
test-only tensor interpreter against independent scalar references, plus unchanged
single-loop source behavior. This is not Triton compilation, GPU correctness, timing,
profiling or target-framework acceptance; those remain R2 evidence. Integration owns
Corpus adoption, the complete Corpus Gate and independently approved Compiler release.
