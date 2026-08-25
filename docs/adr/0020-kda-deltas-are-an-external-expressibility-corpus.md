# ADR 0020: KDA deltas are an external expressibility corpus

Status: accepted, 2026-08-24.

## Outcome and non-goals

The 57-version B200 MoE KDA history is an external, chronological expressibility
corpus. It asks two different questions:

1. can Cake name the mechanism changed by one adjacent version; and
2. can Cake describe the complete kernel or program at that version?

The first may pass while the second fails. At this snapshot no complete KDA version is
expressible, because even v1 needs grouped/ragged GEMM, indexed top-k routing,
quantization-scale relations and scatter semantics that the Schedule vocabulary does not
yet own. The honest complete-version result is therefore 0/57; matching a later tuning
knob must not be reported as version coverage.

This decision does not import the KDA archive, create 57 profiles or modes, duplicate its
measurement ledger, or turn one historical use case into a vocabulary entry. Evidence
about correctness and speed remains owned by KDA. Cake retains only this distilled design
decision and its own independently failing Corpus cases.

## Snapshot and authority

The reviewed snapshot is the local `kda15-three-task-repro/experiment` MoE history:

- version range: v1--v57; 56 judge passes and one v3 failure;
- manifest: `3da671bfa6727eff7b8fb083834a6ee28b3ced66ed6886a54ccf1889fa42e91e`;
- ledger: `14badd410bdd267df7efa68c01c047f3d53168c0679e600754474d85ee499672`;
- analysis: `24c97b6d5e0c8e91d094a12ebbd0573c07d88e57ddfb2f1a3e82963b6cb2296b`;
- version history: `59d21a5c151735bc1a4f87f327491a9e801324a133ba7c5af8035dfc3ae23bdd`.

Those hashes identify the source of this derivation; they do not make the external files
Compiler inputs. New KDA versions require a successor review, not an edit that silently
widens this snapshot.

## Minimal ownership model

| Mechanism observed in KDA | Canonical owner in Cake |
| --- | --- |
| arithmetic, operation dependencies, per-kernel placement and resource commitments | one `Schedule` |
| exact-shape specialist selection | portfolio/dispatch above `Schedule` |
| CUDA Graph, stream ordering and PDL edges between kernels | program composition above `Schedule` |
| correctness, paired timing, NCU and IKET outcomes | external KDA evidence |

This keeps one Schedule equal to one kernel. PDL is not a boolean Schedule knob: v28--v30,
v50--v52 and v56 use it on graph edges between producer and consumer kernels. Shape-only
changes such as v5, v11, v23, v26--v27, v32--v33, v55 and v57 are portfolio evidence, not
new IR operations.

The remaining adjacent deltas group into a small number of mechanisms rather than
version-specific names:

- persistent work acquisition and CLC filtering: v6--v10, v19, v25, v34--v35;
- unary math and reduction implementations: v12--v13, v21--v22, v47--v48;
- DSMEM, fusion and scatter ownership: v24, v40--v44, v49, v53--v54;
- operand reuse and cache intent: v36--v39;
- pipeline-resource lifetime: v46;
- graph capture and inter-kernel PDL: v2--v4, v28--v30, v50--v52, v56.

## Smallest complete slice

The first admitted delta is `ElementwiseOp.TANH`. The exact v11->v12, v12->v13 and
v20->v21 changes independently reuse the same unary primitive for SwiGLU and routing.
One generic operation removes more duplication than it adds; a `kda_v12` mode or a
`swiglu_tanh_formula` would merely encode a finished use case.

The acceptance slice is deliberately narrower than MoE: a standalone SwiGLU Schedule
must parse, pass the Corpus gate, lower through the generic Triton elementwise emitter and
match an independent B200 oracle. A shape-drift companion must fail. This proves the
primitive and its authoring path, not KDA performance or complete-version coverage. KDA
uses `tanh(..., approx=True)`; this slice names the tanh identity but does not yet expose
an accuracy/approximation commitment, so it also does not claim instruction-equivalent
reproduction of v12.

ADR 0021 closes the first prerequisite as a standalone deterministic indexed `top_k`
primitive, including a shape-drift falsifier and B200 correctness observation. That does
not make the v1 chain representable: group formation, selection masks and score updates
still have no composition path. Remaining prerequisites are quantization-scale relations,
routing/group formation, valid-tile work acquisition, and atomic/direct scatter with
DSMEM. ADR 0025 shows that the grouped/ragged contraction itself composes from existing
primitives, so it does not add a grouped-GEMM operation. Only after a baseline
multi-kernel program is representable is a program-DAG/PDL authority justified.
