# ADR 0059: Final release identities bind complete authority

Status: accepted design after independent review, 2026-09-09; implementation and
release remain test-gated. Extends [0049](0049-released-executor-descriptors-reserve-their-identities.md)
and [0050](0050-released-compiler-locks-reserve-their-identities.md).

## Problem

Two worktrees can independently choose the same next ordinal. A checked-in ordinal
reservation file is not an atomic reservation across clones. Source identity alone
also does not distinguish two independent approvals of the same Compiler source.

The following pre-existing aliases remain immutable:

- Executor v48: the ordinary descriptor and `-r2`.
- Executor v55: the ordinary descriptor and `-paired`.
- Executor v63 and v65: the ordinary descriptors and their `-metal` aliases.
- Executor v64: the ordinary descriptor, `-lab-maintainability`, and `-metal`.
- Compiler v62: `compiler/releases/v62/` and `compiler/releases/v62-metal/`.

Readers of these releases must retain their exact descriptor or lock reference.
This decision does not rename, delete, reinterpret or reuse historical identities.

## Decision

The Compiler cycle still prepares a stable `vNN-draft`. Review and the Corpus Gate
bind that draft. After approval, the release builder derives a final id of the form
`vNN+<full-authority-identity>` from the canonical identities of the exact Gate and
approval. The final id is not an input to either authority, so this has no circular
dependency. Different approvals of the same source produce distinct final releases.
An identical complete authority produces the same final id.

The Executor builder derives the same suffix form from its complete source records
and host environment, excluding the draft id and state. The cycle reads the derived
id before installing the create-only descriptor and updating inventory. It refuses
a second descriptor with an already published final id even at another filename.

The suffix uses the full content identity, not a short display prefix. It reuses the
existing release boundary's source and authority records; no separate digest catalog
or source-only proof is introduced. Human-readable ordinals remain ordering aids.
New allocation recognizes both historical plain ordinals and suffixed identities.

An unchanged, verified Compiler or Executor release is a no-op. A changed release
gets a successor while preserving previous authorities. Failed approval still
preserves the current lock and approval. Draft preparation cannot choose a final
Compiler identity or write its approval.

## Acceptance

Use the public builders and cycles to show that distinct complete authorities from
the same ordinal produce distinct final ids; the same authority is deterministic;
an existing final identity cannot acquire a second filename; and unchanged release
verification does not advance inventory. Cover legacy ordinal reservation, repeated
Compiler preparation, independent approval, and source changes after publication.
Historical aliases stay intact. No host qualification or GPU result is inferred from
these CPU release-protocol fixtures.
