# Keep documents understandable and accurate

[中文原文](../../wiki/maintaining.md) · [English home](../README.md)

Use short sentences, small examples, and unchanged code identifiers. This guide explains existing authorities; it does not create another contract or release database.

[Guide index](README.md) · [Bilingual catalog](../../README.md)

| Information | Owner |
| --- | --- |
| Purpose and responsibilities | Architecture and Contexts |
| Formal vocabulary | The Chinese source Glossary; English is its translation |
| Mathematics, shapes, oracle, measurement | Workload and Study Contracts |
| IR fields and admitted forms | Authoring Contract, code, Corpus, and release |
| Current released identities | Generated status from locks and inventory |
| Experimental observations | External Evidence and the bound report |
| Durable rationale | ADRs |

Explain an operator's purpose and inputs/outputs, use a small numerical example, then link the complete contract or Schedule. Preserve rounding and state semantics. Dates in old tutorials are not current execution guarantees.

New primitives need their guide section; new Workload contracts need a catalog entry. Link and coverage checks cannot decide whether a human understands a paragraph, so review the wording too.

## Verification

The existing source guide lists these checks from the repository root:

```bash
.venv/bin/python -m unittest tests.contracts.test_documentation tests.contracts.test_cli
.venv/bin/python tools/render_current_status.py --check
git diff --check
```

They cover guide navigation, links, primitive/workload coverage, tutorial behavior, and CLI JSON/Chinese output. They do not prove GPU performance or replace the Compiler Corpus Gate. A changed executable example needs its own relevant verification, with new temporary outputs and clear expected rejections.

Ordinary explanations do not re-release the Compiler. Bound tools or runtime source use their successor path. Current status is regenerated rather than hand-edited. Documents travel with the repository rather than a separately copied website.

## Languages

The [catalog](../../README.md) pairs every original page with a Chinese and English reading route. Existing original paths remain stable. Missing Chinese explanations live under zh-CN; English counterparts to original Chinese guides live here under en. A translation references the source and does not independently update its experimental verdict. Historical reports keep original dates and identities. When a stable guide changes, update its paired reading page and check both link paths. This is a documentation workflow, not another release authority.
