# Legacy source set

`source_set.json` declares the bounded legacy facts and behavior provenance admitted from
the clean final Stage 6 tree
`cake-repro@2fa79092c143fd8c2d9caa93fd84ad79a7504836`.

`legacy_manifest.jsonl` is generated from that declaration. Its first record binds the repository commit and tree;
each file record binds Git object ID, raw bytes and canonical JSON when applicable; each tree record binds a Git
tree object and recursive file count. Paths are locators, never identity.

The revision was initially observed 68 commits ahead of `origin/main`; the external push later completed and
`origin/main` now equals `2fa79092...`. The complete Git history is additionally retained at
`migration/bundles/cake-repro-final-2fa79092.bundle`, SHA256 `d826124b...ad9ac8`.

Current source-set SHA256: `d70bfafb747b389c73c5acf3fd63193b5c3be408086b0e8fcc8da3fc4a640042`.
Current 146-record manifest SHA256: `cc2592d8878ab2ac56a9a220c16b1b026d1b9c641e1ad755feeb2ad7f932ad7a`.

Dispositions mean:

- `translate_behavior`: create new canonical corpus data with parity evidence.
- `reimplement`: preserve observable behavior behind the new Module Interface.
- `distill_contract`: separate canonical facts into Workload or Study contracts.
- `reference_fixture`: retain exact historical bytes for replay without copying runtime topology.
- `translate_tests`: rewrite as public-Interface contract tests.
- `reference_only`: retain provenance in pinned Git; never import into active source.

The manifest contains no secret bytes and does not authorize deletion of legacy state.
