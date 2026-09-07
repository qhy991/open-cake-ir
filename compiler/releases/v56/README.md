# Compiler v56 identity reservation

The four release authority files in this directory are unchanged copies from
`bfc446e384b87634daf5a289a35bc1e36bbad9ff` on
`codex/b300-support-20260907`. That commit owns the complete source tree for replay.
This directory reserves its already released Compiler identity under ADR 0050.

The AMD successor does not adopt the B300 implementation or its evidence. Load this
release only with its original source commit. The accompanying B200 Executor identity
is retained unchanged at
[`runtime/executors/open-cake-ir-b200-v52.json`](../../../runtime/executors/open-cake-ir-b200-v52.json)
for the same reason; it does not describe the current AMD runtime sources.
