# AKA IR-gap semantic clustering proposal

Source: `gap-cluster-input.json`, the embedded compact projection of 326 verifier-accepted `ir_gap` cases grouped under 296 exact candidate names.

The proposed semantics-preserving partition contains 202 clusters. It has 51 repeated-candidate clusters covering 174 cases, 150 singleton-or-distinct clusters covering 150 cases, and one `conflicted_needs_review` cluster covering 2 cases. The conflict is the indivisible `segmented_scan` exact-name group: its two members assign opposite inclusion behavior to a marked boundary in a reverse scan.

The largest repeated semantic clusters are:

- elementwise FP32 fused multiply-add: 12 cases / 6 exact names;
- typed runtime scalar parameters: 9 cases / 6 exact names;
- computed FP32 atomic add to caller-owned state: 7 cases / 5 exact names;
- INT32-indexed FP32 atomic scatter-add: 7 cases / 4 exact names;
- segmented indirect FP32 sum: 7 cases / 6 exact names;
- zero-initialized INT32-mapped FP32 scatter reduction: 6 cases / 5 exact names;
- unique INT64 scatter-store into FP32 state: 6 cases / 6 exact names; and
- elementwise FP32 natural logarithm: 6 cases / 2 exact names.

The partition lists all 326 unique case IDs and all 296 unique exact names exactly once; `unclustered_case_ids` and `duplicate_case_ids` are empty. Merges preserve material distinctions such as dtype and index width, fused or ordered rounding, reduction identity and tie behavior, atomic versus non-atomic effects, output initialization versus caller-state preservation, aliasing/uniqueness, scalar ABI, and invalid-input or masked-memory behavior.

This dataset is proposal-only evidence. No new IR primitive is approved, implemented, released, GPU-tested, or performance-qualified by this clustering. Per-item reviewer conclusions remain evidence, and the semantic merges do not grant Compiler, lowering, release, or performance authority.
