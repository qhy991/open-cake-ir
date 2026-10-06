# Workload Contract ownership

`flash-kmeans-assign-v2.json` is the current Flash-KMeans authority because it adds the retained b32 materialization.
`flash-kmeans-assign.json` is the historical v1 input boundary required to replay r37/r39 identities; new Studies
must not reference it. `tinygemm2-stage4-v2.json` is the current TinyGEMM2 authority because it pins input, oracle
and parent-output bytes. `tinygemm2-stage4.json` is immutable historical v1 and must not be used by new Studies.

`qwen25-omni-audio-avg-pool1d-k2-s2-bf16.json` is the independent authority for the
even-frame Qwen2.5-Omni audio pooling slice. It fixes BF16 input/output, FP32 tap-ordered
accumulation, division by two, and one final BF16 round-to-nearest-ties-to-even conversion.
Odd input-frame counts and their tail policy are outside this contract.
